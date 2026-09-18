"""
LangGraph Checkpoint 持久化（AsyncPostgresSaver）。

为什么必须是 Postgres 而不是 MemorySaver（DESIGN §8.7）
--------
    1. HITL interrupt 会把图**挂起**并等待用户操作（可能几分钟、
       甚至等用户第二天再回来看）—— 进程内存里的状态撑不过 api 重启。
    2. thread_id = session_id：同一会话多轮对话的上下文天然延续，
       应用重启后用户还能接着问。

表管理
--------
    `setup_on_start: true` 时首次启动自动建 checkpoints 等 3 张表。
    表会随时间增长，30 天前的 thread 由 scripts/cleanup_checkpoints.py 清理。

对应设计文档：DESIGN.md §8.7
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_config, get_env
from app.core.logging import get_logger

logger = get_logger(__name__)

_saver: Any = None
_pool: Any = None


def _postgres_dsn() -> str:
    """
    组装 psycopg 用的 DSN。

    ⚠️ LangGraph 的 AsyncPostgresSaver 用 psycopg3，而业务库用 asyncpg ——
       两套驱动并存是有意的：checkpointer 是 LangGraph 的领地，
       不与 SQLAlchemy 的连接池混用，避免事务语义互相干扰。
       EnvSettings.checkpoint_database_url 就是为此预留的
       （裸 postgresql:// 前缀，与 database_url 的 +asyncpg 区分）。
    """
    return get_env().checkpoint_database_url


async def init_checkpointer() -> Any:
    """
    创建（模块级单例的）AsyncPostgresSaver。

    ⚠️ 为什么不用 `AsyncPostgresSaver.from_conn_string()`：
       它是 async context manager（async with 语法糖）。如果我们只取
       `__aenter__()` 的返回值而把上下文对象丢掉，GC 回收这个 async
       生成器时会向它抛 GeneratorExit → 触发 `async with conn` 的
       清理路径 → **连接被静默关闭**。表现为：图跑到第一次 checkpoint
       put 时报 `OperationalError: the connection is closed`。
       （2026-09-18 整图探针实测踩到。）

       所以这里手动 connect，连接的生命周期由本模块显式管理
       （close_checkpointer() 里关闭），与容器生命周期一致。
    """
    global _saver
    if _saver is not None:
        return _saver

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    dsn = _postgres_dsn()

    # 用连接池而不是单连接：LangGraph 每个 put/get 都借还连接，
    # 单连接在并发 ainvoke（多会话同时问答）时会互相踩。
    # min_size=1 起步即可 —— 问答并发量个位数，池的意义是复用而非吞吐。
    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=5,
        open=False,                      # 手动 open（ psycopg_pool 3.x 要求显式）
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()

    saver = AsyncPostgresSaver(pool)

    cfg = get_config()
    if bool(cfg.get_path("agent.checkpoint.setup_on_start", True)):
        await saver.setup()          # 首次自动建 checkpoints 等 3 张表
        logger.info("LangGraph checkpoint 表已就绪（setup 完成）")

    _saver = saver
    _pool = pool
    return _saver


async def close_checkpointer() -> None:
    global _saver, _pool
    if _saver is not None:
        _saver = None
    if _pool is not None:
        try:
            await _pool.close()
        except Exception as exc:  # noqa: BLE001 —— 退出路径尽力而为
            logger.warning("关闭 checkpoint 连接池异常（忽略）: %s", exc)
        _pool = None
