"""
FastAPI 应用入口。

启动顺序（lifespan），顺序本身是有意义的：

    1. 初始化日志        —— 必须先做，后面每一步都要能打日志
    2. ★ 锁层级自检       —— fail-fast，不匹配直接退出（见下）
    3. 建立 Redis 连接    —— 主动建连，让配置错误在启动时暴露，而不是第一个请求时
    4. 挂路由与异常处理器
    5. 关闭时释放连接池

为什么锁自检放在第 2 位而不是更后面：
    它是全项目唯一一个「配错了不会立刻报错，而是几小时后以随机 OOM 的形式爆炸」
    的检查项。随机 OOM 的排查成本极高（日志里看不出是谁占的显存），
    所以宁可启动失败，也要把它挡在门口。详见 app/core/lock_factory.py。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.router import api_router
from app.core.config import get_config, get_env
from app.core.exceptions import register_exception_handlers
from app.core.lock_factory import LockBackendMismatchError, preflight
from app.core.logging import get_logger, setup_logging
from app.core.redis_client import close_redis, get_redis
from app.db.session import dispose_engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # ---------- 1. 日志 ----------
    setup_logging()
    cfg = get_config()
    env = get_env()
    logger.info("启动 %s v%s (env=%s, log_level=%s)", cfg.app.name, __version__, cfg.app.env, env.log_level)

    # ---------- 2. ★ 锁层级自检（fail-fast）----------
    try:
        plan = preflight()
    except LockBackendMismatchError:
        logger.critical("锁层级与 MinerU 设备不匹配，拒绝启动。修复方式见上方异常信息。")
        raise
    logger.info("锁自检通过: backend=%s device=%s", plan.backend, plan.device)

    # ---------- 3. 连接池 ----------
    await get_redis()
    logger.info("Redis 连接就绪")

    logger.info("应用启动完成，监听中")
    yield

    # ---------- 5. 收尾 ----------
    logger.info("正在关闭连接池 ...")
    await close_redis()
    await dispose_engine()
    logger.info("已退出")


def create_app() -> FastAPI:
    """工厂函数（测试时可以造多个互不干扰的实例）。"""
    # 建 app 之前就要能用日志，否则下面的配置读取报错会没法看
    setup_logging()

    cfg = get_config()
    api_prefix = str(cfg.get_path("app.api_prefix", "/api/v1"))

    app = FastAPI(
        title="ppt-agentic-rag API",
        description=(
            "基于 LangGraph 的多模态 PPT Agentic RAG 系统 —— "
            "离线入库（解析 → 切分 → 向量化）与在线问答（自省驱动的迭代检索）。"
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # 前端在 8080（nginx）或 5173（vite dev）访问本服务，需要跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in (cfg.get_path("app.cors_origins", []) or [])],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # SSE 场景下浏览器需要读到这些头
        expose_headers=["X-Request-Id"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=api_prefix)

    return app


app = create_app()


if __name__ == "__main__":
    # 本地直接 `python -m app.main` 调试用；生产走 uvicorn 命令
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
