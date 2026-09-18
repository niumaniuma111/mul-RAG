"""
数据库连接（SQLAlchemy 2.x 异步引擎）。

⚠️ 本项目有两套 PG 连接串，用途不同，绝不能混用：

    DATABASE_URL              postgresql+asyncpg://...   业务库，SQLAlchemy 异步引擎
    CHECKPOINT_DATABASE_URL   postgresql://...           LangGraph AsyncPostgresSaver

LangGraph 的 checkpointer 走 psycopg3（libpq 协议），它不认 `+asyncpg` 方言前缀。
如果顺手把 DATABASE_URL 填给它，启动时会报 "unknown dialect" 或连接挂起。
反过来，asyncpg 也不认 psycopg3 的连接串格式。

池参数来自 configs/config.yaml 的 postgres 段。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_config, get_env

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """进程内单例引擎。"""
    global _engine
    if _engine is None:
        pg = get_config().get_path("postgres", {}) or {}
        _engine = create_async_engine(
            get_env().database_url,
            pool_size=int(pg.get("pool_size", 10)),
            max_overflow=int(pg.get("max_overflow", 20)),
            pool_timeout=int(pg.get("pool_timeout", 30)),
            # 连接被 PG 侧掐断后（重启 / idle timeout）自动重建，
            # 否则第一次查询会抛 "connection was closed"。
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=bool(pg.get("echo", False)),
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,   # commit 后仍可读对象属性，避免 DetachedInstanceError
            autoflush=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI 依赖：每个请求一个 session。

    出错自动回滚，正常路径自动提交 —— 调用方不用自己管事务边界。
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ping() -> bool:
    """供 /health/deep 使用。"""
    from sqlalchemy import text

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 —— 健康检查不该因依赖故障而抛异常
        return False


async def dispose_engine() -> None:
    """应用关闭时释放连接池。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


__all__ = [
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_session_factory",
    "ping",
]
