"""
Redis 连接（redis.asyncio）。

同一个 Redis 实例承担四类用途，靠不同的 db index 隔离
（db 划分见 configs/config.yaml 的 redis 段）：

    db0  broker   Celery 任务队列
    db1  lock     GPU 串行锁
    db2  cache    语义缓存 / active_version 缓存
    db3  result   Celery 结果后端

分 db 不是为了性能（单线程 Redis 里 db 只是命名空间），而是为了：
    1. 排障时 `SELECT 1` 只看锁，不会被几万个缓存键淹没；
    2. 锁键绝不能被 cache 的 `FLUSHDB` 顺手清掉 —— 那会造成双占卡。
"""

from __future__ import annotations

import logging

from redis.asyncio import Redis, from_url

from app.core.config import get_config, get_env

logger = logging.getLogger(__name__)

_redis: Redis | None = None
_lock_redis: Redis | None = None


def _build(db: int | None = None) -> Redis:
    return from_url(
        get_env().redis_url,
        db=db,
        encoding="utf-8",
        decode_responses=True,      # 直接拿 str，省掉到处 .decode()
        health_check_interval=30,   # 长连接保活，防被中间设备静默掐断
        socket_keepalive=True,
    )


def get_redis() -> Redis:
    """通用连接（缓存 / 队列 / 结果），db 由 REDIS_URL 决定（默认 0）。"""
    global _redis
    if _redis is None:
        _redis = _build()
    return _redis


def get_lock_redis() -> Redis:
    """
    GPU 锁专用连接，固定指向 redis.lock_db（默认 1）。

    单独开一个连接池而不是复用 get_redis()，是为了避免锁键和缓存键
    落在同一个 db 里 —— 见模块头的第 2 条理由。
    """
    global _lock_redis
    if _lock_redis is None:
        lock_db = int(get_config().get_path("redis.lock_db", 1))
        _lock_redis = _build(db=lock_db)
    return _lock_redis


async def ping() -> bool:
    """供 /health/deep 使用。"""
    try:
        return bool(await get_redis().ping())
    except Exception:  # noqa: BLE001
        return False


async def close_redis() -> None:
    """应用关闭时释放连接池。"""
    global _redis, _lock_redis
    for client in (_redis, _lock_redis):
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("关闭 Redis 连接池失败", exc_info=True)
    _redis = None
    _lock_redis = None


__all__ = ["close_redis", "get_lock_redis", "get_redis", "ping"]
