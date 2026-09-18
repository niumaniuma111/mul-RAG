"""
跨容器 GPU 串行锁（Redis 实现）。

什么时候真正用到
----------------
只有在 MinerU 走 GPU（`ingest.mineru.device: cuda`）时才参与主链路 ——
那时 MinerU 容器也在吃显存，进程内的 asyncio.Lock 管不到它，必须跨容器互斥。

CPU 模式下锁层级是 inproc（见 lock_factory），本模块退居兜底，
但保留完整实现：一是 `lock_backend` 可以显式指定 redis，二是将来若加了
第二个吃显存的容器，不用重写。

设计要点
--------
1. 阻塞式获取，等待上限 acquire_timeout_sec，超时抛 GpuBusyError（不是无限等）。
2. 锁自动过期（TTL），防止进程崩溃后留下永久死锁。
3. 持有期间定期续约，覆盖长任务（GPU 模式解析可达 30 分钟）。
4. 释放与续约都用 Lua 脚本做 token 校验，避免「把别人的锁释放掉」——
   这是分布式锁最经典的一个 bug：A 的任务超时锁自动过期，B 拿到锁，
   此时 A 执行释放，就把 B 的锁删了。

对应设计文档：DESIGN.md §10.2
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class GpuBusyError(RuntimeError):
    """等待 GPU 锁超时。上层可转成 HTTP 3002（GPU 繁忙）。"""


# Lua：只有持有者本人才能释放 / 续约
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""


class RedisGpuLock:
    """
    跨容器 GPU 串行锁。

    用法（异步上下文管理器）::

        async with RedisGpuLock(redis, key="lock:gpu:heavy", ttl=600):
            ...  # 独占 GPU 的推理或解析

    :param redis: 已连接的 redis.asyncio 客户端
    :param key: 锁键名，所有吃显存的推理/解析都用同一个 key
    :param ttl: 锁自动过期秒数。GPU 模式下解析是长任务，必须放大到 1800
    :param acquire_timeout: 最长等待秒数
    :param renew_interval: 续约间隔秒数，应显著小于 ttl
    """

    def __init__(
        self,
        redis: Redis,
        key: str = "lock:gpu:heavy",
        ttl: int = 600,
        acquire_timeout: int = 180,
        renew_interval: int = 30,
    ) -> None:
        self.redis = redis
        self.key = key
        self.ttl = int(ttl)
        self.acquire_timeout = int(acquire_timeout)
        # 续约间隔至少要比 TTL 小一半，否则来不及续就过期了
        self.renew_interval = max(1, min(int(renew_interval), self.ttl // 2))

        self.token: str | None = None
        self._renew_task: asyncio.Task[None] | None = None
        self._wait_seconds: float = 0.0

    # ---------------- 上下文管理器 ----------------
    async def __aenter__(self) -> "RedisGpuLock":
        token = uuid4().hex
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.acquire_timeout
        waited_from = loop.time()

        while True:
            acquired = await self.redis.set(self.key, token, nx=True, ex=self.ttl)
            if acquired:
                self.token = token
                self._wait_seconds = loop.time() - waited_from
                if self._wait_seconds > 1:
                    logger.info(
                        "获取 GPU 锁成功（等了 %.1fs）: key=%s", self._wait_seconds, self.key
                    )
                self._renew_task = asyncio.create_task(self._auto_renew())
                return self

            if loop.time() >= deadline:
                raise GpuBusyError(
                    f"GPU 忙，等待超过 {self.acquire_timeout}s（key={self.key}）。"
                    f"可能是有解析任务在跑，或上一个持锁进程异常退出后锁尚未过期。"
                )
            await asyncio.sleep(0.5)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self._stop_renew()
        if self.token is None:
            return
        try:
            await self.redis.eval(_RELEASE_SCRIPT, 1, self.key, self.token)
        except Exception:  # noqa: BLE001 —— 释放失败不该掩盖业务异常
            logger.warning("释放 GPU 锁失败（key=%s），将由 TTL 自动过期", self.key, exc_info=True)
        finally:
            self.token = None

    # ---------------- 续约 ----------------
    async def _auto_renew(self) -> None:
        """持有期间周期性把 TTL 续回满值。"""
        try:
            while True:
                await asyncio.sleep(self.renew_interval)
                if self.token is None:
                    return
                renewed = await self.redis.eval(_RENEW_SCRIPT, 1, self.key, self.token, self.ttl)
                if not renewed:
                    logger.warning(
                        "GPU 锁续约失败（key=%s）：锁已不在自己名下，可能已过期。"
                        "后续操作不再受互斥保护，请检查是否有任务卡死。",
                        self.key,
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("GPU 锁续约任务异常退出（key=%s）", self.key, exc_info=True)

    async def _stop_renew(self) -> None:
        if self._renew_task is None:
            return
        self._renew_task.cancel()
        try:
            await self._renew_task
        except (asyncio.CancelledError, Exception):  # noqa: B014
            pass
        self._renew_task = None

    # ---------------- 观测 ----------------
    @property
    def wait_seconds(self) -> float:
        """本次获取锁等待了多久（已获取时才有意义）。"""
        return self._wait_seconds

    async def current_holder(self) -> str | None:
        """当前持锁者 token，用于排障。"""
        return await self.redis.get(self.key)


__all__ = ["GpuBusyError", "RedisGpuLock"]
