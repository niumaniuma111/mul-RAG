"""
Celery worker 用的「进程级常驻事件循环」。

为什么必须有这个模块
-------------------
Celery 的 prefork worker 里，任务是**同步函数**，而项目其余部分全是 async
（SQLAlchemy asyncpg 引擎、httpx.AsyncClient、qdrant-client）。

最直觉的写法是每次任务 `asyncio.run(coro)` —— 但那会**每次新建一个事件循环**，
而下面这些模块级单例在第一次使用时就把自己绑到了当时那个循环上：

    app/db/session.py       AsyncEngine + asyncpg 连接池
    app/services/model_client.py   httpx.AsyncClient（内含连接池）
    app/tasks/ingest_tasks.py      QdrantClient（gRPC channel）

第二个任务跑在**新循环**里、却复用了**绑在旧循环**的连接 → 典型报错：

    RuntimeError: Task got Future attached to a different loop
    RuntimeError: Event loop is closed

这类故障的恶劣之处在于**第一个任务永远成功**，从第二个才开始炸 ——
非常容易被误判成"偶发"。

解决办法：每个 worker 进程持有**一个**常驻循环，
`run_until_complete` 反复复用它。`worker_max_tasks_per_child=20`
导致进程回收时，新进程会自然建一个新循环，符合预期。

对应设计文档：DESIGN.md §7.2
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    """取（或建）本进程的常驻事件循环。"""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)
            logger.debug("已为本进程创建常驻事件循环")
        return _loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    在常驻循环上跑一个协程并返回结果。

    ⚠️ 只能在**同步上下文**里调用（Celery 任务、脚本入口）。
       如果当前线程已经有正在运行的循环（例如从 FastAPI 请求处理里调），
       直接 run_until_complete 会抛 `This event loop is already running`。
       这种情况属于误用，这里显式报错而不是静默死锁 —— 死锁排查起来要痛苦得多。
    """
    loop = get_loop()
    if loop.is_running():
        raise RuntimeError(
            "run_async() 不能在已有事件循环运行中被调用。"
            "在 FastAPI 异步端点里请直接 await 协程，不要走这个函数。"
        )
    return loop.run_until_complete(coro)


def shutdown_loop() -> None:
    """关闭本进程的常驻循环（worker 进程退出时调）。"""
    global _loop
    with _loop_lock:
        if _loop is not None and not _loop.is_closed():
            _loop.close()
            logger.debug("常驻事件循环已关闭")
        _loop = None


__all__ = ["get_loop", "run_async", "shutdown_loop"]
