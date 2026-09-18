"""
SSE 事件总线 —— LangGraph 节点与 SSE 响应流之间的桥。

为什么需要它
--------
    qa.py 的 SSE 响应是「一边跑图一边推事件」，但图在
    `graph.ainvoke()` 里整体执行 —— 节点无法"返回"中间事件。
    所以节点在关键时刻（token / hitl / 阶段切换）publish 到这里，
    SSE 生成器 subscribe 同一个队列逐条转发。
    这是典型的"生产者-消费者解耦"：节点只管推，不管谁在听。

会话生命周期
--------
    subscribe() 先于 ainvoke() 建立，ainvoke 结束后由 qa.py 调
    close_session() 清理。队列有上限：慢客户端 + 长答案时丢最老的
    token（保序不保量），避免无界队列吃光内存。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# 单会话队列上限。一条 token 事件 ~几十字节，1024 条足够容纳
# 最长答案（1024 token / token_flush_size=2 ≈ 512 条）还留一倍余量。
_MAX_QUEUE = 1024

_sessions: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def subscribe(session_id: str) -> asyncio.Queue[dict[str, Any]]:
    """为会话建立事件队列（qa.py 在启动图之前调用）。"""
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE)
    _sessions[session_id] = q
    return q


def close_session(session_id: str) -> None:
    """会话结束/断连后清理（防泄漏：每个 SSE 连接一个队列）。"""
    _sessions.pop(session_id, None)


def publish(session_id: str, event: str, data: dict[str, Any]) -> None:
    """
    推一条事件（图节点调用，非阻塞）。

    ⚠️ 队列满时丢**最老的** token（get_nowait 腾位）——
       SSE 是"最新进度最有价值"的语义；断连/超时的会话直接丢弃事件。
    """
    q = _sessions.get(session_id)
    if q is None:
        return                      # 会话已断开：静默丢弃，节点不该因此失败
    item = {"event": event, "data": data}
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            q.get_nowait()          # 丢最老
            q.put_nowait(item)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass


def has_subscriber(session_id: str) -> bool:
    return session_id in _sessions
