"""
在线问答 API —— SSE 流式问答 + HITL 断点恢复。

SSE 事件协议（DESIGN §12）
--------
    event: session   data: {"session_id": "..."}          首帧，会话标识
    event: stage     data: {"node": "decide"}             节点推进（前端时间线）
    event: token     data: {"text": "..."}                生成 token（攒批推送）
    event: hitl      data: {reason, max_score, ...}       挂起等用户决策
    event: reflect   data: {round, verdict, reason}       自省进度
    event: final     data: {answer, citations, ...}       终帧（答案 + 引用）
    event: error     data: {"message": "..."}             异常终帧

两条流的分工
--------
    ask    全新提问：建 session → astream 跑完整图
    resume HITL 恢复：Command(resume=...) 唤醒挂起的 hitl 节点 → 续跑

    图的执行放在 producer task（astream），节点内的 token/reflect/final
    经 sse_broker 进入本端点的 SSE 生成器；astream 的节点完成事件
    在 producer 里转成 stage 事件也推进 broker —— 单队列单通道，
    顺序天然一致。

对应设计文档：DESIGN.md §8.6 / §12
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agent.graph import get_graph
from app.core.config import get_config
from app.core.deps import get_current_user
from app.core.logging import get_logger
from app.services import sse_broker

logger = get_logger(__name__)

router = APIRouter(prefix="/qa", tags=["qa"])


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None            # 不传 = 新会话
    doc_ids: list[str] | None = None         # 限定检索范围
    version_override: dict[str, int] | None = None   # 历史版本问答
    history: list[dict[str, str]] | None = None      # 最近几轮对话


class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    action: str = Field(..., pattern="^(rewrite_query|use_current_context)$")
    new_query: str | None = Field(None, max_length=2000)


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _ensure_session(session_id: str, user_id: int, title: str) -> None:
    """
    幂等建会话行。

    为什么必须有这一步：`chat_messages.session_id` 是指向 `chat_sessions.id`
    的外键，finalize 落库时**没有这行就整条事务回滚** —— 表象是"问答落库失败
    但不影响回答"的 WARNING，实际连 `qa_traces` 也一起没写进去（同一事务），
    观测数据全丢。

    HITL 探针一类非 UUID 的 thread_id 直接跳过（与 finalize 的判据一致）。
    title 取首问前 50 字，纯展示用途。
    """
    try:
        uuid.UUID(session_id)
    except ValueError:
        return

    try:
        from sqlalchemy import text

        from app.db.session import get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO chat_sessions (id, user_id, title, thread_id)
                    VALUES (CAST(:sid AS uuid), :uid, :title, :thread)
                    ON CONFLICT (thread_id)
                    DO UPDATE SET updated_at = now()
                    """
                ),
                {
                    # ⚠️ id 与 thread_id 值相同但**必须是两个独立参数名**：
                    #    asyncpg 对同一个占位符只推断一种类型，而这里是
                    #    uuid 列 vs varchar 列 → 共用 :sid 会报
                    #    AmbiguousParameterError: inconsistent types deduced
                    #    for parameter $1 (uuid versus character varying)。
                    "sid": session_id,
                    "thread": session_id,
                    "uid": user_id,
                    "title": (title or "新对话")[:50],
                },
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 —— 建会话失败不该拦住问答
        logger.warning("建会话失败（继续问答）: %s", exc)


def _run_graph_producer(session_id: str, graph_input: Any) -> asyncio.Task:
    """
    后台跑图，把节点推进 / interrupt 转成 broker 事件。

    producer 与 SSE 生成器解耦的关键：SSE 断连时 producer 不必立刻死
    （checkpoint 已持久化，损失可控）；producer 异常时经 error 事件送达。
    """

    async def producer() -> None:
        try:
            graph = await get_graph()
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": int(get_config().get_path("agent.recursion_limit", 30)),
            }
            async for updates in graph.astream(graph_input, config=config):
                for node, update in updates.items():
                    if node == "__interrupt__":
                        # interrupt 的 update 是 Interrupt 元组，value 即 hitl payload
                        for intr in update or ():
                            payload = getattr(intr, "value", None) or {}
                            sse_broker.publish(session_id, "hitl", dict(payload))
                    else:
                        # 节点完成 → 阶段事件（detail 取 trace 里的 detail 更省）
                        sse_broker.publish(
                            session_id, "stage", {"node": node}
                        )
        except Exception as exc:  # noqa: BLE001 —— 图异常转 error 终帧
            logger.exception("问答图执行异常 session=%s", session_id)
            sse_broker.publish(session_id, "error", {"message": str(exc)})
        finally:
            sse_broker.publish(session_id, "__done__", {})   # 哨兵：SSE 收尾

    return asyncio.create_task(producer(), name=f"qa-producer-{session_id}")


async def _sse_response(session_id: str, task: asyncio.Task, heartbeat_sec: float):
    """SSE 生成器：读 broker 队列逐帧下发，哨兵帧后收尾。"""
    q = sse_broker.subscribe(session_id)

    async def gen():
        yield _sse("session", {"session_id": session_id})
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=heartbeat_sec)
                except asyncio.TimeoutError:
                    # 心跳：防中间层闲置断连（Nginx proxy_read_timeout）
                    yield b": heartbeat\n\n"
                    continue
                event = item["event"]
                if event == "__done__":
                    break
                yield _sse(event, item["data"])
        finally:
            sse_broker.close_session(session_id)
            # 图可能还有收尾清理（已到 finalize 之外的情况），给 2s 优雅期
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                task.cancel()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ask")
async def ask(req: AskRequest, user: Any = Depends(get_current_user)) -> Any:
    """流式问答。SSE 响应，事件协议见模块头。"""
    session_id = (req.session_id or str(uuid.uuid4())).strip()
    heartbeat = float(get_config().get_path("agent.sse.heartbeat_sec", 15))
    user_id = int(getattr(user, "id"))
    # ★ 建会话要**先于**跑图：finalize 落库依赖 chat_sessions 里已有这行
    await _ensure_session(session_id, user_id, req.query)

    graph_input = {
        "session_id": session_id,
        # ⚠️ get_current_user 返回 ORM User 对象（不是 dict）—— 取属性而非下标
        "user_id": user_id,
        "raw_query": req.query,
        "doc_ids": req.doc_ids,
        "version_override": req.version_override,
        "history": req.history or [],
        "started_at": time.monotonic(),
    }
    task = _run_graph_producer(session_id, graph_input)
    return await _sse_response(session_id, task, heartbeat)


@router.post("/resume")
async def resume(req: ResumeRequest, user: Any = Depends(get_current_user)) -> Any:
    """HITL 恢复：唤醒挂起的图，返回续跑的 SSE 流。"""
    heartbeat = float(get_config().get_path("agent.sse.heartbeat_sec", 15))
    graph_input = Command(
        resume={
            "action": req.action,
            "new_query": (req.new_query or "").strip() or None,
        }
    )
    task = _run_graph_producer(req.session_id.strip(), graph_input)
    return await _sse_response(req.session_id.strip(), task, heartbeat)
