"""
会话历史 Schema（B2）。

为什么需要这一组接口
--------
    S3/S4 时前端把会话列表与"历史正文"都寄存在浏览器 `localStorage`，
    后端虽然有 `chat_sessions` / `chat_messages` 两张表并且**一直在真实落库**
    （见 `services/qa_persistence.py`），但**没有任何读接口**。

    这直接导致一个用户可见的缺陷（HANDOFF 坑 37）：
    `chat.ts` 的 `pickSession()` 切会话时只能 `messages.value = []`
    —— 因为拿不回历史正文 —— 于是"点一下当前会话，整轮问答从界面消失"。
    数据没丢（库里都在），但用户看不到，体验上等同于丢失。

    B2 补上这两个只读端点后，前端 `pickSession` 才能从"清空"改成"加载历史"。

字段设计
--------
    刻意做扁平（与 tasks.py 同风格）：前端直接解构，不做深层嵌套。
    `citations` / `reflect_info` 保持 JSONB 原样透出 —— 它们是 Agent 产物，
    后端不该重新解释结构（前端 `CitationCard` 已有自己的渲染契约）。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionItem(BaseModel):
    """会话列表项。"""

    session_id: uuid.UUID
    title: str
    message_count: int = Field(default=0, description="该会话下的消息条数（user + assistant）")
    created_at: datetime
    updated_at: datetime


class MessageItem(BaseModel):
    """会话内单条消息。role 只会是 user / assistant。"""

    message_id: uuid.UUID
    role: str = Field(description="user / assistant")
    content: str
    citations: list[dict[str, Any]] | None = Field(
        default=None, description="assistant 消息的引用卡片数据；user 消息为 null"
    )
    reflect_info: dict[str, Any] | None = Field(
        default=None, description="assistant 消息的自省结论 {verdict, reason, rounds, ...}"
    )
    created_at: datetime


class SessionMessagesResult(BaseModel):
    """GET /qa/sessions/{session_id}/messages 的结果。

    `title` 一并返回，前端切会话时可以直接把标题同步过去，
    不必再依赖 localStorage 里那份可能过期的副本。
    """

    session_id: uuid.UUID
    title: str
    items: list[MessageItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    size: int = 50


__all__ = [
    "MessageItem",
    "SessionItem",
    "SessionMessagesResult",
]
