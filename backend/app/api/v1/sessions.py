"""
会话历史读接口（B2）。

    GET /qa/sessions                         会话列表（当前用户）
    GET /qa/sessions/{session_id}/messages   某会话的消息（按时间正序）

为什么放在 qa.py 之外
--------
    `qa.py` 的职责是 **SSE 流式问答**（ask / resume），协议是 event-stream；
    本模块是**普通 JSON 只读查询**，响应包装是 `{code, message, data}`。
    两者协议、缓存语义、错误处理都不同，混在一个文件里会让 qa.py 的
    "SSE 三条流分工"说明失去焦点。但路由前缀同为 `/qa`（会话是问答的子资源），
    所以这里 `prefix="/qa"` 而路径写 `/sessions`。

安全
--------
    两个端点都必须校验 `session.user_id == current_user.id`。
    越权时返回「会话不存在」而不是 403 —— 不向调用者泄露"这个 id 是存在的"。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Path
from sqlalchemy import case, func, select

from app.core.deps import CurrentUser, DbSession, Pagination
from app.core.exceptions import BadRequestError, ok
from app.core.logging import get_logger
from app.db.models import ChatMessage, ChatSession
from app.schemas.common import ApiResponse, Page
from app.schemas.sessions import MessageItem, SessionItem, SessionMessagesResult

logger = get_logger(__name__)

router = APIRouter(prefix="/qa", tags=["qa"])


async def _get_session(session: DbSession, user_id: int, session_id: uuid.UUID) -> ChatSession:
    """
    取会话并校验归属。

    越权与不存在返回同一句话 —— 否则攻击者可以用 404/403 的差异
    枚举出"哪些 session_id 真实存在"，而会话 id 出现在前端 URL/日志里。
    """
    row = await session.scalar(select(ChatSession).where(ChatSession.id == session_id))
    if row is None or int(row.user_id) != int(user_id):
        if row is not None:
            logger.warning(
                "越权访问会话: user=%s session=%s owner=%s", user_id, session_id, row.user_id
            )
        raise BadRequestError("会话不存在")
    return row


@router.get(
    "/sessions",
    response_model=ApiResponse[Page[SessionItem]],
    summary="会话列表",
)
async def list_sessions(
    user: CurrentUser,
    session: DbSession,
    page: Pagination,
) -> dict:
    """
    当前用户的会话列表，按 `updated_at` 倒序（最近说过的排最前）。

    `message_count` 用一条 GROUP BY 子查询算出来 —— 不用 N+1，
    也不依赖前端的本地缓存。这样前端切掉 localStorage 后列表依然完整。
    """
    uid = int(user.id)

    total = await session.scalar(
        select(func.count()).select_from(ChatSession).where(ChatSession.user_id == uid)
    )

    # 消息数：一次聚合出全部会话的计数，再在外层 join —— 避免逐行查询
    counts = (
        select(
            ChatMessage.session_id.label("sid"),
            func.count().label("cnt"),
        )
        .where(ChatMessage.user_id == uid)
        .group_by(ChatMessage.session_id)
        .subquery()
    )

    rows = await session.execute(
        select(ChatSession, func.coalesce(counts.c.cnt, 0))
        .outerjoin(counts, counts.c.sid == ChatSession.id)
        .where(ChatSession.user_id == uid)
        .order_by(ChatSession.updated_at.desc())
        .offset(page.offset)
        .limit(page.size)
    )

    items = [
        {
            "session_id": s.id,
            "title": s.title,
            "message_count": int(cnt),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s, cnt in rows.all()
    ]

    return ok({"items": items, "total": int(total or 0), "page": page.page, "size": page.size})


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ApiResponse[SessionMessagesResult],
    summary="会话消息",
)
async def list_session_messages(
    user: CurrentUser,
    session: DbSession,
    page: Pagination,
    session_id: uuid.UUID = Path(description="会话 id（UUID）"),
) -> dict:
    """
    某会话的消息，**按 created_at 正序**（对话的自然阅读顺序）。

    为什么正序 + 分页而不是"倒序取最近 N 条"：
    前端切会话时要按顺序渲染整段对话，倒序返回还得前端再翻一次；
    而分页参数让超长会话也能安全拉取（`size` 上限 200）。

    ⚠️ `created_at` 相同的行（同一次问答的 user/assistant 两行是同一条 SQL
       插入的，时间戳一致）需要**稳定的次级排序键**，否则翻页会出现
       同一行重复或漏掉。这里用 `role` 做次级键：'assistant' < 'user'
       字典序反了 —— 所以显式用 CASE 把 user 排前。
    """
    uid = int(user.id)
    chat = await _get_session(session, uid, session_id)

    total = await session.scalar(
        select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
    )

    role_order = case((ChatMessage.role == "user", 0), else_=1)

    rows = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), role_order.asc())
        .offset(page.offset)
        .limit(page.size)
    )

    items = [
        {
            "message_id": m.id,
            "role": m.role,
            "content": m.content,
            "citations": m.citations,
            "reflect_info": m.reflect_info,
            "created_at": m.created_at,
        }
        for m in rows.scalars().all()
    ]

    return ok(
        {
            "session_id": chat.id,
            "title": chat.title,
            "items": items,
            "total": int(total or 0),
            "page": page.page,
            "size": page.size,
        }
    )


__all__ = ["router"]
