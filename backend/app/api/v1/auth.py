"""
认证接口（DESIGN.md §11.1）。

    POST /auth/register   注册
    POST /auth/login      登录，换取 access_token
    GET  /auth/me         当前用户

⚠️ 唯一的跨用户安全边界在这里：所有文档/任务/会话接口都靠
   `get_current_user` 拿到 user_id，再在 SQL 与 Qdrant filter 里强制带上它。
   S3/S4 新增接口时必须沿用这个模式，不能信任请求体里传来的 user_id。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from app.core.deps import CurrentUser, DbSession
from app.core.exceptions import BadRequestError, UnauthorizedError, ok
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.schemas.auth import (
    LoginRequest,
    MeResult,
    RegisterRequest,
    RegisterResult,
    TokenResult,
)
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)


@router.post(
    "/register",
    response_model=ApiResponse[RegisterResult],
    status_code=status.HTTP_201_CREATED,
    summary="注册",
)
async def register(payload: RegisterRequest, session: DbSession) -> dict[str, Any]:
    """
    注册新用户。

    ⚠️ 用户名唯一性由**数据库唯一索引**保证，这里先查一次只是为了给出友好提示。
    并发注册时仍可能两个请求同时通过预检，此时后者会撞到索引 —— 所以
    还要捕获 IntegrityError（见下方）。
    """
    existing = await session.scalar(select(User).where(User.username == payload.username))
    if existing is not None:
        raise BadRequestError("用户名已被占用")

    if payload.email:
        email_taken = await session.scalar(select(User).where(User.email == payload.email))
        if email_taken is not None:
            raise BadRequestError("邮箱已被注册")

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    try:
        await session.flush()  # 拿到自增 id，但不提交（由 get_session 统一提交）
    except Exception as exc:  # noqa: BLE001
        await session.rollback()
        # 唯一索引冲突在并发下必然出现，归一成 1001 而不是 5000
        if "uq" in str(exc).lower() or "unique" in str(exc).lower():
            raise BadRequestError("用户名或邮箱已被占用") from exc
        raise

    logger.info("用户注册成功: id=%s username=%s", user.id, user.username)
    result = RegisterResult(user_id=user.id, username=user.username)
    return ok(result.model_dump())


@router.post("/login", response_model=ApiResponse[TokenResult], summary="登录")
async def login(payload: LoginRequest, session: DbSession) -> dict[str, Any]:
    """
    登录。用户名或密码错误都返回同一个消息 —— 不区分"用户不存在"与"密码错"，
    否则会变成用户名枚举接口。
    """
    user = await session.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise UnauthorizedError("用户名或密码错误")
    if not user.is_active:
        raise UnauthorizedError("账号已被禁用")

    token, expires_in = create_access_token(user.id, user.username)
    logger.info("用户登录: id=%s username=%s", user.id, user.username)
    result = TokenResult(
        access_token=token,
        expires_in=expires_in,
        user_id=user.id,
        username=user.username,
    )
    return ok(result.model_dump())


@router.get("/me", response_model=ApiResponse[MeResult], summary="当前用户")
async def me(user: CurrentUser) -> dict[str, Any]:
    result = MeResult(
        user_id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
    )
    return ok(result.model_dump())


__all__ = ["router"]
