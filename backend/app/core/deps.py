"""
FastAPI 依赖注入：数据库会话、当前用户、分页参数。

为什么把「当前用户」做成依赖而不是在函数里读 header
--------------------------------------------------
1. 每个需要鉴权的端点声明 `user: User = Depends(get_current_user)`，
   漏写就是"没有用户变量"，静态可查 —— 比在函数体里读 header 更容易发现遗漏；
2. OpenAPI 会自动带上 security 声明，前端对接时能看到需要 Bearer token。

对应设计文档：DESIGN.md §11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_session

# auto_error=False：没带 token 时不要由 FastAPI 抛 403，改用我们自己的 1002
_bearer = HTTPBearer(auto_error=False, description="Authorization: Bearer <access_token>")

DbSession = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: DbSession,
) -> User:
    """
    解析 Bearer token → 查出 User。

    三道校验：token 有效 → 用户存在 → 用户未被禁用。
    任何一道失败都返回 1002 未认证 / 1003 无权限，绝不返回 500。
    """
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("缺少 Authorization: Bearer <access_token>")

    payload = decode_access_token(credentials.credentials)
    raw_sub = payload.get("sub")
    if raw_sub is None:
        raise UnauthorizedError("令牌缺少 sub 字段")

    try:
        user_id = int(raw_sub)
    except (TypeError, ValueError) as exc:
        raise UnauthorizedError("令牌的 sub 不是合法用户 id") from exc

    user = await session.get(User, user_id)
    if user is None:
        raise UnauthorizedError("用户不存在或已被删除")
    if not user.is_active:
        raise ForbiddenError("账号已被禁用")

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


@dataclass(slots=True)
class PageParams:
    """分页参数。`page` 从 1 开始，`size` 上限 200（防一次拉爆内存）。"""

    page: int
    size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


async def get_page_params(
    page: Annotated[int, Query(ge=1, description="页码，从 1 开始")] = 1,
    size: Annotated[int, Query(ge=1, le=200, description="每页条数")] = 20,
) -> PageParams:
    return PageParams(page=page, size=size)


Pagination = Annotated[PageParams, Depends(get_page_params)]

__all__ = [
    "CurrentUser",
    "DbSession",
    "PageParams",
    "Pagination",
    "get_current_user",
    "get_page_params",
]
