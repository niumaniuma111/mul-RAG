"""
认证相关 Schema（DESIGN.md §11.1）。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, description="登录名，唯一")
    email: EmailStr | None = Field(default=None, description="可选")
    password: str = Field(min_length=8, max_length=128, description="至少 8 位")


class RegisterResult(BaseModel):
    user_id: int
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResult(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="有效期（秒）")
    user_id: int
    username: str


class MeResult(BaseModel):
    user_id: int
    username: str
    email: str | None = None
    created_at: datetime


__all__ = [
    "LoginRequest",
    "MeResult",
    "RegisterRequest",
    "RegisterResult",
    "TokenResult",
]
