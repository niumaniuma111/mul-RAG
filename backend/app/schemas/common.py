"""
通用响应与分页模型。

⚠️ 关于响应包装
--------------
DESIGN.md §11 规定所有非 SSE 接口返回 `{code, message, data}`。
本模块把这个结构建模成 `ApiResponse[T]`，端点用
`response_model=ApiResponse[XxxResult]` 声明 —— 这样 OpenAPI 文档
能显示真实的 data 结构，而不是一个 `object`。

运行时仍由 `app.core.exceptions.ok()` 构造 dict，两者结构一致。
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """统一响应包装。`code=0` 表示成功，非 0 见 ErrorCode。"""

    code: int = Field(default=0, description="0=成功；非 0 见错误码表")
    message: str = Field(default="ok")
    data: T | None = None


class PageMeta(BaseModel):
    """分页元信息。"""

    total: int = Field(description="符合条件的总条数")
    page: int = Field(ge=1, description="当前页码，从 1 开始")
    size: int = Field(ge=1, le=200, description="每页条数")


class Page(BaseModel, Generic[T]):
    """分页结果：`items` + `meta`。"""

    items: list[T] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    size: int = 20


__all__ = ["ApiResponse", "Page", "PageMeta"]
