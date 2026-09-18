"""
业务异常与统一错误码。

错误码对照 DESIGN.md §11.4：

    0     成功
    1001  参数错误
    1002  未认证
    1003  无权限
    2001  文件格式不支持
    2002  文件过大
    3001  解析失败
    3002  GPU 繁忙
    4001  会话不存在
    5000  服务内部错误

所有非 SSE 接口都返回 `{code, message, data}` 结构，见 `ok()` / `fail()`。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# ⚠️ 别名导入：底层锁实现抛的 GpuBusyError 与下面 HTTP 层的 GpuBusyError 是
#    两个不同的类（一个在 core.redis_lock，一个在 core.exceptions）。
#    不加别名的话下面的类定义会把这个 import 覆盖掉，异常处理器就挂错了。
from app.core.redis_lock import GpuBusyError as LockGpuBusyError

logger = logging.getLogger(__name__)


class ErrorCode:
    """集中定义，避免错误码散落在各处的字面量。"""

    SUCCESS = 0
    BAD_REQUEST = 1001
    UNAUTHORIZED = 1002
    FORBIDDEN = 1003
    UNSUPPORTED_FILE_TYPE = 2001
    FILE_TOO_LARGE = 2002
    PARSE_FAILED = 3001
    GPU_BUSY = 3002
    SESSION_NOT_FOUND = 4001
    INTERNAL_ERROR = 5000


class AppError(Exception):
    """所有业务异常的基类。子类只需覆盖 code / http_status / message。"""

    code: int = ErrorCode.INTERNAL_ERROR
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "服务内部错误"

    def __init__(self, message: str | None = None, *, data: Any = None) -> None:
        self.message = message or self.message
        self.data = data
        super().__init__(self.message)


class BadRequestError(AppError):
    code = ErrorCode.BAD_REQUEST
    http_status = status.HTTP_400_BAD_REQUEST
    message = "参数错误"


class UnauthorizedError(AppError):
    code = ErrorCode.UNAUTHORIZED
    http_status = status.HTTP_401_UNAUTHORIZED
    message = "未认证或登录已过期"


class ForbiddenError(AppError):
    code = ErrorCode.FORBIDDEN
    http_status = status.HTTP_403_FORBIDDEN
    message = "无权限访问该资源"


class UnsupportedFileTypeError(AppError):
    code = ErrorCode.UNSUPPORTED_FILE_TYPE
    http_status = status.HTTP_400_BAD_REQUEST
    message = "文件格式不支持"


class FileTooLargeError(AppError):
    code = ErrorCode.FILE_TOO_LARGE
    http_status = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    message = "文件过大"


class ParseFailedError(AppError):
    code = ErrorCode.PARSE_FAILED
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    message = "文档解析失败"


class GpuBusyError(AppError):  # type: ignore[no-redef]
    """
    GPU 繁忙（3002）。

    注意这里继承的是 AppError 而非 redis_lock.GpuBusyError ——
    redis_lock 的那个是底层锁实现抛的，HTTP 层不认识它。
    两者在 register_exception_handlers 里做了桥接转换。
    """

    code = ErrorCode.GPU_BUSY
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    message = "GPU 繁忙，请稍后重试"


class SessionNotFoundError(AppError):
    code = ErrorCode.SESSION_NOT_FOUND
    http_status = status.HTTP_404_NOT_FOUND
    message = "会话不存在"


# ============================================================
# 统一响应包装
# ============================================================
def ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    return {"code": ErrorCode.SUCCESS, "message": message, "data": data}


def fail(code: int, message: str, data: Any = None) -> dict[str, Any]:
    return {"code": code, "message": message, "data": data}


# ============================================================
# 全局异常处理器
# ============================================================
def register_exception_handlers(app: FastAPI) -> None:
    """在 main.py 里挂上，保证任何异常都返回统一结构而不是裸 500。"""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=fail(exc.code, exc.message, exc.data),
        )

    @app.exception_handler(LockGpuBusyError)
    async def _handle_gpu_busy(_: Request, exc: LockGpuBusyError) -> JSONResponse:
        """底层锁（core.redis_lock）抛的 GpuBusyError → HTTP 3002。"""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=fail(ErrorCode.GPU_BUSY, str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=fail(ErrorCode.BAD_REQUEST, "参数校验失败", exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # 未预期的异常必须打完整堆栈，否则线上没法查
        logger.exception("未捕获异常: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=fail(ErrorCode.INTERNAL_ERROR, "服务内部错误"),
        )


__all__ = [
    "AppError",
    "BadRequestError",
    "ErrorCode",
    "FileTooLargeError",
    "ForbiddenError",
    "GpuBusyError",
    "ParseFailedError",
    "SessionNotFoundError",
    "UnauthorizedError",
    "UnsupportedFileTypeError",
    "fail",
    "ok",
    "register_exception_handlers",
]
