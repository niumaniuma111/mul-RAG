"""
Pydantic Schema 包。

约定：**请求体用强校验模型，响应体用 `ApiResponse[T]` 包装**。
响应运行时由 `app.core.exceptions.ok()` 构造，类型声明仅用于 OpenAPI 文档。
"""

from app.schemas.auth import (
    LoginRequest,
    MeResult,
    RegisterRequest,
    RegisterResult,
    TokenResult,
)
from app.schemas.common import ApiResponse, Page, PageMeta
from app.schemas.files import (
    ActivateResult,
    ChunkPreviewItem,
    ChunkPreviewPage,
    DeleteResult,
    FileDetailResult,
    FileListItem,
    UploadResult,
    VersionItem,
)
from app.schemas.tasks import (
    TaskDetailResult,
    TaskItem,
    TaskLogsResult,
    TaskRetryResult,
)

__all__ = [
    "ActivateResult",
    "ApiResponse",
    "ChunkPreviewItem",
    "ChunkPreviewPage",
    "DeleteResult",
    "FileDetailResult",
    "FileListItem",
    "LoginRequest",
    "MeResult",
    "Page",
    "PageMeta",
    "RegisterRequest",
    "RegisterResult",
    "TaskDetailResult",
    "TaskItem",
    "TaskLogsResult",
    "TaskRetryResult",
    "TokenResult",
    "UploadResult",
    "VersionItem",
]
