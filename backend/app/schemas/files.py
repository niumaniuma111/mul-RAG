"""
文档与版本管理 Schema（DESIGN.md §11.2）。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================
# 上传
# ============================================================
class UploadResult(BaseModel):
    """POST /files/upload 的返回。上传接口不阻塞，解析在 Celery 里异步做。"""

    doc_id: uuid.UUID = Field(description="逻辑文档 id（chunk metadata 里的 file_id）")
    version: int = Field(description="本次上传产生的版本号，从 1 开始")
    version_id: int = Field(description="file_versions.id（物理版本行 id）")
    task_id: uuid.UUID = Field(description="ingest_tasks.id，前端用它轮询进度")
    is_new_doc: bool = Field(description="true=新建文档；false=给已有文档加新版本")


# ============================================================
# 列表 / 详情
# ============================================================
class FileListItem(BaseModel):
    doc_id: uuid.UUID
    file_name: str
    file_ext: str
    current_version: int = Field(description="最新上传的版本号")
    active_version: int = Field(description="当前生效版本（可手动切换）")
    parse_status: str = Field(description="待解析版本的状态：pending/parsing/success/failed")
    chunk_count: int = 0
    page_count: int | None = None
    created_at: datetime
    updated_at: datetime


class VersionItem(BaseModel):
    version: int
    is_active: bool
    parse_status: str
    chunk_count: int = 0
    page_count: int | None = None
    file_size: int | None = None
    error_message: str | None = None
    created_at: datetime
    parsed_at: datetime | None = None


class FileDetailResult(BaseModel):
    """GET /files/{doc_id}"""

    doc_id: uuid.UUID
    file_name: str
    file_ext: str
    current_version: int
    active_version: int
    is_deleted: bool = False
    created_at: datetime
    updated_at: datetime
    versions: list[VersionItem] = Field(default_factory=list)


class ActivateResult(BaseModel):
    """POST /files/{doc_id}/versions/{version}/activate"""

    doc_id: uuid.UUID
    active_version: int
    updated_chunks: int = Field(description="被改动的向量点数（is_active 翻转）")
    updated_vectors: int = Field(description="同 updated_chunks，保留给前端展示双指标")


class ChunkPreviewItem(BaseModel):
    """GET /files/{doc_id}/versions/{version}/chunks"""

    chunk_index: int
    chunk_type: str = Field(description="text / table / image_caption")
    page_num: int | None = None
    content: str = Field(description="已截断的预览文本")
    image_path: str | None = None


class ChunkPreviewPage(BaseModel):
    """
    chunk 预览的一页。

    ⚠️ 分页用 `next_offset` 游标而不是页码 —— Qdrant 的 scroll 是
       "从某个点继续往下读"的语义，硬换算成页码会在过滤变化时重复或漏读。
       前端把上次拿到的 next_offset 原样回传即可。
    """

    items: list[ChunkPreviewItem] = Field(default_factory=list)
    next_offset: str | None = Field(
        default=None, description="为 null 表示已到末尾"
    )


class DeleteResult(BaseModel):
    """
    DELETE /files/{doc_id} —— **硬删除**（不可恢复）。

    清理范围：PG 行（documents / file_versions）、Qdrant 向量点、
    磁盘文件（uploads 原件 + parsed 解析产物）。
    ⚠️ ingest_tasks **保留**（2026-09-19 修：任务历史是操作审计，
       不随文档删除），只解绑 version_id 并快照 file_name。
    """

    doc_id: uuid.UUID
    deleted_versions: int = Field(description="物理删除的 file_versions 行数")
    deleted_tasks: int = Field(
        0, description="解除关联（保留记录）的 ingest_tasks 行数，非删除数"
    )
    deleted_vectors: int = Field(0, description="从 Qdrant 物理删除的向量点数")
    deleted_files: int = Field(0, description="物理删除的磁盘文件数（原件+解析产物）")
    warnings: list[str] = Field(
        default_factory=list,
        description="非致命清理失败项；PG 已删成功，但对应载体仍有残留，需人工处理",
    )


__all__ = [
    "ActivateResult",
    "ChunkPreviewItem",
    "ChunkPreviewPage",
    "DeleteResult",
    "FileDetailResult",
    "FileListItem",
    "UploadResult",
    "VersionItem",
]
