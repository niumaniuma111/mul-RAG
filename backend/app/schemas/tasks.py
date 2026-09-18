"""
入库任务 Schema（DESIGN.md §11.3）。

前端文档管理页每 2s 轮询 GET /tasks/{task_id}，所以字段刻意做扁，
避免嵌套导致前端解构麻烦。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class TaskItem(BaseModel):
    task_id: uuid.UUID
    doc_id: uuid.UUID
    # 可空：文档被硬删后任务记录保留，version_id 被置 NULL（见 version_service.hard_delete）
    version_id: int | None = Field(
        default=None, description="关联的 file_versions.id；NULL 表示文档已删除"
    )
    version: int | None = Field(default=None, description="由 version_id 关联出的版本号")
    file_name: str | None = None
    status: str = Field(description="pending/queued/running/success/failed")
    progress: int = Field(ge=0, le=100, description="0-100")
    current_stage: str | None = Field(
        default=None, description="parsing/captioning/chunking/embedding"
    )
    chunk_count: int = 0
    retry_count: int = 0
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskDetailResult(TaskItem):
    """GET /tasks/{task_id} —— 比列表多一个 error_log 摘要。"""

    error_log: str | None = None


class TaskLogsResult(BaseModel):
    """GET /tasks/{task_id}/logs —— 失败日志全文。"""

    task_id: uuid.UUID
    status: str
    retry_count: int
    error_log: str | None = None


class TaskRetryResult(BaseModel):
    task_id: uuid.UUID
    new_celery_task_id: str | None = None
    retry_count: int
    status: str


__all__ = [
    "TaskDetailResult",
    "TaskItem",
    "TaskLogsResult",
    "TaskRetryResult",
]
