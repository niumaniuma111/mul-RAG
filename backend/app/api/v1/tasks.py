"""
入库任务接口（DESIGN.md §11.3）。

     GET    /tasks
     GET    /tasks/{task_id}
     GET    /tasks/{task_id}/logs
     POST   /tasks/{task_id}/retry
     DELETE /tasks/{task_id}

前端文档管理页每 2s 轮询 GET /tasks/{task_id} 显示进度条，
所以这个接口必须**轻**：一条 SQL 出结果，不做任何额外调用。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.core.deps import CurrentUser, DbSession, Pagination
from app.core.exceptions import BadRequestError, ok
from app.core.logging import get_logger
from app.db.models import Document, FileVersion, IngestTask
from app.schemas.common import ApiResponse, Page
from app.schemas.tasks import (
    TaskDetailResult,
    TaskItem,
    TaskLogsResult,
    TaskRetryResult,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def _get_task(session: DbSession, user_id: int, task_id: uuid.UUID) -> IngestTask:
    """
    取任务并校验归属。

    注意用 doc_id → documents.user_id 反查归属，因为 ingest_tasks 表里
    虽然有 user_id 冗余列，但那一列在 schema 里没有外键约束，
    以它为准会让"删库后残留任务"变成可被他人读到的数据。
    """
    task = await session.scalar(select(IngestTask).where(IngestTask.id == task_id))
    if task is None:
        raise BadRequestError("任务不存在")
    if int(task.user_id) != int(user_id):
        logger.warning("越权访问任务: user=%s task=%s owner=%s", user_id, task_id, task.user_id)
        raise BadRequestError("任务不存在")
    return task


def _to_item(
    task: IngestTask,
    *,
    version: int | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "doc_id": task.doc_id,
        "version_id": task.version_id,
        "version": version,
        "file_name": file_name,
        "status": task.status,
        "progress": task.progress,
        "current_stage": task.current_stage,
        "chunk_count": task.chunk_count,
        "retry_count": task.retry_count,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


@router.get("", response_model=ApiResponse[Page[TaskItem]], summary="任务列表")
async def list_tasks(
    user: CurrentUser,
    session: DbSession,
    pagination: Pagination,
    status: str | None = Query(default=None, description="pending/queued/running/success/failed"),
) -> dict[str, Any]:
    base = select(IngestTask).where(IngestTask.user_id == int(user.id))
    if status:
        base = base.where(IngestTask.status == status)

    total = int(await session.scalar(select(func.count()).select_from(base.subquery())) or 0)
    rows = list(
        (
            await session.scalars(
                base.order_by(IngestTask.created_at.desc())
                .offset(pagination.offset)
                .limit(pagination.size)
            )
        ).all()
    )

    items = await _decorate(session, rows)
    return ok(
        {
            "items": items,
            "total": total,
            "page": pagination.page,
            "size": pagination.size,
        }
    )


async def _decorate(session: DbSession, tasks: list[IngestTask]) -> list[dict[str, Any]]:
    """
    给任务补上 version 号与文件名（列表页要显示）。

    一次查出所有关联行再在内存里映射，避免 N+1 ——
    列表页每 2s 被轮询，N+1 会很快把 PG 连接池打满。
    """
    if not tasks:
        return []

    version_ids = {t.version_id for t in tasks}
    doc_ids = {t.doc_id for t in tasks}

    version_rows = list(
        (
            await session.scalars(
                select(FileVersion).where(FileVersion.id.in_(version_ids))
            )
        ).all()
    )
    version_map = {int(v.id): v for v in version_rows}

    doc_rows = list(
        (await session.scalars(select(Document).where(Document.doc_id.in_(doc_ids)))).all()
    )
    name_map = {str(d.doc_id): d.file_name for d in doc_rows}

    items: list[dict[str, Any]] = []
    for task in tasks:
        ver = version_map.get(int(task.version_id)) if task.version_id is not None else None
        # file_name 回退链：documents 存续期走 join → 文档已删走任务行快照
        # （hard_delete 时写入）→ 兜底占位符。version_id 为 NULL 即"文档已删"。
        file_name = (
            name_map.get(str(task.doc_id))
            or task.file_name
            or "（文档已删除）"
        )
        items.append(
            _to_item(
                task,
                version=ver.version if ver else None,
                file_name=file_name,
            )
        )
    return items


@router.get("/{task_id}", response_model=ApiResponse[TaskDetailResult], summary="任务详情")
async def get_task(
    user: CurrentUser, session: DbSession, task_id: uuid.UUID
) -> dict[str, Any]:
    task = await _get_task(session, int(user.id), task_id)
    decorated = (await _decorate(session, [task]))[0]
    # 详情页额外带上 error_log 摘要（前端 2s 轮询，不宜传全文）
    decorated["error_log"] = (task.error_log or "")[:2000] or None
    return ok(decorated)


@router.get(
    "/{task_id}/logs", response_model=ApiResponse[TaskLogsResult], summary="任务失败日志"
)
async def get_task_logs(
    user: CurrentUser, session: DbSession, task_id: uuid.UUID
) -> dict[str, Any]:
    task = await _get_task(session, int(user.id), task_id)
    result = TaskLogsResult(
        task_id=task.id,
        status=task.status,
        retry_count=task.retry_count,
        error_log=task.error_log,
    )
    return ok(result.model_dump())


@router.post(
    "/{task_id}/retry", response_model=ApiResponse[TaskRetryResult], summary="重试失败任务"
)
async def retry_task(
    user: CurrentUser, session: DbSession, task_id: uuid.UUID
) -> dict[str, Any]:
    """
    重新派发解析任务。

    ⚠️ 只允许重试 `failed` 的任务。重试 running 中的任务会导致两个 worker
       同时解析同一版本 → 写入同一批 point id（幂等，不会重复数据），
       但会白烧一倍 GPU/CPU 时间，且进度条会来回跳。
    """
    task = await _get_task(session, int(user.id), task_id)
    if task.status != "failed":
        raise BadRequestError(f"只有 failed 状态可以重试，当前状态：{task.status}")
    # version_id 已被置 NULL = 文档（连同版本）已被硬删，重试无意义
    # （_load_context 查不到行只会静默 skip，用户却看到任务"转了一圈原地不动"）
    if task.version_id is None:
        raise BadRequestError("文档已删除，无法重试")

    task.retry_count = int(task.retry_count) + 1
    task.status = "pending"
    task.progress = 0
    task.current_stage = "pending"
    task.error_log = None
    task.started_at = None
    task.finished_at = None
    await session.commit()

    new_celery_id: str | None = None
    try:
        from app.tasks.celery_app import celery_app

        async_result = celery_app.send_task(
            "ingest.full_pipeline", args=[str(task_id), int(task.version_id)]
        )
        new_celery_id = str(async_result.id)
    except Exception as exc:  # noqa: BLE001
        logger.error("重试派发失败: task=%s %s", task_id, exc)

    result = TaskRetryResult(
        task_id=task_id,
        new_celery_task_id=new_celery_id,
        retry_count=task.retry_count,
        status=task.status,
    )
    logger.info("任务重试: task=%s retry_count=%s", task_id, task.retry_count)
    return ok(result.model_dump())


@router.delete("/{task_id}", summary="删除任务记录（不删文件）")
async def delete_task(
    user: CurrentUser, session: DbSession, task_id: uuid.UUID
) -> dict[str, Any]:
    task = await _get_task(session, int(user.id), task_id)
    if task.status == "running":
        raise BadRequestError("任务正在运行，不能删除记录")

    await session.delete(task)
    await session.commit()
    logger.info("任务记录已删除: %s", task_id)
    return ok({"task_id": str(task_id), "deleted_at": datetime.now(UTC).isoformat()})


__all__ = ["router"]
