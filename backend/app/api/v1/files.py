"""
文档与版本管理接口（DESIGN.md §11.2）。

     POST   /files/upload
     GET    /files
     GET    /files/{doc_id}
     GET    /files/{doc_id}/versions
     POST   /files/{doc_id}/versions/{version}/activate
     GET    /files/{doc_id}/versions/{version}/chunks
     DELETE /files/{doc_id}          ← 硬删除（PG + Qdrant + 磁盘，不可恢复）

⚠️ 所有接口的第一道防线都是 `CurrentUser`（见 app/core/deps.py）。
   多租户隔离靠它给出的 user_id，**绝不信任请求体里传来的 user_id**。
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Query, UploadFile, status
from sqlalchemy import select

from app.core.config import get_config
from app.core.deps import CurrentUser, DbSession, Pagination
from app.core.exceptions import BadRequestError, ok
from app.core.logging import get_logger
from app.db.models import FileVersion
from app.schemas.common import ApiResponse, Page
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
from app.services import ingest_service, version_service
from app.services.ingest_service import UploadOutcome

logger = get_logger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


# ============================================================
# 上传
# ============================================================
async def _stream_to_disk(upload: UploadFile, dest: Path) -> tuple[int, str]:
    """
    把上传流写入磁盘，边写边算 md5。

    为什么必须流式写（而不是 `await upload.read()` 一次读完）：
        上限 200MB，而 api 容器的 mem_limit 只有 512MB、还会开 2 个 worker 进程。
        一次性读入内存会让两个并发上传直接把容器打爆（表现是 OOMKilled，
        日志里什么都看不到）。

    `asyncio.to_thread` 是为了不让磁盘写阻塞事件循环 ——
    否则一个 200MB 上传会让所有其他请求（含 /health）全部卡住。
    """
    digest = hashlib.md5()  # noqa: S324 —— 仅作去重标识
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _open() -> Any:
        return dest.open("wb")

    fp = await asyncio.to_thread(_open)
    try:
        while True:
            chunk = await upload.read(1 << 20)      # 1MB
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            await asyncio.to_thread(fp.write, chunk)
        await asyncio.to_thread(fp.flush)
    finally:
        await asyncio.to_thread(fp.close)

    return size, digest.hexdigest()


@router.post(
    "/upload",
    response_model=ApiResponse[UploadResult],
    status_code=status.HTTP_201_CREATED,
    summary="上传文档（立即返回，解析异步）",
)
async def upload(
    user: CurrentUser,
    session: DbSession,
    file: UploadFile = File(..., description="pptx / ppt / pdf"),
) -> dict[str, Any]:
    """
    上传并登记版本，随即派发 Celery 解析任务。

    ⚠️ 本接口**不等待解析**：CPU 模式下 MinerU 解析几十页 PPT 要 10-20 分钟，
       同步等待必然超时。返回 task_id 由前端轮询 /tasks/{id}。

    落盘分两阶段（因为目标路径需要 doc_id，而 doc_id 在登记后才生成）：
        1. 先写到 `uploads/_tmp/` 下的临时文件（同时算出 size 与 md5）
        2. 登记拿到 doc_id/version 后，移动到 `uploads/{user}/{doc}/v{n}_{名}`
    """
    cfg_upload_dir = Path(
        str(get_config().get_path("paths.upload_dir", "/app/storage/uploads"))
    )

    if not file.filename:
        raise BadRequestError("缺少文件名")

    # 先用文件名+扩展名做前置校验（大小未知，先传 0 跳过大小校验）
    file_ext = ingest_service.validate_upload(file.filename, 1)

    # ---------- 1. 落到临时文件 ----------
    tmp_dir = cfg_upload_dir / "_tmp"
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{file_ext}"
    size, md5 = await _stream_to_disk(file, tmp_path)

    # 拿到真实大小后**再校验一次**（这才是有效的上限检查）
    try:
        ingest_service.validate_upload(file.filename, size)
    except Exception:
        tmp_path.unlink(missing_ok=True)     # 校验失败要清理，否则 _tmp 会攒垃圾
        raise

    # ---------- 2. 登记（内部 commit，见 ingest_service 模块头）----------
    outcome: UploadOutcome = await ingest_service.register_upload(
        session,
        user_id=int(user.id),
        file_name=file.filename,
        file_ext=file_ext,
        file_path=str(tmp_path),             # 先占位，下一步改成最终路径
        file_size=size,
        file_md5=md5,
    )

    # ---------- 3. 移动到最终路径并回填 ----------
    final_path = ingest_service.upload_target_path(
        int(user.id), outcome.doc_id, outcome.version, file.filename
    )
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(tmp_path.replace, final_path)
        await session.execute(
            FileVersion.__table__.update()   # type: ignore[attr-defined]
            .where(FileVersion.id == outcome.version_id)
            .values(file_path=str(final_path))
        )
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        # 移动失败 → 保留临时路径（解析仍能进行），但必须记录，否则事后无法定位
        logger.error("上传文件移动失败，继续使用临时路径 %s: %s", tmp_path, exc)

    # ---------- 4. 派发解析任务（commit 之后，见 ingest_service 模块头）----------
    celery_task_id = _dispatch_ingest(outcome)

    logger.info(
        "上传完成: user=%s doc=%s v%s size=%sB task=%s",
        user.id, outcome.doc_id, outcome.version, size, outcome.task_id,
    )
    result = UploadResult(
        doc_id=outcome.doc_id,
        version=outcome.version,
        version_id=outcome.version_id,
        task_id=outcome.task_id,
        is_new_doc=outcome.is_new_doc,
    )
    payload = result.model_dump()
    payload["celery_task_id"] = celery_task_id     # 便于排障，前端可忽略
    return ok(payload)


def _dispatch_ingest(outcome: UploadOutcome) -> str | None:
    """
    派发 Celery 任务。

    ⚠️ 这里刻意**吞掉异常只记日志**：任务派发失败不应让"上传"这个动作失败 ——
       文件已经落盘、版本已登记，用户可以点"重试"（POST /tasks/{id}/retry）。
       若让异常冒出来，用户看到上传失败但文件其实已存，会更困惑。
    """
    from app.tasks.celery_app import celery_app

    try:
        async_result = celery_app.send_task(
            "ingest.full_pipeline",
            args=[str(outcome.task_id), outcome.version_id],
        )
        return str(async_result.id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Celery 任务派发失败（任务保持 pending，可手动重试）: task=%s %s",
            outcome.task_id, exc,
        )
        return None


# ============================================================
# 列表 / 详情
# ============================================================
@router.get("", response_model=ApiResponse[Page[FileListItem]], summary="文档列表")
async def list_files(
    user: CurrentUser,
    session: DbSession,
    pagination: Pagination,
    keyword: str | None = Query(default=None, description="按文件名模糊搜索"),
) -> dict[str, Any]:
    items, total = await version_service.list_files(
        session,
        user_id=int(user.id),
        keyword=keyword,
        page=pagination.page,
        size=pagination.size,
    )
    return ok({"items": items, "total": total, "page": pagination.page, "size": pagination.size})


@router.get("/{doc_id}", response_model=ApiResponse[FileDetailResult], summary="文档详情+版本")
async def get_file(
    user: CurrentUser, session: DbSession, doc_id: uuid.UUID
) -> dict[str, Any]:
    detail = await version_service.get_file_detail(
        session, user_id=int(user.id), doc_id=doc_id
    )
    return ok(detail)


@router.get(
    "/{doc_id}/versions",
    response_model=ApiResponse[list[VersionItem]],
    summary="版本列表",
)
async def list_versions(
    user: CurrentUser, session: DbSession, doc_id: uuid.UUID
) -> dict[str, Any]:
    versions = await version_service.list_versions(
        session, user_id=int(user.id), doc_id=doc_id
    )
    return ok(versions)


# ============================================================
# 切换 / 预览
# ============================================================
@router.post(
    "/{doc_id}/versions/{version}/activate",
    response_model=ApiResponse[ActivateResult],
    summary="切换生效版本",
)
async def activate_version(
    user: CurrentUser, session: DbSession, doc_id: uuid.UUID, version: int
) -> dict[str, Any]:
    outcome = await version_service.activate_version(
        session, user_id=int(user.id), doc_id=doc_id, version=version
    )
    result = ActivateResult(
        doc_id=doc_id,
        active_version=outcome.active_version,
        updated_chunks=outcome.updated_chunks,
        updated_vectors=outcome.updated_chunks,
    )
    return ok(result.model_dump())


@router.get(
    "/{doc_id}/versions/{version}/chunks",
    response_model=ApiResponse[ChunkPreviewPage],
    summary="版本 chunk 预览",
)
async def preview_chunks(
    user: CurrentUser,
    session: DbSession,
    doc_id: uuid.UUID,
    version: int,
    limit: int = Query(default=20, ge=1, le=200),
    offset: str | None = Query(default=None, description="游标，回传上次返回的 next_offset"),
) -> dict[str, Any]:
    # 先做归属校验：preview 直接读 Qdrant，不校验就等于任何人可读任意 doc_id
    await version_service.get_doc(session, int(user.id), doc_id)

    payloads, next_offset = await version_service.preview_chunks(
        doc_id=doc_id, version=version, offset=offset, limit=limit
    )
    items = [
        ChunkPreviewItem(
            chunk_index=int(p.get("chunk_index", 0)),
            chunk_type=str(p.get("chunk_type", "text")),
            page_num=p.get("page_num"),
            content=str(p.get("content", ""))[:500],
            image_path=p.get("image_path"),
        )
        for p in payloads
    ]
    page_result = ChunkPreviewPage(
        items=items,
        next_offset=str(next_offset) if next_offset is not None else None,
    )
    return ok(page_result.model_dump())


# ============================================================
# 删除
# ============================================================
@router.delete(
    "/{doc_id}",
    response_model=ApiResponse[DeleteResult],
    summary="硬删除文档（PG + 向量 + 磁盘文件，不可恢复）",
)
async def delete_file(
    user: CurrentUser, session: DbSession, doc_id: uuid.UUID
) -> dict[str, Any]:
    """
    ⚠️ **不可恢复**：同时清掉 PG 三张表的行、Qdrant 全部版本的点、
    以及 uploads/parsed 下该文档的整个目录。

    `warnings` 非空表示 PG 已删成功但某些载体没清干净（如 Qdrant 短暂不可用），
    前端应把它显式展示给用户，而不是当作成功悄悄吞掉。
    """
    outcome = await version_service.hard_delete(
        session, user_id=int(user.id), doc_id=doc_id
    )
    result = DeleteResult(
        doc_id=doc_id,
        deleted_versions=outcome.deleted_versions,
        deleted_tasks=outcome.deleted_tasks,
        deleted_vectors=outcome.deleted_vectors,
        deleted_files=outcome.deleted_files,
        warnings=outcome.warnings,
    )
    return ok(result.model_dump())


# ============================================================
# 供 tasks 路由复用的小工具
# ============================================================
async def get_version_row(session: DbSession, version_id: int) -> FileVersion | None:
    return await session.scalar(select(FileVersion).where(FileVersion.id == version_id))


__all__ = ["router"]
