"""
版本管理服务（DESIGN.md §11.2）。

核心概念：PG 是唯一事实来源
-------------------------
`documents.active_version` 决定"检索时该用哪个版本"。
`file_versions.is_active` 是它的逐版本展开（同一时刻只有一个为 true），
Qdrant 的 `is_active` payload 又是 `file_versions` 的投影。

三层递进，一致性策略：
    PG（强一致，事务内维护）
      └─ file_versions.is_active（事务内与 active_version 一起更新）
           └─ Qdrant payload（事务外 best-effort；PG 是唯一事实来源）

切换版本时先改 PG，再改 Qdrant；**不让 Qdrant 反过来影响 PG**。
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_config
from app.core.exceptions import BadRequestError, ForbiddenError
from app.core.logging import get_logger
from app.db.models import Document, FileVersion, IngestTask
from app.services.vector_store import get_vector_store

logger = get_logger(__name__)


@dataclass(slots=True)
class ActivateOutcome:
    active_version: int
    updated_chunks: int


@dataclass(slots=True)
class DeleteOutcome:
    """硬删除的结果。`warnings` 非空 = PG 已删成功但载体有残留，需人工处理。"""

    deleted_versions: int
    deleted_tasks: int
    deleted_vectors: int
    deleted_files: int
    warnings: list[str] = field(default_factory=list)


async def get_doc(session: AsyncSession, user_id: int, doc_id: uuid.UUID) -> Document:
    """
    取文档并校验归属。

    ⚠️ 归属校验必须在这里做，而不是在 SQL 里 `WHERE doc_id=? AND user_id=?` 后
       返回 None —— 两者都能防住越权，但前者能区分"不存在"与"不是你的"。
       对外统一报 404（不泄露资源是否存在），内部日志区分，便于排查。
    """
    doc = await session.scalar(
        select(Document).where(Document.doc_id == doc_id).with_for_update()
    )
    if doc is None or doc.is_deleted:
        raise BadRequestError("文档不存在或已删除")
    if int(doc.user_id) != int(user_id):
        logger.warning("越权访问文档: user=%s doc=%s owner=%s", user_id, doc_id, doc.user_id)
        raise ForbiddenError("无权访问该文档")
    return doc


# ============================================================
# 列表 / 详情
# ============================================================
async def list_files(
    session: AsyncSession,
    *,
    user_id: int,
    keyword: str | None,
    page: int,
    size: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    分页列出文档，并带上**当前生效版本**的解析状态与块数。

    为什么状态取 active_version 而不是 current_version：
        用户看到"已就绪"应该意味着"现在问它是有内容的"。
        刚传完的新版本还在解析中，此时 current_version 是 pending 而
        active_version（旧版本）是 success —— 后者才是可用的状态。
    """
    base = select(Document).where(Document.user_id == user_id, Document.is_deleted.is_(False))
    if keyword:
        base = base.where(Document.file_name.ilike(f"%{keyword}%"))

    total = int(
        await session.scalar(
            select(func.count()).select_from(base.subquery())
        )
        or 0
    )

    docs = list(
        (
            await session.scalars(
                base.order_by(Document.updated_at.desc())
                .offset((page - 1) * size)
                .limit(size)
            )
        ).all()
    )
    if not docs:
        return [], total

    # 一次查出这些文档所有 active_version 的版本行（避免 N+1）
    doc_ids = [d.doc_id for d in docs]
    active_pairs = [(d.doc_id, d.active_version) for d in docs]
    rows = list(
        (
            await session.scalars(
                select(FileVersion).where(
                    FileVersion.doc_id.in_(doc_ids),
                    FileVersion.version.in_({v for _, v in active_pairs}),
                )
            )
        ).all()
    )
    version_map = {(r.doc_id, r.version): r for r in rows}

    items: list[dict[str, Any]] = []
    for doc in docs:
        ver = version_map.get((doc.doc_id, doc.active_version))
        items.append(
            {
                "doc_id": doc.doc_id,
                "file_name": doc.file_name,
                "file_ext": doc.file_ext,
                "current_version": doc.current_version,
                "active_version": doc.active_version,
                "parse_status": ver.parse_status if ver else "pending",
                "chunk_count": ver.chunk_count if ver else 0,
                "page_count": ver.page_count if ver else None,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at,
            }
        )
    return items, total


async def get_file_detail(
    session: AsyncSession, *, user_id: int, doc_id: uuid.UUID
) -> dict[str, Any]:
    doc = await get_doc(session, user_id, doc_id)
    versions = await list_versions(session, user_id=user_id, doc_id=doc_id)
    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "file_ext": doc.file_ext,
        "current_version": doc.current_version,
        "active_version": doc.active_version,
        "is_deleted": doc.is_deleted,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "versions": versions,
    }


async def list_versions(
    session: AsyncSession, *, user_id: int, doc_id: uuid.UUID
) -> list[dict[str, Any]]:
    await get_doc(session, user_id, doc_id)
    rows = list(
        (
            await session.scalars(
                select(FileVersion)
                .where(FileVersion.doc_id == doc_id)
                .order_by(FileVersion.version.desc())
            )
        ).all()
    )
    return [
        {
            "version": r.version,
            "is_active": r.is_active,
            "parse_status": r.parse_status,
            "chunk_count": r.chunk_count,
            "page_count": r.page_count,
            "file_size": r.file_size,
            "error_message": r.error_message,
            "created_at": r.created_at,
            "parsed_at": r.parsed_at,
        }
        for r in rows
    ]


# ============================================================
# 切换生效版本
# ============================================================
async def activate_version(
    session: AsyncSession, *, user_id: int, doc_id: uuid.UUID, version: int
) -> ActivateOutcome:
    """
    把 `active_version` 切到指定版本（DESIGN §11.2 的事务语义）。

    ⚠️ 只允许切到 **parse_status='success'** 的版本。
       切到一个 failed/pending 的版本会让检索直接空结果，
       用户看到的现象是"问答突然什么都答不出来" —— 极难定位。
    """
    doc = await get_doc(session, user_id, doc_id)          # 内含 FOR UPDATE

    target = await session.scalar(
        select(FileVersion).where(
            FileVersion.doc_id == doc_id, FileVersion.version == version
        )
    )
    if target is None:
        raise BadRequestError(f"版本 {version} 不存在")
    if target.parse_status != "success":
        raise BadRequestError(
            f"版本 {version} 解析状态为 {target.parse_status}，只有 success 的版本可以切换"
        )

    # PG 侧原子更新：新的 active_version 与逐版本 is_active 一起改
    doc.active_version = version
    await session.execute(
        update(FileVersion)
        .where(FileVersion.doc_id == doc_id)
        .values(is_active=(FileVersion.version == version))
    )
    await session.commit()

    # Qdrant 侧最终一致（失败不回滚 PG —— PG 才是事实来源）
    updated = 0
    try:
        updated = get_vector_store().set_active_by_version(doc_id, version)
    except Exception as exc:  # noqa: BLE001
        # 修复手段：再切一次同一版本即可 —— set_active_by_version 是幂等的
        # （先把该 doc 全部置 false，再把目标版本置 true）。
        logger.warning("Qdrant 投影失败（重切同一版本可重建）: doc=%s %s", doc_id, exc)

    logger.info("版本已切换: doc=%s → v%s（Qdrant 影响 %s 点）", doc_id, version, updated)
    return ActivateOutcome(active_version=version, updated_chunks=updated)


# ============================================================
# 硬删除（不可恢复）
# ============================================================
def _purge_paths(targets: list[Path], roots: list[Path]) -> tuple[int, list[str]]:
    """
    递归删除目录，返回 `(删除的文件数, 非致命告警)`。

    ⚠️ 每个 target 必须落在某个允许的 root **之内** —— 删除操作里最贵的
       错误就是"路径拼错删到别处"。所以在真正 `rmtree` 之前先用
       `resolve()` 做前缀校验（resolve 会解析 symlink 与 `..`，
       比裸字符串 startswith 可靠）。
    """
    removed = 0
    warnings: list[str] = []
    resolved_roots = [r.resolve() for r in roots]

    for target in targets:
        if not target.exists():
            continue
        try:
            resolved = target.resolve()
            if not any(resolved == r or r in resolved.parents for r in resolved_roots):
                warnings.append(f"路径越界，跳过删除: {target}")
                logger.error("硬删除路径越界，已跳过: %s", target)
                continue
            removed += sum(1 for p in resolved.rglob("*") if p.is_file())
            shutil.rmtree(resolved)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"目录清理失败 {target}: {exc}")
            logger.warning("硬删除目录失败: %s %s", target, exc)

    return removed, warnings


async def hard_delete(
    session: AsyncSession, *, user_id: int, doc_id: uuid.UUID
) -> DeleteOutcome:
    """
    **硬删除**（不可恢复）：PG 行 + Qdrant 向量点 + 磁盘文件一起清。

    顺序纪律：**PG 先提交**（它是唯一事实来源），Qdrant 与磁盘后清且
    best-effort —— 它们失败只记日志并回给前端 `warnings`，不回滚 PG。
    反过来（先删文件再删 PG）的话，PG 事务一旦失败，文档还挂在列表里
    但已经读不出内容，用户看到"文档在、问答空了"这种最难解释的状态；
    而现在最坏情况只是留下几个孤儿文件/孤儿点，可解释、可重跑。

    ⚠️ 与"软删除"的差别不只是省磁盘，而是**软删的残留点真的能被重新召回**：

       1. 常规检索（无 `version_override`）→ `build_filter` 的 must 里带
          `is_active=true`（retrieval_service 末尾那条 append），软删点
          （is_active=false）不会被召回 —— 这一层软删是安全的。
       2. 但 `version_override` 分支是例外：被点名的 doc 走
          `(doc_id AND version)` 精确命中，**刻意不要求 is_active**
          （见该函数注释：点名历史版本时该版本本就是 false，必须绕过）。
          而 `doc_ids` 与 `version_override` 都直接来自请求体、服务端不做
          存在性校验 → 只要客户端点名一个已**软删**的 doc，它的点就会
          重新参与召回并出现在引用里。硬删除从根上消掉这个可能。

    删除范围：
        PG    : file_versions → documents（ingest_tasks **保留**，只解绑+快照，见下）
        Qdrant: 该 doc 全部版本的点（真删，非 is_active=false）
        磁盘  : `uploads/{user}/{doc}/` 与 `parsed/{doc}/`

    ⚠️ ingest_tasks 不删（2026-09-19 修：删文档连累任务历史）。
       任务历史是用户的操作审计，文档删了历史也应留下。
       处理方式：删 file_versions **之前**先把该 doc 的任务行
       `version_id 置 NULL`（外键是 ON DELETE SET NULL，本行也已无引用可级联）
       并把 file_name 快照进任务行 —— 否则 documents 行删除后，
       任务列表会显示不出文件名。
    """
    doc = await get_doc(session, user_id, doc_id)  # 内含归属校验 + FOR UPDATE
    # ⚠️ 必须在 DELETE 之前把要用的字段取出来：commit 后 ORM 对象被 expire，
    #    再访问 doc.file_name 会触发 refresh，而那一行已经不存在了
    #    → ObjectDeletedError（纯粹是"日志里想打印个名字"引发的 500）。
    doc_name = doc.file_name

    cfg = get_config()
    upload_root = Path(str(cfg.get_path("paths.upload_dir", "/app/storage/uploads")))
    parsed_root = Path(str(cfg.get_path("paths.parsed_dir", "/app/storage/parsed")))
    targets = [upload_root / str(user_id) / str(doc_id), parsed_root / str(doc_id)]

    # ---------- 1. PG：一个事务内删干净 ----------
    # ingest_tasks 保留（2026-09-19 修）：只把该 doc 的任务行与版本解绑
    # （version_id → NULL，外键为 ON DELETE SET NULL；显式置空是为了在
    #  删 file_versions 之前完成，避免依赖级联时机）并快照文件名，
    # 让任务历史在文档删除后仍可读。原先"事务内连 ingest_tasks 一起删"的
    # 行为已移除 —— 任务历史是操作审计，不该随文档消失。
    tasks_detached = int(
        (
            await session.execute(
                update(IngestTask)
                .where(IngestTask.doc_id == doc_id)
                .values(version_id=None, file_name=doc_name)
            )
        ).rowcount
        or 0
    )
    versions_deleted = int(
        (
            await session.execute(delete(FileVersion).where(FileVersion.doc_id == doc_id))
        ).rowcount
        or 0
    )
    await session.execute(delete(Document).where(Document.doc_id == doc_id))
    await session.commit()

    # ---------- 2. Qdrant：真删点 ----------
    deleted_vectors = 0
    warnings: list[str] = []
    try:
        deleted_vectors = get_vector_store().delete_by_doc(doc_id)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"向量清理失败（可能有残留点）: {exc}")
        logger.error("硬删除 Qdrant 清理失败: doc=%s %s", doc_id, exc)

    # ---------- 3. 磁盘：原件目录 + 解析产物目录 ----------
    # 放 to_thread：目录可能几百个文件，同步遍历会卡住事件循环（同 upload 的理由）。
    deleted_files, purge_warnings = await asyncio.to_thread(
        _purge_paths, targets, [upload_root, parsed_root]
    )
    warnings.extend(purge_warnings)

    logger.info(
        "文档硬删除: doc=%s name=%s versions=%s tasks_detached=%s vectors=%s files=%s warn=%s",
        doc_id, doc_name, versions_deleted, tasks_detached,
        deleted_vectors, deleted_files, len(warnings),
    )
    return DeleteOutcome(
        deleted_versions=versions_deleted,
        deleted_tasks=tasks_detached,
        deleted_vectors=deleted_vectors,
        deleted_files=deleted_files,
        warnings=warnings,
    )


# ============================================================
# chunk 预览
# ============================================================
async def preview_chunks(
    *, doc_id: uuid.UUID, version: int, offset: Any = None, limit: int = 20
) -> tuple[list[dict[str, Any]], Any]:
    """
    从 Qdrant 读回某版本的 chunk（供前端"版本内容"面板）。

    ⚠️ 分页用 Qdrant 的 **scroll offset**（游标），不是页码。
       原因：scroll 是"从某个点继续往下读"，与"第 N 页"不是同一语义，
       硬做页码换算会在过滤/排序变化时出现重复或遗漏。
       前端拿到 `next_offset` 后原样回传即可续拉。
    """
    store = get_vector_store()
    return store.scroll_chunks(doc_id, version, offset=offset, limit=limit)


__all__ = [
    "ActivateOutcome",
    "DeleteOutcome",
    "activate_version",
    "get_doc",
    "get_file_detail",
    "hard_delete",
    "list_files",
    "list_versions",
    "preview_chunks",
]
