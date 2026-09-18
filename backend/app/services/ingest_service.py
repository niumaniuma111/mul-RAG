"""
入库编排：上传登记 + 派发异步任务（DESIGN.md §7.1）。

时序（与设计文档 §7.1 一一对应）
------------------------------
    1. 校验扩展名 / 大小（在 api 层做）
    2. 计算 md5，按 (user_id, file_name) 查 documents
         不存在 → INSERT documents(version=1, active_version=1)
         存在   → current_version += 1
    3. INSERT file_versions(version=N, is_active=TRUE, parse_status='pending')
    4. 事务内下线旧版本：旧版本 is_active=FALSE，documents.active_version=N
    5. 事务外异步对齐 Qdrant 的 is_active（失败不影响上传成功）
    6. INSERT ingest_tasks(status='pending')
    7. 返回 {doc_id, version, version_id, task_id}

⚠️ 关于「先 commit 再派发 Celery」这个顺序
----------------------------------------
Celery worker 是**独立进程**，它按 task_id 去数据库读任务行。
如果先 send_task 再 commit，worker 可能抢先执行 → 查不到行 → 任务失败。
所以 `register_upload` 内部**显式 commit**，把事务边界收在函数内，
之后调用方才派发任务。这是本项目里唯一一处服务层主动 commit 的地方，
原因是它有跨进程的副作用，不能等请求结束。

⚠️ PG 与 Qdrant 的 is_active 更新不在同一事务（跨存储无法原子）
    PG 是唯一事实来源，Qdrant 是投影。步骤 5 失败只记日志不抛错 ——
    检索侧会用 PG 的 active_version 二次校验；要修投影时**重新切一次同一
    版本**即可（`set_active_by_version` 幂等：先全置 false，再点目标版本）。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_config
from app.core.exceptions import BadRequestError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.db.models import Document, FileVersion, IngestTask
from app.services.vector_store import get_vector_store

logger = get_logger(__name__)

# 文件名安全化：只保留字母数字、下划线、横线、点、常见中文字符范围
_UNSAFE_RE = re.compile(r"[^0-9A-Za-z._\-\u4e00-\u9fff\u3040-\u30ff]+")


@dataclass(slots=True)
class UploadOutcome:
    doc_id: uuid.UUID
    version: int
    version_id: int
    task_id: uuid.UUID
    is_new_doc: bool


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """
    清洗文件名，防止路径穿越（`../`）与非法字符。

    ⚠️ 绝不能直接用用户传来的文件名拼路径 —— `../../etc/passwd` 这类输入
       会让上传接口变成任意文件写。这里把不安全字符统一替换成下划线，
       并且只取 basename（丢掉任何目录成分）。
    """
    base = Path(name).name           # 丢掉目录成分，这是防穿越的关键一步
    cleaned = _UNSAFE_RE.sub("_", base).strip("._") or "upload"
    return cleaned[:max_len]


def compute_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """
    计算文件 md5。

    分块读（1MB）而不是一次性读入内存 —— 允许上传 200MB 的文件，
    而 api 容器的 mem_limit 只有 512MB。
    """
    digest = hashlib.md5()  # noqa: S324 —— 仅用于去重标识，非安全用途
    with open(path, "rb") as fp:
        while True:
            block = fp.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_upload(file_name: str, file_size: int) -> str:
    """
    校验扩展名与大小，返回规范化后的扩展名（小写、带点）。

    两处都按 config 判定，不写死 —— 改策略只改 config.yaml。
    """
    cfg = get_config()
    allowed = [str(e).lower() for e in (cfg.get_path("ingest.allowed_ext", []) or [])]
    max_mb = int(cfg.get_path("ingest.max_file_size_mb", 200))

    suffix = Path(file_name).suffix.lower()
    if not suffix or suffix not in allowed:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型 {suffix or '(无扩展名)'}，允许：{', '.join(allowed)}"
        )
    if file_size <= 0:
        raise BadRequestError("文件为空")
    if file_size > max_mb * 1024 * 1024:
        raise BadRequestError(f"文件超过 {max_mb}MB 上限")
    return suffix


def upload_target_path(user_id: int, doc_id: uuid.UUID, version: int, file_name: str) -> Path:
    """
    上传文件的落盘路径：`{upload_dir}/{user_id}/{doc_id}/v{version}_{安全名}`

    按 user/doc 分目录而不是全平铺：单目录几十万个文件会让 ls / 备份 / 排障都变得难受。
    """
    root = Path(str(get_config().get_path("paths.upload_dir", "/app/storage/uploads")))
    safe = sanitize_filename(file_name)
    return root / str(user_id) / str(doc_id) / f"v{version}_{safe}"


# ============================================================
# 主流程
# ============================================================
async def register_upload(
    session: AsyncSession,
    *,
    user_id: int,
    file_name: str,
    file_ext: str,
    file_path: str,
    file_size: int,
    file_md5: str,
) -> UploadOutcome:
    """
    登记一次上传，返回 doc_id / version / version_id / task_id。

    ⚠️ 本函数**自己 commit**（理由见模块头）。
    """
    # ---------- 2. 查/建 documents ----------
    doc = await session.scalar(
        select(Document)
        .where(Document.user_id == user_id, Document.file_name == file_name)
        .with_for_update()          # 行锁：并发上传同名文件时避免版本号撞车
    )

    is_new_doc = doc is None
    if doc is None:
        doc = Document(
            user_id=user_id,
            file_name=file_name,
            file_ext=file_ext,
            current_version=1,
            active_version=1,
            is_deleted=False,
        )
        session.add(doc)
        await session.flush()        # 拿到 doc_id
        version = 1
        logger.info("新建文档: doc=%s name=%s", doc.doc_id, file_name)
    else:
        # 历史兼容分支：文档删除已改为**硬删除**（行直接消失，见
        # version_service.hard_delete），所以新数据不会再有 is_deleted=True。
        # 这里保留复位逻辑，只为兜住早期软删留下的存量行 + 字段本身是 NOT NULL。
        if doc.is_deleted:
            doc.is_deleted = False
            logger.info("恢复已删除文档: doc=%s", doc.doc_id)
        version = int(doc.current_version) + 1
        doc.current_version = version
        doc.active_version = version
        logger.info("为文档追加新版本: doc=%s version=%s", doc.doc_id, version)

    # ---------- 3. 建 file_versions ----------
    version_row = FileVersion(
        doc_id=doc.doc_id,
        user_id=user_id,
        version=version,
        file_path=file_path,
        file_size=file_size,
        file_md5=file_md5,
        parse_status="pending",
        is_active=True,
    )
    session.add(version_row)
    await session.flush()
    version_id = int(version_row.id)

    # ---------- 4. 事务内下线旧版本 ----------
    if not is_new_doc:
        await session.execute(
            update(FileVersion)
            .where(FileVersion.doc_id == doc.doc_id, FileVersion.version < version)
            .values(is_active=False)
        )

    # ---------- 6. 建任务 ----------
    task = IngestTask(
        doc_id=doc.doc_id,
        version_id=version_id,
        user_id=user_id,
        status="pending",
        progress=0,
        current_stage="pending",
    )
    session.add(task)
    await session.flush()
    task_id = task.id

    # ★ 显式提交（跨进程副作用要求，见模块头）
    await session.commit()

    outcome = UploadOutcome(
        doc_id=doc.doc_id,
        version=version,
        version_id=version_id,
        task_id=task_id,
        is_new_doc=is_new_doc,
    )

    # ---------- 5. 事务外对齐 Qdrant（best-effort） ----------
    try:
        get_vector_store().set_active_by_version(doc.doc_id, version)
    except Exception as exc:  # noqa: BLE001 —— 投影失败不能影响上传成功
        logger.warning(
            "Qdrant is_active 对齐失败（不影响上传，检索侧会二次校验）: doc=%s %s",
            doc.doc_id, exc,
        )

    logger.info(
        "上传登记完成: doc=%s version=%s task=%s is_new=%s",
        outcome.doc_id, outcome.version, outcome.task_id, outcome.is_new_doc,
    )
    return outcome


async def count_documents(session: AsyncSession, user_id: int, keyword: str | None = None) -> int:
    """列表页总数。keyword 走 file_name 的 ILIKE（数据量不大，无需全文索引）。"""
    stmt = select(func.count()).select_from(Document).where(Document.user_id == user_id)
    if keyword:
        stmt = stmt.where(Document.file_name.ilike(f"%{keyword}%"))
    return int(await session.scalar(stmt) or 0)


__all__ = [
    "UploadOutcome",
    "compute_md5",
    "count_documents",
    "register_upload",
    "sanitize_filename",
    "upload_target_path",
    "validate_upload",
]
