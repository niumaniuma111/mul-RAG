"""
SQLAlchemy ORM 模型 —— 与 deploy/postgres/init/01_schema.sql 严格一一对应。

为什么要有 ORM 模型（而不是裸 SQL）
-----------------------------------
建表仍然由 `01_schema.sql` 负责（它是权威，且只在数据卷为空时跑一次）；
ORM 的职责是**读写**。两者必须保持同步 —— 改表结构时两边都要动。

⚠️ 但**不要**用 `Base.metadata.create_all()` 建表。理由：
    1. 库里还有 LangGraph 的 checkpoint 表与触发器，SQL 文件是完整事实来源；
    2. 生产改表要走 alembic（requirements 里已装），而不是 ORM 自动建。

对应设计文档：DESIGN.md §5.1
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def _uuid_pk() -> Mapped[uuid.UUID]:
    """UUID 主键：由 PG 的 gen_random_uuid() 生成，Python 侧不赋初值。"""
    return mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )


# ============================================================
# 1. users
# ============================================================
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    email: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# ============================================================
# 2. documents —— 逻辑文档（版本切换的锚点）
# ============================================================
class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("user_id", "file_name", name="uq_doc_user_name"),
    )

    doc_id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # 展示名，同名即同文档 —— 这是「重传 = 新版本」的判定依据
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_ext: Mapped[str] = mapped_column(String(16), nullable=False)
    # 最新上传到的版本号
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    # ★ 当前生效版本 —— 检索过滤 is_active=true 对应的就是它
    active_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    versions: Mapped[list[FileVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="FileVersion.version",
    )


# ============================================================
# 3. file_versions —— 物理版本
# ============================================================
class FileVersion(Base):
    __tablename__ = "file_versions"
    __table_args__ = (
        UniqueConstraint("doc_id", "version", name="uq_version"),
        CheckConstraint(
            "parse_status IN ('pending','parsing','success','failed')",
            name="ck_parse_status",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    # 冗余字段：检索/权限判断时省一次 documents 的 join
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_md5: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # pending / parsing / success / failed
    parse_status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # ★ 与 Qdrant 的 is_active 同步（PG 是事实来源，Qdrant 是投影）
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document: Mapped[Document] = relationship(back_populates="versions")


# ============================================================
# 4. ingest_tasks —— 入库任务（前端轮询）
# ============================================================
class IngestTask(Base):
    __tablename__ = "ingest_tasks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # ⚠️ 刻意不建 FK 到 documents：doc_id 在 documents 表里是 UUID 主键，
    #    但 ingest_tasks.doc_id 在 schema 里是无约束列（见 01_schema.sql），
    #    这样即使文档被硬删，任务记录仍可用于排障。
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # 可空 + ON DELETE SET NULL：文档硬删时任务行**保留**，只解除与版本的关联
    # （2026-09-19 修：原先 CASCADE 会把任务历史连着一起删掉）。
    version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("file_versions.id", ondelete="SET NULL"), nullable=True
    )
    # 文件名快照：documents 行删除后，任务列表靠它仍能显示文件名
    # （快照在 hard_delete 时写入；存续期显示以 documents.file_name 为准）
    file_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # pending / queued / running / success / failed
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    progress: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    # parsing / captioning / chunking / embedding
    current_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ============================================================
# 5. 会话与消息（S3 在线问答使用，此处先定义以保持模型完整）
# ============================================================
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, server_default="新对话")
    # LangGraph checkpoint 的 thread_id（两者必须一致，否则恢复不了）
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    reflect_info: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# ============================================================
# 6. qa_traces —— Agent 链路追踪
# ============================================================
class QaTrace(Base):
    __tablename__ = "qa_traces"

    id: Mapped[uuid.UUID] = _uuid_pk()
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    normalized_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_keywords: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    retrieval_scores: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    hitl_triggered: Mapped[bool] = mapped_column(Boolean, server_default="false")
    hitl_choice: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reflect_rounds: Mapped[int] = mapped_column(SmallInteger, server_default="0")
    reflect_verdict: Mapped[str | None] = mapped_column(String(8), nullable=True)
    reflect_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


__all__ = [
    "Base",
    "ChatMessage",
    "ChatSession",
    "Document",
    "FileVersion",
    "IngestTask",
    "QaTrace",
    "User",
]
