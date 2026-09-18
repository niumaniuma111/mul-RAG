-- ============================================================
-- 01_schema.sql — 业务库建表
-- ------------------------------------------------------------
-- 由 postgres 容器首次启动时自动执行（/docker-entrypoint-initdb.d/）。
-- ⚠️ 只在数据卷为空时执行一次。改了本文件要重建卷：docker compose down -v
--
-- 注意：LangGraph 的 checkpoint 表（checkpoints / checkpoint_writes /
--       checkpoint_blobs）不在这里手写，由 AsyncPostgresSaver.setup() 自动创建。
--
-- 对应设计文档：DESIGN.md §5.1
-- ============================================================

-- ============================================================
-- 1. 用户表
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    email         VARCHAR(128) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 2. 逻辑文档表（版本切换的锚点）
--    doc_id 即 chunk metadata 中的 file_id
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name      VARCHAR(512) NOT NULL,           -- 展示名，同名即同文档
    file_ext       VARCHAR(16)  NOT NULL,           -- pptx / pdf
    current_version INT         NOT NULL DEFAULT 1, -- 最新上传的版本号
    active_version  INT         NOT NULL DEFAULT 1, -- 当前生效版本（可手动切换）
    is_deleted     BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_doc_user_name UNIQUE (user_id, file_name)
);

COMMENT ON COLUMN documents.active_version IS '检索过滤 is_active=true 对应的版本；切换历史版本时改写此字段';

-- ============================================================
-- 3. 物理版本表
-- ============================================================
CREATE TABLE IF NOT EXISTS file_versions (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        UUID    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    user_id       BIGINT  NOT NULL,                  -- 冗余，便于直接过滤
    version       INT     NOT NULL,
    file_path     TEXT    NOT NULL,                  -- 容器内绝对路径
    file_size     BIGINT,
    file_md5      CHAR(32),
    parse_status  VARCHAR(16) NOT NULL DEFAULT 'pending',
    -- pending / parsing / success / failed
    chunk_count   INT     NOT NULL DEFAULT 0,
    page_count    INT,
    is_active     BOOLEAN NOT NULL DEFAULT FALSE,    -- ★ 与 Qdrant 同步
    error_message TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parsed_at     TIMESTAMPTZ,
    CONSTRAINT uq_version UNIQUE (doc_id, version),
    CONSTRAINT ck_parse_status CHECK (
        parse_status IN ('pending','parsing','success','failed')
    )
);

COMMENT ON COLUMN file_versions.is_active IS '旧版本置 false 但保留行与向量，支持溯源与手动切回';

-- ============================================================
-- 4. 入库任务表（前端轮询）
-- ============================================================
CREATE TABLE IF NOT EXISTS ingest_tasks (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id         UUID    NOT NULL,
    -- 可空 + SET NULL：文档硬删时任务行保留，只解除与版本的关联
    -- （2026-09-19 修：原先 CASCADE 会把任务历史连着一起删掉）
    version_id     BIGINT  REFERENCES file_versions(id) ON DELETE SET NULL,
    -- 文件名快照：documents 行删除后任务列表仍能显示文件名
    file_name      VARCHAR(512),
    user_id        BIGINT  NOT NULL,
    celery_task_id VARCHAR(64),
    status         VARCHAR(16) NOT NULL DEFAULT 'pending',
    -- pending / queued / running / success / failed
    progress       SMALLINT    NOT NULL DEFAULT 0,   -- 0-100
    current_stage  VARCHAR(32),                      -- parsing/captioning/chunking/embedding
    chunk_count    INT         NOT NULL DEFAULT 0,
    error_log      TEXT,
    retry_count    SMALLINT    NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ
);

-- ============================================================
-- 5. 会话与消息
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      VARCHAR(255) NOT NULL DEFAULT '新对话',
    thread_id  VARCHAR(64)  NOT NULL UNIQUE,   -- LangGraph checkpoint thread_id
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id      BIGINT NOT NULL,
    role         VARCHAR(16) NOT NULL,          -- user / assistant
    content      TEXT NOT NULL,
    citations    JSONB,                         -- 引用列表快照
    reflect_info JSONB,                         -- 自省结论快照
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 6. Agent 链路追踪（前端展示推理链路）
-- ============================================================
CREATE TABLE IF NOT EXISTS qa_traces (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id          UUID REFERENCES chat_messages(id) ON DELETE CASCADE,
    session_id          UUID NOT NULL,
    user_id             BIGINT NOT NULL,
    normalized_query    TEXT,
    retrieval_keywords  JSONB,                  -- 检索决策节点改写后的关键词
    retrieval_scores    JSONB,                  -- rerank 分数（含每路召回）
    hitl_triggered      BOOLEAN DEFAULT FALSE,
    hitl_choice         VARCHAR(32),
    reflect_rounds      SMALLINT DEFAULT 0,
    reflect_verdict     VARCHAR(8),             -- PASS / FAIL / PENDING（HITL 挂起未恢复）
    reflect_reason      TEXT,
    total_latency_ms    INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 7. LangGraph checkpoint
--    由 AsyncPostgresSaver.setup() 自动建表，此处仅说明，不手写。
--    表名固定：checkpoints / checkpoint_writes / checkpoint_blobs
-- ============================================================
