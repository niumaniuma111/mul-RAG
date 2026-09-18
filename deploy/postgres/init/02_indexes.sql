-- ============================================================
-- 02_indexes.sql — 索引与触发器
-- ------------------------------------------------------------
-- 对应设计文档：DESIGN.md §5.2
--
-- 索引设计要点：
--   · idx_documents_user / idx_versions_active 是「部分索引」(WHERE)，
--     只索引活跃行，比全量索引更小更快。
--   · 版本切换是高频操作（activate 接口），所以 (doc_id, version DESC)
--     和 (user_id, is_active) 两条都必须存在。
-- ============================================================

-- ---- documents ----
CREATE INDEX IF NOT EXISTS idx_documents_user
    ON documents(user_id) WHERE is_deleted = FALSE;

-- ---- file_versions ----
CREATE INDEX IF NOT EXISTS idx_versions_doc
    ON file_versions(doc_id, version DESC);

CREATE INDEX IF NOT EXISTS idx_versions_active
    ON file_versions(user_id, is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_versions_status
    ON file_versions(parse_status);

-- ---- ingest_tasks ----
CREATE INDEX IF NOT EXISTS idx_tasks_user_created
    ON ingest_tasks(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tasks_status
    ON ingest_tasks(status);

-- ---- chat ----
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON chat_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON chat_sessions(user_id, updated_at DESC);

-- ---- qa_traces ----
CREATE INDEX IF NOT EXISTS idx_traces_session
    ON qa_traces(session_id);

-- ============================================================
-- updated_at 自动维护
-- ------------------------------------------------------------
-- DESIGN.md §5.2 的注释写的是「索引与触发器」，但未给出触发器定义，
-- 这里按标准做法补齐。理由：updated_at 由 DB 层维护比应用层可靠 ——
-- 应用层容易在某个 repository 里漏掉赋值，且并发写入时会互相覆盖。
-- ============================================================
CREATE OR REPLACE FUNCTION trg_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at_users ON users;
CREATE TRIGGER set_updated_at_users
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

DROP TRIGGER IF EXISTS set_updated_at_documents ON documents;
CREATE TRIGGER set_updated_at_documents
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

DROP TRIGGER IF EXISTS set_updated_at_chat_sessions ON chat_sessions;
CREATE TRIGGER set_updated_at_chat_sessions
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();
