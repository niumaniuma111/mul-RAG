/**
 * 后端 API 的 TS 类型定义 —— 与 backend/app/schemas/*.py 一一对应。
 * 字段名以 schemas 为准（serde 由 pydantic 保证 snake_case）。
 */

// ---------- 通用 ----------
export interface ApiEnvelope<T> {
  code: number // 0 = 成功
  message: string
  data: T | null
}

export interface Page<T> {
  items: T[]
  total: number
  page: number
  size: number
}

// ---------- 认证 ----------
export interface TokenResult {
  access_token: string
  token_type: string
  expires_in: number
  user_id: number
  username: string
}

export interface MeResult {
  user_id: number
  username: string
  email: string | null
  created_at: string
}

// ---------- 文档 / 版本 ----------
export interface UploadResult {
  doc_id: string
  version: number
  version_id: number
  task_id: string
  is_new_doc: boolean
}

export interface FileListItem {
  doc_id: string
  file_name: string
  file_ext: string
  current_version: number
  active_version: number
  parse_status: string // pending/parsing/success/failed
  chunk_count: number
  page_count: number | null
  created_at: string
  updated_at: string
}

export interface VersionItem {
  version: number
  is_active: boolean
  parse_status: string
  chunk_count: number
  page_count: number | null
  file_size: number | null
  error_message: string | null
  created_at: string
  parsed_at: string | null
}

export interface FileDetailResult {
  doc_id: string
  file_name: string
  file_ext: string
  current_version: number
  active_version: number
  is_deleted: boolean
  created_at: string
  updated_at: string
  versions: VersionItem[]
}

export interface ActivateResult {
  doc_id: string
  active_version: number
  updated_chunks: number
  updated_vectors: number
}

export interface ChunkPreviewItem {
  chunk_index: number
  chunk_type: string
  page_num: number | null
  content: string
  image_path: string | null
}

export interface ChunkPreviewPage {
  items: ChunkPreviewItem[]
  next_offset: string | null
}

/**
 * DELETE /files/{doc_id} —— 硬删除结果（不可恢复）。
 * warnings 非空 = PG 已删成功但载体有残留（如 Qdrant 短暂不可用），前端须显式提示。
 */
export interface DeleteResult {
  doc_id: string
  deleted_versions: number
  deleted_tasks: number
  deleted_vectors: number
  deleted_files: number
  warnings: string[]
}

// ---------- 任务 ----------
export interface TaskItem {
  task_id: string
  doc_id: string
  version_id: number
  version: number | null
  file_name: string | null
  status: string // pending/queued/running/success/failed
  progress: number
  current_stage: string | null
  chunk_count: number
  retry_count: number
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface TaskDetailResult extends TaskItem {
  error_log: string | null
}

export interface TaskLogsResult {
  task_id: string
  status: string
  retry_count: number
  error_log: string | null
}

// ---------- 问答 ----------
export interface Citation {
  index: number
  point_id?: string
  doc_id?: string
  file_name?: string
  page_num?: number
  version?: number
  chunk_type?: 'text' | 'table' | 'image_caption'
  snippet?: string
  table_html?: string | null
  image_path?: string | null
  rerank_score?: number
}

export interface FinalPayload {
  answer: string
  citations: Citation[]
  finish_reason: 'pass' | 'max_rounds' | 'hitl_skip' | 'no_retrieval' | 'error'
}

export interface HitlPayload {
  reason: string
  max_score: number
  threshold: number
  query: string
  options: string[]
}

export interface ReflectPayload {
  round: number
  verdict: 'PASS' | 'FAIL' | string
  reason: string
}

// ---------- 推理链路 ----------
export interface TraceStep {
  node: string
  label: string
  status: 'running' | 'done' | 'error'
}
