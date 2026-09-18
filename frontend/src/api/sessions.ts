import { http } from './http'
import type { Citation, ReflectPayload } from '../types/api'

/**
 * 会话历史（B2）。
 *
 * 历史：
 *   S4 时后端没有读接口，会话列表与标题都存在 `localStorage`，
 *   切会话只能把消息清空（因为没有正文可取）—— 这直接导致坑 37：
 *   点一下当前会话就把正在看的一轮问答从界面抹掉。
 *   本模块接上 `GET /qa/sessions` 与 `GET /qa/sessions/{id}/messages` 后，
 *   `chat.ts` 才能把"清空"改成"加载历史"。
 *
 * 注意：这里的 `http` 已由拦截器解包成 `data`，所以返回类型直接是 data 的结构。
 */

export interface SessionItem {
  session_id: string
  title: string
  message_count: number
  created_at: string
  updated_at: string
}

export interface SessionPage {
  items: SessionItem[]
  total: number
  page: number
  size: number
}

export interface MessageItem {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[] | null
  reflect_info: ReflectPayload | null
  created_at: string
}

export interface SessionMessages {
  session_id: string
  title: string
  items: MessageItem[]
  total: number
  page: number
  size: number
}

/** 会话列表（当前用户，按最近更新倒序） */
export function listSessions(page = 1, size = 50) {
  return http.get<SessionPage>('/qa/sessions', { params: { page, size } }).then((r) => r.data)
}

/**
 * 某会话的历史消息（按对话顺序正序）。
 *
 * `size` 默认给大一点（100）：切换会话是"一次性把这段对话铺出来"的交互，
 * 分页反而会让用户看到半截对话。真有超长会话再用分页。
 */
export function listMessages(sessionId: string, page = 1, size = 100) {
  return http
    .get<SessionMessages>(`/qa/sessions/${sessionId}/messages`, { params: { page, size } })
    .then((r) => r.data)
}
