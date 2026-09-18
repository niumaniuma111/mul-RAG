import { http, unwrap } from './http'
import type { Page, TaskItem } from '../types/api'

/**
 * 问答相关的轻量封装。
 *
 * ⚠️ 会话历史的接口**不在这里**，在 `./sessions.ts`（B2 新增）：
 *      GET /qa/sessions                     会话列表
 *      GET /qa/sessions/{id}/messages       会话内的历史消息
 *   SSE 流式问答不走 axios，见 `composables/useSSE.ts`。
 *
 * 本模块只提供文档列表的轻量封装给「文档范围选择器」复用。
 */
export type { Page, TaskItem }

export function listSessionsFallback() {
  // 兼容旧调用点：现在会话列表走 api/sessions.ts，这里是空实现。
  return Promise.resolve<{ items: never[] }>({ items: [] })
}

export { http, unwrap }
