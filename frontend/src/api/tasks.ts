import { http, unwrap } from './http'
import type { Page, TaskDetailResult, TaskItem, TaskLogsResult, TaskRetryResult } from '../types/api'

/** 任务列表（可按状态过滤） */
export function listTasks(params: { page?: number; size?: number; status?: string } = {}) {
  return unwrap<Page<TaskItem>>(http.get('/tasks', { params }))
}

/** 任务详情（轮询用：进度 / 阶段 / 状态） */
export function getTask(taskId: string) {
  return unwrap<TaskDetailResult>(http.get(`/tasks/${taskId}`))
}

/** 失败日志全文 */
export function getTaskLogs(taskId: string) {
  return unwrap<TaskLogsResult>(http.get(`/tasks/${taskId}/logs`))
}

/** 重试失败任务 */
export function retryTask(taskId: string) {
  return unwrap<TaskRetryResult>(http.post(`/tasks/${taskId}/retry`))
}

/** 删除任务记录（不删文件） */
export function deleteTask(taskId: string) {
  return http.delete(`/tasks/${taskId}`)
}
