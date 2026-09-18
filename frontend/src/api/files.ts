import { http, unwrap } from './http'
import type {
  ActivateResult,
  ChunkPreviewPage,
  DeleteResult,
  FileDetailResult,
  FileListItem,
  Page,
  UploadResult,
} from '../types/api'

/** 上传文档（multipart，字段名 file）。立即返回，解析在 Celery 异步做 */
export function uploadFile(file: File) {
  const form = new FormData()
  form.append('file', file)
  return unwrap<UploadResult>(
    http.post('/files/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300_000, // 大文件上传给足时间
    }),
  )
}

/** 文档列表（含解析状态） */
export function listFiles(page = 1, size = 20, keyword = '') {
  return unwrap<Page<FileListItem>>(
    http.get('/files', { params: { page, size, keyword: keyword || undefined } }),
  )
}

/** 文档详情 + 版本列表 */
export function getFileDetail(docId: string) {
  return unwrap<FileDetailResult>(http.get(`/files/${docId}`))
}

/** 激活某历史版本（切换后问答走该版本） */
export function activateVersion(docId: string, version: number) {
  return unwrap<ActivateResult>(
    http.post(`/files/${docId}/versions/${version}/activate`),
  )
}

/** chunk 预览（游标分页：把上次返回的 next_offset 原样回传） */
export function listChunks(docId: string, version: number, offset?: string, limit = 20) {
  return unwrap<ChunkPreviewPage>(
    http.get(`/files/${docId}/versions/${version}/chunks`, {
      params: { offset: offset || undefined, limit },
    }),
  )
}

/**
 * 硬删除文档（**不可恢复**）：PG 行 + Qdrant 向量点 + 磁盘文件一起清。
 * 返回体里的 warnings 若非空，表示 PG 已删但某些载体有残留，必须展示给用户。
 */
export function deleteFile(docId: string) {
  return unwrap<DeleteResult>(http.delete(`/files/${docId}`))
}
