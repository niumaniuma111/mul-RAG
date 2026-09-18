import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as filesApi from '../api/files'
import type { FileListItem } from '../types/api'

/**
 * 文档状态：列表缓存 + 「问答范围选择」用的已解析文档集合。
 * Chat 页的 DocScopeSelector 与 Documents 页共享这份数据。
 */
export const useDocumentsStore = defineStore('documents', () => {
  const items = ref<FileListItem[]>([])
  const total = ref(0)
  const loading = ref(false)
  /** 勾选为问答范围的 doc_id 列表（空 = 不限定，检索全部） */
  const selectedDocIds = ref<string[]>([])

  async function refresh(page = 1, size = 50) {
    loading.value = true
    try {
      const resp = await filesApi.listFiles(page, size)
      items.value = resp.items
      total.value = resp.total
    } finally {
      loading.value = false
    }
  }

  /** 只挑解析成功的文档供问答选择 */
  const readyDocs = () => items.value.filter((d) => d.parse_status === 'success')

  function toggleSelect(docId: string) {
    const idx = selectedDocIds.value.indexOf(docId)
    if (idx >= 0) selectedDocIds.value.splice(idx, 1)
    else selectedDocIds.value.push(docId)
  }

  return { items, total, loading, selectedDocIds, refresh, readyDocs, toggleSelect }
})
