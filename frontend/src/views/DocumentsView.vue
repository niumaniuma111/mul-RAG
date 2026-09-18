<script setup lang="ts">
/**
 * 文档管理页：上传 / 列表（轮询刷新）/ 版本切换 / chunk 预览 / 硬删除。
 * 轮询策略见 DESIGN §13.4：上传后 2s 轮询任务详情，success/failed 停。
 */
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as filesApi from '../api/files'
import * as tasksApi from '../api/tasks'
import type { ChunkPreviewItem, FileDetailResult, FileListItem } from '../types/api'
import { useDocumentsStore } from '../stores/documents'
import { usePolling } from '../composables/usePolling'
import UploadDialog from '../components/UploadDialog.vue'
import TaskStageTag from '../components/TaskStageTag.vue'

const docs = useDocumentsStore()

const uploadVisible = ref(false)
/** 上传后的任务 id：非空时轮询 */
const activeTaskId = ref('')
const activeTaskStatus = ref('')

// ---------- 轮询：任务进度 + 列表刷新 ----------
const snapshot = ref('')
async function pollTask() {
  if (!activeTaskId.value) return
  const t = await tasksApi.getTask(activeTaskId.value)
  activeTaskStatus.value = t.status
  const snap = `${t.status}:${t.progress}:${t.current_stage}`
  pollCtl.markChanged(snap)
  if (t.status === 'success') {
    ElMessage.success('解析完成，文档已可问答')
    activeTaskId.value = ''
    docs.refresh()
  } else if (t.status === 'failed') {
    ElMessage.error('解析失败，可在「任务历史」查看日志并重试')
    activeTaskId.value = ''
    docs.refresh()
  } else {
    docs.refresh()
  }
}
const pollCtl = usePolling(pollTask, () => !!activeTaskId.value)

function onUploaded(result: { task_id: string }) {
  activeTaskId.value = result.task_id
  activeTaskStatus.value = 'pending'
  docs.refresh()
}

// ---------- 版本详情 / 预览 ----------
const detail = ref<FileDetailResult | null>(null)
const detailLoading = ref(false)

async function showDetail(row: FileListItem) {
  detailLoading.value = true
  try {
    detail.value = await filesApi.getFileDetail(row.doc_id)
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    detailLoading.value = false
  }
}

// chunk 预览（游标分页）
const previewVisible = ref(false)
const previewItems = ref<ChunkPreviewItem[]>([])
const previewOffset = ref<string | null>(null)
const previewDoc = ref<FileListItem | null>(null)
const previewLoading = ref(false)

async function showPreview(row: FileListItem) {
  previewDoc.value = row
  previewItems.value = []
  previewOffset.value = null
  previewVisible.value = true
  await loadChunks(row, null)
}

async function loadChunks(row: FileListItem, offset: string | null) {
  previewLoading.value = true
  try {
    const page = await filesApi.listChunks(row.doc_id, row.active_version, offset ?? undefined)
    previewItems.value.push(...page.items)
    previewOffset.value = page.next_offset
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    previewLoading.value = false
  }
}

// ---------- 操作 ----------
async function activate(row: FileListItem, version: number) {
  // ⚠️ confirm 必须自己包 try：它在用户点「取消」时以 reject('cancel'|'close')
  //    结束，不捕获就是未处理的 Promise rejection → 控制台飘红。
  //    这里的 catch 只吞这一种语义明确的"取消"，不是空吞错误面。
  try {
    await ElMessageBox.confirm(
      `切换后问答将使用 v${version} 的切片（当前 v${row.active_version}）。确认切换？`,
      '切换生效版本',
      { type: 'warning' },
    )
  } catch {
    return
  }
  try {
    const r = await filesApi.activateVersion(row.doc_id, version)
    ElMessage.success(`已切到 v${r.active_version}，更新 ${r.updated_chunks} 个切片`)
    docs.refresh()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

async function remove(row: FileListItem) {
  // ⚠️ 硬删除不可恢复：文案必须把"删了什么"讲清楚，而不是只说"删除"。
  //    用户在确认框里唯一能获取信息的渠道就是这段文字。
  // ⚠️ confirm 在用户点「取消」时以 reject('cancel'|'close') 结束，
  //    不捕获就是未处理的 Promise rejection → 控制台飘红。
  try {
    await ElMessageBox.confirm(
      `将永久删除「${row.file_name}」——数据库记录、向量索引、磁盘文件（原件与解析产物）一并清除，且无法恢复。确认删除？`,
      '硬删除文档（不可恢复）',
      {
        type: 'warning',
        confirmButtonText: '永久删除',
        cancelButtonText: '取消',
        confirmButtonClass: 'el-button--danger',
      },
    )
  } catch {
    return
  }
  try {
    const r = await filesApi.deleteFile(row.doc_id)
    // 后端把"PG 删成功但向量/文件没清干净"回报为 warnings —— 不能当成功吞掉，
    // 否则用户以为删干净了，实际库里还有可被召回的残留点。
    if (r.warnings?.length) {
      ElMessage.warning(`已删除，但存在残留：${r.warnings.join('；')}`)
    } else {
      ElMessage.success(
        `已永久删除（版本 ${r.deleted_versions} · 切片 ${r.deleted_vectors} · 文件 ${r.deleted_files}）`,
      )
    }
    docs.refresh()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

onMounted(() => docs.refresh())
</script>

<template>
  <div class="doc-page">
    <div class="toolbar">
      <span class="title">文档管理</span>
      <el-button type="primary" @click="uploadVisible = true">上传文档</el-button>
    </div>

    <el-table
      :data="docs.items"
      v-loading="docs.loading"
      class="table"
      :header-cell-style="{ background: '#f5f7fa' }"
    >
      <el-table-column prop="file_name" label="文件名" min-width="220" show-overflow-tooltip />
      <el-table-column label="版本" width="140">
        <template #default="{ row }">
          生效 v{{ row.active_version }} / 共 {{ row.current_version }}
        </template>
      </el-table-column>
      <el-table-column label="解析状态" width="200">
        <template #default="{ row }">
          <TaskStageTag :status="row.parse_status === 'success' ? 'success' : row.parse_status === 'failed' ? 'failed' : 'running'" />
        </template>
      </el-table-column>
      <el-table-column label="切片 / 页" width="110">
        <template #default="{ row }">{{ row.chunk_count }} / {{ row.page_count ?? '-' }}</template>
      </el-table-column>
      <el-table-column prop="updated_at" label="更新时间" width="170">
        <template #default="{ row }">{{ new Date(row.updated_at).toLocaleString() }}</template>
      </el-table-column>
      <el-table-column label="操作" width="230" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" @click="showDetail(row)">版本</el-button>
          <el-button link type="primary" @click="showPreview(row)">切片预览</el-button>
          <el-button link type="danger" @click="remove(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <UploadDialog v-model="uploadVisible" @uploaded="onUploaded" />

    <!-- 版本详情 -->
    <el-dialog :model-value="!!detail" title="版本管理" width="720px" align-center @closed="detail = null">
      <template v-if="detail">
        <p class="d-name">{{ detail.file_name }}</p>
        <el-table :data="detail.versions" size="small" :header-cell-style="{ background: '#f5f7fa' }">
          <el-table-column prop="version" label="版本" width="70" />
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <TaskStageTag :status="row.parse_status === 'success' ? 'success' : row.parse_status === 'failed' ? 'failed' : 'running'" />
            </template>
          </el-table-column>
          <el-table-column label="切片" width="70">
            <template #default="{ row }">{{ row.chunk_count }}</template>
          </el-table-column>
          <el-table-column label="生效中" width="90">
            <template #default="{ row }">
              <el-tag v-if="row.is_active" type="success" size="small">生效</el-tag>
              <el-button
                v-else
                link
                type="primary"
                size="small"
                @click="activate({ doc_id: detail!.doc_id, active_version: detail!.active_version } as FileListItem, row.version)"
              >
                切换
              </el-button>
            </template>
          </el-table-column>
          <el-table-column prop="created_at" label="上传时间" min-width="150">
            <template #default="{ row }">{{ new Date(row.created_at).toLocaleString() }}</template>
          </el-table-column>
          <el-table-column prop="error_message" label="错误" min-width="140" show-overflow-tooltip />
        </el-table>
      </template>
    </el-dialog>

    <!-- 切片预览 -->
    <el-dialog v-model="previewVisible" title="切片预览（当前生效版本）" width="760px" align-center top="6vh">
      <div v-if="previewDoc" class="preview-meta">
        {{ previewDoc.file_name }} · v{{ previewDoc.active_version }} · {{ previewItems.length }} 条
      </div>
      <div v-for="c in previewItems" :key="c.chunk_index" class="chunk-item">
        <div class="chunk-head">
          <el-tag size="small" effect="plain" type="info">{{ c.chunk_type }}</el-tag>
          <span class="chunk-idx">#{{ c.chunk_index }}</span>
          <span v-if="c.page_num != null" class="chunk-page">第 {{ c.page_num + 1 }} 页</span>
        </div>
        <pre class="chunk-body">{{ c.content }}</pre>
      </div>
      <div class="load-more">
        <el-button v-if="previewOffset && previewDoc" :loading="previewLoading" size="small" @click="loadChunks(previewDoc, previewOffset)">
          加载更多
        </el-button>
        <span v-else-if="!previewLoading" class="no-more">没有更多了</span>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.doc-page {
  height: 100%;
  padding: 16px;
  overflow-y: auto;
}
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.title {
  font-size: 16px;
  font-weight: 600;
}
.table {
  border-radius: 8px;
}
.d-name {
  margin-bottom: 10px;
  font-weight: 600;
}
.preview-meta {
  margin-bottom: 10px;
  font-size: 12px;
  color: #909399;
}
.chunk-item {
  margin-bottom: 10px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 8px 12px;
}
.chunk-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.chunk-idx,
.chunk-page {
  font-size: 12px;
  color: #909399;
}
.chunk-body {
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  color: #606266;
  white-space: pre-wrap;
  max-height: 120px;
  overflow-y: auto;
}
.load-more {
  text-align: center;
  padding: 6px 0;
}
.no-more {
  font-size: 12px;
  color: #c0c4cc;
}
</style>
