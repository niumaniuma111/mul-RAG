<script setup lang="ts">
/**
 * 任务历史页：列表（运行中自动轮询）/ 失败日志 / 重试 / 删除。
 */
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as tasksApi from '../api/tasks'
import type { TaskItem } from '../types/api'
import { usePolling } from '../composables/usePolling'
import TaskStageTag from '../components/TaskStageTag.vue'

const items = ref<TaskItem[]>([])
const total = ref(0)
const page = ref(1)
const size = 20
const statusFilter = ref('')
const loading = ref(false)

async function refresh() {
  loading.value = true
  try {
    const resp = await tasksApi.listTasks({ page: page.value, size, status: statusFilter.value || undefined })
    items.value = resp.items
    total.value = resp.total
    // 快照：无变化时 usePolling 自动退避
    pollCtl.markChanged(resp.items.map((t) => `${t.task_id}:${t.status}:${t.progress}`).join(','))
  } finally {
    loading.value = false
  }
}

const hasRunning = () => items.value.some((t) => ['pending', 'queued', 'running'].includes(t.status))
const pollCtl = usePolling(refresh, hasRunning, 3000)

// 日志 / 重试 / 删除
const logVisible = ref(false)
const logText = ref('')
const logTaskId = ref('')

async function showLogs(taskId: string) {
  try {
    const r = await tasksApi.getTaskLogs(taskId)
    logTaskId.value = taskId
    logText.value = r.error_log || '（无日志）'
    logVisible.value = true
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

async function retry(row: TaskItem) {
  try {
    await tasksApi.retryTask(row.task_id)
    ElMessage.success('已重新排队')
    refresh()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

async function remove(row: TaskItem) {
  // 用户点「取消」会让 confirm reject → 必须捕获，否则控制台报 'cancel'
  try {
    await ElMessageBox.confirm('仅删除任务记录，不删除文件', '删除任务', { type: 'warning' })
  } catch {
    return
  }
  await tasksApi.deleteTask(row.task_id)
  refresh()
}

onMounted(refresh)
</script>

<template>
  <div class="task-page">
    <div class="toolbar">
      <span class="title">任务历史</span>
      <el-select v-model="statusFilter" placeholder="全部状态" clearable style="width: 140px" @change="page = 1; refresh()">
        <el-option label="运行中" value="running" />
        <el-option label="成功" value="success" />
        <el-option label="失败" value="failed" />
        <el-option label="等待中" value="pending" />
      </el-select>
    </div>

    <el-table :data="items" v-loading="loading" class="table" :header-cell-style="{ background: '#f5f7fa' }">
      <el-table-column prop="file_name" label="文件" min-width="200" show-overflow-tooltip>
        <template #default="{ row }">{{ row.file_name || row.doc_id.slice(0, 8) }}</template>
      </el-table-column>
      <el-table-column prop="version" label="版本" width="70" />
      <el-table-column label="状态" width="190">
        <template #default="{ row }">
          <TaskStageTag :status="row.status" :stage="row.current_stage" :progress="row.progress" />
        </template>
      </el-table-column>
      <el-table-column prop="progress" label="进度" width="160">
        <template #default="{ row }">
          <el-progress :percentage="row.progress" :stroke-width="8" :status="row.status === 'failed' ? 'exception' : row.status === 'success' ? 'success' : undefined" />
        </template>
      </el-table-column>
      <el-table-column prop="retry_count" label="重试" width="70" />
      <el-table-column prop="created_at" label="创建时间" width="170">
        <template #default="{ row }">{{ new Date(row.created_at).toLocaleString() }}</template>
      </el-table-column>
      <el-table-column label="操作" width="220" fixed="right">
        <template #default="{ row }">
          <el-button v-if="row.status === 'failed'" link type="primary" @click="showLogs(row.task_id)">日志</el-button>
          <el-button v-if="row.status === 'failed'" link type="warning" @click="retry(row)">重试</el-button>
          <el-button link type="danger" @click="remove(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="pager">
      <el-pagination
        v-model:current-page="page"
        layout="prev, pager, next, total"
        :total="total"
        :page-size="size"
        @current-change="refresh"
      />
    </div>

    <el-dialog v-model="logVisible" title="失败日志" width="720px" align-center>
      <p class="log-task">task: {{ logTaskId }}</p>
      <pre class="log-body">{{ logText }}</pre>
      <template #footer>
        <el-button @click="logVisible = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.task-page {
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
.pager {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}
.log-task {
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
}
.log-body {
  margin: 0;
  max-height: 50vh;
  overflow-y: auto;
  background: #1d2129;
  color: #d4d7de;
  padding: 12px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
}
</style>
