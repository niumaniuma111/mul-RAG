<script setup lang="ts">
/**
 * 任务阶段展示：stage 中英映射 + 状态色。
 * DocumentsView / TasksView 共用。
 */
import { computed } from 'vue'
import type { TagProps } from 'element-plus'

// 用运行时形式声明 props（理由见 CitationCard.vue 顶部注释）：
// 类型参数形式的 defineProps 依赖构建期 TS 类型解析，容易被静默跳过。
const props = defineProps({
  status: { type: String, required: true },
  stage: { type: String as () => string | null, default: null },
  progress: { type: Number, default: undefined },
})

const STATUS_LABELS: Record<string, string> = {
  pending: '等待中',
  queued: '排队中',
  running: '解析中',
  success: '成功',
  failed: '失败',
}

const STAGE_LABELS: Record<string, string> = {
  parsing: '文档解析',
  captioning: '图片描述',
  chunking: '语义切分',
  embedding: '向量化入库',
}

const statusLabel = computed(() => STATUS_LABELS[props.status] || props.status)
const stageLabel = computed(() => (props.stage ? STAGE_LABELS[props.stage] || props.stage : ''))

const tagType = computed<TagProps['type']>(() => {
  switch (props.status) {
    case 'success':
      return 'success'
    case 'failed':
      return 'danger'
    case 'running':
      return 'primary'
    default:
      return 'info'
  }
})
</script>

<template>
  <span class="task-stage">
    <el-tag :type="tagType" size="small" :effect="status === 'running' ? 'dark' : 'light'">
      {{ statusLabel }}
    </el-tag>
    <span v-if="stageLabel && status === 'running'" class="stage-text">{{ stageLabel }} {{ progress ?? 0 }}%</span>
  </span>
</template>

<style scoped>
.task-stage {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.stage-text {
  font-size: 12px;
  color: #909399;
}
</style>
