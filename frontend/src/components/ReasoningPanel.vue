<script setup lang="ts">
/**
 * 推理链路面板：取最后一条助手消息的 steps 时间线 + 自省结论。
 * 节点顺序即 LangGraph 7 节点执行序（后端 producer 的 stage 事件）。
 */
import { computed } from 'vue'
import { useChatStore } from '../stores/chat'
import { Check, Loading } from '@element-plus/icons-vue'

const chat = useChatStore()

const lastAssistant = computed(() => {
  const arr = chat.messages
  for (let i = arr.length - 1; i >= 0; i--) {
    if (arr[i].role === 'assistant') return arr[i]
  }
  return null
})

const steps = computed(() => lastAssistant.value?.steps ?? [])
const reflect = computed(() => lastAssistant.value?.reflect ?? null)
</script>

<template>
  <el-card shadow="never" class="panel">
    <template #header>Agent 推理链路</template>
    <el-empty v-if="steps.length === 0" description="提问后展示节点推进" :image-size="48" />
    <el-steps v-else direction="vertical" :active="steps.length" :space="44">
      <el-step
        v-for="s in steps"
        :key="s.node"
        :title="s.label"
        :status="s.status === 'done' ? 'success' : 'process'"
      />
    </el-steps>

    <div v-if="reflect" class="reflect">
      <el-divider content-position="left">自省</el-divider>
      <el-alert
        :type="reflect.verdict === 'PASS' ? 'success' : 'warning'"
        :closable="false"
        show-icon
      >
        第 {{ reflect.round }} 轮：{{ reflect.verdict }}
        <p v-if="reflect.reason" class="reason">{{ reflect.reason }}</p>
      </el-alert>
    </div>

    <div v-if="chat.sending && steps.length" class="running">
      <el-icon class="is-loading"><Loading /></el-icon>
    </div>
  </el-card>
</template>

<style scoped>
.panel {
  border-radius: 10px;
}
.reflect .reason {
  margin-top: 4px;
  font-size: 12px;
  color: inherit;
}
.running {
  margin-top: 10px;
  color: var(--el-color-primary);
}
</style>
