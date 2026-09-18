<script setup lang="ts">
/**
 * HITL 弹窗（DESIGN §13.2）：检索相关性偏低时挂起等用户决策。
 *   ① 改写 query 重新检索（可输入新检索词）
 *   ② 直接基于现有上下文生成答案
 * 数据来自 SSE hitl 事件 payload：{reason, max_score, threshold, query, options}
 */
import { ref, watch } from 'vue'
import { useChatStore } from '../stores/chat'

const chat = useChatStore()
const newQuery = ref('')

watch(
  () => chat.hitl,
  (v) => {
    if (v) newQuery.value = String(v.query || '')
  },
)

function submit(action: 'rewrite_query' | 'use_current_context') {
  chat.resumeHitl(action, action === 'rewrite_query' ? newQuery.value : undefined)
}
</script>

<template>
  <el-dialog
    :model-value="!!chat.hitl"
    title="⚠ 检索相关性偏低"
    width="520px"
    :close-on-click-modal="false"
    :show-close="false"
    align-center
  >
    <template v-if="chat.hitl">
      <el-alert type="warning" :closable="false" show-icon class="mb">
        <p>{{ chat.hitl.reason }}</p>
        <p>
          最高相关性 <b>{{ chat.hitl.max_score }}</b>，低于阈值 <b>{{ chat.hitl.threshold }}</b>
        </p>
      </el-alert>

      <div class="option">
        <el-radio-group :model-value="'rewrite_query'">
          <el-radio value="rewrite_query">① 改写 query 重新检索</el-radio>
        </el-radio-group>
        <el-input v-model="newQuery" placeholder="输入新的检索词..." class="query-input" />
      </div>

      <div class="option">
        <el-button plain class="full" @click="submit('use_current_context')">
          ② 直接基于现有上下文生成答案
        </el-button>
      </div>
    </template>

    <template #footer>
      <el-button @click="submit('use_current_context')">跳过（用现有上下文）</el-button>
      <el-button type="primary" :disabled="!newQuery.trim()" @click="submit('rewrite_query')">
        用新检索词重查
      </el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.mb {
  margin-bottom: 14px;
}
.option {
  margin-bottom: 14px;
}
.query-input {
  margin-top: 8px;
}
.full {
  width: 100%;
}
</style>
