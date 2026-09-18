<script setup lang="ts">
/**
 * 消息气泡：用户 / 助手两态。
 * 助手正文里的 [n] 渲染成可点击上标，点击滚动到对应引用卡片。
 * 生成中显示光标动画；结束后展示引用卡片列表。
 */
import { computed } from 'vue'
import type { PropType } from 'vue'
import type { ChatMessage } from '../stores/chat'
import CitationCard from './CitationCard.vue'

// 用运行时形式声明 props（理由见 CitationCard.vue 顶部注释）：
// 类型参数形式的 defineProps 依赖构建期 TS 类型解析，容易被静默跳过，
// 产物里会留下裸的 defineProps → 运行时 ReferenceError → 组件白屏。
const props = defineProps({
  message: { type: Object as PropType<ChatMessage>, required: true },
})

const FINISH_LABELS: Record<string, string> = {
  pass: '自省通过',
  max_rounds: '达到自省上限，仅供参考',
  hitl_skip: 'HITL：基于现有上下文',
  no_retrieval: '闲聊 / 未检索',
  error: '生成异常',
}

const finishLabel = computed(() =>
  props.message.finishReason ? FINISH_LABELS[props.message.finishReason] || props.message.finishReason : '',
)

/** 把正文按 [n] 切成 (文本 | 引用) 片段序列 */
const segments = computed(() => {
  const parts: Array<{ type: 'text' | 'cite'; value: string }> = []
  const re = /\[(\d{1,2})\]/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(props.message.text))) {
    if (m.index > last) parts.push({ type: 'text', value: props.message.text.slice(last, m.index) })
    parts.push({ type: 'cite', value: m[1] })
    last = m.index + m[0].length
  }
  if (last < props.message.text.length) parts.push({ type: 'text', value: props.message.text.slice(last) })
  return parts
})

function scrollToCitation(n: string) {
  document.getElementById(`citation-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
}
</script>

<template>
  <div class="bubble-row" :class="message.role">
    <div class="bubble">
      <!-- 用户 -->
      <template v-if="message.role === 'user'">{{ message.text }}</template>

      <!-- 助手 -->
      <template v-else>
        <div v-if="message.text || message.streaming" class="answer">
          <template v-for="(seg, i) in segments" :key="i">
            <span v-if="seg.type === 'text'">{{ seg.value }}</span>
            <sup v-else class="cite-sup" @click="scrollToCitation(seg.value)">[{{ seg.value }}]</sup>
          </template>
          <span v-if="message.streaming" class="cursor">▍</span>
        </div>

        <!-- 自省结论 -->
        <el-tag
          v-if="message.reflect"
          size="small"
          :type="message.reflect.verdict === 'PASS' ? 'success' : 'warning'"
          class="reflect-tag"
        >
          自省 R{{ message.reflect.round }}：{{ message.reflect.verdict }}
          <span v-if="message.reflect.reason"> · {{ message.reflect.reason }}</span>
        </el-tag>

        <!-- 引用卡片 -->
        <div v-if="message.citations?.length" class="citations">
          <CitationCard v-for="c in message.citations" :key="c.index" :citation="c" />
        </div>

        <!-- 完成原因 -->
        <div v-if="finishLabel" class="finish">
          <el-tag size="small" :type="message.finishReason === 'error' ? 'danger' : 'info'" effect="plain">
            {{ finishLabel }}
          </el-tag>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.bubble-row {
  display: flex;
  margin-bottom: 14px;
}
.bubble-row.user {
  justify-content: flex-end;
}
.bubble {
  max-width: 82%;
  padding: 10px 14px;
  border-radius: 10px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}
.bubble-row.user .bubble {
  background: var(--el-color-primary);
  color: #fff;
  border-bottom-right-radius: 2px;
}
.bubble-row.assistant .bubble {
  background: #f5f7fa;
  border-bottom-left-radius: 2px;
}
.answer {
  white-space: pre-wrap;
}
.cursor {
  animation: blink 0.8s infinite;
  color: var(--el-color-primary);
}
@keyframes blink {
  50% {
    opacity: 0;
  }
}
.citations {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.reflect-tag {
  margin-top: 8px;
}
.finish {
  margin-top: 6px;
}
</style>
