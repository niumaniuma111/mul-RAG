<script setup lang="ts">
/**
 * 核心对话页（DESIGN §13.2）：三栏布局
 *   左：会话列表 + 新建 + 文档范围选择
 *   中：消息流（打字机 + 引用卡片 + HITL 提示） + 输入框
 *   右：Agent 推理链路可视化
 */
import { nextTick, onMounted, ref, watch } from 'vue'
import { useChatStore } from '../stores/chat'
import { useDocumentsStore } from '../stores/documents'
import SessionList from '../components/SessionList.vue'
import DocScopeSelector from '../components/DocScopeSelector.vue'
import MessageBubble from '../components/MessageBubble.vue'
import HitlDialog from '../components/HitlDialog.vue'
import ReasoningPanel from '../components/ReasoningPanel.vue'

const chat = useChatStore()
const docs = useDocumentsStore()

const input = ref('')
const inputRef = ref<HTMLTextAreaElement>()
const streamEl = ref<HTMLElement>()

onMounted(() => {
  docs.refresh() // 供范围选择器使用
})

// 新 token 到达时滚动到底部
watch(
  () => chat.messages.map((m) => m.text.length).join(','),
  async () => {
    await nextTick()
    streamEl.value?.scrollTo({ top: streamEl.value.scrollHeight })
  },
)

async function onSend() {
  const q = input.value.trim()
  if (!q || chat.sending) return
  input.value = ''
  await chat.ask(q, docs.selectedDocIds)
}

function onStop() {
  chat.abort()
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    onSend()
  }
}
</script>

<template>
  <div class="chat-page">
    <!-- 左栏：会话 + 范围 -->
    <aside class="left-col">
      <SessionList />
      <DocScopeSelector />
    </aside>

    <!-- 中栏：消息流 -->
    <section class="mid-col">
      <div ref="streamEl" class="stream">
        <!-- 切会话时拉历史正文的加载态（B2）：避免"点了没反应"的空窗 -->
        <el-skeleton v-if="chat.loadingHistory" :rows="4" animated />
        <el-empty
          v-else-if="chat.messages.length === 0"
          description="上传文档后，在这里提问。支持追问，检索范围可在左侧限定。"
        />
        <MessageBubble v-for="m in chat.messages" :key="m.id" :message="m" />
      </div>

      <div class="composer">
        <el-input
          ref="inputRef"
          v-model="input"
          type="textarea"
          :rows="3"
          resize="none"
          placeholder="输入问题，Enter 发送 / Shift+Enter 换行"
          :disabled="chat.sending"
          @keydown="onKeydown"
        />
        <div class="composer-bar">
          <span v-if="chat.streaming" class="streaming-hint">生成中…</span>
          <span v-else class="hint">首次提问可能需要 30-60s（模型冷启动）</span>
          <el-button v-if="chat.streaming" type="danger" plain size="small" @click="onStop">停止</el-button>
          <el-button v-else type="primary" size="small" :disabled="!input.trim()" @click="onSend">发送</el-button>
        </div>
      </div>
    </section>

    <!-- 右栏：推理链路 -->
    <aside class="right-col">
      <ReasoningPanel />
    </aside>

    <HitlDialog />
  </div>
</template>

<style scoped>
.chat-page {
  height: 100%;
  display: grid;
  grid-template-columns: 260px 1fr 280px;
  gap: 12px;
  padding: 12px;
  overflow: hidden;
}
.left-col,
.right-col {
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.mid-col {
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: #fff;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #e4e7ed;
}
.stream {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
.composer {
  flex: 0 0 auto;
  border-top: 1px solid #e4e7ed;
  padding: 12px;
}
.composer-bar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 8px;
}
.hint {
  font-size: 12px;
  color: #909399;
  margin-right: auto;
}
.streaming-hint {
  font-size: 12px;
  color: var(--el-color-primary);
  margin-right: auto;
}

/* 窄屏折叠右栏 */
@media (max-width: 1100px) {
  .chat-page {
    grid-template-columns: 240px 1fr;
  }
  .right-col {
    display: none;
  }
}
</style>
