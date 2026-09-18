<script setup lang="ts">
/**
 * 会话列表（B2 后由服务端驱动）。
 *
 * 权威数据源是 `GET /qa/sessions`；本组件在挂载时拉一次，
 * 每次问答结束后由 store 的 `onDone` 再刷一次。
 * localStorage 只在拉取失败时兜底（见 stores/chat.ts 的 refreshSessions）。
 */
import { onMounted } from 'vue'
import { Close } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useChatStore } from '../stores/chat'

const chat = useChatStore()

onMounted(() => {
  void chat.refreshSessions()
})

function onNew() {
  chat.newSession()
}

async function onPick(sessionId: string, title: string) {
  // 点当前会话 → 短路。store 的 pickSession 内部也会 no-op，
  // 但这里必须提前 return：否则会弹一句「已切到该会话」而实际什么都没发生，纯误导。
  if (sessionId === chat.currentSessionId) return
  // ⚠️ 必须 await：pickSession 现在是异步的（要拉历史正文），
  //    不等它完成就弹提示会说谎 —— 用户可能看到"已切换"但消息还在加载。
  await chat.pickSession({ sessionId, title, createdAt: 0 })
  ElMessage.success('已切到该会话')
}

async function onRemove(sessionId: string) {
  // 用户点「取消」会让 confirm reject → 必须捕获，否则控制台报 'cancel'
  try {
    await ElMessageBox.confirm('仅移除列表项，不影响后端会话数据', '删除会话', { type: 'warning' })
  } catch {
    return
  }
  chat.removeSession(sessionId)
}
</script>

<template>
  <el-card shadow="never" class="panel">
    <template #header>
      <div class="head">
        <span>会话</span>
        <el-button size="small" type="primary" plain @click="onNew">新建</el-button>
      </div>
    </template>
    <el-empty
      v-if="chat.sessions.length === 0"
      :description="chat.sessionsLoading ? '加载中…' : '暂无会话'"
      :image-size="48"
    />
    <div
      v-for="s in chat.sessions"
      :key="s.sessionId"
      class="session-item"
      :class="{ active: s.sessionId === chat.currentSessionId }"
      @click="onPick(s.sessionId, s.title)"
    >
      <span class="s-title">{{ s.title || '未命名会话' }}</span>
      <el-icon class="del" @click.stop="onRemove(s.sessionId)"><Close /></el-icon>
    </div>
  </el-card>
</template>

<style scoped>
.panel {
  border-radius: 10px;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: #303133;
}
.session-item:hover {
  background: #f5f7fa;
}
.session-item.active {
  background: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
}
.s-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.del {
  color: #c0c4cc;
}
.del:hover {
  color: var(--el-color-danger);
}
</style>
