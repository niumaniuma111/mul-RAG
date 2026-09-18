import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useSSE } from '../composables/useSSE'
import * as sessionsApi from '../api/sessions'
import type { Citation, HitlPayload, ReflectPayload, TraceStep } from '../types/api'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]
  finishReason?: string
  reflect?: ReflectPayload
  steps?: TraceStep[]
  streaming?: boolean
  error?: boolean
}

export interface SessionMeta {
  sessionId: string
  title: string
  createdAt: number
}

const SESSIONS_KEY = 'pptrag.sessions'

/**
 * 本地会话列表 —— 仅作**降级备份**。
 * B2 之后权威来源是 `GET /qa/sessions`（refreshSessions 会覆盖这份），
 * 留着它是为了「后端暂时不可达时列表不至于空白」。
 */
function loadSessions(): SessionMeta[] {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY)
    return raw ? (JSON.parse(raw) as SessionMeta[]) : []
  } catch {
    return []
  }
}

const NODE_LABELS: Record<string, string> = {
  preprocess: 'Query 预处理',
  decide: '检索决策',
  retrieve: '混合检索 + 重排',
  hitl: '等待人工确认',
  generate: '生成答案',
  reflect: '自省校验',
  finalize: '收尾定稿',
}

/**
 * 问答状态机：
 *   ask  → POST /qa/ask        （SSE）
 *   hitl → 弹窗 → POST /qa/resume（SSE 续流，同一 session）
 *
 * 事件协议（backend qa.py）：session/stage/token/hitl/reflect/final/error/__done__
 */
export const useChatStore = defineStore('chat', () => {
  const sessions = ref<SessionMeta[]>(loadSessions())
  const currentSessionId = ref<string>('')
  const messages = ref<ChatMessage[]>([])
  const sending = ref(false)
  /** 切换会话时正在拉取历史正文（供 UI 显示骨架/加载态） */
  const loadingHistory = ref(false)
  /** HITL 弹窗状态：非 null 时显示 */
  const hitl = ref<(HitlPayload & { sessionId: string }) | null>(null)

  const { streaming, start, abort } = useSSE()

  const currentMessages = computed(() => messages.value)

  // ---------- 会话管理 ----------
  // B2（2026-09-18）：会话列表改为**服务端权威**。
  //   此前用 localStorage 存 {session_id, title}，但标题其实后端一直在落
  //   （chat_sessions.title = 首轮 query），本地那份只是副本、还会过期。
  //   现在 refreshSessions() 从 GET /qa/sessions 拉，localStorage 仅作为
  //   「拉取失败时的降级显示」保留。
  const sessionsLoading = ref(false)

  /** 拉取会话列表（服务端权威，按 updated_at 倒序） */
  async function refreshSessions() {
    sessionsLoading.value = true
    try {
      const data = await sessionsApi.listSessions(1, 50)
      sessions.value = data.items.map((s) => ({
        sessionId: s.session_id,
        title: s.title,
        createdAt: new Date(s.created_at).getTime(),
      }))
      persistSessions() // 写回本地做降级备份
    } catch {
      // 拉取失败（离线/未登录）→ 保留本地那份，别把列表清空
    } finally {
      sessionsLoading.value = false
    }
  }

  function persistSessions() {
    try {
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.value.slice(0, 100)))
    } catch {
      /* 隐私模式下 setItem 会抛，忽略 */
    }
  }

  function newSession() {
    currentSessionId.value = ''
    messages.value = []
    hitl.value = null
    activeAssistantId = null
  }

  /**
   * 切换到某个会话 —— 从服务端**加载历史正文**（B2 的核心改动）。
   *
   * 改动前：`messages.value = []`（清空）。因为当时没有读接口，拿不回正文，
   * 只能清空 —— 于是「点一下当前会话，整轮问答从界面消失」（HANDOFF 坑 37）。
   * 改动后：先短路"点自己"，再从 `GET /qa/sessions/{id}/messages` 取回正文。
   *
   * ⚠️ 加载历史只用于**展示**。多轮上下文由 LangGraph checkpointer 按
   *    thread_id 维护，不依赖这里回填的 messages —— 所以即使加载失败，
   *    继续追问依然能延续上下文，只是聊天记录看不到而已。
   */
  async function pickSession(meta: SessionMeta) {
    // 点自己 → 无操作。既能避免无谓请求，也彻底杜绝"误点清空"这一类 bug。
    if (meta.sessionId === currentSessionId.value) return

    currentSessionId.value = meta.sessionId
    hitl.value = null
    messages.value = []
    activeAssistantId = null // 换会话 → 丢弃上一轮的气泡归属
    loadingHistory.value = true

    try {
      const data = await sessionsApi.listMessages(meta.sessionId)
      // 竞态防护：加载期间用户又切了别的会话 → 丢弃这次结果
      if (currentSessionId.value !== meta.sessionId) return

      messages.value = data.items.map(toChatMessage)
      // 顺手用服务端标题刷新列表项（本地那份可能过期）
      const item = sessions.value.find((s) => s.sessionId === meta.sessionId)
      if (item && data.title) item.title = data.title
    } catch {
      // 加载失败不清空提示之外的任何东西；保留会话可继续追问
      messages.value = []
    } finally {
      loadingHistory.value = false
    }
  }

  /** 服务端 MessageItem → 前端 ChatMessage */
  function toChatMessage(m: sessionsApi.MessageItem): ChatMessage {
    return {
      id: m.message_id,
      role: m.role,
      text: m.content,
      citations: (m.citations as Citation[] | null) ?? undefined,
      // ⚠️ 字段名不同：后端 reflect_info 用 `rounds`，前端 ReflectPayload 用 `round`
      reflect: m.reflect_info
        ? {
            round: Number((m.reflect_info as unknown as { rounds?: number }).rounds ?? 0),
            verdict: m.reflect_info.verdict,
            reason: m.reflect_info.reason,
          }
        : undefined,
      // 历史消息不还原节点时间线（qa_traces 才存了完整链路，v1 不回溯）
      steps: [],
      streaming: false,
    }
  }

  function saveSession(sid: string, title: string) {
    if (sessions.value.some((s) => s.sessionId === sid)) return
    sessions.value.unshift({ sessionId: sid, title, createdAt: Date.now() })
    persistSessions()
  }

  function removeSession(sid: string) {
    // ⚠️ 仅移除本地列表项（后端会话数据不动）—— 与 SessionList 的确认文案一致。
    //    后端若要真删，应另开 DELETE /qa/sessions/{id}，不要在这里偷偷调。
    sessions.value = sessions.value.filter((s) => s.sessionId !== sid)
    persistSessions()
    if (currentSessionId.value === sid) newSession()
  }

  // ---------- SSE 事件处理 ----------

  /**
   * 当前这一轮问答对应**哪一个助手消息**。
   *
   * ⚠️ 为什么要有这个显式 id，而不是"看最后一条是不是 assistant 且 streaming"：
   *    后端事件顺序是 `... → final → stage(finalize)`，
   *    即 **`final` 之后还有一个 `finalize` stage**。
   *    `final` 会把 `msg.streaming` 置 false，于是随后到达的 `stage(finalize)`
   *    一旦按 streaming 判断就会**再新建一个空白助手气泡**，
   *    把真正的答案挤在上面、看起来像"前端没回答"。
   *    用固定 id 锁定同一条消息，整轮事件（含 finalize）都落到同一个气泡里。
   */
  let activeAssistantId: string | null = null

  function ensureAssistantMsg(): ChatMessage {
    if (activeAssistantId) {
      const found = messages.value.find((m) => m.id === activeAssistantId)
      if (found) return found
    }
    const msg: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', text: '', streaming: true, steps: [] }
    messages.value.push(msg)
    activeAssistantId = msg.id
    return msg
  }

  function handleEvent(event: string, data: Record<string, unknown>) {
    const msg = ensureAssistantMsg()
    switch (event) {
      case 'session': {
        currentSessionId.value = String(data.session_id || '')
        break
      }
      case 'stage': {
        const node = String(data.node || '')
        // 节点完成 → 该节点标 done；下一个未知节点追加 running
        const existing = msg.steps?.find((s) => s.node === node)
        if (existing) existing.status = 'done'
        else {
          msg.steps?.push({ node, label: NODE_LABELS[node] || node, status: 'done' })
        }
        break
      }
      case 'token': {
        msg.text += String(data.text || '')
        break
      }
      case 'hitl': {
        msg.streaming = false
        hitl.value = { ...(data as unknown as HitlPayload), sessionId: currentSessionId.value }
        break
      }
      case 'reflect': {
        msg.reflect = data as unknown as ReflectPayload
        break
      }
      case 'final': {
        msg.text = String(data.answer || msg.text) // 终帧以 answer 为准
        msg.citations = (data.citations as Citation[]) || []
        msg.finishReason = String(data.finish_reason || '')
        msg.streaming = false
        break
      }
      case 'error': {
        msg.text = `⚠️ ${String(data.message || '服务异常')}`
        msg.error = true
        msg.streaming = false
        break
      }
      // __done__ 哨兵由流关闭自然处理，无需单独动作
    }
  }

  /** 发送新问题 */
  async function ask(query: string, docIds: string[]) {
    if (!query.trim() || sending.value) return
    messages.value.push({ id: crypto.randomUUID(), role: 'user', text: query })
    sending.value = true
    activeAssistantId = null // 新一轮 → 下一个事件必须新建气泡

    await start(
      '/api/v1/qa/ask',
      {
        query,
        session_id: currentSessionId.value || undefined,
        doc_ids: docIds.length ? docIds : undefined,
      },
      {
        onEvent: handleEvent,
        onError: (err) => {
          const msg = ensureAssistantMsg()
          msg.text = `⚠️ ${err.message}`
          msg.error = true
          msg.streaming = false
        },
        onDone: () => {
          const last = messages.value[messages.value.length - 1]
          if (last?.streaming) last.streaming = false
          sending.value = false
          // 先把本地列表补上（用户立刻看到新会话），再异步拉服务端校准。
          // 顺序有讲究：saveSession 是同步的、列表立刻发生变化；
          // refreshSessions 会带回服务端标题（= 完整首轮 query，比本地的 30 字截断更准）。
          if (currentSessionId.value) {
            saveSession(currentSessionId.value, query.slice(0, 30))
            void refreshSessions()
          }
        },
      },
    )
    sending.value = false
  }

  /** HITL 决策提交 → resume 续流 */
  async function resumeHitl(action: 'rewrite_query' | 'use_current_context', newQuery?: string) {
    if (!hitl.value) return
    const { sessionId } = hitl.value
    hitl.value = null
    sending.value = true
    // resume 后是同一助手消息的延续
    await start(
      '/api/v1/qa/resume',
      { session_id: sessionId, action, new_query: newQuery || undefined },
      {
        onEvent: handleEvent,
        onError: (err) => {
          const msg = ensureAssistantMsg()
          msg.text = `⚠️ ${err.message}`
          msg.error = true
          msg.streaming = false
        },
        onDone: () => {
          const last = messages.value[messages.value.length - 1]
          if (last?.streaming) last.streaming = false
          sending.value = false
        },
      },
    )
    sending.value = false
  }

  return {
    sessions, currentSessionId, messages: currentMessages, sending, hitl, streaming,
    loadingHistory, sessionsLoading,
    newSession, pickSession, saveSession, removeSession, refreshSessions, ask, resumeHitl, abort,
  }
})
