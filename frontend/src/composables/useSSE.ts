import { ref } from 'vue'
import { useUserStore } from '../stores/user'

/**
 * SSE 客户端（DESIGN §13.3）
 * ----------
 * 为什么不用 EventSource：它只支持 GET 且**放不了 Authorization header**，
 * 而 /qa/ask 是 POST + Bearer —— 必须手写 fetch + ReadableStream。
 *
 * 关键点：**半包处理**。TCP 分包可能把一个事件块（以 \n\n 结尾）截成两半，
 * 必须把残块缓存（buf）到下一次 read 拼起来，否则偶发 JSON 解析失败。
 * 心跳帧是注释行（": heartbeat"），解析时跳过即可。
 */

export type SseHandlers = {
  onEvent: (event: string, data: Record<string, unknown>) => void
  onError?: (err: Error) => void
  onDone?: () => void
}

/** 解析单个事件块（"event: xxx\ndata: {...}"）→ [event, data] */
function parseEventBlock(block: string): [string, Record<string, unknown>] | null {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue // 心跳 / 注释
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return null
  try {
    return [event, JSON.parse(dataLines.join('\n'))]
  } catch {
    return null // 单块损坏直接丢弃，不让一条坏帧打断整条流
  }
}

export function useSSE() {
  const streaming = ref(false)
  let ctrl: AbortController | null = null

  async function start(url: string, body: unknown, handlers: SseHandlers): Promise<void> {
    const user = useUserStore()
    ctrl = new AbortController()
    streaming.value = true

    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${user.token}`,
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      })

      if (!resp.ok || !resp.body) {
        // 非 200：后端可能返回 JSON 错误体（如 422 / 500）
        let message = `SSE 请求失败（HTTP ${resp.status}）`
        try {
          const j = await resp.json()
          message = (j as { message?: string }).message || message
        } catch { /* 保留默认文案 */ }
        throw new Error(message)
      }

      const reader = resp.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        // 按 \n\n 切完整事件块，最后一段可能是半包 → 留在 buf
        const blocks = buf.split('\n\n')
        buf = blocks.pop() ?? ''
        for (const block of blocks) {
          if (!block.trim()) continue
          const parsed = parseEventBlock(block)
          if (parsed) handlers.onEvent(parsed[0], parsed[1])
        }
      }
      handlers.onDone?.()
    } catch (err) {
      // AbortError 是用户主动停止，不算错误
      if ((err as Error).name === 'AbortError') {
        handlers.onDone?.()
      } else {
        handlers.onError?.(err as Error)
      }
    } finally {
      streaming.value = false
      ctrl = null
    }
  }

  /** 中断当前流（前端停止生成） */
  function abort() {
    ctrl?.abort()
  }

  return { streaming, start, abort }
}
