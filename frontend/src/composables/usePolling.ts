import { onUnmounted } from 'vue'

/**
 * 轮询管理（DESIGN §13.4）
 * ----------
 * - 基础间隔 2s；连续 10 次结果无变化 → 指数退避到最长 5s
 * - 组件卸载自动清理定时器，防泄漏
 *
 * 用法：
 *   const stop = usePolling(fn, isActive)
 *   // isActive() 为 false 时轮询暂停（但不清理，恢复 true 后继续）
 */
export function usePolling(fn: () => void | Promise<void>, isActive: () => boolean, baseMs = 2000) {
  let timer: ReturnType<typeof setTimeout> | null = null
  let unchanged = 0
  let lastSnapshot = ''

  async function tick() {
    if (isActive()) {
      try {
        await fn()
      } catch { /* 轮询失败静默，下轮再试 */ }
    }
    schedule()
  }

  function schedule() {
    // 无变化 → 指数退避：2s → 2.5s → 3.1s ... 封顶 5s
    const ms = Math.min(baseMs * Math.pow(1.25, unchanged), 5000)
    timer = setTimeout(tick, ms)
  }

  /** 调用方在 fn 里对比数据快照：有变化时调这个重置退避 */
  function markChanged(snapshot: string) {
    if (snapshot !== lastSnapshot) {
      lastSnapshot = snapshot
      unchanged = 0
    } else {
      unchanged++
    }
  }

  function stop() {
    if (timer) clearTimeout(timer)
    timer = null
  }

  timer = setTimeout(tick, baseMs)
  onUnmounted(stop)

  return { stop, markChanged }
}
