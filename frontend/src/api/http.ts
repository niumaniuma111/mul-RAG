import axios, { AxiosError } from 'axios'
import type { ApiEnvelope } from '../types/api'
import { useUserStore } from '../stores/user'
import router from '../router'

/**
 * 统一 axios 实例。
 *
 * 职责：
 *  1. baseURL /api/v1 + 自动注入 Bearer token
 *  2. 响应解包：只把 data 交给调用方（非 SSE 接口全部是 {code,message,data} 包装）
 *  3. 401 拦截：清 token → 跳登录页（DESIGN §13.1）
 *
 * SSE 不走 axios —— fetch + ReadableStream 见 composables/useSSE.ts
 */
export const http = axios.create({
  baseURL: '/api/v1',
  timeout: 60_000,
})

http.interceptors.request.use((config) => {
  const user = useUserStore()
  if (user.token) {
    config.headers.Authorization = `Bearer ${user.token}`
  }
  return config
})

http.interceptors.response.use(
  (resp) => {
    const body = resp.data as ApiEnvelope<unknown>
    // 统一解包：业务失败（code != 0）在这里抛出，调用方只处理成功数据
    if (body && typeof body === 'object' && 'code' in body) {
      if (body.code !== 0) {
        return Promise.reject(new Error(body.message || `业务错误 ${body.code}`))
      }
      // 保持 resp.data = 解包后的 data，调用方直接用
      resp.data = body.data
    }
    return resp
  },
  (error: AxiosError<ApiEnvelope<unknown>>) => {
    if (error.response?.status === 401) {
      const user = useUserStore()
      user.logout()
      router.replace('/login')
      return Promise.reject(new Error('登录已过期，请重新登录'))
    }
    const msg =
      (error.response?.data as ApiEnvelope<unknown> | undefined)?.message ||
      error.message ||
      '网络错误'
    return Promise.reject(new Error(msg))
  },
)

/** 便捷取 data 的辅助 */
export async function unwrap<T>(promise: Promise<{ data: T }>): Promise<T> {
  const resp = await promise
  return resp.data
}
