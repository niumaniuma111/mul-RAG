import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as authApi from '../api/auth'

const TOKEN_KEY = 'pptrag.token'
const USER_KEY = 'pptrag.username'
const UID_KEY = 'pptrag.uid'

/**
 * 用户状态：token 持久化到 localStorage（刷新不丢）。
 * 401 清理由 http.ts 拦截器统一做。
 */
export const useUserStore = defineStore('user', () => {
  const token = ref<string>(localStorage.getItem(TOKEN_KEY) || '')
  const username = ref<string>(localStorage.getItem(USER_KEY) || '')
  const userId = ref<number>(Number(localStorage.getItem(UID_KEY) || 0))

  function setAuth(t: { access_token: string; user_id: number; username: string }) {
    token.value = t.access_token
    userId.value = t.user_id
    username.value = t.username
    localStorage.setItem(TOKEN_KEY, t.access_token)
    localStorage.setItem(UID_KEY, String(t.user_id))
    localStorage.setItem(USER_KEY, t.username)
  }

  function logout() {
    token.value = ''
    username.value = ''
    userId.value = 0
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(UID_KEY)
    localStorage.removeItem(USER_KEY)
  }

  /** 启动时校验 token 是否仍有效（无效即清） */
  async function refreshMe() {
    if (!token.value) return false
    try {
      const info = await authApi.me()
      username.value = info.username
      localStorage.setItem(USER_KEY, info.username)
      return true
    } catch {
      logout()
      return false
    }
  }

  return { token, username, userId, setAuth, logout, refreshMe }
})
