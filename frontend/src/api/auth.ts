import { http, unwrap } from './http'
import type { MeResult, TokenResult } from '../types/api'

/** 注册。返回新建用户的 id/username（不自动登录，让用户走一遍登录） */
export function register(username: string, password: string, email?: string) {
  return unwrap<{ user_id: number; username: string }>(
    http.post('/auth/register', { username, password, email: email || null }),
  )
}

/** 登录 → token */
export function login(username: string, password: string) {
  return unwrap<TokenResult>(http.post('/auth/login', { username, password }))
}

/** 当前用户信息 */
export function me() {
  return unwrap<MeResult>(http.get('/auth/me'))
}
