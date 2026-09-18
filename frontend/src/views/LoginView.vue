<script setup lang="ts">
/**
 * 登录页：登录 / 注册双 Tab。
 * 登录成功 → 写入 user store → 跳 redirect 或 /chat。
 */
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import type { FormInstance, FormRules } from 'element-plus'
import * as authApi from '../api/auth'
import { useUserStore } from '../stores/user'

const router = useRouter()
const route = useRoute()
const user = useUserStore()

const activeTab = ref<'login' | 'register'>('login')
const loading = ref(false)
// 两个 Tab 各持一个 ref：共用一个 ref 时后者会覆盖前者，
// 导致「点登录」校验的是注册表单（confirm 为空）→ 静默失败、不发请求、不报错。
const loginFormRef = ref<FormInstance>()
const registerFormRef = ref<FormInstance>()

const loginForm = reactive({ username: '', password: '' })
const registerForm = reactive({ username: '', password: '', confirm: '', email: '' })

const loginRules: FormRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

const registerRules: FormRules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 64, message: '3-64 个字符', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 8, max: 128, message: '至少 8 位', trigger: 'blur' },
  ],
  confirm: [
    {
      validator: (_r, v: string, cb) => {
        if (v !== registerForm.password) cb(new Error('两次密码不一致'))
        else cb()
      },
      trigger: 'blur',
    },
  ],
}

async function onLogin() {
  const ok = await loginFormRef.value?.validate().catch(() => false)
  if (ok === false) return
  loading.value = true
  try {
    const t = await authApi.login(loginForm.username, loginForm.password)
    user.setAuth(t)
    ElMessage.success('登录成功')
    router.replace((route.query.redirect as string) || '/chat')
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    loading.value = false
  }
}

async function onRegister() {
  const ok = await registerFormRef.value?.validate().catch(() => false)
  if (ok === false) return
  loading.value = true
  try {
    await authApi.register(registerForm.username, registerForm.password, registerForm.email)
    ElMessage.success('注册成功，请登录')
    activeTab.value = 'login'
    loginForm.username = registerForm.username
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <el-card class="login-card" shadow="hover">
      <template #header>
        <div class="title">
          <span>PPT 文档智能问答</span>
          <el-tag size="small" type="info" effect="plain">离线 RAG</el-tag>
        </div>
      </template>

      <el-tabs v-model="activeTab" stretch>
        <el-tab-pane label="登录" name="login">
          <el-form ref="loginFormRef" :model="loginForm" :rules="loginRules" label-position="top" @submit.prevent="onLogin" @keyup.enter="onLogin">
            <el-form-item label="用户名" prop="username">
              <el-input v-model="loginForm.username" placeholder="用户名" autocomplete="username" />
            </el-form-item>
            <el-form-item label="密码" prop="password">
              <el-input v-model="loginForm.password" type="password" show-password placeholder="密码" autocomplete="current-password" />
            </el-form-item>
            <el-button type="primary" native-type="submit" class="submit" :loading="loading" @click="onLogin">登录</el-button>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="注册" name="register">
          <el-form ref="registerFormRef" :model="registerForm" :rules="registerRules" label-position="top" @submit.prevent="onRegister" @keyup.enter="onRegister">
            <el-form-item label="用户名" prop="username">
              <el-input v-model="registerForm.username" placeholder="3-64 个字符" />
            </el-form-item>
            <el-form-item label="邮箱（可选）" prop="email">
              <el-input v-model="registerForm.email" placeholder="选填" />
            </el-form-item>
            <el-form-item label="密码" prop="password">
              <el-input v-model="registerForm.password" type="password" show-password placeholder="至少 8 位" />
            </el-form-item>
            <el-form-item label="确认密码" prop="confirm">
              <el-input v-model="registerForm.confirm" type="password" show-password placeholder="再输一次" />
            </el-form-item>
            <el-button type="primary" class="submit" :loading="loading" @click="onRegister">注册</el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<style scoped>
.login-page {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1d2129 0%, #2b3a55 100%);
}
.login-card {
  width: 400px;
  border-radius: 12px;
}
.title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 16px;
  font-weight: 600;
}
.submit {
  width: 100%;
  margin-top: 8px;
}
</style>
