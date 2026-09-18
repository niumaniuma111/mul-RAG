<script setup lang="ts">
/**
 * 应用壳：登录页全屏独立，其余页面共享顶部导航。
 *
 * 路由守卫已保证 /chat /documents /tasks 必有 token，
 * 所以这里只需按路由隐藏导航即可，不做重复鉴权逻辑。
 */
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore } from './stores/user'

const route = useRoute()
const router = useRouter()
const user = useUserStore()

const isLogin = computed(() => route.name === 'login')

const navs = [
  { path: '/chat', label: '智能问答' },
  { path: '/documents', label: '文档管理' },
  { path: '/tasks', label: '任务历史' },
]

function onLogout() {
  user.logout()
  router.replace('/login')
}
</script>

<template>
  <div class="app-shell">
    <!-- 登录页：无导航，独占全屏 -->
    <router-view v-if="isLogin" />

    <!-- 业务页：顶部导航 + 页面内容 -->
    <template v-else>
      <header class="app-nav">
        <div class="brand">PPT 智能问答</div>
        <nav class="links">
          <router-link
            v-for="n in navs"
            :key="n.path"
            :to="n.path"
            class="nav-link"
            active-class="nav-link--active"
          >
            {{ n.label }}
          </router-link>
        </nav>
        <div class="user-area">
          <span class="username">{{ user.username }}</span>
          <el-button link type="danger" @click="onLogout">退出</el-button>
        </div>
      </header>
      <main class="app-main">
        <router-view />
      </main>
    </template>
  </div>
</template>

<style scoped>
.app-shell {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.app-nav {
  height: 52px;
  flex: 0 0 52px;
  display: flex;
  align-items: center;
  gap: 32px;
  padding: 0 20px;
  background: #1d2129;
  color: #fff;
}
.brand {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.5px;
}
.links {
  display: flex;
  gap: 8px;
  flex: 1;
}
.nav-link {
  color: rgba(255, 255, 255, 0.72);
  text-decoration: none;
  font-size: 14px;
  padding: 6px 14px;
  border-radius: 6px;
  transition: background 0.2s, color 0.2s;
}
.nav-link:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}
.nav-link--active {
  color: #fff;
  background: var(--el-color-primary);
}
.user-area {
  display: flex;
  align-items: center;
  gap: 12px;
}
.username {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.85);
}
.app-main {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
</style>
