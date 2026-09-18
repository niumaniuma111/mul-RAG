import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发期：vite dev server 直接把 /api 透到本地后端容器，避免跨域问题。
// 该 proxy 与后端配置互补 —— 后端 CORS 已经放开了 5173，proxy 是为了少一层预检。
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
})
