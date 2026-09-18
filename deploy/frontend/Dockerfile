# ============================================================
# deploy/frontend/Dockerfile — 多阶段构建（node build → nginx serve）
# ------------------------------------------------------------
# 阶段拆分的收益：最终镜像里没有 node_modules、没有源码、没有 npm ——
# 只留几十 MB 的静态产物 + nginx。
#
# S1 阶段 frontend/ 里只有一个占位页（保证镜像能构建、链路能通）；
# 真正的 4 个页面在 S4 落地。
# ============================================================

# ---------- 阶段 1：构建 ----------
FROM node:22-alpine AS builder

WORKDIR /app

# 依赖单独一层，改业务代码时不用重装 node_modules
COPY frontend/package.json ./
# ⚠️ S1 阶段还没有 package-lock.json，所以用 npm install。
#    S4 生成 lock 文件后应改成 `npm ci`（严格按 lock 安装，构建可复现）。
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ---------- 阶段 2：运行 ----------
FROM nginx:alpine

# 静态产物
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx 配置打进镜像，保证镜像单独跑也能工作（compose 里再挂载一份方便调参）
COPY deploy/nginx/nginx.conf   /etc/nginx/nginx.conf
COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# healthcheck 由 docker-compose.yml 统一定义
CMD ["nginx", "-g", "daemon off;"]
