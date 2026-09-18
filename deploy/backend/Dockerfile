# ============================================================
# deploy/backend/Dockerfile — FastAPI + Celery
# ------------------------------------------------------------
# api 与 worker 共用这个镜像，只是 CMD 不同（见 docker-compose.yml）。
#
# ⚠️ 构建上下文是项目根目录（`.`），不是 deploy/。所以 COPY 路径都从根写起：
#      docker compose build api
#    （compose 里已写好 build.context: .）
#
# ⚠️ 本镜像刻意不含 torch / transformers —— api 与 worker 都不直接加载模型，
#    全部通过 HTTP 调 model-service。这使得镜像从 ~3.5GB 降到 ~600MB，
#    构建时间与磁盘占用都大幅下降。
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖：
#   curl   —— compose healthcheck 要用（python:slim 不带 curl）
#   libpq5 —— psycopg3（LangGraph checkpointer）的运行时动态库
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 依赖先装 —— 单独一层，改代码时不会触发重装（利用构建缓存）
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 再拷代码
COPY backend/ /app/

# 非 root 运行（容器逃逸的纵深防御，也避免写文件时权限混乱）
RUN useradd -m -u 1000 -s /bin/bash appuser \
    && mkdir -p /app/storage /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 默认是 api；worker 由 compose 覆盖 command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
