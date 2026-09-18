# ============================================================
# deploy/model-service/Dockerfile — 模型推理服务
# ------------------------------------------------------------
# ⚠️ 这是全项目**唯一**加载模型权重的容器。这个约束不是风格问题：
#    torch.load 是进程级行为，两个容器共用同一份权重文件，仍会各自
#    把权重拷进自己的 CUDA context。8GB 卡上 VLM 权重 2.4GB × 2 份
#    就已经吃掉一半显存 —— 这是拆分本容器的唯一理由。
#    （详见 DESIGN.md §14.1 的「模型挂载的作用边界」）
#
# ⚠️ 模型权重不打进镜像，走宿主只读挂载 ./models:/models:ro。
#    收益是「换模型免重建镜像」，而**不是**省显存 —— 别搞混。
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    TZ=Asia/Shanghai \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# 系统依赖：
#   curl          —— healthcheck
#   libgl1/libglib2.0-0 —— 图像预处理链
#   libgomp1      —— OpenMP 运行时
# 注：CUDA 运行时不在这里装 —— 由 nvidia-container-toolkit 从宿主机注入
#     （Docker Desktop 的 WSL2 后端会自动完成，无需在 WSL 里另装 toolkit）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/model-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复用 backend 的 app.core（配置加载 + 锁工厂），保证锁层级只有一处实现
COPY backend/app/core/ /app/app/core/
COPY backend/app/__init__.py /app/app/__init__.py
COPY backend/model_service/ /app/model_service/

EXPOSE 8002

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8002"]
