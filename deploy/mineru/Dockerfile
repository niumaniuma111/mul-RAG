# ============================================================
# deploy/mineru/Dockerfile — MinerU 解析服务（CPU 版）
# ------------------------------------------------------------
# 这是全项目最重的镜像（约 7GB）。体积换来的收益是：
#   · 与后端依赖完全隔离（MinerU 的 torch/opencv/PDF 库很容易和后端打架）
#   · 与在线问答**真并行**（CPU 模式不占显存、不持锁）
#
# ⚠️ 不要在这个镜像里装 vllm / GPU 版依赖。MinerU 默认走 CPU；
#    需要 GPU 加速时用 docker-compose.gpu.yml 叠加（只加设备预留，不改镜像）。
#    那才是设计里"GPU 是开关、不是默认"的落地方式。
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    TZ=Asia/Shanghai \
    MINERU_DEVICE=cpu \
    OMP_NUM_THREADS=6

WORKDIR /app

# ★ PyPI 镜像源（2026-09-17 构建实测后新增）
#   实测：容器内直连 PyPI 下载速度仅 0.3~0.6MB/s，而本镜像依赖最多
#   （mineru[core] 全家桶含 magika/lxml/fonttools/opencv 等），
#   直连构建 15 分钟仍未下完；清华镜像本机实测可达且更快。
#   需改回官方源时：
#     docker compose build --build-arg PIP_INDEX=https://pypi.org/simple mineru
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 系统依赖：
#   curl              —— healthcheck
#   libgl1            —— opencv（MinerU 图像处理链依赖）
#   libglib2.0-0      —— opencv 的 GLib 依赖，缺了会报 "libgthread not found"
#   libgomp1          —— OpenMP 运行时（torch CPU 版多线程）
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ★ 先装 CPU 版 torch —— 必须在 mineru 之前。
#   原因：PyPI 上 Linux 的 torch 默认是 CUDA 版，会多拉约 2.5GB 的 CUDA 运行时，
#   而本项目 MinerU 默认走 CPU，那些库纯属浪费磁盘和构建时间。
#   先把 CPU 版装好，后面 mineru 解析依赖时已满足，就不会再引入 CUDA 版。
#
#   ⚠️ 这里必须保留 pytorch 官方源 —— 国内 PyPI 镜像**只有 CUDA 版 torch**
#      （它们是 PyPI 的镜像，而 PyPI 上 Linux 的 torch 就是 CUDA 构建），
#      想拿 +cpu 版本只能走 download.pytorch.org。
#
#   ⚠️ 加重试的原因（2026-09-17 实测）：本机容器内解析 download.pytorch.org
#      可能返回**全是 IPv6** 而没有 A 记录，容器又无 IPv6 路由 → Errno 101
#      Network is unreachable。该连通性是**间歇性**的（同一轮构建重试后能过），
#      但 pip 默认 5 次重试可能不够，故显式放宽。
#
#   ⚠️ 若 MinerU 对 torch 版本区间有更严格要求，pip 仍可能把它换掉。
#      构建完成后请核对：
#        docker compose exec mineru pip show torch | grep -E "Version|Location"
#      若看到 +cu 后缀，说明被换成 CUDA 版了，需要在 requirements 里加约束。
#   ⚠️ 不要为了绕开下面的哈希问题给本命令加 `--extra-index-url` 指向国内 PyPI 镜像！
#      国内镜像就是 PyPI 的镜像，而 **PyPI 上 Linux 的 torch 是 CUDA 构建**：
#      两个源都提供同名同版本的 torch wheel，pip 选哪个是未定义行为 ——
#      可能悄悄装上 CUDA 版，白拉 2.5GB，且要在很久之后才发现。
#
#   ⚠️ 2026-09-17 实测踩坑（这次构建失败的真因）：
#      download.pytorch.org 对**传递依赖**（jinja2 / markupsafe / mpmath 等）
#      也提供副本，但它上面 `markupsafe-3.0.3` 的 wheel 内容与索引声明的
#      sha256 不一致 → pip 报
#        "THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE"
#      直接构建失败。修法见下面那行「预装纯 Python 依赖」。
RUN pip install --no-cache-dir \
        -i "$PIP_INDEX" \
        filelock typing-extensions sympy networkx jinja2 fsspec

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --retries 10 \
        --timeout 120 \
        torch

COPY deploy/mineru/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -i "$PIP_INDEX" -r /app/requirements.txt

COPY deploy/mineru/server.py /app/server.py

EXPOSE 8001

# 说明：这里不设 healthcheck，由 docker-compose.yml 统一定义，
#       避免同一件事在两个地方各写一份。
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
