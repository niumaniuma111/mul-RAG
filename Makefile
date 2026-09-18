# ============================================================
# Makefile — 常用命令封装
# ------------------------------------------------------------
# ⚠️ 推荐在 WSL 发行版内部执行（不是 /mnt/e 下）：
#      cd ~/projects/ppt-agentic-rag && make up
#
#    本文此前写「必须从 WSL 内跑，Windows 侧会失败」—— 该断言已于
#    2026-09-17 被实测推翻。真正的门槛是 Docker Desktop 的
#    Settings → Resources → WSL Integration 开关，不是 Windows 本身。
#    开启后从 Windows 侧跑，相对路径会解析成 \\wsl.localhost\... 并正常挂载。
#
#    注意：本文件的 SHELL := /bin/bash 与 recipe 语法是 POSIX 的，
#    从 Windows 终端跑 make 需要 GNU make + bash 环境；
#    若只想从 Windows 侧启动，直接调 docker compose 更省事。
# ============================================================

SHELL   := /bin/bash
COMPOSE := docker compose

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "ppt-agentic-rag 常用命令"
	@echo ""
	@echo "  初始化"
	@echo "    make env              从 .env.example 生成 .env（已存在则不覆盖）"
	@echo "    make models           下载模型权重（约 12GB，走 ModelScope）"
	@echo "    make init-qdrant      建 Qdrant collection 与 payload index"
	@echo ""
	@echo "  启动 / 停止"
	@echo "    make up               启动全部服务（MinerU 走 CPU）"
	@echo "    make up-gpu           启动并把 MinerU 切到 GPU（叠加覆盖文件）"
	@echo "    make down             停止（保留数据卷）"
	@echo "    make down-v           停止并删除数据卷（!! 清空 PG / Qdrant 数据）"
	@echo ""
	@echo "  观测"
	@echo "    make ps               服务状态"
	@echo "    make stats            资源占用（对比 mem_limit）"
	@echo "    make logs             跟踪全部日志"
	@echo "    make logs S=api       跟踪单个服务日志"
	@echo "    make health           调 /health/deep 看各依赖状态"
	@echo "    make gpu              看显存占用"
	@echo ""
	@echo "  开发"
	@echo "    make build            重新构建镜像"
	@echo "    make sh S=api         进入容器 shell"
	@echo "    make rebuild S=api    只重建并重启一个服务"

# ---------------- 初始化 ----------------
.PHONY: env
env:
	@if [ -f .env ]; then \
		echo ".env 已存在，未覆盖。要重做请先手工删除。"; \
	else \
		cp .env.example .env; \
		echo ".env 已生成 —— 请务必修改里面的 CHANGE_ME 再启动。"; \
	fi

.PHONY: models
models:
	python scripts/download_models.py --root ./models

.PHONY: init-qdrant
init-qdrant:
	$(COMPOSE) exec -T api python scripts/init_qdrant.py

# ---------------- 启动 / 停止 ----------------
.PHONY: up
up:
	$(COMPOSE) up -d
	@echo ""
	@echo "启动中。首次启动前请确认已执行 make env 并改过密钥。"
	@echo "查看状态：make ps   查看日志：make logs"

.PHONY: up-gpu
up-gpu:
	@echo "★ MinerU 切 GPU 模式 —— 上传文档后的 1-3 分钟内问答会排队等锁。"
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d

.PHONY: down
down:
	$(COMPOSE) down

.PHONY: down-v
down-v:
	@echo "⚠️  这会删除 pgdata / qdrant_data / redis_data 三个卷，数据不可恢复。"
	@read -p "确认？输入 yes 继续: " ok && [ "$$ok" = "yes" ] && $(COMPOSE) down -v

# ---------------- 观测 ----------------
.PHONY: ps
ps:
	$(COMPOSE) ps

.PHONY: stats
stats:
	docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"

.PHONY: logs
logs:
	$(COMPOSE) logs -f --tail=100 $(S)

.PHONY: health
health:
	@curl -s http://localhost:8000/api/v1/health/deep | python -m json.tool 2>/dev/null \
		|| curl -s http://localhost:8000/api/v1/health/deep

.PHONY: gpu
gpu:
	@nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
		--format=csv 2>/dev/null || echo "nvidia-smi 不可用"

# ---------------- 开发 ----------------
.PHONY: build
build:
	$(COMPOSE) build

.PHONY: rebuild
rebuild:
	$(COMPOSE) up -d --build $(S)

.PHONY: sh
sh:
	$(COMPOSE) exec $(S) /bin/bash
