"""
模型推理服务入口。

端点规划
--------
    GET  /health            存活 + 锁层级 + 当前驻留模型
    POST /embed             批量向量化（BGE-large-zh-v1.5，1024 维）
    POST /rerank            精排（BGE-reranker-large）
    POST /vlm/caption       图片 caption
    POST /vlm/generate      VLM 流式生成（Qwen2.5-VL-3B 4bit）              —— S3
    POST /admin/unload      手动触发换出（排障用）

为什么单独一个容器（而不是让 worker 直接加载模型）
--------------------------------------------------
`torch.load` 是**进程级**行为。worker 和 Agent 各自加载一份 VLM，
权重文件只有一份没用 —— 显存里会是两份，8GB 卡上直接 OOM。
所以全项目只有这一个容器碰权重。

启动顺序里锁自检同样是第 2 位（理由见 app/core/lock_factory.py）。

错误语义（必须与 app/services/model_client.py 的解析对齐）
--------------------------------------------------------
    503  → 客户端视为 GpuBusyError，**可重试**（排队等锁超时、模型加载失败、
           子进程意外死亡 —— 这三种都是"稍后重试就能好"，语义一致）
    500  → 客户端视为可重试的 5xx，但重试后仍失败会放弃（单次推理失败）
    4xx  → 客户端视为**不可重试**，直接抛错（请求本身有问题，如路径越权）

⚠️ 一个容易踩的坑：`/vlm/caption` 收到的是**容器内路径**（不是 base64），
   所以 model-service 必须挂载 storage 目录才能读到图片。
   docker-compose.yml 里已加 `./storage:/app/storage:ro` ——
   这个挂载漏掉的话，端点会在"打开图片"这一步失败，
   而报错看起来像模型问题，实际是挂载问题。

对应设计文档：DESIGN.md §10.2 / §10.3 / §11
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_config, get_env, load_prompt, resolve_mineru_device
from app.core.exceptions import ErrorCode
from app.core.lock_factory import (
    LockBackendMismatchError,
    build_gpu_lock,
    preflight,
)
from app.core.logging import get_logger, setup_logging
from app.core.redis_client import close_redis, get_lock_redis
from app.core.redis_lock import GpuBusyError as LockGpuBusyError

logger = get_logger(__name__)

# 进程级单例（由 lifespan 填充）
_plan: Any = None
_manager: Any = None

# 允许的模型 key（与 ModelManager._MODEL_KEYS 一致，此处复制是为了
# 请求校验不依赖 manager 是否已初始化）
_MODEL_KEYS = ("embedding", "reranker", "vlm")


def _fail(status_code: int, detail: str, code: int = ErrorCode.INTERNAL_ERROR) -> JSONResponse:
    """
    统一错误响应体。

    ⚠️ 必须用 `detail` 作为字段名：model_client._detail() 优先读 detail/message，
       前端与排障日志都依赖这个字段拿到可读原因。
    """
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": detail, "detail": detail, "data": None},
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _plan, _manager

    # ---------- 1. 日志 ----------
    setup_logging()
    cfg = get_config()
    env = get_env()
    logger.info("启动 model-service (models_root=%s)", env.models_root)

    # ---------- 2. ★ 锁层级自检（fail-fast）----------
    try:
        _plan = preflight()
    except LockBackendMismatchError:
        logger.critical("锁层级与 MinerU 设备不匹配，拒绝启动。")
        raise

    # redis 模式下需要真实连接来建锁；inproc 模式不需要，但连上也没坏处
    redis_client = None
    if _plan.backend == "redis":
        redis_client = await get_lock_redis()
        logger.info("已建立 GPU 锁的 Redis 连接（key=%s, ttl=%ss）", _plan.key, _plan.ttl)

    # ---------- 3. 显存管理器（含 GPU 锁）----------
    from model_service.manager import ModelManager

    gpu_lock = build_gpu_lock(redis_client=redis_client) if _plan.backend else None
    _manager = ModelManager(gpu_lock=gpu_lock)

    # ---------- 4. 预热 ----------
    # config 里 warmup_on_start: ["embedding"]。预热放在后台任务里做：
    # 阻塞 lifespan 会让 /health 在加载期间不可达，healthcheck 会一直等到
    # start_period 结束，容器状态看起来像"起不来"。
    warmup = list(cfg.get_path("models.manager.warmup_on_start", []) or [])
    if warmup:
        logger.info("预热计划（后台执行）: %s", warmup)
    await _manager.start(warmup=warmup)

    logger.info(
        "model-service 就绪 | mineru.device=%s lock_backend=%s",
        resolve_mineru_device(),
        _plan.backend,
    )
    yield

    # ---------- 5. 收尾 ----------
    if _manager is not None:
        await _manager.shutdown()
    if redis_client is not None:
        await close_redis()
    logger.info("model-service 已退出")


app = FastAPI(
    title="ppt-agentic-rag model-service",
    version="0.2.0",
    description="全项目唯一加载模型权重的容器（VLM / embedder / reranker）",
    lifespan=lifespan,
)


# ============================================================
# 健康检查
# ============================================================
@app.get("/health")
async def health() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "service": "model-service",
        "lock_backend": getattr(_plan, "backend", None),
        "lock_key": getattr(_plan, "key", None),
        "mineru_device": resolve_mineru_device(),
        "residents": [],
        "resident_gb": 0.0,
    }
    if _manager is not None:
        payload.update(_manager.snapshot())
    return payload


# ============================================================
# 请求模型
# ============================================================
class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    normalize: bool = True


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    top_n: int = 3


class CaptionRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    context: str = ""
    max_new_tokens: int | None = None


class UnloadRequest(BaseModel):
    key: str | None = None      # 不传或 null = 全部卸载


# ============================================================
# 异常 → HTTP 状态码的映射（集中一处，避免每个端点重复 try/except）
# ============================================================
async def _run(key: str, payload: dict[str, Any], *, op: str | None = None) -> Any:
    """
    调 ModelManager.infer 并把异常翻译成 HTTP 响应。

    返回 dict（成功）或 JSONResponse（失败）—— 用 isinstance 判断，
    这样端点里可以写成 `if isinstance(result, JSONResponse): return result`。
    """
    from model_service.manager import (
        ModelInferenceError,
        ModelLoadError,
        WorkerDeadError,
    )

    try:
        return await _manager.infer(key, payload, op=op)
    except LockGpuBusyError as exc:
        # 等 GPU 锁超时 —— 前端会显示"排队中"，Celery 会退避重试
        logger.warning("%s 等待 GPU 锁超时: %s", key, exc)
        return _fail(503, f"GPU 繁忙：{exc}", code=ErrorCode.GPU_BUSY)
    except ModelLoadError as exc:
        logger.error("%s 加载失败: %s", key, exc)
        return _fail(503, f"模型加载失败：{exc}")
    except WorkerDeadError as exc:
        logger.error("%s 子进程异常: %s", key, exc)
        return _fail(503, f"模型子进程异常：{exc}")
    except ModelInferenceError as exc:
        logger.error("%s 推理失败: %s", key, exc)
        return _fail(500, f"推理失败：{exc}")
    except asyncio.TimeoutError as exc:
        logger.error("%s 调用超时: %s", key, exc)
        return _fail(500, f"推理超时：{exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s 未预期异常", key)
        return _fail(500, f"未预期异常：{exc}")


def _guard_ready() -> JSONResponse | None:
    return None if _manager is not None else _fail(503, "model-service 尚未完成初始化")


# ============================================================
# 推理端点
# ============================================================
@app.post("/embed", response_model=None)
async def embed(req: EmbedRequest) -> Any:
    """
    批量向量化。返回 `{"vectors": [[...]], "dim": 1024, "count": n}`。

    ⚠️ 返回体里的键名必须是 `vectors`：model_client.embed() 按这个键取值
       （兼容 `embeddings`），键名变了整条入库流水线会静默拿到 0 条向量。
    """
    if (early := _guard_ready()) is not None:
        return early

    batch_size = int(get_config().get_path("models.embedding.batch_size", 16) or 16)
    result = await _run(
        "embedding",
        {"texts": req.texts, "normalize": req.normalize, "batch_size": batch_size},
    )
    if isinstance(result, JSONResponse):
        return result

    vectors = result.get("vectors") or []
    if len(vectors) != len(req.texts):
        return _fail(
            500,
            f"向量数量不匹配：请求 {len(req.texts)} 条，模型返回 {len(vectors)} 条",
        )
    return {"vectors": vectors, "dim": int(result.get("dim") or 0), "count": len(vectors)}


@app.post("/rerank", response_model=None)
async def rerank(req: RerankRequest) -> Any:
    """精排。返回 `{"results": [{index, score, document}]}`，score 已过 sigmoid（0~1）。"""
    if (early := _guard_ready()) is not None:
        return early

    result = await _run(
        "reranker",
        {"query": req.query, "documents": req.documents, "top_n": req.top_n},
    )
    if isinstance(result, JSONResponse):
        return result
    return {"results": list(result.get("results") or [])}


def _allowed_image_roots() -> list[Path]:
    """
    允许 VLM 读取的图片根目录（按配置取，去重保序）。

    ★ 为什么根是 `paths.storage_root` 而不是 `paths.image_dir`：
      真正写图片的是 MinerU 解析链路，它把图片外置到
      `parsed/{doc}/v{n}/images/`（见 deploy/mineru/server.py 的
      `_materialize_images`）。而 `paths.image_dir`（默认 /app/storage/images）
      是设计初期的设想值，**没有任何写入方在用它**。
      2026-09-17 端到端实测：只按 image_dir 校验时，每一张图都被
      400 拒绝（`image_path 越界`）→ caption 全灭 → 整个入库任务判失败。

      安全边界应当对齐 model-service **实际挂载**的存储根
      （compose：`./storage:/app/storage:ro`）—— 它既能挡住
      `../../etc/passwd`，又不会绑死到某个具体子目录。
      image_dir 仍旧保留在允许集合里，这样即便有人为了别的用途把图片
      放在那里（或把 image_dir 配到 storage 之外），行为也不会被这次修改打破。
    """
    cfg = get_config()
    roots: list[Path] = []
    for key, default in (
        ("paths.storage_root", "/app/storage"),
        ("paths.image_dir", "/app/storage/images"),
    ):
        raw = str(cfg.get_path(key, default) or "").strip()
        if not raw:
            continue
        root = Path(raw).resolve()
        if root not in roots:          # 去重：两者常配成父子关系
            roots.append(root)
    return roots


def _resolve_image_path(raw: str) -> Path:
    """
    把请求里的图片路径解析成真实路径，并做越界校验。

    为什么必须校验：model-service 是无鉴权的内网服务，
    但它的 `image_path` 来自上游 Celery 任务。若上游被污染（或未来开放给前端），
    一个 `../../etc/passwd` 就能让 VLM 去读任意文件。
    校验方式是「解析后必须落在允许的图片根之内」——
    用 resolve() 消掉 `..` 之后再比，软链接也被展开。
    """
    from app.core.exceptions import BadRequestError

    path = Path(raw)
    if not path.is_absolute():
        raise BadRequestError(f"image_path 必须是绝对路径: {raw}")

    roots = _allowed_image_roots()

    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BadRequestError(f"图片不可读: {raw} ({exc})") from exc

    if not any(resolved == root or root in resolved.parents for root in roots):
        allowed = " 或 ".join(str(root) for root in roots)
        raise BadRequestError(f"image_path 越界（必须位于 {allowed} 之内）: {resolved}")

    if not resolved.is_file():
        raise BadRequestError(f"image_path 不是文件: {resolved}")

    return resolved


@app.post("/vlm/caption", response_model=None)
async def vlm_caption(req: CaptionRequest) -> Any:
    """
    图片 caption。返回 `{"caption": "..."}`。

    返回体键名必须是 `caption`（model_client.caption() 按它取值）。

    Prompt 由两段拼成：
        system = configs/prompts/caption_system.txt   （描述规则，主指令）
        user   = configs/prompts/caption_user.txt + 本页上下文
    外置在 prompts/ 下，调 prompt 不需要重建镜像。
    """
    if (early := _guard_ready()) is not None:
        return early

    from app.core.exceptions import BadRequestError

    try:
        image_path = _resolve_image_path(req.image_path)
    except BadRequestError as exc:
        # 4xx：客户端不可重试（路径问题重试也没用）
        return _fail(400, str(exc), code=ErrorCode.BAD_REQUEST)

    cfg = get_config()
    caption_cfg = cfg.get_path("models.caption", {}) or {}

    system_prompt = load_prompt("caption_system.txt")
    user_prompt = load_prompt("caption_user.txt").replace(
        "{context}", (req.context or "").strip() or "（无）"
    )
    max_new_tokens = int(
        req.max_new_tokens
        if req.max_new_tokens is not None
        else caption_cfg.get("max_new_tokens", 256)
    )

    result = await _run(
        "vlm",
        {
            "system_prompt": system_prompt,
            "prompt": user_prompt,
            "image_path": str(image_path),
            "max_new_tokens": max_new_tokens,
            "temperature": float(caption_cfg.get("temperature", 0.3)),
            # ★ caption 阶段图片再压一档（显存关键，见 DESIGN.md 风险 R14）
            "max_pixels": int(caption_cfg.get("max_pixels", 501760)),
        },
        op="caption",
    )
    if isinstance(result, JSONResponse):
        return result
    return {"caption": str(result.get("text") or "").strip()}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str = ""
    image_path: str | None = None            # 可选：多模态提问时带上
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    max_pixels: int | None = None            # 带图时限制视觉 token 数


@app.post("/vlm/generate", response_model=None)
async def vlm_generate(req: GenerateRequest) -> Any:
    """
    流式生成（SSE）。事件协议：

        event: delta\\ndata: {"text": "..."}\\n\\n     （若干次）
        event: done\\ndata: {"text": 全文}\\n\\n        （恰好一次）
        event: error\\ndata: {"message": "..."}\\n\\n   （异常时一次）

    为什么用 SSE 而不是 JSON Lines：api 容器里的消费方（qa.py）用
    httpx 的 `aiter_lines()` 解析，SSE 的 `data:` 前缀天然分行且
    与前端 EventSource 语义一致，两跳（model-service → api → 浏览器）
    共用一套解析器，出错面最小。

    ⚠️ 生成期间 VLM 持有 GPU 锁与 _call_lock，本端点的并发调用天然排队
       （见 manager.infer_stream 文档）。agent.max_wall_clock_sec 是更上层
       的墙钟兜底，这里只管模型层超时。
    """
    if (early := _guard_ready()) is not None:
        return early

    from fastapi.responses import StreamingResponse

    timeout = float(
        get_config().get_path("models.manager.timeouts.generate", 900) or 900
    )

    async def event_stream() -> AsyncIterator[bytes]:
        payload = {
            "prompt": req.prompt,
            "system_prompt": req.system_prompt or "",
            "image_path": req.image_path,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
            "max_pixels": req.max_pixels,
            "stream": True,          # ★ workers._do_vlm 的流式开关
        }
        try:
            async for chunk in _manager.infer_stream(
                "vlm", payload, op="generate", timeout=timeout
            ):
                event = chunk.get("event", "done")
                text = chunk.get("text", "")
                data = json.dumps({"text": text}, ensure_ascii=False)
                yield f"event: {event}\ndata: {data}\n\n".encode("utf-8")
        except Exception as exc:  # noqa: BLE001 —— 流中断也要给客户端一个终态
            logger.exception("流式生成失败")
            data = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n".encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Nginx/中间层不得缓冲 —— 否则前端看到的是"一次性吐全篇"
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/admin/unload", response_model=None)
async def admin_unload(req: UnloadRequest | None = None) -> Any:
    """
    手动触发换出。排障用 —— 当怀疑显存没释放干净时，
    调它把常驻模型也卸掉，再看 `nvidia-smi` 是否归零。

    验收方式（S2）：调 `{"key": null}` 全卸 → `nvidia-smi` 里
    python 进程（含 CUDA context）应整体消失，而不只是显存数字变小。
    那正是"进程级卸载 vs del+empty_cache"的判别点。
    """
    if (early := _guard_ready()) is not None:
        return early

    key = (req.key if req is not None else None) or None
    if key is not None and key not in _MODEL_KEYS:
        return _fail(400, f"未知的模型 key: {key!r}，只允许 {list(_MODEL_KEYS)}",
                     code=ErrorCode.BAD_REQUEST)

    try:
        if key is None:
            unloaded = await _manager.unload_all(reason="admin")
        else:
            unloaded = [key] if await _manager.unload(key, reason="admin") else []
    except Exception as exc:  # noqa: BLE001
        logger.exception("手动换出失败")
        return _fail(500, f"换出失败：{exc}")

    return {
        "unloaded": unloaded,
        "residents": _manager.residents(),
        "resident_gb": round(_manager.current_gb(), 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_service.main:app", host="0.0.0.0", port=8002)
