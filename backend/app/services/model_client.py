"""
model-service HTTP 客户端（api / worker 侧使用）。

为什么 worker 不直接加载模型
---------------------------
`torch.load` 是进程级行为。worker 与 model-service 各加载一份 VLM，权重文件
只有一份也没用 —— 显存里是两份，8GB 卡直接 OOM。所以全项目只有一个容器
碰权重，其余一律走 HTTP。见 DESIGN.md §2 D-决策 2。

错误语义（决定 Celery 是重试还是直接失败）
----------------------------------------
    连接失败 / 5xx        → 可重试（服务重启中、临时过载）
    503 GPU 繁忙          → 可重试，但要看等待时长；映射成 GpuBusyError(3002)
    4xx                   → 不可重试（请求本身有问题，重试也没用）→ 直接抛

⚠️ 关于 caption 的图片传递方式
    本客户端传的是**容器内路径**（不是 base64），因此 model-service
    必须能读到 storage 目录 —— compose 里给它挂了 `./storage:/app/storage:ro`。
    这样做的理由是避免把几 MB 的图片 base64 后塞进 JSON（体积 +33%，且要全量驻留内存）。

对应设计文档：DESIGN.md §7.2 Stage 2/4
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from app.core.config import get_config, get_env
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.core.redis_lock import GpuBusyError

logger = get_logger(__name__)


class ModelServiceError(AppError):
    """model-service 调用失败（可重试类）。"""

    code = 3001  # 复用 PARSE_FAILED 语义位不对，这里用通用内部错误更合适
    message = "模型服务调用失败"


class ModelClient:
    """统一的模型服务客户端。

    ⚠️ 不要在多处 new 这个类：它内部持有 httpx 连接池。
       api 侧用模块级单例 `get_model_client()`。
    """

    def __init__(self) -> None:
        cfg = get_config()
        env = get_env()
        self.base_url: str = str(
            cfg.get_path("models.service_url", env.model_service_url)
        ).rstrip("/")
        self.timeout_sec: float = float(cfg.get_path("models.service_timeout_sec", 300))
        self.max_retries: int = int(cfg.get_path("models.service_max_retries", 3))
        self._client: httpx.AsyncClient | None = None

    # ---------------- 连接 ----------------
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_sec, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ---------------- 通用请求（含重试） ----------------
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST 一个 JSON 并返回解析后的 dict。

        重试策略：仅对「网络层错误」与「5xx」重试（指数退避 1s/2s/4s）；
        4xx 立即失败 —— 请求体本身有问题，重试只是浪费时间。
        """
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await client.post(path, json=payload)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "model-service 连接失败 (%s 第 %s/%s 次): %s",
                    path, attempt, self.max_retries, exc,
                )
            else:
                if resp.status_code == 503:
                    # GPU 锁没抢到 —— 上层（Celery / Agent）决定是否等待重试
                    raise GpuBusyError(_detail(resp, "模型服务繁忙（GPU 锁被占用）"))
                if 400 <= resp.status_code < 500:
                    raise ModelServiceError(
                        f"model-service 拒绝请求 {path} ({resp.status_code}): {_detail(resp)}"
                    )
                if resp.status_code >= 500:
                    last_exc = ModelServiceError(
                        f"model-service {path} 返回 {resp.status_code}: {_detail(resp)}"
                    )
                    logger.warning(
                        "model-service 5xx (%s 第 %s/%s 次): %s",
                        path, attempt, self.max_retries, resp.status_code,
                    )
                else:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise ModelServiceError(f"model-service 返回非 JSON: {exc}") from exc

            if attempt < self.max_retries:
                await asyncio.sleep(2 ** (attempt - 1))

        raise ModelServiceError(f"model-service 调用失败（已重试 {self.max_retries} 次）: {last_exc}")

    # ---------------- 业务端点 ----------------
    async def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """
        单次批量向量化。

        ⚠️ 调用方负责控制批大小（configs 的 ingest.batch.embed_batch）。
           一次塞几千条会把 model-service 的请求体撑爆并长时间占住 GPU 锁。
        """
        if not texts:
            return []
        data = await self._post("/embed", {"texts": texts, "normalize": normalize})
        vectors = data.get("vectors") or data.get("embeddings") or []
        if len(vectors) != len(texts):
            raise ModelServiceError(
                f"向量数量不匹配: 请求 {len(texts)} 条，返回 {len(vectors)} 条"
            )
        return vectors

    async def embed_all(
        self,
        texts: list[str],
        batch_size: int | None = None,
        on_batch: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> list[list[float]]:
        """
        分批向量化，用于大批量入库。

        `on_batch(done, total)` 让 Celery 任务能推进 progress 进度条 ——
        否则一次 30 分钟的任务在数据库里全程停在 90%，用户会以为卡死。
        """
        cfg = get_config()
        size = int(batch_size or cfg.get_path("ingest.batch.embed_batch", 32))
        size = max(size, 1)

        results: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, size):
            chunk = texts[start : start + size]
            results.extend(await self.embed(chunk))
            if on_batch is not None:
                await on_batch(min(start + size, total), total)
        return results

    async def caption(
        self,
        image_path: str,
        context: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        """
        给单张图生成中文 caption。

        `context` 传图片所在页的周边文本 —— 这是让 caption 与文档语义对齐的关键，
        否则通用 caption（如"一张流程图"）在检索时几乎没有区分度。
        """
        cfg = get_config()
        payload = {
            "image_path": image_path,
            "context": context[:2000],       # 防止把整页正文塞进去撑爆 context
            "max_new_tokens": int(
                max_new_tokens or cfg.get_path("models.caption.max_new_tokens", 256)
            ),
        }
        data = await self._post("/vlm/caption", payload)
        return str(data.get("caption") or "").strip()

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 3
    ) -> list[dict[str, Any]]:
        """精排。返回 `[{index, score, document}]`，已按 score 降序。"""
        if not documents:
            return []
        data = await self._post(
            "/rerank", {"query": query, "documents": documents, "top_n": top_n}
        )
        return list(data.get("results") or [])

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        image_path: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[dict[str, str]]:
        """
        流式生成（消费 model-service 的 SSE）。逐段 yield::

            {"event": "delta", "text": "片段"}
            {"event": "done",  "text": "全文"}

        ⚠️ 不走 _post（它有重试包装）—— 流式响应**不可重试**：
            重试意味着把已经推给用户的半截答案再推一遍。
            连接失败直接抛 ModelServiceError，由上层决定话术。

        SSE 解析：按行迭代，`event:` 行记录事件类型，`data:` 行携带 JSON。
        空行是事件分隔符，解析器靠「遇 data 即处理」天然兼容两种空行风格。
        """
        import json as _json

        cfg = get_config()
        payload: dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "image_path": image_path,
            "max_new_tokens": int(
                max_new_tokens
                or cfg.get_path("models.generate.max_new_tokens", 1024)
            ),
            "temperature": float(
                temperature
                or cfg.get_path("models.generate.temperature", 0.2)
            ),
            "stream": True,
        }
        timeout = httpx.Timeout(
            float(cfg.get_path("models.manager.timeouts.generate", 900)),
            connect=10.0,
        )

        client = self._get_client()
        current_event = "message"
        try:
            async with client.stream(
                "POST", "/vlm/generate", json=payload, timeout=timeout
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise ModelServiceError(
                        f"/vlm/generate {resp.status_code}: {body[:300]}"
                    )
                async for raw_line in resp.aiter_lines():
                    line = raw_line.rstrip("\r")
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = _json.loads(line[len("data:"):].strip())
                    except _json.JSONDecodeError:
                        logger.warning("SSE data 行解析失败: %r", line[:120])
                        continue
                    yield {
                        "event": current_event if current_event != "message" else "delta",
                        "text": str(data.get("text") or ""),
                    }
                    current_event = "message"
        except httpx.HTTPError as exc:
            raise ModelServiceError(f"/vlm/generate 流中断: {exc}") from exc

    async def health(self) -> bool:
        try:
            client = self._get_client()
            resp = await client.get("/health", timeout=5.0)
            return resp.status_code < 500
        except httpx.HTTPError:
            return False


def _detail(resp: httpx.Response, fallback: str = "") -> str:
    """从错误响应里提取可读信息，并截断。"""
    try:
        body = resp.json()
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("message") or fallback)[:300]
    except ValueError:
        pass
    text = (resp.text or "").strip()
    return text[:300] if text else fallback


# ============================================================
# 模块级单例
# ============================================================
_client: ModelClient | None = None


def get_model_client() -> ModelClient:
    global _client
    if _client is None:
        _client = ModelClient()
    return _client


async def close_model_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


__all__ = [
    "ModelClient",
    "ModelServiceError",
    "close_model_client",
    "get_model_client",
]
