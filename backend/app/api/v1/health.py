"""
健康检查。

两个端点，用途完全不同，不要搞混：

    GET /api/v1/health        轻量存活检查 —— 只证明进程还活着。
                              给 docker-compose 的 healthcheck 用，必须快且不依赖外部服务。
    GET /api/v1/health/deep   逐项探测依赖（PG / Redis / Qdrant / model-service / MinerU）。
                              排障与联调用。

⚠️ 为什么不让 compose healthcheck 调 deep 端点：
    PG 或 Qdrant 抖一下，deep 就返回非 200，compose 会把 api 容器反复重启 ——
    而重启解决不了下游抖动，只会把一次小故障放大成雪崩。
    依赖的健康状态应该是可观测指标，不是重启触发器。
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter

from app import __version__
from app.core import redis_client
from app.core.config import get_env, resolve_mineru_device
from app.core.exceptions import ok
from app.db import session as db_session

router = APIRouter()

_STARTED_AT = time.monotonic()


@router.get("/health", summary="存活检查（轻量）")
async def health() -> dict[str, Any]:
    return ok(
        {
            "status": "ok",
            "service": "ppt-agentic-rag-api",
            "version": __version__,
            "uptime_sec": round(time.monotonic() - _STARTED_AT, 1),
        }
    )


@router.get("/health/deep", summary="依赖深检（排障用）")
async def health_deep() -> dict[str, Any]:
    env = get_env()

    checks: dict[str, str] = {
        "postgres": "ok" if await db_session.ping() else "fail",
        "redis": "ok" if await redis_client.ping() else "fail",
        "qdrant": await _probe(f"{env.qdrant_url}/readyz"),
        "model_service": await _probe(f"{env.model_service_url}/health"),
        "mineru": await _probe(f"{env.mineru_url}/health"),
    }

    all_ok = all(v == "ok" for v in checks.values())
    return ok(
        {
            "status": "ok" if all_ok else "degraded",
            "mineru_device": resolve_mineru_device(),
            "checks": checks,
        }
    )


async def _probe(url: str, timeout: float = 3.0) -> str:
    """
    探活一个 HTTP 依赖。

    4xx 也算「服务在」—— 比如 model-service 的 /health 若因路径拼错返回 404，
    我们想知道的是「进程在不在」，不是「这个路径对不对」。
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
        return "ok" if resp.status_code < 500 else f"fail({resp.status_code})"
    except Exception:  # noqa: BLE001
        return "unreachable"


__all__ = ["router"]
