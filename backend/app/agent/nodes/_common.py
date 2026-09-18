"""
节点公共工具：打点、计时、LLM JSON 调用。

为什么 decide / preprocess / reflect 共用一个 LLM-JSON 通道
--------
    三个节点都要"小模型输出结构化判定"。3B 模型对自由文本的指令遵循
    本就不稳，对 JSON 更甚 —— 所以统一走：低温 + 强制 JSON prompt +
    宽松抽取（首个 { 到末个 }），再由各节点自己做字段校验。
    反复重试问模型"你输出的 JSON 哪里坏了"是 3B 上的负优化。

对应设计文档：DESIGN.md §8.5
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.agent.state import AgentState, TraceEvent
from app.core.config import get_config
from app.core.logging import get_logger
from app.services.model_client import get_model_client

logger = get_logger(__name__)

# 匹配首个 JSON 对象（容忍模型在 JSON 前后加说明文字）
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def now_ts() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def trace_event(node: str, status: str, detail: dict[str, Any], t0: float) -> list[TraceEvent]:
    """造一条 trace（节点返回 dict 时直接 `"trace": trace_event(...)`）。"""
    return [
        TraceEvent(
            node=node,
            status=status,  # type: ignore[typeddict-item]
            detail=detail,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            ts=now_ts(),
        )
    ]


async def llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """
    让 VLM 输出 JSON 并解析。失败（无 JSON / 字段缺失）抛 ValueError，
    由调用方决定降级行为 —— 各节点的降级策略不同，这里不代劳。
    """
    cfg = get_config()
    mc = get_model_client()
    pieces: list[str] = []
    async for chunk in mc.generate_stream(
        user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=float(cfg.get_path("models.generate.temperature", 0.2)),
    ):
        if chunk.get("event") == "delta":
            pieces.append(chunk.get("text", ""))
        elif chunk.get("event") == "error":
            raise ValueError(f"LLM 流式返回 error: {chunk.get('text')}")

    raw = "".join(pieces).strip()
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"LLM 未输出 JSON: {raw[:200]!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM 输出的 JSON 无法解析: {exc}: {raw[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM 输出不是 JSON 对象: {type(parsed).__name__}")
    return parsed


def wall_clock_exceeded(state: AgentState) -> bool:
    """★ 防护层 3：单次问答墙钟上限。在 reflect / retrieve 入口都查一次。"""
    limit = float(get_config().get_path("agent.max_wall_clock_sec", 120))
    started = state.get("started_at") or 0.0
    return started > 0 and (time.monotonic() - started) > limit
