"""
节点1：Query 预处理 —— 指代消解 + 意图分类。

为什么指代消解要用 LLM 而不是规则
--------
    "它的毛利率呢？" 里"它"指什么，取决于 history 里的实体。
    规则能覆盖的（"这个/它/上面"）只能标记不能消解，
    而消解错了检索词就全错 —— 这一步是后续所有节点的地基。

降级策略
--------
    LLM 失败时：normalized_query = raw_query 原样透传、intent=lookup。
    指代消解失败只是"多轮对话的第二轮变差"，不该炸掉整次问答。

对应设计文档：DESIGN.md §8.5（preprocess 行）
"""

from __future__ import annotations

import time

from app.agent.nodes._common import llm_json, trace_event
from app.agent.state import AgentState

_SYSTEM = """你是问答系统的查询预处理器。根据对话历史，把用户最新的问题改写成
【独立完整、不依赖上下文】的查询，并判断意图类型。

规则：
1. 若问题含指代词（它/这个/该/上述等），必须用历史中的具体实体替换。
2. 若问题独立完整，原样保留，不要改写。
3. intent 只能是：lookup（查事实）/ compare（对比）/ summarize（总结）/
   chat（闲聊寒暄，与文档无关）/ unclear（无法理解）。
4. 只输出 JSON，不要任何解释。"""

_USER = """对话历史：
{history}

用户最新问题：{query}

输出格式：{{"normalized_query": "...", "intent": "lookup|compare|summarize|chat|unclear"}}"""


def _history_block(history: list[dict[str, str]]) -> str:
    """把最近几轮对话压成纯文本（超出 max_history_messages 的丢弃）。"""
    if not history:
        return "（首轮，无历史）"
    lines = []
    for turn in history:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}: {turn.get('content', '')[:200]}")
    return "\n".join(lines) if lines else "（首轮，无历史）"


async def preprocess_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    raw = str(state.get("raw_query") or "").strip()
    history = list(state.get("history") or [])

    prompt = _USER.format(history=_history_block(history), query=raw)
    try:
        parsed = await llm_json(_SYSTEM, prompt, max_new_tokens=256)
    except Exception:  # noqa: BLE001 —— 降级：原样透传
        parsed = {"normalized_query": raw, "intent": "lookup"}

    normalized = str(parsed.get("normalized_query") or raw).strip() or raw
    intent = str(parsed.get("intent") or "lookup")
    if intent not in ("lookup", "compare", "summarize", "chat", "unclear"):
        intent = "lookup"

    return {
        "normalized_query": normalized,
        "query_intent": intent,
        "history": history,
        "trace": trace_event(
            "preprocess", "end",
            {"intent": intent, "normalized": normalized[:80], "has_history": bool(history)},
            t0,
        ),
    }
