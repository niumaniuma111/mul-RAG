"""
节点2：检索决策 —— 单次规划（Routing 模式），不是 ReAct 循环。

⚠️ 口径纪律（DESIGN §3.1 / 坑 6）
--------
    本节点只在图入口执行一次，**没有任何回边指向它**。
    它回答的是"这次要不要检索 + 用什么关键词"，不是"下一步调什么工具"。
    讲解时不要说"Agent 决定调用检索工具"——那是 ReAct 的说法。

输出强制 JSON 的原因
--------
    3B 模型的自由文本规划无法机器消费；JSON 让 need_retrieval/keywords
    变成可判定的字段。关键词要求是**检索用短语**（名词性）而非原句 ——
    它们会同时进 BM25（分词）与 rerank 改写。

对应设计文档：DESIGN.md §8.5（decide 行）
"""

from __future__ import annotations

import time

from app.agent.nodes._common import llm_json, trace_event
from app.agent.state import AgentState

_SYSTEM = """你是 RAG 问答系统的检索决策器。判断用户的问题是否需要检索文档库。

判定规则：
1. 闲聊、寒暄、常识问答（"你好""你是谁"）→ need_retrieval=false
2. 询问文档内容相关的事实、数据、对比、总结 → need_retrieval=true
3. keywords 是 2~5 个**检索用短语**（名词性短语，如"毛利率 环比变化"），
   不是原句复述；include 同义词/相关词以提高召回。
4. 只输出 JSON。"""

_USER = """用户问题：{query}
问题类型：{intent}

输出格式：{{"need_retrieval": true/false, "keywords": ["短语1", "短语2"], "reason": "一句话理由"}}"""


async def decide_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    query = str(state.get("normalized_query") or state.get("raw_query") or "")
    intent = state.get("query_intent", "lookup")

    # 闲聊/无法理解：跳过 LLM，直接不检索（省一次前向，语义也明确）
    if intent in ("chat", "unclear"):
        return {
            "need_retrieval": False,
            "retrieval_keywords": [],
            "decide_reason": f"intent={intent}，无需检索",
            "decide_elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "trace": trace_event("decide", "end", {"need_retrieval": False, "skip": intent}, t0),
        }

    prompt = _USER.format(query=query, intent=intent)
    try:
        parsed = await llm_json(_SYSTEM, prompt, max_new_tokens=256)
        need = bool(parsed.get("need_retrieval", True))
        keywords = [str(k) for k in (parsed.get("keywords") or []) if str(k).strip()]
        reason = str(parsed.get("reason") or "")
    except Exception as exc:  # noqa: BLE001 —— 降级：默认检索原句
        need, keywords, reason = True, [query], f"decide 失败降级: {exc}"

    return {
        "need_retrieval": need,
        "retrieval_keywords": keywords[:5] or [query],
        "decide_reason": reason,
        "decide_elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "trace": trace_event(
            "decide", "end",
            {"need_retrieval": need, "keywords": keywords[:5]},
            t0,
        ),
    }
