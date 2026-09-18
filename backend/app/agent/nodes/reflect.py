"""
节点6：自省校验 —— 前置规则闸门 + LLM 复核（Self-RAG 的外置实现）。

两段式的分工（DESIGN §8.5.1）
--------
    规则闸门（纯程序，微秒级）：引用编号闭合、来源合法、表格格式、
    否定一致 —— 这些是**符号操作**，代码一查就准。
    LLM 复核（3B 弱项）：结论与原文的语义比对 —— 只有规则全过才调模型。

为什么这对 3B 模型尤其重要
--------
    自评能力是 3B 的短板（风险 R19）；把能确定性判定的部分交给代码，
    模型 prompt 更聚焦更短，还能省一次前向推理。

对应设计文档：DESIGN.md §8.5.1
"""

from __future__ import annotations

import re
import time

from app.agent.nodes._common import llm_json, trace_event, wall_clock_exceeded
from app.agent.state import AgentState
from app.core.config import get_config
from app.services.sse_broker import publish

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_NEGATIVE_RE = re.compile(r"未提及|没有提及|文档中未|未提供|未包含")
_TABLE_HEADER_RE = re.compile(r"\|.*-.*-.*\||\|---")


def _rule_gate(state: AgentState) -> list[str]:
    """纯程序校验。返回违规列表（空 = 全过）。"""
    violations: list[str] = []
    answer = str(state.get("draft_answer") or "")
    citations = state.get("citations") or []
    context = state.get("context_chunks") or []

    cited = {int(m) for m in _CITE_RE.findall(answer)}
    valid = {int(c.get("index", 0)) for c in citations}

    # 1. 引用编号闭合：正文里的 [n] 必须都能在 citations 里找到
    for n in sorted(cited - valid):
        violations.append(f"正文引用 [{n}] 不存在于引用列表（共 {len(valid)} 条）")

    # 2. 引用来源合法：citations 必须来自本次检索命中
    hit_ids = {str(h.get("point_id")) for h in state.get("reranked_hits") or []}
    for c in citations:
        if str(c.get("point_id")) not in hit_ids:
            violations.append(f"引用 [{c.get('index')}] 的来源不在本次检索结果中")
            break                                   # 一条足以判定，别刷屏

    # 3. 表格格式合规（D6 决策的落地校验）
    #    ★ 只在回答**实际引用了**表格片段时才要求 Markdown 表格 ——
    #      "上下文里恰好带出一个表格块"不代表这个问题的答案需要表格
    #      （Q1 问毛利率文字原因时，表格规则误报过一次，白白烧掉一轮重检）
    table_chunk_indexes = {
        i
        for i, h in enumerate(state.get("context_chunks") or [], start=1)
        if str(h.get("chunk_type")) == "table"
    }
    cited_now = {int(m) for m in _CITE_RE.findall(answer)}
    has_table_chunk = bool(cited_now & table_chunk_indexes)
    if has_table_chunk and not _TABLE_HEADER_RE.search(answer):
        violations.append("回答引用了表格片段但未用 Markdown 表格渲染")

    # 4. 否定一致：说"未提及"却给了引用 —— 自相矛盾
    if _NEGATIVE_RE.search(answer) and not cited and citations:
        violations.append("回答声称未提及但引用列表非空")

    return violations


_SYSTEM = """你是 RAG 答案的质检员。逐条核对这些结论是否有文档片段支撑。

判定标准：
- PASS：所有关键结论都能在片段中找到依据，且引用编号正确。
- FAIL：存在无依据的结论、数字错误、或把片段没有的内容当成了事实。

只输出 JSON。"""

_USER_TEMPLATE = """用户问题：{query}

文档片段：
{context}

助手回答：
{answer}

输出格式：{{"verdict": "PASS|FAIL", "reason": "一句话",
"unsupported_claims": ["无依据的结论", ...],
"new_keywords": ["若 FAIL，给出补检索用的关键词短语", ...]}}"""


async def reflect_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    cfg = get_config()
    session_id = str(state.get("session_id") or "")
    rounds = int(state.get("reflect_round", 0))

    answer = str(state.get("draft_answer") or "")
    if not answer or not state.get("context_chunks"):
        # 没有答案可检，或走的是"无上下文兜底话术"（闲聊/检索失败）——
        # 这两种都没有证据可比对，进自省只会把兜底话术误判成 FAIL
        # 然后触发一轮毫无意义的重检（Q2 闲聊实测多花 35s）。
        return {
            "reflect_verdict": "PASS",
            "reflect_round": rounds,
            "finish_reason": "error" if not answer else state.get("finish_reason"),
            "trace": trace_event(
                "reflect", "end",
                {"skip": "no answer" if not answer else "no context"},
                t0,
            ),
        }

    # ---------- 第一段：规则闸门 ----------
    violations = _rule_gate(state)
    if violations:
        verdict, reason = "FAIL", "；".join(violations[:3])
        publish(session_id, "reflect", {"round": rounds, "verdict": verdict, "reason": reason})
        return {
            "reflect_verdict": verdict,
            "reflect_reason": reason,
            "rule_violations": violations,
            # ★ FAIL 意味着要重检 → 本轮 retry 已消耗，必须 +1，
            #   否则 route_after_reflect 的轮次上限永远打不到（死循环源）
            "reflect_round": rounds + 1,
            # 规则 FAIL 不给 refined_keywords —— 让路由按"没新词"收尾，
            # 因为引用类违规重检也修不了（是生成问题，不是检索问题）
            "refined_keywords": [],
            "trace": trace_event("reflect", "end", {"gate": violations}, t0),
        }

    # ---------- 墙钟 / 轮次前哨：超了直接放行（防护层 2/3） ----------
    if wall_clock_exceeded(state) or rounds >= int(cfg.get_path("agent.max_reflect_rounds", 2)):
        return {
            "reflect_verdict": "PASS",
            "reflect_reason": "到达轮次/墙钟上限，强制放行",
            "reflect_round": rounds,
            "terminate_reason": "max_rounds" if not wall_clock_exceeded(state) else "timeout",
            "trace": trace_event("reflect", "end", {"forced": True, "rounds": rounds}, t0),
        }

    # ---------- 第二段：LLM 语义复核 ----------
    context = "\n\n".join(
        f"[{i}] {str(c.get('content') or '')[:400]}"
        for i, c in enumerate(state.get("context_chunks") or [], start=1)
    )
    prompt = _USER_TEMPLATE.format(
        query=state.get("normalized_query") or state.get("raw_query"),
        context=context or "（无）",
        answer=answer[:2000],
    )

    try:
        parsed = await llm_json(_SYSTEM, prompt, max_new_tokens=512)
        verdict = str(parsed.get("verdict") or "PASS").upper()
        verdict = "FAIL" if verdict == "FAIL" else "PASS"
        reason = str(parsed.get("reason") or "")
        unsupported = [str(x) for x in (parsed.get("unsupported_claims") or [])][:5]
        new_keywords = [str(x) for x in (parsed.get("new_keywords") or []) if str(x).strip()][:5]
    except Exception as exc:  # noqa: BLE001 —— 自省失败不能卡住用户，放行
        verdict, reason, unsupported, new_keywords = "PASS", f"自省失败放行: {exc}", [], []

    publish(session_id, "reflect", {"round": rounds, "verdict": verdict, "reason": reason})
    return {
        "reflect_verdict": verdict,
        "reflect_reason": reason,
        "unsupported_claims": unsupported,
        "refined_keywords": new_keywords if verdict == "FAIL" else [],
        # ★ 同上：FAIL → 重检 → 本轮 retry 已消耗
        "reflect_round": rounds + 1 if verdict == "FAIL" else rounds,
        "trace": trace_event(
            "reflect", "end",
            {"verdict": verdict, "new_keywords": new_keywords},
            t0,
        ),
    }
