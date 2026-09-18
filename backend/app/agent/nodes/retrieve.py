"""
节点3：混合检索 —— 图里唯一不碰 LLM 的节点。

关键词 → 查询文本的口径
--------
    retrieval_keywords 是 decide/preprocess/自省 累加的短语池。
    dense 路用「原查询 + 关键词拼接」—— 向量模型对完整句子的表征
    比碎片短语更稳；sparse 路（BM25）天然受益于短语化分词。
    重检轮次（HITL 改写 / 自省 refined_keywords）会把新词并进查询。

超时与降级
--------
    retrieval_service 的网络异常在这里捕获：检索失败 → 空结果 +
    trace error，走 generate 的"文档中未提及"话术，而不是炸掉整图。
    （在线问答对可用性的要求高于入库 —— 入库失败可以重跑，问答不行。）

对应设计文档：DESIGN.md §9.1-§9.4
"""

from __future__ import annotations

import time

from app.agent.nodes._common import trace_event, wall_clock_exceeded
from app.agent.state import AgentState
from app.services.retrieval_service import get_retrieval_service


def _query_text(state: AgentState) -> str:
    """组装 dense 路的查询文本：原查询在前（保句子语义），关键词在后（补召回）。"""
    parts = [str(state.get("normalized_query") or state.get("raw_query") or "").strip()]
    for kw in state.get("retrieval_keywords") or []:
        kw = str(kw).strip()
        if kw and kw not in parts:
            parts.append(kw)
    return " ".join(parts[:6])   # 防御：异常长的关键词池截断


async def retrieve_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    # ★ 防护层 3 前哨：墙钟已超时就不再花一次检索的钱，让流程尽快收尾
    if wall_clock_exceeded(state):
        return {
            "retrieval_round": state.get("retrieval_round", 0) + 1,
            "context_chunks": [],
            "max_relevance_score": 0.0,
            "retrieval_elapsed_ms": 0,
            "terminate_reason": "timeout",
            "trace": trace_event("retrieve", "error", {"reason": "wall_clock"}, t0),
        }

    svc = get_retrieval_service()
    query = _query_text(state)

    try:
        result = await svc.search(
            query,
            user_id=int(state["user_id"]),
            doc_ids=state.get("doc_ids") or None,
            version_override=state.get("version_override") or None,
        )
        fused = result["fused_hits"]
        reranked = result["reranked_hits"]
        max_score = result["max_relevance_score"]
        error = None
    except Exception as exc:  # noqa: BLE001 —— 检索失败不炸图
        fused, reranked, max_score = [], [], 0.0
        error = f"retrieval failed: {exc}"

    return {
        "retrieval_round": state.get("retrieval_round", 0) + 1,
        "retrieval_filter": result.get("filter", {}) if not error else {},
        "fused_hits": fused,
        "reranked_hits": reranked,
        "context_chunks": reranked,
        "max_relevance_score": max_score,
        "retrieval_elapsed_ms": result["elapsed_ms"] if not error else 0,
        "error": error,
        "trace": trace_event(
            "retrieve", "error" if error else "end",
            {
                "query": query[:80],
                "fused": len(fused),
                "reranked": len(reranked),
                "max_score": round(max_score, 4),
                "round": state.get("retrieval_round", 0) + 1,
                **({"error": error} if error else {}),
            },
            t0,
        ),
    }
