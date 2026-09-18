"""
节点4：HITL 人工断点 —— LangGraph interrupt() 的落点。

幂等性（DESIGN §8.6 明文要求）
--------
    interrupt() 恢复时，**本节点会从头重新执行**（LangGraph 的语义）。
    所以节点体必须满足：interrupt 之前的代码无副作用、
    靠 state 里已有的字段（hitl_triggered 等）判断"这是恢复后的第二次进入"。
    判断依据：`hitl_resolved=True` 说明 resume 已注入 choice，直接放行 ——
    此时再调 interrupt() 会立即返回 resume 值（LangGraph 缓存语义），
    但显式分支让意图更清晰，也少依赖框架内部行为。

interrupt payload 与前端弹窗字段的对应
--------
    {reason, max_score, threshold, query, options} → el-dialog 的文案与按钮。

对应设计文档：DESIGN.md §8.6
"""

from __future__ import annotations

import time

from langgraph.types import interrupt

from app.agent.nodes._common import trace_event
from app.agent.state import AgentState
from app.core.config import get_config
from app.services.qa_persistence import record_hitl_pending

_OPTIONS = ["rewrite_query", "use_current_context"]


async def hitl_node(state: AgentState) -> dict:
    t0 = time.perf_counter()

    # ---- 恢复路径：resume 已注入，直接消费选择 ----
    if state.get("hitl_resolved"):
        return {
            "trace": trace_event(
                "hitl", "end",
                {"resolved": True, "choice": state.get("hitl_choice")},
                t0,
            ),
        }

    # ---- 首次进入：挂起图，等前端 POST /qa/resume ----
    threshold = float(get_config().get_path("retrieval.score_threshold", 0.35))
    payload = {
        "reason": "检索结果与问题的相关性过低，请选择处理方式",
        "max_score": round(float(state.get("max_relevance_score", 0.0)), 4),
        "threshold": threshold,
        "query": state.get("normalized_query") or state.get("raw_query"),
        "options": _OPTIONS,
    }

    # ★ 坑 31：interrupt() 会把图停在这里，finalize 永不执行 →
    #   挂起的会话在 qa_traces 里**完全没有痕迹**（A21 两次复现缺席）。
    #   所以必须在挂起**之前**落一条 PENDING_HITL 行；resume 后 finalize 会把
    #   它 UPDATE 成完整行（见 services/qa_persistence）。用户不点 / 超时 / 断连
    #   这些最该被发现的"卡住"场景，靠这条记录才可见。
    await record_hitl_pending(state)

    # interrupt() 会把图挂起并把 payload 写进 checkpoint；SSE 层据此推 event:hitl。
    # 恢复时这个调用返回用户提交的 {"action": ..., "new_query": ...}。
    resumed = interrupt(payload)
    resumed = resumed or {}

    action = str(resumed.get("action") or "use_current_context")
    if action not in _OPTIONS:
        action = "use_current_context"
    new_query = str(resumed.get("new_query") or "").strip() or None

    return {
        "hitl_triggered": True,       # ★ 本轮问答只允许打断一次（routing 判据）
        "hitl_choice": action,
        "hitl_new_query": new_query,
        "hitl_resolved": True,
        # ★ 用户思考期间不该计入墙钟：resume 后重置起点，
        #   否则挂起超过 max_wall_clock_sec 的会话一恢复就超时
        "started_at": time.monotonic(),
        # 用户改写的新查询要真正生效：覆盖 normalized_query 并登记关键词
        "normalized_query": new_query if (action == "rewrite_query" and new_query)
        else state.get("normalized_query"),
        "retrieval_keywords": [new_query] if (action == "rewrite_query" and new_query) else [],
        "trace": trace_event(
            "hitl", "end",
            {"action": action, "new_query": (new_query or "")[:60]},
            t0,
        ),
    }
