"""
条件边 —— 图的"路由器"（Routing 模式的全部逻辑都在这 4 个函数里）。

口径纪律（DESIGN §8.2/§8.3）
--------
    这是 **Workflow 的固定路由**，不是 ReAct 的自主决策：
    每个函数都是纯逻辑（读 state → 返回分支名），不调 LLM、无副作用。
    图里没有任何回边指向 decide —— 面试/讲解时要主动说明这一点。

对应设计文档：DESIGN.md §8.3 / §8.4
"""

from __future__ import annotations

from app.agent.state import AgentState
from app.core.config import get_config


def _cfg() -> tuple[int, bool, float, int]:
    """集中取 4 个路由参数，避免每个函数重复 get_config。"""
    cfg = get_config()
    return (
        int(cfg.get_path("agent.max_total_retrieval", 4)),
        bool(cfg.get_path("retrieval.hitl_enabled", True)),
        float(cfg.get_path("retrieval.score_threshold", 0.35)),
        int(cfg.get_path("agent.max_reflect_rounds", 2)),
    )


def route_after_decide(state: AgentState) -> str:
    """检索决策之后：要检索吗？（覆盖总次数上限）"""
    max_total, _, _, _ = _cfg()
    # ★ 防护层 1：总检索次数上限 —— HITL 重检 + 自省重检共用这个池子，
    #   双循环叠加时靠它兜底
    if state.get("retrieval_round", 0) >= max_total:
        return "generate"
    return "retrieve" if state.get("need_retrieval", True) else "generate"


def route_after_retrieve(state: AgentState) -> str:
    """检索之后：分数够吗？不够且 HITL 可用 → 挂起问用户。"""
    _, hitl_enabled, threshold, _ = _cfg()
    # 一轮都没召回到 → 直接生成（生成节点会走"文档中未提及"话术）
    if not state.get("context_chunks"):
        return "generate"
    # 分数过低 + HITL 开启 + 本轮没用过 → 拉起人工断点
    # （hitl_triggered 防止 resume → retrieve → 再次 interrupt 的死循环）
    if (
        hitl_enabled
        and not state.get("hitl_triggered")
        and state.get("max_relevance_score", 0.0) < threshold
    ):
        return "hitl"
    return "generate"


def route_after_hitl(state: AgentState) -> str:
    """HITL 恢复之后：用户选了改写重检，还是凑合用现有上下文？"""
    if state.get("hitl_choice") == "rewrite_query" and state.get("hitl_new_query"):
        return "retrieve"
    return "generate"


def route_after_reflect(state: AgentState) -> str:
    """自省之后：PASS / 超轮次 / 没给新词 → 收尾；否则用新词重检。"""
    _, _, _, max_rounds = _cfg()
    verdict = state.get("reflect_verdict", "PASS")
    rounds = state.get("reflect_round", 0)
    if verdict == "PASS":
        return "finalize"
    # ★ 防护层 2：自省重试轮次上限
    if rounds >= max_rounds:
        return "finalize"
    # 没给出新关键词 → 重检也不会更好，直接收尾
    if not state.get("refined_keywords"):
        return "finalize"
    return "retrieve"
