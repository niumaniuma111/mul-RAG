"""
图构建 —— Routing + Evaluator-optimizer 工作流（非 ReAct，见 DESIGN §3.1）。

拓扑一览
--------
    START → preprocess → decide ─┬→ retrieve ─┬→ generate → reflect ─┬→ finalize → END
                                 │            └→ hitl ─┬→ retrieve    │
                                 └→ generate           └→ generate    └→ retrieve（重检）

四条条件边的判定全在 routing.py（纯函数）；本文件只负责装配。

recursion_limit
--------
    agent.recursion_limit=30 是死循环的最终兜底（防护层 4）：
    本图理论最长路径 ≈ 2×(retrieve+generate+reflect) + 7 ≈ 13 步，
    30 留了一倍余量 —— 触发它说明路由写错了，fail-fast 比静默截断好。

对应设计文档：DESIGN.md §8.2
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.decide import decide_node
from app.agent.nodes.finalize import finalize_node
from app.agent.nodes.generate import generate_node
from app.agent.nodes.hitl import hitl_node
from app.agent.nodes.preprocess import preprocess_node
from app.agent.nodes.reflect import reflect_node
from app.agent.nodes.retrieve import retrieve_node
from app.agent.routing import (
    route_after_decide,
    route_after_hitl,
    route_after_reflect,
    route_after_retrieve,
)
from app.agent.state import AgentState
from app.core.config import get_config
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_graph(checkpointer: BaseCheckpointSaver) -> Any:
    """装配问答图。checkpointer 传 AsyncPostgresSaver（HITL 挂起恢复的前提）。"""
    g: StateGraph = StateGraph(AgentState)

    g.add_node("preprocess", preprocess_node)
    g.add_node("decide", decide_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("hitl", hitl_node)
    g.add_node("generate", generate_node)
    g.add_node("reflect", reflect_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "preprocess")
    g.add_edge("preprocess", "decide")

    # ── 分支1：要不要检索（闲聊直达生成）──
    g.add_conditional_edges(
        "decide", route_after_decide, {"retrieve": "retrieve", "generate": "generate"}
    )
    # ── 分支2：检索后分数是否够（低分且未用过 HITL → 挂起问用户）──
    g.add_conditional_edges(
        "retrieve", route_after_retrieve, {"hitl": "hitl", "generate": "generate"}
    )
    # ── 分支3：HITL 恢复后走向 ──
    g.add_conditional_edges(
        "hitl", route_after_hitl, {"retrieve": "retrieve", "generate": "generate"}
    )

    g.add_edge("generate", "reflect")

    # ── 分支4：自省 PASS → 收尾；FAIL 且有新词 → 重检 ──
    g.add_conditional_edges(
        "reflect", route_after_reflect, {"retrieve": "retrieve", "finalize": "finalize"}
    )

    g.add_edge("finalize", END)

    limit = int(get_config().get_path("agent.recursion_limit", 30))
    logger.info("问答图装配完成（recursion_limit=%s）", limit)

    return g.compile(checkpointer=checkpointer)


_graph: Any = None


async def get_graph() -> Any:
    """模块级单例（图 + checkpointer 全程复用，不要每请求重建）。"""
    global _graph
    if _graph is None:
        from app.agent.checkpointer import init_checkpointer

        _graph = build_graph(await init_checkpointer())
    return _graph


async def close_graph() -> None:
    global _graph
    if _graph is not None:
        from app.agent.checkpointer import close_checkpointer

        await close_checkpointer()
        _graph = None
