"""
Agent 状态定义 —— 全图唯一的"共享内存"。

设计要点（对应 DESIGN.md §8.1）
--------
1. 用 ``TypedDict(total=False)`` 而不是 pydantic：LangGraph 的 State 通道
   语义（reducer / 覆盖）依赖 TypedDict 的 Annotated 标注，pydantic 模型
   反而会在每节点间做全量校验拷贝，白白费内存。

2. **Annotated reducer 是"累加"语义的唯一来源**：
   - ``retrieval_keywords`` 用 ``merge_keywords`` —— 自省阶段追加新词时
     去重合并，而不是覆盖（覆盖会丢掉 decide 轮给的关键词）；
   - ``dense_hits / sparse_hits / trace`` 用 ``operator.add`` —— 多轮检索
     的命中与观测打点必须**累积**，否则第 2 轮会把第 1 轮的痕迹抹掉；
   - 其余字段无标注 = 默认"最后写入者胜"。

3. v1.1.4 已移除 ``messages`` 槽（ReAct 口径的遗留），本图没有 ToolNode 消费它。

对应设计文档：DESIGN.md §8.1
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


def merge_keywords(old: list[str], new: list[str]) -> list[str]:
    """关键词累加去重，保留顺序（供自省阶段追加新词）。"""
    seen: set[str] = set()
    out: list[str] = []
    for k in (old or []) + (new or []):
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


class RetrievedChunk(TypedDict, total=False):
    """单条检索结果（最终渲染成引用卡片）。

    ⚠️ 字段名与 ``vector_store.upsert_chunks`` 写入的 payload 键**逐字对齐**：
       doc_id / file_name / page_num / version / chunk_type / content /
       table_html / image_path。DESIGN 早期示例里的 ``file_id`` 是旧名，
       以这里为准 —— 错一个键检索结果就会静默丢字段。
    """

    point_id: str
    doc_id: str
    file_name: str
    page_num: int
    version: int
    chunk_type: Literal["text", "table", "image_caption"]
    content: str
    table_html: Optional[str]
    image_path: Optional[str]

    dense_score: Optional[float]     # 服务端 RRF 融合后不可得，置 None
    sparse_score: Optional[float]    # 同上
    rrf_score: float                 # 融合分（量纲 ~0.016-0.033，勿与阈值比）
    rerank_score: Optional[float]    # ★ 重排相关性分（HITL 判据，sigmoid 0~1）
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


class Citation(TypedDict, total=False):
    """引用卡片（finalize 产出，前端渲染溯源）。"""

    index: int                       # 正文里的 [1] [2]
    point_id: str
    doc_id: str
    file_name: str
    page_num: int
    version: int
    chunk_type: str
    snippet: str
    table_html: Optional[str]
    image_path: Optional[str]
    rerank_score: float


class TraceEvent(TypedDict, total=False):
    """全链路打点（前端"推理链路可视化"的数据源）。"""

    node: str
    status: Literal["start", "end", "error"]
    detail: dict[str, Any]
    elapsed_ms: int
    ts: str


class AgentState(TypedDict, total=False):
    # ───────── 入参 ─────────
    session_id: str
    user_id: int
    raw_query: str
    doc_ids: Optional[list[str]]                  # 限定检索范围（可空=全部）
    version_override: Optional[dict[str, int]]    # ★ 指定某文档用历史版本
    started_at: float                             # ★ 墙钟起点（防护层 3 判据）
    terminate_reason: Optional[str]               # ★ pass / max_rounds / timeout

    # ───────── 节点1：Query 预处理 ─────────
    normalized_query: str
    query_intent: Literal["lookup", "compare", "summarize", "chat", "unclear"]
    history: list[dict[str, str]]                 # 最近 N 轮对话（供指代消解）

    # ───────── 节点2：检索决策（单次规划，非 ReAct 循环）─────────
    need_retrieval: bool
    retrieval_keywords: Annotated[list[str], merge_keywords]   # ★ 累加 reducer
    decide_reason: str
    decide_elapsed_ms: int

    # ───────── 节点3：混合检索 ─────────
    retrieval_filter: dict[str, Any]              # 实际下发的 filter（便于观测）
    retrieval_round: int                          # 检索轮次（HITL/自省重检 +1）
    dense_hits: Annotated[list[RetrievedChunk], operator.add]
    sparse_hits: Annotated[list[RetrievedChunk], operator.add]
    fused_hits: list[RetrievedChunk]
    reranked_hits: list[RetrievedChunk]
    context_chunks: list[RetrievedChunk]          # 最终进 prompt 的 top3
    max_relevance_score: float                    # ★ HITL 判据
    retrieval_elapsed_ms: int

    # ───────── 节点4：HITL ─────────
    hitl_triggered: bool
    hitl_reason: str
    hitl_options: list[str]                       # ["rewrite_query", "use_current_context"]
    hitl_choice: Optional[str]
    hitl_new_query: Optional[str]
    hitl_resolved: bool

    # ───────── 节点5：生成 ─────────
    draft_answer: str
    stream_chunks: int                            # SSE 已推 token 数
    generate_elapsed_ms: int

    # ───────── 节点6：自省 ─────────
    reflect_round: int                            # ★ 已重试轮数
    reflect_verdict: Literal["PASS", "FAIL", ""]
    reflect_reason: str
    unsupported_claims: list[str]                 # 无证据支撑的结论
    refined_keywords: list[str]                   # 自省给出的新检索词
    rule_violations: list[str]                    # ★ 前置规则闸门的命中记录

    # ───────── 节点7：收尾 ─────────
    final_answer: str
    citations: list[Citation]
    finish_reason: Literal[
        "pass", "max_rounds", "hitl_skip", "no_retrieval", "error"
    ]

    # ───────── 观测 ─────────
    trace: Annotated[list[TraceEvent], operator.add]   # ★ 全链路打点
    error: Optional[str]
