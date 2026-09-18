"""
节点7：收尾 —— 引用对齐、终稿定稿、落库。

[cn] 与 citations 的对齐规则
--------
    generate 阶段的 citations 骨架按 context_chunks 顺序编号，模型被要求
    用相同编号引用。但模型可能漏引（正文没有 [2] 却有 3 条 citations）。
    收尾时**只保留正文实际引用了的**条目并重新编号，保证前端渲染的
    引用卡片与正文 [n] 严格一致 —— 这是溯源可信的最低要求。

finish_reason 的语义
--------
    pass          正常通过自省
    max_rounds    自省连续 FAIL 到上限（答案仍会给出，但前端可标记"仅供参考"）
    hitl_skip     用户在 HITL 选了"用现有上下文"
    no_retrieval  闲聊/无需检索
    error         检索或生成失败（答案为兜底话术）

对应设计文档：DESIGN.md §8.5（finalize 行）
"""

from __future__ import annotations

import re
import time

from app.agent.nodes._common import trace_event
from app.agent.state import AgentState, Citation
from app.services.qa_persistence import record_final
from app.services.sse_broker import publish

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


async def finalize_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    answer = str(state.get("draft_answer") or "")
    citations = list(state.get("citations") or [])

    # ---------- 引用对齐：只留正文实际引用的，并重编号 ----------
    used = sorted({int(m) for m in _CITE_RE.findall(answer)})
    by_index = {int(c.get("index", 0)): c for c in citations}
    aligned: list[Citation] = []
    remap: dict[int, int] = {}
    for new_idx, old_idx in enumerate(used, start=1):
        remap[old_idx] = new_idx
        if old_idx in by_index:
            c = dict(by_index[old_idx])
            c["index"] = new_idx
            aligned.append(Citation(**{k: v for k, v in c.items() if k in Citation.__annotations__}))

    # 正文里的旧编号同步替换（[3]→[2] 这类），无效引用 [9] 原样保留交给 reflect 记录
    if remap and used != list(range(1, len(citations) + 1)):
        answer = _CITE_RE.sub(
            lambda m: f"[{remap.get(int(m.group(1)), int(m.group(1)))}]", answer
        )

    # ---------- finish_reason ----------
    if state.get("error"):
        finish = "error"
    elif state.get("terminate_reason") == "timeout":
        finish = "max_rounds"
    elif not state.get("need_retrieval", True):
        finish = "no_retrieval"
    elif state.get("hitl_choice") == "use_current_context":
        finish = "hitl_skip"
    elif state.get("reflect_verdict") == "FAIL":
        finish = "max_rounds"
    else:
        finish = "pass"

    final = {
        "final_answer": answer,
        "citations": aligned,
        "finish_reason": finish,
        "trace": trace_event(
            "finalize", "end",
            {"citations": len(aligned), "finish": finish},
            t0,
        ),
    }

    # ---------- 落库（尽力而为，失败不影响返回） ----------
    # 逻辑已下沉到 services/qa_persistence.record_final —— 因为 HITL 挂起路径
    # 也要写 qa_traces（坑 31），两处必须共用同一套 SQL，否则会插出重复行。
    await record_final(
        state,
        answer=answer,
        citations=[dict(c) for c in aligned],
        finish_reason=finish,
    )

    # ---------- 推终帧 ----------
    publish(
        str(state.get("session_id") or ""),
        "final",
        {
            "answer": answer,
            "citations": [dict(c) for c in aligned],
            "finish_reason": finish,
        },
    )
    return final
