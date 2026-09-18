"""
节点5：答案初稿生成 —— 全项目唯一的流式 LLM 消费点。

事件流经两跳 SSE
--------
    model-service /vlm/generate（delta 片段）
      → 本节点消费并 publish 到 sse_broker（event: token）
        → qa.py 的 SSE 生成器转发浏览器

引用标注的契约
--------
    强制模型用 [1] [2] 标注 —— finalize 阶段会把编号与 citations 对齐；
    证据不足时明确说"文档中未提及"（reflect 的否定一致性规则依赖这句）。
    表格数据用 Markdown 表格渲染（reflect 规则闸门会检查 |--- 表头）。

阈值外防线（A26/A27）
--------
    score_threshold 拦不住的两条：A26（资产负债率 0.723）、A27（竞争对手 0.560）。
    抬阈值不可行 —— 能拦住它们时可答用例也被拦光（见 S5 标定：0.725 起误拦 25.9%）。
    所以防线放在 prompt 层：把「未提及」的判据写成**字段级**而非话题级，
    并明确"话题相近但缺该项数据 → 报未提及"。3B 模型在 prompt 到位时能照做
    （A26/A27 在旧 prompt 下最终答案就已正确，此处是把这条行为固化+泛化）。

对应设计文档：DESIGN.md §8.5（generate 行）/ §12 SSE 协议
"""

from __future__ import annotations

import time

from app.agent.nodes._common import now_ts, trace_event
from app.agent.state import AgentState, Citation
from app.core.config import get_config
from app.services.model_client import get_model_client
from app.services.sse_broker import publish

_SYSTEM = """你是基于文档的问答助手。严格遵守：
1. 只依据提供的文档片段回答；片段中没有的信息，明确说"文档中未提及"。
   ★ 判断标准是**字段级**的，不是"话题级"的：片段里出现了相关话题（同一家公司、
   同一年度、同一份报告），但**没有问到的那一项具体数据/事实**时，一律按
   "文档中未提及"处理，绝不可以用话题相近的其它数字替代，也不要从其它字段
   推算或估计。
   示例：问"资产负债率"，片段里只有营业收入/毛利率 → 答"文档中未提及资产负债率"，
   不能因为同属财务指标就给出别的比例。
   示例：问"主要竞争对手"，片段里只有客户集中度/供应商集中度 → 答"文档中未提及
   主要竞争对手"。
2. ★ 每一个依据片段得出的事实性结论，句末必须带来源标注 [1] [2]，
   示例："毛利率环比提升约两个百分点 [1]。"漏标视为错误回答。
   但"文档中未提及"这类**无依据**的答复**不要**加 [n] 标注 —— 没有来源可标。
3. 涉及表格数据时用 Markdown 表格渲染。
4. 用简体中文，简洁准确。"""

_USER_TEMPLATE = """文档片段：
{context}

用户问题：{query}

请回答。先确认片段里是否**直接包含**回答该问题所需的那一项数据或事实：
- 包含 → 据实回答，并在每个事实性结论句末用 [n] 标注来源；
- 不包含（哪怕片段讨论了相近话题）→ 只回"文档中未提及"，不要给替代数字。"""


def _context_block(state: AgentState, max_tokens: int) -> tuple[str, list[Citation]]:
    """
    把 context_chunks 拼成 prompt 上下文，同时预生成 citations 骨架。
    编号 = chunk 在列表中的序号 —— 与正文的 [n] 一一对应。
    """
    cfg = get_config()
    # 粗略截断：1 个中文字 ≈ 1 token（Qwen tokenizer 对中文略合算），留 20% 余量
    budget = int(max_tokens * 0.8)
    lines: list[str] = []
    citations: list[Citation] = []
    used = 0

    for idx, chunk in enumerate(state.get("context_chunks") or [], start=1):
        content = str(chunk.get("content") or "")
        header = f"[{idx}] （{chunk.get('file_name', '')} 第 {chunk.get('page_num', '?')} 页）"
        piece = f"{header}\n{content}"
        if used + len(piece) > budget and lines:
            break
        lines.append(piece)
        used += len(piece)
        citations.append(
            Citation(
                index=idx,
                point_id=str(chunk.get("point_id", "")),
                doc_id=str(chunk.get("doc_id", "")),
                file_name=str(chunk.get("file_name", "")),
                page_num=int(chunk.get("page_num", 0)),
                version=int(chunk.get("version", 0)),
                chunk_type=str(chunk.get("chunk_type", "text")),
                snippet=content[:120],
                table_html=chunk.get("table_html"),
                image_path=chunk.get("image_path"),
                rerank_score=float(chunk.get("rerank_score") or 0.0),
            )
        )
    return "\n\n".join(lines), citations


async def generate_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    cfg = get_config()
    session_id = str(state.get("session_id") or "")

    # ---- 无上下文：两种走法到此（闲聊 / 检索失败），话术分流 ----
    context, citations = _context_block(
        state, int(cfg.get_path("retrieval.context_max_tokens", 3000))
    )
    if not context:
        if state.get("query_intent") in ("chat", "unclear"):
            answer = "你好！我是 PPT 文档问答助手，可以就你上传的文档内容回答问题。"
        else:
            answer = "文档中未提及相关内容。你可以尝试换一种问法，或确认文档已上传并解析完成。"
        return {
            "draft_answer": answer,
            "citations": [],
            "stream_chunks": 0,
            "generate_elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "trace": trace_event("generate", "end", {"no_context": True}, t0),
        }

    # ---- 正常生成：流式消费并转发 token ----
    user_prompt = _USER_TEMPLATE.format(context=context, query=state.get("normalized_query"))
    mc = get_model_client()
    pieces: list[str] = []
    flush = int(cfg.get_path("agent.sse.token_flush_size", 2))
    pending = 0
    stream_error: str | None = None

    try:
        async for chunk in mc.generate_stream(
            user_prompt,
            system_prompt=_SYSTEM,
            max_new_tokens=int(cfg.get_path("models.generate.max_new_tokens", 1024)),
        ):
            if chunk.get("event") == "error":
                stream_error = str(chunk.get("text") or "生成失败")
                break
            piece = chunk.get("text") or ""
            if chunk.get("event") != "delta" or not piece:
                continue
            pieces.append(piece)
            pending += 1
            # ★ 按 token_flush_size 攒批推 —— 每字符一条会把 SSE 帧开销放大 10 倍
            if pending >= flush:
                publish(session_id, "token", {"text": "".join(pieces[-pending:])})
                pending = 0
    except Exception as exc:  # noqa: BLE001 —— 生成失败降级为错误话术
        stream_error = f"生成中断: {exc}"

    # 尾批未满 flush 的 token 也要推出去
    if pending > 0:
        publish(session_id, "token", {"text": "".join(pieces[-pending:])})

    if stream_error:
        publish(session_id, "error", {"message": stream_error})

    draft = "".join(pieces).strip()
    return {
        "draft_answer": draft,
        "citations": citations,
        "stream_chunks": len(pieces),
        "generate_elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "error": stream_error,
        "trace": trace_event(
            "generate", "error" if stream_error else "end",
            {"tokens": len(pieces), "citations": len(citations)},
            t0,
        ),
    }
