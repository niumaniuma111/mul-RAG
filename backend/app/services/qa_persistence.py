"""
问答落库：`chat_messages` + `qa_traces` 的唯一写入点。

为什么要单独一个模块（坑 31）
--------
    LangGraph 的 `interrupt()` 会让图**停在 hitl 节点**，`finalize` 根本不执行。
    原来的落库逻辑整个写在 finalize 里 —— 于是"HITL 挂起"的会话
    **在 qa_traces 里完全没有痕迹**（A21 两次复现缺席）。

    但挂起恰恰是最该留痕的时刻：用户没点、超时了、前端断了 —— 这些都是
    运维最需要知道的"卡在哪"。所以拆出这个模块：
      - `record_hitl_pending()`  在 interrupt 那一刻写一条 pending 痕迹
      - `record_final()`         在 finalize 写完整行（同 session 覆盖式更新）

    **不引入新表**：qa_traces 一行一 session，挂起时先插，收尾时 UPDATE。
    这样"挂起未恢复"的行会以 `reflect_verdict = 'PENDING'` 留在库里，
    一眼能扫出来（`reflect_reason` 里带当时的 max_score 与阈值）。

session_id 必须是 UUID（表约束）
--------
    非 UUID 的探针 thread_id 直接跳过落库 —— 与旧行为一致，不报错。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.core.config import get_config
from app.core.logging import get_logger
from app.db.session import get_session_factory

logger = get_logger(__name__)

# ⚠️ reflect_verdict 是 varchar(8) —— 标记必须 ≤8 字符。
#    这里踩过：起初用了 "PENDING_HITL"（12 字符）→ asyncpg
#    StringDataRightTruncationError，整条 INSERT 被回滚，挂起痕迹依然没落上。
#    用 "PENDING"（7）刚好；细节信息全部塞 reflect_reason（TEXT，无长度限制）。
PENDING_MARK = "PENDING"


def _valid_uuid(sid: str) -> bool:
    try:
        uuid.UUID(sid)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _threshold() -> float | None:
    """
    当前生效的 score_threshold —— 落进 qa_traces.retrieval_scores 供事后核对。

    为什么值得单独存：HITL 判据是「max_score < threshold」，而 threshold 在
    config.yaml 里可改。事后看到一条 max_score=0.56 却没触发 HITL 时，必须知道
    **当时**的阈值是多少才能判断是 bug 还是配置变更（S5 标定把 A27 从 0.519
    抬到 0.560，正好卡在 0.55 上沿 —— 这类"差一点"的案例最需要留痕）。
    """
    try:
        return float(get_config().get_path("retrieval.score_threshold", 0.55))
    except Exception:  # noqa: BLE001
        return None


def build_retrieval_scores(state: dict[str, Any]) -> dict[str, Any]:
    """
    从 state 抽取 rerank 分数分布 + 实际下发的 filter。

    只存分数与定位信息，不存正文（正文在 chat_messages.citations 里），
    控制 JSONB 体积。`filter` 必须带上 —— 排查"点名历史版本却拿到当前版本
    数字"这类问题（坑 34）时，version_override 是否真的生效，只有这里能事后验证。
    """
    reranked = list(state.get("reranked_hits") or [])
    return {
        "max_score": round(float(state.get("max_relevance_score") or 0.0), 6),
        "threshold": _threshold(),
        "rounds": int(state.get("retrieval_round", 0) or 0),
        "n_fused": len(state.get("fused_hits") or []),
        "n_reranked": len(reranked),
        "rerank": [
            {
                "point_id": str(h.get("point_id", "")),
                "doc_id": str(h.get("doc_id", "")),
                "version": int(h.get("version", 0) or 0),
                "page_num": int(h.get("page_num", 0) or 0),
                "chunk_type": str(h.get("chunk_type", "text")),
                "rerank_score": round(float(h.get("rerank_score") or 0.0), 6),
            }
            for h in reranked
        ],
        "filter": state.get("retrieval_filter") or {},
    }


def _fit(value: str | None, limit: int) -> str | None:
    """
    按列宽截断 —— 防御 varchar 溢出。

    为什么需要：asyncpg 在值超长时抛 StringDataRightTruncationError 并**回滚整条
    语句**，而落库是 best-effort（外层只 warn）。结果就是"日志里有一条 warning，
    但你想要的那条观测记录悄悄没了" —— 正是坑 31 的形态。与其依赖每个调用点
    记准列宽，不如在这里统一收口。
    """
    if value is None:
        return None
    return value[:limit] if len(value) > limit else value


async def record_hitl_pending(state: dict[str, Any]) -> None:
    """
    在进 interrupt 之前写一条"挂起待恢复"的 qa_traces 行。

    幂等：如果该 session 已有行（重复进入、或恢复后发现已写过），ON CONFLICT
    不重复插。用 session_id 作冲突键需要唯一索引；这里用「先查后插」的土办法
    更稳（避免依赖线上表结构是否有 unique 约束）。
    落库失败只 warn，不打断问答。
    """
    sid = str(state.get("session_id") or "")
    if not _valid_uuid(sid):
        return

    try:
        from sqlalchemy import text

        scores = build_retrieval_scores(state)
        session_factory = get_session_factory()
        async with session_factory() as session:
            exists = await session.execute(
                text("SELECT 1 FROM qa_traces WHERE session_id = :sid LIMIT 1"),
                {"sid": sid},
            )
            if exists.scalar():
                return   # 已有行（已恢复过 / 已 finalize），不覆盖
            await session.execute(
                text(
                    """
                    INSERT INTO qa_traces
                        (session_id, user_id, normalized_query, retrieval_keywords,
                         retrieval_scores, hitl_triggered, reflect_rounds,
                         reflect_verdict, reflect_reason)
                    VALUES
                        (:sid, :uid, :nq, CAST(:kw AS jsonb),
                         CAST(:rsc AS jsonb), true, 0,
                         :mark, :reason)
                    """
                ),
                {
                    "sid": sid,
                    "uid": int(state.get("user_id") or 0),
                    "nq": str(state.get("normalized_query") or ""),
                    "kw": json.dumps(state.get("retrieval_keywords") or [], ensure_ascii=False),
                    "rsc": json.dumps(scores, ensure_ascii=False),
                    "mark": _fit(PENDING_MARK, 8),
                    "reason": f"HITL 挂起待恢复 max_score={scores['max_score']} threshold={scores['threshold']}",
                },
            )
            await session.commit()
            logger.info("HITL 挂起痕迹已落库 session=%s", sid)
    except Exception as exc:  # noqa: BLE001 —— 观测字段，失败不影响问答
        logger.warning("HITL 落库失败（不影响挂起）: %s", exc)


async def record_final(
    state: dict[str, Any],
    *,
    answer: str,
    citations: list[dict[str, Any]],
    finish_reason: str,
) -> None:
    """
    finalize 落库：chat_messages 两行（user/assistant）+ qa_traces 一行。

    qa_traces 用 UPDATE-or-INSERT：HITL 路径可能已经写了 PENDING 行，
    这里要把它补成完整行（覆盖 PENDING 标记），而不是插第二行。
    """
    sid = str(state.get("session_id") or "")
    if not _valid_uuid(sid):
        return

    try:
        from sqlalchemy import text

        reflect_info = {
            "verdict": state.get("reflect_verdict"),
            "reason": state.get("reflect_reason"),
            "rounds": state.get("reflect_round", 0),
            "rule_violations": state.get("rule_violations") or [],
        }
        started = float(state.get("started_at") or 0.0)
        total_ms = int((time.perf_counter() - started) * 1000) if started else None
        scores = build_retrieval_scores(state)

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO chat_messages
                        (session_id, user_id, role, content, citations, reflect_info)
                    VALUES
                        (:sid, :uid, 'user', :q, NULL::jsonb, NULL::jsonb),
                        (:sid, :uid, 'assistant', :a, CAST(:cites AS jsonb),
                         CAST(:rinfo AS jsonb))
                    """
                ),
                {
                    "sid": sid,
                    "uid": int(state.get("user_id") or 0),
                    "q": str(state.get("raw_query") or ""),
                    "a": answer,
                    "cites": json.dumps(citations, ensure_ascii=False),
                    "rinfo": json.dumps(reflect_info, ensure_ascii=False),
                },
            )

            # 一条 SQL 插了两行（user/assistant），按 role 过滤取最新一条
            msg_row = await session.execute(
                text(
                    """
                    SELECT id FROM chat_messages
                     WHERE session_id = :sid AND role = 'assistant'
                     ORDER BY created_at DESC
                     LIMIT 1
                    """
                ),
                {"sid": sid},
            )
            message_id = msg_row.scalar()

            params = {
                "sid": sid,
                "uid": int(state.get("user_id") or 0),
                "mid": message_id,
                "nq": str(state.get("normalized_query") or ""),
                "kw": json.dumps(state.get("retrieval_keywords") or [], ensure_ascii=False),
                "rsc": json.dumps(scores, ensure_ascii=False),
                "hitl": bool(state.get("hitl_triggered")),
                "hchoice": _fit(state.get("hitl_choice"), 32),
                "rounds": int(state.get("reflect_round", 0) or 0),
                "verdict": _fit(state.get("reflect_verdict") or "", 8),
                "reason": state.get("reflect_reason") or "",
                "latency": total_ms,
            }

            updated = await session.execute(
                text(
                    """
                    UPDATE qa_traces SET
                        message_id = :mid, normalized_query = :nq,
                        retrieval_keywords = CAST(:kw AS jsonb),
                        retrieval_scores = CAST(:rsc AS jsonb),
                        hitl_triggered = :hitl, hitl_choice = :hchoice,
                        reflect_rounds = :rounds, reflect_verdict = :verdict,
                        reflect_reason = :reason, total_latency_ms = :latency
                     WHERE session_id = :sid
                    """
                ),
                params,
            )

            if not updated.rowcount:
                await session.execute(
                    text(
                        """
                        INSERT INTO qa_traces
                            (session_id, user_id, message_id, normalized_query,
                             retrieval_keywords, retrieval_scores, hitl_triggered,
                             hitl_choice, reflect_rounds, reflect_verdict,
                             reflect_reason, total_latency_ms)
                        VALUES
                            (:sid, :uid, :mid, :nq, CAST(:kw AS jsonb),
                             CAST(:rsc AS jsonb), :hitl, :hchoice, :rounds,
                             :verdict, :reason, :latency)
                        """
                    ),
                    params,
                )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("问答落库失败（不影响回答）: %s", exc)
