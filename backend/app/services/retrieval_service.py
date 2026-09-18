"""
在线混合检索服务（离线入库之外的另一半检索链路）。

链路（对应 DESIGN.md §9.1，v1.1 融合下沉到 Qdrant 服务端）
--------
    query ─┬─ dense 路：model-service /embed → 1024 维向量
           └─ sparse 路：jieba 分词 → Hashing Trick → SparseVector（复用
              vector_store.build_sparse_vector，保证与入库侧**逐字节一致**）

    ↓ 一次 query_points（服务端完成）
    prefetch[dense top6] + prefetch[bm25 top6] → Fusion.RRF(k=60) → top8

    ↓ model-service /rerank（BGE-reranker-large 交叉编码）
    保留 top3，得到 sigmoid 相关性分 → HITL 判据

三条纪律（都是踩过坑或 DESIGN 明文要求的）
--------
1. **RRF 融合分绝不与 score_threshold 比**（坑 1）：服务端 RRF 分量纲只有
   ~0.016-0.033，阈值 0.35 是给 rerank sigmoid 分用的。两套分数在这里
   就地分离命名（rrf_score vs rerank_score），下游不再有机会混淆。
2. **sparse 向量必须复用入库侧的同一构建函数**：Hashing Trick 的索引空间
   与 TF 公式任何一侧改动，都会让"写入的索引"与"查询的向量"失配，
   BM25 路静默失效（不报错，只是召回变差 —— 最难发现的一类退化）。
3. **filter 必带 user_id + is_active**：多用户隔离靠 Qdrant payload index
   硬过滤，不走应用层二次筛查（性能与正确性双重理由，§9.2）。
   ⚠️ 唯一例外是 `version_override`：点名历史版本时该版本在 Qdrant 里
   已是 `is_active=false`，必须绕过；但这只对该 **被点名的 doc** 生效，
   且是「锁定这一版」而不是「这一版 ∪ 当前版」（坑 34）。

对应设计文档：DESIGN.md §9.1-§9.4
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as M

from app.core.config import get_config, get_env
from app.core.logging import get_logger
from app.services.model_client import get_model_client
from app.services.text_utils import tokenize
from app.services.vector_store import build_sparse_vector, token_index

logger = get_logger(__name__)


class RetrievalService:
    """混合检索：双路召回 → 服务端 RRF → rerank 精排。全异步。"""

    def __init__(self) -> None:
        cfg = get_config()
        env = get_env()
        self.collection: str = str(cfg.get_path("qdrant.collection", "ppt_chunks"))

        raw = env.qdrant_url
        host = raw.split("://", 1)[-1].split(":")[0]
        tail = raw.rsplit(":", 1)[-1]
        port = int(tail) if tail.isdigit() else 6333

        # ⚠️ LangGraph 节点全是 async，这里必须用 AsyncQdrantClient；
        #    复用 VectorStore 的同步 client 会把整个事件循环堵死。
        self.client = AsyncQdrantClient(
            host=host,
            port=port,
            grpc_port=int(cfg.get_path("qdrant.grpc_port", env.qdrant_grpc_port)),
            prefer_grpc=bool(cfg.get_path("qdrant.prefer_grpc", True)),
            timeout=int(cfg.get_path("qdrant.timeout_sec", 60)),
        )

    # ---------------- filter（§9.2） ----------------
    def build_filter(
        self,
        user_id: int,
        doc_ids: list[str] | None = None,
        version_override: dict[str, int] | None = None,
    ) -> M.Filter:
        """
        构造检索过滤条件。规则（§9.2）：

            默认            → user_id + is_active=true (+ doc_ids 限定范围)
            版本覆盖        → user_id + ( (doc=A AND version=1) OR (doc=B AND version=2) ... )
                              ★ 只召回**被点名的那一版**，是「锁定」不是「扩召回」。

        ⚠️ 坑 34（历史版本静默回答当前版本数字）
        ----------------------------------------
        早期实现把 `is_active=true` 塞进了 should 的最后一个分支，于是语义
        从「点名版本」退化成了「点名版本 ∪ 当前激活版本」—— 两版同时在池，
        rerank 分常常只差 0.0001，历史版本的问法会静默拿到最新版本的数字
        （实测 A22: 412 被答成 468、87.6% 被答成 89.2%、11300 被答成 12100）。

        正确语义是**覆盖**：被 override 点名的 doc 用 (doc_id, version) 精确
        锁定，不再要求 is_active（该版本在 Qdrant 里已被翻成 false，本就要
        绕过）；**未被点名**的 doc 继续走 is_active=true 默认规则。

        实现方式是把每个 doc 变成 must 里的一个 should 分支：
            (doc=A AND version=n) OR (doc=其他 AND is_active=true)
        must 里的其他条件（user_id、doc_ids 白名单）对两者**同时**生效，
        所以范围限定与多用户隔离都不会被 override 绕过。
        """
        must: list[Any] = [
            M.FieldCondition(key="user_id", match=M.MatchValue(value=user_id)),
        ]

        # doc_ids 是硬白名单：无论是否有 version_override 都必须生效。
        if doc_ids:
            must.append(M.FieldCondition(key="doc_id", match=M.MatchAny(any=[str(d) for d in doc_ids])))

        if version_override:
            allow = {str(d) for d in (doc_ids or [])}
            locked: set[str] = set()

            per_doc: list[M.Filter] = []
            for doc_id, ver in version_override.items():
                key = str(doc_id)
                # override 点名了 doc_ids 白名单之外的文档 → 该分支与白名单
                # 互斥，放进 should 只会白白扩大召回面，直接丢弃（不报错，
                # 调用方传了不一致的参数，logger 里留痕即可）。
                if allow and key not in allow:
                    logger.warning(
                        "version_override 中的 doc_id=%s 不在 doc_ids 白名单内，已忽略", key,
                    )
                    continue
                per_doc.append(
                    M.Filter(
                        must=[
                            M.FieldCondition(key="doc_id", match=M.MatchValue(value=key)),
                            M.FieldCondition(key="version", match=M.MatchValue(value=int(ver))),
                        ]
                    )
                )
                locked.add(key)

            # ---- 未被点名的文档：继续按 is_active=true 默认规则召回 ----
            # 两个场景需要这个分支：
            #   a) doc_ids 为空（不限范围）→ 除 locked 之外的所有 doc 都算未点名；
            #   b) doc_ids 非空但只点名了其中一部分 → 白名单里的剩余 doc 算未点名。
            # 场景 b 靠 must 里的 doc_id 白名单兜住范围，这里只要把 locked
            # 排除掉即可，无需再显式枚举白名单。
            uncovered: set[str] = set()
            if not allow:
                uncovered = {"*"}                      # 不限范围：除 locked 外全部
            elif allow - locked:
                uncovered = allow - locked             # 白名单内未被点名的部分

            if per_doc and uncovered:
                others = M.Filter(
                    must=[M.FieldCondition(key="is_active", match=M.MatchValue(value=True))],
                    must_not=[
                        M.FieldCondition(
                            key="doc_id", match=M.MatchAny(any=sorted(locked))
                        )
                    ],
                )
                return M.Filter(must=must, should=per_doc + [others])

            # 全部 doc 都被点名锁定（或白名单内无一未点名）→ 纯精确命中，
            # 绝不引入 is_active 分支（这正是坑 34 的病灶）。
            return M.Filter(must=must, should=per_doc)

        must.append(M.FieldCondition(key="is_active", match=M.MatchValue(value=True)))
        return M.Filter(must=must)

    # ---------------- 单路调试（§9.3 debug 开关） ----------------
    async def debug_single_route(
        self, dense_vec: list[float], flt: M.Filter
    ) -> dict[str, list[dict[str, Any]]]:
        """额外发 2 次单路查询拿各自 score 与 rank，排障用（生产关闭）。"""
        cfg = get_config()
        dense_k = int(cfg.get_path("retrieval.dense_top_k", 6))
        sparse_k = int(cfg.get_path("retrieval.sparse_top_k", 6))
        sparse = build_sparse_vector(_last_query[0]) if _last_query[0] else None

        d = await self.client.query_points(
            self.collection, query=dense_vec, using="dense",
            limit=dense_k, query_filter=flt, with_payload=False,
        )
        s = None
        if sparse is not None:
            s = await self.client.query_points(
                self.collection, query=sparse, using="bm25",
                limit=sparse_k, query_filter=flt, with_payload=False,
            )
        return {
            "dense": [{"id": str(p.id), "score": p.score} for p in d.points],
            "sparse": [{"id": str(p.id), "score": p.score} for p in (s.points if s else [])],
        }

    # ---------------- 主链路 ----------------
    async def search(
        self,
        query: str,
        *,
        user_id: int,
        doc_ids: list[str] | None = None,
        version_override: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """
        完整检索一轮。返回::

            {
              "fused_hits":    [...8 条，含 rrf_score],
              "reranked_hits": [...≤top_n 条，含 rerank_score，降序],
              "context_chunks":[同 reranked_hits（当前 top3 直接做上下文）],
              "max_relevance_score": float,   # ★ HITL 判据
              "filter": {...},                # 实际下发条件（观测用）
              "elapsed_ms": int,
            }

        抛错语义：Qdrant / model-service 的网络异常原样上抛，
        由 LangGraph 节点层统一兜底（trace 里记 error）。
        """
        cfg = get_config()
        t0 = time.perf_counter()

        flt = self.build_filter(user_id, doc_ids, version_override)
        filter_view = {
            "user_id": user_id,
            "doc_ids": doc_ids or [],
            "version_override": version_override or {},
        }

        # ---- 1. dense 路：query 向量化（单条，不需要分批）----
        mc = get_model_client()
        vectors = await mc.embed([query])
        dense_vec = vectors[0]

        # ---- 2. sparse 路：与入库侧同一构建函数（纪律 2）----
        sparse = build_sparse_vector(query)

        # ---- 3. 服务端 RRF 融合（一次往返）----
        resp = await self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                M.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=int(cfg.get_path("retrieval.dense_top_k", 6)),
                    filter=flt,
                ),
                M.Prefetch(
                    query=M.SparseVector(
                        indices=list(sparse.indices), values=[float(v) for v in sparse.values]
                    ),
                    using="bm25",
                    limit=int(cfg.get_path("retrieval.sparse_top_k", 6)),
                    filter=flt,
                ),
            ],
            query=M.FusionQuery(fusion=M.Fusion.RRF),   # RRF_K=60 服务端硬编码
            limit=int(cfg.get_path("retrieval.fusion_top_n", 8)),
            query_filter=flt,
            with_payload=True,
            with_vectors=False,                          # 省内存 / 省带宽
        )

        global _last_query
        _last_query = (query,)                           # 供 debug_single_route 取词

        fused: list[dict[str, Any]] = []
        for rank, p in enumerate(resp.points, start=1):
            payload = dict(p.payload or {})
            fused.append(
                {
                    "point_id": str(p.id),
                    # ★ 键名对齐 vector_store.upsert_chunks 实际写入的 payload
                    "doc_id": str(payload.get("doc_id", "")),
                    "file_name": str(payload.get("file_name", "")),
                    "page_num": int(payload.get("page_num", 0)),
                    "version": int(payload.get("version", 0)),
                    "chunk_type": str(payload.get("chunk_type", "text")),
                    "content": str(payload.get("content", "")),
                    "table_html": payload.get("table_html"),
                    "image_path": payload.get("image_path"),
                    "rrf_score": float(p.score),
                    "dense_score": None,   # ⚠️ 服务端融合后单路分不可得
                    "sparse_score": None,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "_fusion_rank": rank,  # 内部用：rerank 打平时的稳定排序
                }
            )

        # ---- 4. rerank 精排（若融合为空则跳过，省一次模型调用）----
        rerank_top_n = int(cfg.get_path("retrieval.rerank_top_n", 3))
        reranked: list[dict[str, Any]] = []
        if fused:
            # 交叉编码打分对**全部** top8 做 —— 只让模型挑出真正相关的 top3，
            # 而不是"在 8 条里任意取 3 条"。rerank 一次 ~30ms，全量打分不贵。
            results = await mc.rerank(
                query, [h["content"] for h in fused], top_n=len(fused)
            )
            for item in results:
                idx = int(item.get("index", -1))
                if idx < 0 or idx >= len(fused):
                    continue
                hit = dict(fused[idx])
                hit["rerank_score"] = float(item.get("score", 0.0))
                reranked.append(hit)
            reranked.sort(key=lambda h: h["rerank_score"], reverse=True)
            reranked = reranked[:rerank_top_n]
            for rank, hit in enumerate(reranked, start=1):
                hit["_rerank_rank"] = rank

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        max_score = max((h["rerank_score"] for h in reranked), default=0.0)

        logger.info(
            "混合检索完成: user=%s fused=%s reranked=%s max_score=%.3f elapsed=%sms",
            user_id, len(fused), len(reranked), max_score, elapsed_ms,
        )

        return {
            "fused_hits": fused,
            "reranked_hits": reranked,
            "context_chunks": reranked,
            "max_relevance_score": max_score,
            "filter": filter_view,
            "elapsed_ms": elapsed_ms,
        }

    async def aclose(self) -> None:
        await self.client.close()


# debug_single_route 需要拿到最近一次的 query 文本（sparse 构建用）。
# 模块级单槽即可 —— 它本来就是排障工具，不追求并发安全。
_last_query: tuple[str] = ("",)

_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """模块级单例（AsyncQdrantClient 持有连接池，不要反复 new）。"""
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service


async def close_retrieval_service() -> None:
    global _service
    if _service is not None:
        await _service.aclose()
        _service = None
