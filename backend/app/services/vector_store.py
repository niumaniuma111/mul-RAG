"""
Qdrant 向量库读写封装。

写入侧的三条硬约定
-----------------
1. **point id 用 uuid5 确定性生成**：`uuid5(NS, f"{doc_id}:{version}:{chunk_index}")`
   同一份内容永远落到同一个 id → 任务重跑天然幂等（覆盖而非新增）。
   配套约定：写入前先 `delete_by_version`，避免本次解析块数变少时留下孤儿点。

2. **is_active 是"投影"不是事实来源**（DESIGN D4/§7.1）
   PG 的 `documents.active_version` 才是唯一事实。Qdrant 的 is_active 用于
   让过滤走索引、少一次 join。二者不一致时以 PG 为准，靠
   `rebuild_consistency()` 幂等修复。

3. **稀疏向量的词表用"哈希技巧"，不落 vocab 文件**
   config 里有个 `paths.vocab_file`，但增量入库场景下**无法**先建全局词表：
   第一份文档入库时词表还不存在。所以这里把 token 哈希成 uint32 作为索引
   （Hashing Trick），全局稳定、无需持久化。
   ⚠️ 绝不能用 Python 内置 `hash()` —— 它受 PYTHONHASHSEED 影响，
      **跨进程不一致**，重建索引后 BM25 会完全失效。这里用 blake2b 保证稳定。

对应设计文档：DESIGN.md §5.3 / §9.3 / §7.1
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
)

from app.core.config import get_config, get_env
from app.core.logging import get_logger
from app.services.chunker import Chunk
from app.services.text_utils import tokenize

logger = get_logger(__name__)

# uuid5 的命名空间：本项目固定值，改它等于让所有历史点失联
_NAMESPACE = uuid.UUID("6f1b6b1e-9c3a-4a7e-9f0e-2f4b8c1d5e70")

# 稀疏向量索引空间：2^32。20 万词的生日碰撞期望 ≈ 200000²/(2×2^32) ≈ 5 次，
# 可忽略；换来的是无需维护全局词表。
_HASH_SPACE = 1 << 32


def point_id(doc_id: str | uuid.UUID, version: int, chunk_index: int) -> str:
    """确定性 point id。任何一处改动都会让重跑产生重复点，勿改。"""
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{version}:{chunk_index}"))


def token_index(token: str) -> int:
    """
    token → uint32 索引（Hashing Trick）。

    ⚠️ 必须用密码学/校验和哈希而不是内置 hash()：
       内置 hash() 对 str 会加随机盐（PYTHONHASHSEED），
       api 与 worker 两个进程算出的索引不同 → 检索永远命中不了。
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _HASH_SPACE


def build_sparse_vector(text: str) -> SparseVector:
    """
    文本 → Qdrant 稀疏向量（BM25 的 TF 部分）。

    IDF 由 Qdrant 的 `Modifier.IDF` 在检索时按集合统计自动算，
    所以这里只提供词频。用**次线性 TF（1+ln tf）**抑制高频词重复出现的影响，
    这是 BM25 的标准做法之一（k1 饱和的简化形式）。

    config 里的 `ingest.bm25.k1/b` 因此只是参考值 —— Qdrant 不接受它们，
    想精确控制只能改用客户端融合（retrieval.fusion_mode=client）。
    """
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counter = Counter(token_index(tok) for tok in tokens)
    indices = sorted(counter)
    values = [1.0 + _ln(counter[idx]) for idx in indices]
    return SparseVector(indices=indices, values=values)


def _ln(value: int) -> float:
    """自然对数。value>=1，所以 ln 不会出现负无穷。"""
    import math

    return math.log(value) if value > 1 else 0.0


class VectorStore:
    """Qdrant 读写封装。collection 名与集合维度来自 config。"""

    def __init__(self) -> None:
        cfg = get_config()
        env = get_env()
        self.collection: str = str(cfg.get_path("qdrant.collection", "ppt_chunks"))
        self.timeout_sec: int = int(cfg.get_path("qdrant.timeout_sec", 60))

        raw = env.qdrant_url
        host = raw.split("://", 1)[-1].split(":")[0]
        tail = raw.rsplit(":", 1)[-1]
        port = int(tail) if tail.isdigit() else 6333

        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=int(cfg.get_path("qdrant.grpc_port", env.qdrant_grpc_port)),
            prefer_grpc=bool(cfg.get_path("qdrant.prefer_grpc", True)),
            timeout=self.timeout_sec,
        )

    # ---------------- 写入 ----------------
    def upsert_chunks(
        self,
        *,
        doc_id: str | uuid.UUID,
        user_id: int,
        version: int,
        file_name: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        is_active: bool = True,
    ) -> int:
        """
        写入一批 chunk。返回写入点数。

        ⚠️ 调用方必须保证 `chunks` 与 `vectors` 等长且顺序一致。
           错位会导致"内容与向量不匹配"——检索结果看起来对但引用是错的，
           是最难查的一类 bug，所以这里做显式校验。
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0

        doc_key = str(doc_id)
        points: list[PointStruct] = []
        for chunk, dense in zip(chunks, vectors, strict=True):
            payload: dict[str, Any] = {
                # ★ 前四个是检索时的强制过滤字段，必须在 init_qdrant.py 里建索引
                "user_id": user_id,
                "doc_id": doc_key,
                "is_active": is_active,
                "version": version,
                "file_name": file_name,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "page_num": chunk.page_num,
                "content": chunk.content,
            }
            # table_html / image_path 只在有值时写入 —— 避免无意义地撑大 payload
            if chunk.table_html:
                payload["table_html"] = chunk.table_html
            if chunk.image_path:
                payload["image_path"] = chunk.image_path

            points.append(
                PointStruct(
                    id=point_id(doc_key, version, chunk.chunk_index),
                    vector={
                        "dense": dense,
                        "bm25": build_sparse_vector(chunk.content),
                    },
                    payload=payload,
                )
            )

        self._upsert_with_retry(points)
        logger.info(
            "写入 Qdrant: doc=%s version=%s points=%s", doc_key, version, len(points)
        )
        return len(points)

    def _upsert_with_retry(self, points: list[PointStruct]) -> None:
        """分批 + 重试。Qdrant 在大批量写入时偶发超时，重试成本远低于重新解析。"""
        cfg = get_config()
        batch = int(cfg.get_path("ingest.batch.upsert_batch", 128))
        retries = int(cfg.get_path("qdrant.upsert_retries", 3))

        for start in range(0, len(points), batch):
            window = points[start : start + batch]
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    self.client.upsert(
                        collection_name=self.collection,
                        points=window,
                        wait=True,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 —— qdrant 客户端异常类型较杂
                    last_exc = exc
                    logger.warning("Qdrant upsert 第 %s/%s 次失败: %s", attempt, retries, exc)
            else:
                raise RuntimeError(f"Qdrant upsert 连续 {retries} 次失败: {last_exc}")

    def delete_by_version(self, doc_id: str | uuid.UUID, version: int) -> int:
        """
        删除某文档某版本的全部点（重解析前的幂等清理），返回删除前的点数。

        用 `delete` + filter 而不是 list_ids + PointIdsList：后者在点数多时要分页。
        ⚠️ Qdrant 的 delete 不回传删除条数，所以先 count 一次用于日志。
        """
        doc_key = str(doc_id)
        removed = self.count_chunks(doc_key, version)

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_key)),
                    FieldCondition(key="version", match=MatchValue(value=version)),
                ]
            ),
            wait=True,
        )
        logger.info("删除旧版本点: doc=%s version=%s removed=%s", doc_key, version, removed)
        return removed

    def delete_by_doc(self, doc_id: str | uuid.UUID) -> int:
        """
        **物理删除**该文档**全部版本**的点，返回删除前的点数。

        用途：文档硬删除（不可恢复）。与 `set_payload(is_active=False)` 的
        软下线有本质区别 —— 软下线的点仍躺在库里，只要一次 filter 放宽就会
        重新参与召回；硬删除是真正让它们消失。

        ⚠️ Qdrant 的 delete 不回传删除条数，所以先 count 一次（与
           `delete_by_version` 同一套口径：count → delete → 记日志）。
        """
        from qdrant_client.models import CountRequest  # noqa: F401 —— 仅为对齐写法

        doc_key = str(doc_id)
        result = self.client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_key))]
            ),
            exact=True,
        )
        removed = int(getattr(result, "count", 0) or 0)

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_key))]
            ),
            wait=True,
        )
        logger.info("物理删除文档全部点: doc=%s removed=%s", doc_key, removed)
        return removed

    # ---------------- is_active 投影 ----------------
    def set_active_by_version(self, doc_id: str | uuid.UUID, active_version: int) -> int:
        """
        把该文档的 is_active 对齐到 `active_version`，返回目标版本的点数。

        ⚠️ 这是**最终一致**操作：即便失败也不影响检索正确性 ——
           检索服务会用 PG 的 active_version 做二次校验（DESIGN §7.1）。

        两步，顺序不能颠倒：
            1. 该文档**全部**版本 is_active=false
            2. 目标版本 is_active=true
        反过来（先 true 再 false）会把刚点亮的版本又关掉。
        """
        doc_key = str(doc_id)

        self.client.set_payload(
            collection_name=self.collection,
            payload={"is_active": False},
            points=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_key))]
            ),
            wait=True,
        )
        count = self._set_active_for_version(doc_key, active_version)
        logger.info("is_active 投影已对齐: doc=%s active_version=%s", doc_key, active_version)
        return count

    def _set_active_for_version(self, doc_key: str, version: int) -> int:
        """只把目标版本翻成 true，并返回受影响的点数。"""
        from qdrant_client.models import FilterSelector

        selector = FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_key)),
                    FieldCondition(key="version", match=MatchValue(value=version)),
                ]
            )
        )
        # set_payload 不返回条数，所以先 count 一次用于日志/返回
        count = self.count_chunks(doc_key, version)
        self.client.set_payload(
            collection_name=self.collection,
            payload={"is_active": True},
            points=selector,
            wait=True,
        )
        return count

    # ---------------- 查询（预览 / 校验） ----------------
    def count_chunks(self, doc_id: str | uuid.UUID, version: int) -> int:
        from qdrant_client.models import CountRequest

        result = self.client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id))),
                    FieldCondition(key="version", match=MatchValue(value=version)),
                ]
            ),
            exact=True,
        )
        return int(getattr(result, "count", 0) or 0)

    def scroll_chunks(
        self,
        doc_id: str | uuid.UUID,
        version: int,
        *,
        offset: Any = None,
        limit: int = 20,
    ) -> tuple[list[dict[str, Any]], Any]:
        """
        按 chunk_index 顺序分页拉取（供前端"版本内容预览"）。

        用 scroll 而不是 search：这里要的是**全量浏览**而非相似度排序。
        Qdrant 的 scroll 不保证有序，所以拉回后在内存里按 chunk_index 排 ——
        单页最多 200 条，排序成本可忽略。
        """
        records, next_offset = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id))),
                    FieldCondition(key="version", match=MatchValue(value=version)),
                ]
            ),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads = [dict(r.payload or {}) for r in records]
        payloads.sort(key=lambda p: int(p.get("chunk_index", 0)))
        return payloads, next_offset

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """模块级单例（QdrantClient 内部持有连接池，不要反复 new）。"""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


__all__ = [
    "VectorStore",
    "build_sparse_vector",
    "get_vector_store",
    "point_id",
    "token_index",
]
