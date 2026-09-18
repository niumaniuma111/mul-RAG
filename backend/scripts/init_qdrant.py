"""
初始化 Qdrant collection 与 payload index。

设计要点（对应 DESIGN.md §5.3）
------------------------------

**双命名向量**：`dense`(1024, Cosine) + `bm25`(sparse, Modifier.IDF)

    Qdrant 的稀疏向量支持 IDF modifier —— 这才是能做出"真 BM25 打分"的关键。
    光靠 `text` 类型的 payload index 只能做过滤，不能参与打分，
    这也是为什么不用「文本字段 + 全文索引」那套方案。

**payload index 必须在检索前建好**

    Qdrant 的执行顺序是「先过滤、后召回」。`user_id` 和 `is_active` 是每次检索
    都带的强制过滤条件（config 里 require_user_id / require_is_active 都为 true），
    没有索引就会退化成全集合扫描 —— 检索延迟直接从毫秒级掉到秒级。

**point id 用 uuid5 确定性生成**：`uuid5(NAMESPACE, f"{doc_id}:{version}:{chunk_index}")`

    好处是任务重跑天然幂等：同一份内容永远落到同一个 id，重复写入是覆盖而非新增。
    配套约定：重解析前先 `delete` 同 doc_id + version 的点，再写入。

用法
----
    # 推荐在容器内跑（依赖已装好）
    docker compose exec api python scripts/init_qdrant.py

    # 重建（!! 会删除该 collection 下的全部数据）
    docker compose exec api python scripts/init_qdrant.py --recreate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# 让 `python scripts/xxx.py` 能 import 到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    SparseVectorParams,
    Modifier,
    VectorParams,
)

from app.core.config import get_config, get_env  # noqa: E402

# ★ 7 个 payload index，对应 DESIGN.md §5.3 的清单。
#   前三个是检索时每次都要用的过滤条件，缺一个都会全表扫。
PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    ("user_id", PayloadSchemaType.INTEGER),      # ★ 强制过滤：租户隔离
    ("doc_id", PayloadSchemaType.KEYWORD),       # ★ 指定文档问答
    ("is_active", PayloadSchemaType.BOOL),       # ★ 强制过滤：版本生效标记
    ("file_name", PayloadSchemaType.KEYWORD),    # 按文件名筛
    ("chunk_type", PayloadSchemaType.KEYWORD),   # text / table / image_caption
    ("version", PayloadSchemaType.INTEGER),      # 历史版本问答
    ("page_num", PayloadSchemaType.INTEGER),     # 溯源定位
]


def build_client() -> QdrantClient:
    env = get_env()
    cfg = get_config()
    prefer_grpc = bool(cfg.get_path("qdrant.prefer_grpc", True))
    grpc_port = int(cfg.get_path("qdrant.grpc_port", 6334))
    timeout = int(cfg.get_path("qdrant.timeout_sec", 60))

    # QdrantClient 接的是 host:port，而 .env 里给的是完整 URL，这里拆开
    raw = env.qdrant_url
    host = raw.split("://", 1)[-1].split(":")[0]
    port = int(raw.rsplit(":", 1)[-1]) if raw.rsplit(":", 1)[-1].isdigit() else 6333

    return QdrantClient(
        host=host,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
    )


def create_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            ),
        },
        sparse_vectors_config={
            # ★ Modifier.IDF 让 Qdrant 按词频-逆文档频率打分，即真 BM25
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
        # ★ 超过 20MB 的段自动 memmap 到磁盘 —— 16GB 内存机器的关键设定
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        # ★ payload 落盘而非驻留内存
        on_disk_payload=True,
    )
    print(f"  ✓ collection 已创建: {name} (dense={dense_dim}d, sparse=bm25/IDF)")


def create_indexes(client: QdrantClient, name: str) -> None:
    for field, schema in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )
        print(f"  ✓ payload index: {field} ({schema})")


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 Qdrant collection")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除已存在的 collection 后重建（!! 会清空该 collection 的全部数据）",
    )
    args = parser.parse_args()

    cfg = get_config()
    env = get_env()
    collection = str(cfg.get_path("qdrant.collection", "ppt_chunks"))
    dense_dim = int(cfg.get_path("models.embedding.dim", 1024))

    print(f"Qdrant: {env.qdrant_url}")
    print(f"collection: {collection} | dense 维度: {dense_dim}")
    print("-" * 60)

    client = build_client()

    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 连不上 Qdrant ({env.qdrant_url}): {exc}")
        return 1

    if collection in existing:
        if not args.recreate:
            print(f"collection '{collection}' 已存在，跳过创建（要重建请加 --recreate）")
            # 索引可能是后补的，仍然尝试建一遍（幂等操作）
            create_indexes(client, collection)
            return 0
        print(f"⚠️  --recreate：删除已存在的 collection '{collection}' ...")
        client.delete_collection(collection)

    create_collection(client, collection, dense_dim)
    create_indexes(client, collection)

    info = client.get_collection(collection)
    print("-" * 60)
    print(f"完成。points_count={info.points_count} status={info.status}")

    # 自检：dense 维度必须与 config 一致，否则写入时会报维度不匹配
    vectors: Any = info.config.params.vectors
    actual_dim = getattr(vectors.get("dense"), "size", None) if isinstance(vectors, dict) else None
    if actual_dim is not None and int(actual_dim) != dense_dim:
        print(f"✗ 维度不一致：collection={actual_dim} config={dense_dim}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
