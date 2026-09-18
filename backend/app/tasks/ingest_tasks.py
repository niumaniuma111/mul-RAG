"""
离线入库任务链（DESIGN.md §7.2）。

    Stage 1  MinerU 解析        → parsing     5%   → 35%
    Stage 2  图片 VLM caption   → captioning  35%  → 65%
    Stage 3  Chunk 切分         → chunking    70%
    Stage 4  Embedding + 写 Qdrant → embedding 90%
    Stage 5  收尾               → success     100%

⚠️ 三个容易踩塌的地方（都在这份代码里做了处理）
------------------------------------------------
1. **事件循环与连接池的绑定**
   Celery 用 prefork，任务是同步函数；而 asyncpg / httpx 的连接池是**绑定事件循环**的。
   如果每次任务都 `asyncio.run()`，会不断创建新循环，而复用的模块级单例
   （engine / httpx client）仍指向旧循环 → 随机报
   `attached to a different loop` / `Event loop is closed`。
   所以本模块用**进程级常驻事件循环**（`_run`），循环随 worker 进程存活。

2. **内存不能随块数线性增长**
   worker 的 mem_limit 只有 512MB。若把全部 chunk 的向量攒在内存里再一次性写 Qdrant，
   一份 50 页 PPT（几千块 × 1024 维）就会直接 OOMKilled。
   所以 Stage 4 **边算边写**：每批 embedding 完成立刻 upsert，向量不累积。

3. **进度写入必须用短连接**
   长任务不能把一条 PG 连接握 20 分钟（连接池会被占死，api 侧查询排队）。
   所以每次推进度都用独立的短会话，用完即释放。

对应设计文档：DESIGN.md §7.1 / §7.2 / §7.3
"""

from __future__ import annotations

import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from app.core.config import get_config
from app.core.logging import get_logger
from app.core.redis_lock import GpuBusyError
from app.db.models import Document, FileVersion, IngestTask
from app.db.session import dispose_engine, get_session_factory
from app.services import chunker
from app.services.mineru_client import MineruClient, ParsedDocument
from app.services.model_client import ModelServiceError, get_model_client
from app.services.vector_store import get_vector_store
from app.tasks.celery_app import celery_app
from app.tasks.loop import run_async

logger = get_logger(__name__)


# ============================================================
# 数据库小工具（全部短会话）
# ============================================================
async def _set_task(task_id: str, **fields: Any) -> None:
    """更新任务行。**独立短会话**，避免长任务把连接池占死。"""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            update(IngestTask).where(IngestTask.id == task_id).values(**fields)
        )
        await session.commit()


async def _set_version(version_id: int, **fields: Any) -> None:
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            update(FileVersion).where(FileVersion.id == version_id).values(**fields)
        )
        await session.commit()


async def _load_context(task_id: str, version_id: int) -> dict[str, Any]:
    """
    读出本次任务需要的全部信息（一次查询，之后不再碰 DB 直到推进度）。

    返回 None 表示任务/版本已不存在（例如用户把文档删了）——
    此时任务应安静失败，而不是抛异常刷满日志。
    """
    factory = get_session_factory()
    async with factory() as session:
        task = await session.scalar(select(IngestTask).where(IngestTask.id == task_id))
        version = await session.scalar(
            select(FileVersion).where(FileVersion.id == version_id)
        )
        if task is None or version is None:
            return {}
        doc = await session.scalar(
            select(Document).where(Document.doc_id == task.doc_id)
        )
        return {
            "doc_id": str(task.doc_id),
            "user_id": int(task.user_id),
            "version": int(version.version),
            "file_path": str(version.file_path),
            "file_name": doc.file_name if doc else "unknown",
        }


def _progress_stage(stage: str, progress: int) -> dict[str, Any]:
    return {"current_stage": stage, "progress": progress, "status": "running"}


# ============================================================
# 图片路径解析
# ============================================================
def _resolve_image_path(raw: str, output_dir: str, image_dir: str | None) -> str:
    """
    MinerU 的 `img_path` 通常是相对路径（如 `images/abc.jpg`）。

    解析优先级：绝对路径原样返回 → image_dir 下 → output_dir 下。
    ⚠️ 这一步必须做对，否则 model-service 读不到图片、
       caption 全部失败，而现象只是"图片块内容都是（此图未生成描述）"。
    """
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return str(path)

    candidates: list[Path] = []
    if image_dir:
        candidates.append(Path(image_dir) / raw)
        candidates.append(Path(image_dir) / path.name)
    candidates.append(Path(output_dir) / raw)
    candidates.append(Path(output_dir) / path.name)
    candidates.append(Path(output_dir) / "images" / path.name)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    logger.warning("图片路径无法解析，caption 将跳过: raw=%s", raw)
    return ""


def _public_image_path(abs_path: str, fallback: str, storage_root: str) -> str:
    """
    规范化**写进 chunk payload** 的 image_path。

    ✅ 落盘位置在 storage_root 内 → 原样返回容器内绝对路径
       （``/app/storage/parsed/{doc}/v{n}/images/x.jpg``）。
       前端 CitationCard 剥掉 ``/app/storage`` 前缀 → ``/storage/...``，
       命中 nginx 的 ``/storage/`` 静态代理（alias 到宿主 ``./storage``）。

    ⚠️ 为什么必须有这一步：``chunk.image_path`` 若沿用 MinerU 的**原始相对路径**
       （``images/x.jpg``），前端会请求 ``/images/x.jpg``，而该路径不存在于
       nginx 的 root → 命中 SPA fallback → 返回 **200 + text/html**（index.html）
       → ``<img>`` 解码失败 → 卡片显示「图片不可预览」。
       现象极具误导性（看起来像图片格式不支持），实为路径映射缺失。

    ⚠️ 越界（图片在 storage_root 之外，如 MinerU 临时目录）时**退回 fallback**：
       前端还有一层用 ``doc_id + version`` 兜底拼接的逻辑；把 ``/tmp/...``
       写进 payload 只会让前端请求到一个 200 的 HTML，比没有路径更糟。
    """
    if not abs_path:
        return fallback
    try:
        Path(abs_path).resolve().relative_to(Path(storage_root).resolve())
    except ValueError:
        logger.warning(
            "图片不在 storage_root 内，image_path 退回原值交由前端兜底: %s (root=%s)",
            abs_path,
            storage_root,
        )
        return fallback
    return abs_path


def _page_text_map(parsed: ParsedDocument) -> dict[int, str]:
    """页号 → 该页文本（截断）。用作 caption 的上下文，让描述与本文语义对齐。"""
    mapping: dict[int, list[str]] = {}
    for block in parsed.text_blocks:
        mapping.setdefault(block.page_num, []).append(block.text)
    return {page: "\n".join(texts)[:1500] for page, texts in mapping.items()}


# ============================================================
# 主任务
# ============================================================
@celery_app.task(
    bind=True,
    name="ingest.full_pipeline",
    max_retries=1,
    acks_late=True,
)
def full_pipeline(self: Any, task_id: str, version_id: int) -> dict[str, Any]:
    """
    入库全流程。Celery 入口是**同步**函数，内部用进程级事件循环跑 async 流程。

    重试策略（max_retries=1）：
        只重试一次。MinerU 解析失败绝大多数是文件本身的问题，
        重试 5 次只会让用户多等 5 倍时间。真正的临时故障（如模型服务重启）
        由用户在界面上手动点重试 —— 那时他有上下文，比自动重试更可控。
    """
    logger.info("开始入库任务: task=%s version_id=%s", task_id, version_id)
    try:
        result = run_async(_pipeline(task_id, version_id))
    except BaseException as exc:  # noqa: BLE001 —— 含 SoftTimeLimitExceeded
        error_log = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        logger.error("入库任务失败: task=%s err=%s", task_id, exc)
        run_async(_mark_failed(task_id, version_id, error_log))
        raise
    logger.info("入库任务完成: task=%s %s", task_id, result)
    return result


async def _pipeline(task_id: str, version_id: int) -> dict[str, Any]:
    context = await _load_context(task_id, version_id)
    if not context:
        logger.warning("任务或版本已不存在，跳过: task=%s", task_id)
        return {"skipped": True}

    doc_id = context["doc_id"]
    user_id = context["user_id"]
    version = context["version"]
    file_path = context["file_path"]

    cfg = get_config()
    output_dir = str(
        Path(str(cfg.get_path("paths.parsed_dir", "/app/storage/parsed")))
        / doc_id
        / f"v{version}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Stage 0：置为 running ----------
    await _set_version(version_id, parse_status="parsing")
    await _set_task(
        task_id,
        **_progress_stage("parsing", 5),
        started_at=datetime.now(UTC),
    )

    # ---------- Stage 1：MinerU 解析 ----------
    mineru = MineruClient()
    parsed = await mineru.parse(file_path, output_dir)
    await _set_version(version_id, page_count=parsed.page_count)
    await _set_task(task_id, **_progress_stage("parsing", 35))

    # ---------- Stage 2：图片 caption ----------
    captions: dict[str, str] = {}
    if parsed.images:
        await _set_task(task_id, **_progress_stage("captioning", 36))
        captions = await _caption_images(task_id, parsed, output_dir)
    await _set_task(task_id, **_progress_stage("captioning", 65))

    # ---------- Stage 3：切分 ----------
    await _set_task(task_id, **_progress_stage("chunking", 70))
    model_client = get_model_client()

    async def _embed_for_split(texts: list[str]) -> list[list[float]]:
        return await model_client.embed(texts)

    chunks = await chunker.build(
        parsed, captions, embed_fn=_embed_for_split, on_page=None
    )
    if not chunks:
        raise RuntimeError("切分结果为空 —— 文件可能是扫描件或纯图片且未识别出任何文本")

    # ---------- Stage 4：embedding + 写 Qdrant（边算边写） ----------
    await _set_task(task_id, **_progress_stage("embedding", 72))
    store = get_vector_store()

    # 幂等：先删本版本旧点，避免重跑后块数变少留下孤儿点
    store.delete_by_version(doc_id, version)

    embed_batch = int(cfg.get_path("ingest.batch.embed_batch", 32))
    file_name = context["file_name"]
    written = 0

    for start in range(0, len(chunks), embed_batch):
        window = chunks[start : start + embed_batch]
        vectors = await model_client.embed([c.content for c in window])
        written += store.upsert_chunks(
            doc_id=doc_id,
            user_id=user_id,
            version=version,
            file_name=file_name,
            chunks=window,
            vectors=vectors,
            is_active=True,          # 新版本上传即生效（旧版本已在 PG 侧下线）
        )
        # 72% → 92% 线性推进，让前端进度条真实反映"算到哪了"
        progress = 72 + int(20 * min(written, len(chunks)) / max(len(chunks), 1))
        await _set_task(task_id, **_progress_stage("embedding", progress))
        # 向量写进 Qdrant 后即可回收，防止内存随块数增长（见模块头第 2 点）
        del vectors

    # ---------- Stage 5：收尾 ----------
    await _set_version(
        version_id,
        parse_status="success",
        chunk_count=written,
        parsed_at=datetime.now(UTC),
    )
    await _set_task(
        task_id,
        status="success",
        progress=100,
        current_stage="success",
        chunk_count=written,
        finished_at=datetime.now(UTC),
    )

    # 对齐 Qdrant 的 is_active 投影（PG 已是事实来源，这里只是让过滤走索引）
    try:
        store.set_active_by_version(doc_id, version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("is_active 投影对齐失败（检索侧会二次校验）: %s", exc)

    return {
        "doc_id": doc_id,
        "version": version,
        "page_count": parsed.page_count,
        "chunks": written,
        "images": len(parsed.images),
        "tables": len(parsed.tables),
    }


async def _caption_images(
    task_id: str, parsed: ParsedDocument, output_dir: str
) -> dict[str, str]:
    """
    逐张生成 caption。

    为什么串行（config 里的 caption_batch=2 说的是单次 VLM 调用的图片数）：
        caption 走的是同一把 GPU 锁，并发发出只会排队，反而因为
        "同时占住 N 个请求的内存"而拉高显存峰值。串行的总耗时基本相同。

    单张失败不阻断整体：某张图 caption 不出来，
    该图片块仍然入库（内容标为"未生成描述"），
    比整个文档解析失败要合理得多。
    """
    model_client = get_model_client()
    page_text = _page_text_map(parsed)
    captions: dict[str, str] = {}
    total = len(parsed.images)
    storage_root = str(get_config().get_path("paths.storage_root", "/app/storage"))

    for index, image in enumerate(parsed.images, start=1):
        raw_path = image.path
        abs_path = _resolve_image_path(raw_path, output_dir, parsed.image_dir)
        if abs_path:
            # ★ 把解析出的落盘绝对路径**写回** image.path。
            #   chunk.image_path 取的就是 image.path，下方 captions 的 key 也用它，
            #   两者必须同源 —— 否则 chunker 里 captions.get(image.path) 取不到
            #   caption，图注会静默丢失。详见 _public_image_path 的说明。
            image.path = _public_image_path(abs_path, raw_path, storage_root)
        if not abs_path:
            continue
        try:
            caption = await model_client.caption(
                abs_path, context=page_text.get(image.page_num, "")
            )
        except GpuBusyError as exc:
            # GPU 被占：不当作致命错误，保留空 caption 继续
            logger.warning("caption 跳过（GPU 繁忙）: %s %s", abs_path, exc)
            continue
        except (ModelServiceError, Exception) as exc:  # noqa: BLE001
            logger.warning("caption 失败（跳过该图）: %s %s", abs_path, exc)
            continue

        if caption:
            captions[image.path] = caption

        progress = 36 + int(28 * index / max(total, 1))
        await _set_task(task_id, **_progress_stage("captioning", progress))

    logger.info("caption 完成: %s/%s 张成功", len(captions), total)

    # ★ 系统性故障闸门：一张都没成功 ≠ "这张图质量差"。
    #   VLM 对任何图都会给出一些描述，所以 total>0 却 0 张成功，
    #   只可能是系统性原因（权重未就绪 / model-service 不可用 /
    #   GPU 锁长期被占 / image_path 全部解析失败）。
    #
    #   为什么必须在这里 fail 而不是"降级入库"：
    #     降级入库会产出一个**没有 image_caption 点的索引**，且 parse_status=success。
    #     而本项目的检索目标就是多模态 —— 图注缺失会让"这张架构图讲了什么"这类
    #     问题永远检索不到，但用户与日志都看不到任何错误。
    #     宁可让任务明确失败（用户可手动重试），也不要静默产出一个残缺索引。
    if total > 0 and not captions:
        raise RuntimeError(
            f"全部 {total} 张图片 caption 均失败，判定为系统性故障（非单图偶发）。"
            f"请检查 model-service 是否可用、模型权重是否就绪"
            f"（curl -s http://model-service:8002/health）。"
            f"每张图的具体原因见上方 WARNING 日志。"
        )

    return captions


async def _mark_failed(task_id: str, version_id: int, error_log: str) -> None:
    """把任务与版本都标成失败。任何异常都不再向外抛（否则会覆盖原始错误）。"""
    try:
        await _set_version(version_id, parse_status="failed", error_message=error_log[:4000])
    except Exception as exc:  # noqa: BLE001
        logger.error("标记版本失败时出错: %s", exc)
    try:
        await _set_task(
            task_id,
            status="failed",
            current_stage="failed",
            error_log=error_log[:20000],
            finished_at=datetime.now(UTC),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("标记任务失败时出错: %s", exc)


# ============================================================
# worker 进程退出时的清理钩子
# ============================================================
@celery_app.on_after_finalize.connect
def _register_worker_shutdown(sender: Any, **_: Any) -> None:  # pragma: no cover
    """worker 进程回收时释放连接池（max_tasks_per_child 会触发）。"""
    from celery.signals import worker_process_shutdown

    @worker_process_shutdown.connect
    def _shutdown(**__: Any) -> None:
        try:
            run_async(dispose_engine())
            run_async(get_model_client().aclose())
        except Exception as exc:  # noqa: BLE001
            logger.warning("worker 收尾清理失败: %s", exc)


__all__ = ["full_pipeline"]
