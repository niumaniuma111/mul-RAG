"""
Chunk 切分器（DESIGN.md §7.3）。

三类 chunk，切分策略完全不同
---------------------------
    text           语义切分：切句 → BGE 相邻句余弦 → 相似度低处断开 → 合并到 ≤512 token
    table          **整块不切分**（PPT 合并单元格多，切开会丢表头语义）
    image_caption  **整块不切分**

为什么用"相邻句余弦"而不是按字数切
--------------------------------
按字数切会把一句话拦腰截断，且切点与语义无关 —— 检索时一个 chunk 里混着
两个不相干的话题，向量被平均掉，两边都召不回。
相邻句余弦低 ⇔ 话题发生切换，这才是语义边界。代价是要为每个句子算一次
embedding（入库耗时的主要来源）。

内存约束（16GB 机器的实际限制，决定了实现方式）
--------------------------------------------
**逐页组算 embedding，而不是把所有句子攒起来一次算。**
一个 100 页 PPT 可能有几千个句子，1024 维向量的 Python list 每个浮点对象
占 ~32 字节 → 5000×1024×32B ≈ 160MB，而 worker 的 mem_limit 只有 512MB。
逐组算把峰值压到「单页句子数 × 1024」的量级（几百 KB）。

对应设计文档：DESIGN.md §7.3
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.core.config import get_config
from app.core.logging import get_logger
from app.services.mineru_client import ImageBlock, ParsedDocument, TableBlock, TextBlock
from app.services.text_utils import estimate_tokens, split_sentences

logger = get_logger(__name__)


@dataclass(slots=True)
class Chunk:
    """一个待入库的切分单元。写入 Qdrant 时会补上 user_id / doc_id / version。"""

    chunk_index: int
    chunk_type: str                 # text / table / image_caption
    content: str                    # 参与向量化与 BM25 的文本
    page_num: int
    table_html: str | None = None   # 仅 table：HTML 原文，供前端渲染
    image_path: str | None = None   # 仅 image_caption：容器内绝对路径（/app/storage/...），前端剥前缀后走 nginx 静态代理
    order: int = 0                  # 阅读顺序（来自 content_list 下标）
    token_estimate: int = 0


# ============================================================
# 余弦相似度（纯 Python 实现）
# ============================================================
def _cosine(a: list[float], b: list[float]) -> float:
    """
    余弦相似度。

    ⚠️ 刻意不依赖 numpy：api/worker 镜像的 requirements 里没有 numpy
       （它只是 qdrant-client 的传递依赖，不可靠）。纯 Python 在
       1024 维、每页十几对向量的量级下完全够用（微秒级）。

    向量已归一化（embed 时 normalize=true），所以本可以只算点积，
    但这里仍做完整计算 —— 万一某次调用关了 normalize，结果不会错。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ============================================================
# 合并与硬切（semantic_split 的后半段）
# ============================================================
def _join(sentences: list[str]) -> str:
    return "".join(sentences).strip()


def _merge_short_groups(
    groups: list[list[str]], min_tokens: int, max_tokens: int
) -> list[list[str]]:
    """
    把过短的组并入**前一组**（不是后一组）。

    为什么往前并：向前合并保持"新话题从新组开始"的语义结构；
    向后合并会把新话题的开头拖进上一组的尾巴。
    """
    merged: list[list[str]] = []
    for group in groups:
        if not group:
            continue
        if merged:
            prev_text = _join(merged[-1])
            candidate = prev_text + _join(group)
            if estimate_tokens(prev_text) < min_tokens and estimate_tokens(candidate) <= max_tokens:
                merged[-1] = merged[-1] + group
                continue
        merged.append(list(group))
    return merged


def _hard_split_group(
    sentences: list[str], max_tokens: int, overlap_tokens: int
) -> list[list[str]]:
    """
    对超长组做滑动窗口硬切。

    只在"单组本身就超过 max_tokens"时触发 —— 此时语义上它已经是一个话题，
    再按语义切没有意义，只能按长度切，并靠 overlap 保住跨界的上下文
    （否则关键句被切在两块中间，两块都召回不全）。
    """
    if estimate_tokens(_join(sentences)) <= max_tokens:
        return [sentences]

    windows: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        if current and estimate_tokens(_join(current + [sentence])) > max_tokens:
            windows.append(current)
            # 从窗口尾部回取约 overlap_tokens 的内容作为下一窗口的前缀
            tail: list[str] = []
            acc = 0
            for prev in reversed(current):
                acc += estimate_tokens(prev)
                tail.insert(0, prev)
                if acc >= overlap_tokens:
                    break
            current = list(tail)
        current.append(sentence)
    if current:
        windows.append(current)
    return windows


# ============================================================
# 语义切分（单个文本组）
# ============================================================
async def semantic_split(
    text: str,
    embeddings: list[list[float]],
    threshold: float,
) -> list[str]:
    """
    对**已切句并算好 embedding**的一段文本做语义切分。

    ⚠️ embedding 由调用方传入而不是在这里算：这样调用方可以按页批量算，
       避免每页都发一次 HTTP（省掉几十倍的请求开销）。
    """
    cfg = get_config()
    max_tokens = int(cfg.get_path("ingest.chunk.text.max_tokens", 512))
    min_tokens = int(cfg.get_path("ingest.chunk.text.min_tokens", 80))
    overlap = int(cfg.get_path("ingest.chunk.text.overlap_tokens", 64))

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        # 单句（或空）：仍可能本身超长，交给硬切处理
        windows = _hard_split_group(sentences, max_tokens, overlap)
        joined_windows = [_join(w) for w in windows]
        return [w for w in joined_windows if w]

    if len(embeddings) != len(sentences):
        # embedding 数量对不上（理论上不该发生）→ 退化为整段一块，不阻断入库
        logger.warning(
            "句子数(%s)与向量数(%s)不匹配，退化为整段不切分",
            len(sentences), len(embeddings),
        )
        return [text.strip()] if text.strip() else []

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        joined = _join(groups[-1] + [sentences[i]])
        if sim >= threshold and estimate_tokens(joined) <= max_tokens:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    groups = _merge_short_groups(groups, min_tokens, max_tokens)

    result: list[str] = []
    for group in groups:
        result.extend(_join(w) for w in _hard_split_group(group, max_tokens, overlap))
    return [r for r in result if r]


# ============================================================
# 文本块 → 按页分组
# ============================================================
def _group_text_blocks_by_page(
    blocks: list[TextBlock],
) -> list[tuple[int, int, str]]:
    """
    把**连续的**同页文本块合并成一段。

    返回 [(page_num, order, text), ...]，order 取该组第一块的 order
    （用于最终按阅读顺序排序）。

    为什么只合并"连续"的同页块：MinerU 的 content_list 是阅读顺序，
    中间隔着表格/图片的两段文字在版面上并不相邻，合并会让语义切分跨越图片，
    产生"上半页 + 下半页"的怪块。
    """
    groups: list[tuple[int, int, str]] = []
    current_page: int | None = None
    current_order = 0
    buffer: list[str] = []

    for block in sorted(blocks, key=lambda b: b.order):
        if current_page is None:
            current_page, current_order = block.page_num, block.order
        elif block.page_num != current_page:
            groups.append((current_page, current_order, "\n".join(buffer)))
            current_page, current_order, buffer = block.page_num, block.order, []
        buffer.append(block.text)

    if buffer and current_page is not None:
        groups.append((current_page, current_order, "\n".join(buffer)))
    return groups


# ============================================================
# 表格线性化
# ============================================================
def _linearize_table(html: str, caption: str = "") -> str:
    """
    把 HTML 表格转成一行一行的线性文本，供向量化与 BM25。

    格式：`表头: 值 | 表头: 值`
    —— 带上表头是关键。只输出值的话，"30%" 这种单元格脱离表头完全没有语义，
    既召不回也无法被人读懂。

    首行若为 `th` 则作为表头；否则用第一行当表头（PPT 表格常见做法）。
    """
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001 —— 解析器缺失等情况退化为纯文本
        soup = BeautifulSoup(html, "html.parser")

    rows = soup.find_all("tr")
    if not rows:
        text = soup.get_text(" ", strip=True)
        return f"{caption}\n{text}".strip() if caption else text

    def cells_of(tr) -> list[str]:
        return [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]

    header = cells_of(rows[0])
    lines: list[str] = []
    for tr in rows[1:]:
        values = cells_of(tr)
        if not values:
            continue
        pairs = []
        for i, value in enumerate(values):
            if not value:
                continue
            key = header[i] if i < len(header) and header[i] else f"列{i + 1}"
            pairs.append(f"{key}: {value}")
        if pairs:
            lines.append(" | ".join(pairs))

    if not lines:
        # 单行表格（只有表头）→ 只输出表头
        lines = [" | ".join(header)] if any(header) else []

    body = "\n".join(lines)
    head = f"表格：{caption}" if caption else "表格"
    return f"{head}\n{body}".strip()


# ============================================================
# 主入口
# ============================================================
async def build(
    parsed: ParsedDocument,
    captions: dict[str, str],
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
    on_page: Callable[[int, int], Awaitable[None]] | None = None,
) -> list[Chunk]:
    """
    构建全部 chunk。

    参数
    ----
    parsed      MinerU 解析结果
    captions    image_path → VLM 生成的 caption（缺失则退回 MinerU 自带提示）
    embed_fn    批量向量化函数（注入而非直接调 client，便于单测替换）
    on_page     每完成一页文本的回调，用于推进进度

    返回按**阅读顺序**排序、chunk_index 从 0 连续编号的 chunk 列表。
    """
    cfg = get_config()
    threshold = float(cfg.get_path("ingest.chunk.text.similarity_threshold", 0.65))

    chunks: list[Chunk] = []

    # ---------- 1. 文本 ----------
    page_groups = _group_text_blocks_by_page(parsed.text_blocks)
    total_groups = len(page_groups)
    for done, (page_num, order, text) in enumerate(page_groups, start=1):
        if not text.strip():
            continue
        sentences = split_sentences(text)
        embeddings: list[list[float]] = []
        if len(sentences) > 1:
            # 逐页组算 embedding（内存约束见模块头）
            embeddings = await embed_fn(sentences)

        pieces = await semantic_split(text, embeddings, threshold)
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_index=-1,          # 最后统一编号
                    chunk_type="text",
                    content=piece,
                    page_num=page_num,
                    order=order,
                    token_estimate=estimate_tokens(piece),
                )
            )
        if on_page is not None and total_groups:
            await on_page(done, total_groups)

    # ---------- 2. 表格（整块） ----------
    for table in parsed.tables:
        content = _linearize_table(table.html, table.caption)
        note_parts = [p for p in (table.footnote,) if p]
        if note_parts:
            content = content + "\n" + " ".join(note_parts)
        chunks.append(
            Chunk(
                chunk_index=-1,
                chunk_type="table",
                content=content,
                page_num=table.page_num,
                table_html=table.html,
                order=table.order,
                token_estimate=estimate_tokens(content),
            )
        )

    # ---------- 3. 图片 caption（整块） ----------
    for image in parsed.images:
        caption = (captions.get(image.path) or image.caption_hint or "").strip()
        if not caption:
            # 没有 caption 的图片无法检索 —— 但仍然入库并标出路径，
            # 这样前端溯源时能看到"这页有张图没被描述"，而不是静默丢失。
            caption = "（此图未生成描述）"
        content = caption
        if image.footnote:
            content = f"{caption}\n{image.footnote}"
        chunks.append(
            Chunk(
                chunk_index=-1,
                chunk_type="image_caption",
                content=content,
                page_num=image.page_num,
                image_path=image.path,
                order=image.order,
                token_estimate=estimate_tokens(content),
            )
        )

    # ---------- 4. 统一排序与编号 ----------
    # 排序键：(order, 类型优先级)。同一 order 不会出现（order 是唯一下标），
    # 加类型优先级只为排序稳定，避免相同 order 时顺序随机。
    type_priority = {"text": 0, "table": 1, "image_caption": 2}
    chunks.sort(key=lambda c: (c.order, type_priority.get(c.chunk_type, 9)))
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index

    logger.info(
        "切分完成: 共 %s 块（text=%s table=%s image=%s）",
        len(chunks),
        sum(1 for c in chunks if c.chunk_type == "text"),
        sum(1 for c in chunks if c.chunk_type == "table"),
        sum(1 for c in chunks if c.chunk_type == "image_caption"),
    )
    return chunks


__all__ = [
    "Chunk",
    "build",
    "semantic_split",
]
