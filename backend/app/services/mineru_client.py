"""
MinerU 解析客户端 + `content_list.json` 解析器。

两段职责
--------
    1. HTTP 调用：把文件路径丢给 mineru 容器，等它解析完返回产物路径
    2. 产物解析：把 MinerU 的 content_list.json 归一化成三类块
       （text / table / image_caption），供 chunker 消费

关于 page_num 的 0/1 基（容易错，且错了很难发现）
-----------------------------------------------
MinerU 的 `content_list.json` 用 **`page_idx`，0 基**；
而本项目的 `chunks.payload.page_num` 与前端溯源都按**人类习惯的 1 基**。
本模块统一在入口处 `+1` 转换，**下游一律按 1 基处理**。
如果只在一部分地方转换，会出现"页码整体差 1"这种低级但难查的 bug。

关于 MinerU 版本
-------------------
⚠️ **实测装的是 MinerU 4.0.0**（`requirements.txt` 刻意未锁版本，
   首次构建后才该回填），不是设计文档当初假设的 2.x。两者差异很大，
   本模块按 **4.0 的 Content List V1** 写，但对 2.x 的同名字段做了兼容：

| 概念 | 2.x | 4.0（本模块针对） |
|---|---|---|
| 档位 | `pipeline` / `vlm` | **`flash` / `basic` / `standard` / `advanced`** |
| OCR 开关 | `formula_enable` / `table_enable` | **`ocr_mode`: auto / txt / ocr** |
| 输出 | `content_list.json` | Markdown + **Content List V1/V2** |

**好消息**：4.0 的 Content List **V1** 条目字段与 2.x **同名**
（`page_idx` / `text` / `text_level` / `table_body` / `table_caption` /
`img_path` / `image_caption` / `image_footnote`），所以本模块的解析逻辑
基本原样可用 —— 这是选择"跟随 4.0"而不是"锁回 2.x"的重要依据之一。

**两处必须注意的 4.0 特性**：
    1. `table_body` 依然是 **HTML `<table>`**（对应 DESIGN.md D6：不转 Markdown，
       以保住合并单元格）—— 实测 4.0 在 flash 档下也满足这条。
    2. ★ `img_path` 在 MinerU 未外置图片时是 **`data:image/jpeg;base64,...`**
       内联串（实测一张柱状图 36411 字符）。本模块**显式拦截**它，
       因为把 base64 写进 Qdrant payload 会造成索引膨胀 + 检索结果不可读。
       外置动作由 `deploy/mineru/server.py` 的 `_materialize_images` 完成，
       外置后这里是相对路径 `images/xxx.jpg`，由 ingest_tasks 解析成绝对路径。

对应设计文档：DESIGN.md §7.2 Stage 1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_config, get_env
from app.core.exceptions import ParseFailedError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 归一化后的块结构
# ============================================================
@dataclass(slots=True)
class TextBlock:
    text: str
    page_num: int
    # ★ content_list 里的原始下标，用于还原**阅读顺序**
    #   （PPT 同一页常见"文 → 图 → 文"混排，丢掉顺序会让溯源引用错位）
    order: int = 0


@dataclass(slots=True)
class TableBlock:
    """表格整块不切分（D6）。html 是原文，线性化文本另存供向量化。"""

    html: str
    caption: str
    footnote: str
    page_num: int
    order: int = 0


@dataclass(slots=True)
class ImageBlock:
    """图片在 chunker 阶段会补上 VLM caption，本结构只带"上下文提示"。"""

    path: str
    caption_hint: str          # MinerU 自带的 image_caption（多数情况下为空）
    footnote: str
    page_num: int
    order: int = 0


@dataclass(slots=True)
class ParsedDocument:
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    images: list[ImageBlock] = field(default_factory=list)
    page_count: int = 0
    markdown_path: str | None = None
    content_list_path: str | None = None
    image_dir: str | None = None
    elapsed_sec: float = 0.0

    @property
    def total_blocks(self) -> int:
        return len(self.text_blocks) + len(self.tables) + len(self.images)


# ============================================================
# 小工具：MinerU 的 caption 字段是 list[str]，统一拼成一段
# ============================================================
def _join_caption(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _page_num(item: dict[str, Any]) -> int:
    """把 MinerU 的 0 基 page_idx 转成 1 基 page_num（见模块头）。"""
    for key in ("page_idx", "page_index", "page_num"):
        if key in item and item[key] is not None:
            try:
                raw = int(item[key])
            except (TypeError, ValueError):
                continue
            # page_idx 是 0 基；若字段名本来就叫 page_num 则视为已是 1 基
            return raw + 1 if key != "page_num" else max(raw, 1)
    return 1


class MineruClient:
    """MinerU 解析服务的 HTTP 客户端。"""

    def __init__(self) -> None:
        cfg = get_config()
        env = get_env()
        self.base_url: str = str(
            cfg.get_path("ingest.mineru.base_url", env.mineru_url)
        ).rstrip("/")
        # CPU 模式解析慢，超时按 config 放宽到 30 分钟
        self.timeout_sec: int = int(cfg.get_path("ingest.mineru.timeout_sec", 1800))

    # ---------------- 解析 ----------------
    async def parse(self, file_path: str, output_dir: str) -> ParsedDocument:
        """
        调用 MinerU 解析文件。

        ⚠️ 超时给得很宽（默认 1800s）：CPU 模式下几十页 PPT 解析 10-20 分钟是正常的。
        容易踩的坑是把这里设成 60s，然后表现成"随机解析失败"。
        """
        cfg = get_config()
        # ⚠️ 字段名与 MinerU 4.0 的 `mineru.parser.parse()` 签名**一一对应**。
        #    2.x 的 `backend` / `formula_enable` / `table_enable` / `page_batch_size`
        #    在 4.0 里不存在 —— 传了会被 pydantic 静默丢弃（请求模型会忽略未知字段），
        #    于是表现为"配置改了但不生效"。改配置时务必对照这里。
        payload = {
            "file_path": file_path,
            "output_dir": output_dir,
            "tier": str(cfg.get_path("ingest.mineru.tier", "flash")),
            "ocr_mode": str(cfg.get_path("ingest.mineru.ocr_mode", "txt")),
            # "all" = 全部页。**不要**改成空串依赖默认值：
            # mineru CLI 的 --pages 默认只解析前 10 页，这个差异会静默截断长文档。
            "page_range": str(cfg.get_path("ingest.mineru.page_range", "all")),
            "image_analysis": bool(cfg.get_path("ingest.mineru.image_analysis", True)),
        }

        url = f"{self.base_url}/parse"
        logger.info("调用 MinerU 解析: file=%s → %s", file_path, output_dir)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ParseFailedError(f"无法连接 MinerU ({url}): {exc}") from exc

        if resp.status_code >= 400:
            detail = _safe_detail(resp)
            raise ParseFailedError(f"MinerU 返回 {resp.status_code}: {detail}")

        data = resp.json()
        if data.get("status") not in {"ok", "success", None}:
            raise ParseFailedError(f"MinerU 解析失败: {data.get('status')}")

        content_list_path = data.get("content_list_path")
        if not content_list_path:
            raise ParseFailedError("MinerU 未返回 content_list_path，无法继续入库")

        parsed = self.load_content_list(content_list_path)
        parsed.page_count = int(data.get("page_count") or parsed.page_count)
        parsed.markdown_path = data.get("markdown_path")
        parsed.content_list_path = content_list_path
        parsed.image_dir = data.get("image_dir")
        parsed.elapsed_sec = float(data.get("elapsed_sec") or 0.0)

        logger.info(
            "MinerU 解析完成: pages=%s text=%s table=%s image=%s",
            parsed.page_count,
            len(parsed.text_blocks),
            len(parsed.tables),
            len(parsed.images),
        )
        return parsed

    # ---------------- 产物解析 ----------------
    def load_content_list(self, path: str) -> ParsedDocument:
        """
        读取并归一化 content_list.json。

        设计取舍：**未知 type 直接跳过而不是报错**。
        MinerU 会输出 `equation`（公式）等本系统不处理的类型，
        因为 config 里 formula_enable=false，遇到就跳过更稳；
        若未来开启公式，再补一个 equation 分支即可。
        """
        file = Path(path)
        if not file.is_file():
            raise ParseFailedError(f"content_list.json 不存在: {path}")

        try:
            items = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ParseFailedError(f"content_list.json 解析失败: {exc}") from exc

        if not isinstance(items, list):
            raise ParseFailedError("content_list.json 顶层不是数组")

        doc = ParsedDocument()
        max_page = 0
        skipped = 0

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                skipped += 1
                continue
            item_type = str(item.get("type", "")).strip().lower()
            page = _page_num(item)
            max_page = max(max_page, page)

            if item_type in {"text", "title", "paragraph"}:
                text = str(item.get("text") or "").strip()
                if text:
                    doc.text_blocks.append(TextBlock(text=text, page_num=page, order=idx))

            elif item_type == "table":
                html = str(
                    item.get("table_body") or item.get("html") or item.get("text") or ""
                ).strip()
                if html:
                    doc.tables.append(
                        TableBlock(
                            html=html,
                            caption=_join_caption(item.get("table_caption")),
                            footnote=_join_caption(item.get("table_footnote")),
                            page_num=page,
                            order=idx,
                        )
                    )

            elif item_type in {"image", "figure", "chart"}:
                # ★ `chart`（图表）是 4.0 新增类型，结构与 image 同源，
                #   只有 caption 字段名不同（chart_caption / chart_footnote）。
                #   PPT 里图表极多，不合并处理会把**整类内容静默丢掉**
                #   —— 旧实现只认 {"image","figure"}，chart 会落到 else 被跳过。
                img_path = str(item.get("img_path") or item.get("image_path") or "").strip()
                if img_path.startswith("data:"):
                    # mineru 侧应已把内联图片外置成文件（见模块头第 2 点）。
                    # 走到这里说明外置失败 —— 宁可少一个图片块，
                    # 也绝不能把 36KB 的 base64 写进 Qdrant payload。
                    logger.warning(
                        "图片仍为内联 data URI，已跳过（mineru 侧外置失败？）: type=%s page=%s idx=%s",
                        item_type, page, idx,
                    )
                    skipped += 1
                elif img_path:
                    doc.images.append(
                        ImageBlock(
                            path=img_path,
                            caption_hint=_join_caption(
                                item.get("image_caption") or item.get("chart_caption")
                            ),
                            footnote=_join_caption(
                                item.get("image_footnote") or item.get("chart_footnote")
                            ),
                            page_num=page,
                            order=idx,
                        )
                    )

            else:
                # equation / 等本系统不处理的类型
                skipped += 1

        doc.page_count = max_page
        if skipped:
            logger.info("content_list 跳过 %s 个非文本/表格/图片条目", skipped)
        return doc

    # ---------------- 健康 ----------------
    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
            return resp.status_code < 500
        except httpx.HTTPError:
            return False


def _safe_detail(resp: httpx.Response, limit: int = 300) -> str:
    """截断错误详情，避免把整页 HTML 塞进数据库 error_log。"""
    try:
        return resp.text[:limit]
    except Exception:  # noqa: BLE001
        return "<无法读取响应体>"


__all__ = [
    "ImageBlock",
    "MineruClient",
    "ParsedDocument",
    "TableBlock",
    "TextBlock",
]
