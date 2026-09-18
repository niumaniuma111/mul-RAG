"""
MinerU HTTP 包装服务 —— **MinerU 4.0** 实现。

为什么需要这层包装
------------------
MinerU 本身是 CLI / Python 库，而离线入库流水线跑在 Celery 里。中间加一个 HTTP
服务边界的理由有三条：

    1. **隔离** —— MinerU 的依赖（torch / opencv / 一堆 PDF 解析库）非常重且容易
       与后端冲突，独立容器可以各用各的依赖树。
    2. **配合全局 GPU 锁** —— device=cuda 时解析全程要持有跨容器锁
       （lock:gpu:heavy），这个「抢锁 → 解析 → 放锁」的临界区必须有统一入口。
    3. **统一报错** —— CLI 的退出码和 stderr 格式不适合直接透给 Celery 任务。

GPU 模式的并发语义
------------------
    device=cpu  → 不持锁，可与在线问答完全并行（这是它作默认值的理由）
    device=cuda → 解析全程持锁，期间所有问答请求排队等待（1-3 分钟）

⚠️⚠️ 本文件已按 **MinerU 4.0.0** 重写，不是 2.x 的写法（2026-09-17 实测）
================================================================
`deploy/mineru/requirements.txt` 刻意未锁版本（「首次构建成功后再回填」），
实际装出来的是 **4.0.0**，而它的架构与 2.x 完全不同：

| | MinerU 2.x | **MinerU 4.0**（本文件针对的版本） |
|---|---|---|
| 入口 | `magic-pdf` 命令 / `pipeline` | `mineru.parser.parse()`，或 `mineru` CLI |
| CLI 语义 | 直接解析 | **`parse` 只是客户端**，要先 `mineru server start` |
| 输出 | `content_list.json` | Markdown / Content List **V1 / V2** / middle_json |
| 档位 | pipeline / vlm | **flash / basic / standard / advanced** |

**为什么走进程内 API 而不是 `mineru server`**：
    `mineru.parser.parse()` 是纯函数式入口，不需要常驻守护进程 ——
    少一个需要健康检查、需要守护重启的进程，也少一处故障点。
    （若改用 CLI，就必须先起 `mineru server`，KV 里那条
     `server_not_running` 错误就是踩过这个坑的证据。）

**关于 flash 档（本项目默认）**
--------------------------------
`flash` 对 OOXML 做**原生解析**（`docvortex.analyzers.native.PptxModel`），
不加载任何模型、不占显存、不需要 LibreOffice，因此**可完全离线**。
2026-09-17 实测：`--network none` 下解析带文字/表格/图片的 3 页 pptx 成功。

    收益：PPT 正文是**精确原文**而非 OCR 结果；PPT 表格还原成**真 HTML 表格**
          （保住合并单元格，对应 DESIGN.md D6）；内嵌图片取出**原始位图**。
    代价：**不做 OCR** → 扫描件 PDF 会缺文本。故 tier 做成入参，
          将来按文件类型改传 `standard` 即可，**不必换引擎**。

对应设计文档：DESIGN.md §7.2 Stage 1
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# 环境
# ============================================================
# ⚠️ MINERU_DEVICE 是**我们自己的**变量，MinerU 不认它（MinerU 自己的模型源
#    变量叫 MINERU_MODEL_SOURCE）。它只用来决定「要不要持跨容器 GPU 锁」。
MINERU_DEVICE = (os.getenv("MINERU_DEVICE", "cpu") or "cpu").strip().lower()
OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "")

GPU_LOCK_REQUIRED = MINERU_DEVICE == "cuda"
GPU_LOCK_KEY = os.getenv("GPU_LOCK_KEY", "lock:gpu:heavy")
GPU_LOCK_TTL_SEC = int(os.getenv("GPU_LOCK_TTL_SEC", "1800"))
GPU_LOCK_ACQUIRE_TIMEOUT_SEC = int(os.getenv("GPU_LOCK_ACQUIRE_TIMEOUT_SEC", "600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# 走 visual 分支的 content_list 类型
_VISUAL_TYPES = {"image", "chart", "table"}

# data URI 形如： data:image/jpeg;base64,<payload>
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;,]*)(?P<b64>;base64)?,(?P<payload>.*)$", re.S)
_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


# ============================================================
# 并发闸门
# ------------------------------------------------------------
# MinerU 解析是 CPU/内存密集型。worker 侧虽然是 -c 1，但 api 侧的手动重试
# 或未来的并发上传仍可能同时打进来。mem_limit=3g 下，两个解析并行很容易被
# OOM killer 干掉（表现是容器重启 + 任务莫名失败）。
#
# 用 asyncio.Lock（而非 threading.Lock）：解析本身在 to_thread 里跑，
# 持锁方是事件循环，这样锁的语义与 HTTP 请求一一对应，不会被线程池大小影响。
# ============================================================
_parse_lock = asyncio.Lock()


# ============================================================
# 跨容器 GPU 锁（仅 device=cuda 时使用）
# ------------------------------------------------------------
# 复用 backend 的 key（lock:gpu:heavy）与 TTL 下限，保证两侧看得见同一把锁。
# 为什么必须续期：MinerU 解析可能远超初始 TTL（50 页 PPT 十几分钟），
# 不续期就会出现「锁已过期但我还在跑」→ 另一侧拿到锁 → 两个模型同时上卡 → OOM。
# 释放用 Lua 比对 token，避免删掉别人的锁。
# ============================================================
_LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class GpuLock:
    """跨容器 GPU 锁的极简实现（阻塞式获取 + 后台续期）。"""

    def __init__(self, key: str, ttl_sec: int, owner: str) -> None:
        self.key = key
        self.ttl_sec = ttl_sec
        self.owner = owner
        self._client: Any = None
        self._renew_task: asyncio.Task[None] | None = None

    async def acquire(self, timeout_sec: int) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(REDIS_URL, decode_responses=True)
        deadline = time.monotonic() + timeout_sec
        while True:
            ok = await self._client.set(self.key, self.owner, nx=True, ex=self.ttl_sec)
            if ok:
                self._renew_task = asyncio.create_task(self._renew_loop())
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待 GPU 锁超时（{timeout_sec}s）：key={self.key}。"
                    f"通常是上一个问答/解析任务卡住未释放。"
                )
            await asyncio.sleep(1.0)

    async def _renew_loop(self) -> None:
        interval = max(self.ttl_sec // 3, 30)
        try:
            while True:
                await asyncio.sleep(interval)
                # 只有仍是自己持有才续期，避免把别人的锁续下去
                if await self._client.get(self.key) == self.owner:
                    await self._client.expire(self.key, self.ttl_sec)
        except asyncio.CancelledError:
            raise

    async def release(self) -> None:
        if self._renew_task is not None:
            self._renew_task.cancel()
            try:
                await self._renew_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._renew_task = None
        if self._client is not None:
            try:
                await self._client.eval(_LUA_RELEASE, 1, self.key, self.owner)
            finally:
                await self._client.aclose()
                self._client = None


@asynccontextmanager
async def _gpu_lock_if_needed() -> AsyncIterator[None]:
    if not GPU_LOCK_REQUIRED:
        yield
        return
    lock = GpuLock(GPU_LOCK_KEY, GPU_LOCK_TTL_SEC, f"mineru:{uuid.uuid4().hex}")
    await lock.acquire(GPU_LOCK_ACQUIRE_TIMEOUT_SEC)
    try:
        yield
    finally:
        await lock.release()


# ============================================================
# 请求 / 响应模型
# ============================================================
class ParseRequest(BaseModel):
    """
    解析请求。字段与 `configs/config.yaml` 的 `ingest.mineru` 段一一对应。

    ⚠️ `tier` / `ocr_mode` 是 MinerU 4.0 的概念，2.x 的 `backend` /
       `formula_enable` / `table_enable` 在这里**不存在**，不要照着旧代码传。
    """

    file_path: str = Field(..., description="容器内绝对路径（mineru 与 worker 共用 ./storage 挂载）")
    output_dir: str = Field(..., description="输出目录（容器内绝对路径）")
    tier: str = Field("flash", description="flash | basic | standard | advanced")
    ocr_mode: str = Field("txt", description="auto | txt | ocr（flash 档对 Office 恒为 txt）")
    # ⚠️ `page_range` **仅 PDF 生效**（实测：对 pptx 传值会抛
    #    `InvalidRequestError: Page range is only supported for PDF files`）。
    #    服务端会按扩展名自动忽略它，传了也不会报错。
    #    默认写 "all" 而不是空串，是因为 mineru CLI 的 --pages 默认**只解析前 10 页**，
    #    显式 "all" 可以避免"50 页 PPT 只入库 10 页"这种极难发现的静默截断。
    page_range: str = Field("all", description="仅 PDF 生效；\"all\" 或空 = 全部（CLI 默认仅前 10 页）")
    image_analysis: bool = Field(True, description="是否启用版面内图片分析")


class ParseResponse(BaseModel):
    status: str
    markdown_path: str | None = None
    content_list_path: str | None = None
    image_dir: str | None = None
    page_count: int = 0
    elapsed_sec: float = 0.0
    tier: str = "flash"
    counts: dict[str, int] = Field(default_factory=dict)


# ============================================================
# 图片外置（data URI → 文件）
# ============================================================
def _materialize_images(
    items: list[dict[str, Any]], output_dir: Path
) -> tuple[int, dict[str, str]]:
    """
    把 content_list 里内联的 base64 图片落盘，并把 `img_path` 改写成**相对路径**。

    为什么必须做这一步（2026-09-17 实测）：
        MinerU 4.0 的 `render_content_list()` 在未 `save()` 时，图片块的
        `img_path` 是 `data:image/jpeg;base64,...` —— 实测一张柱状图有
        **36411 个字符**。若原样透传给下游，会被写进 Qdrant 的 payload，
        造成：索引体积爆炸 + 检索结果里塞满 base64 + model-service 读不到图。

    为什么改写成**相对路径**（如 `images/page_2_image_body_1.jpg`）：
        下游 `ingest_tasks._resolve_image_path()` 已按
        「绝对路径 → image_dir → output_dir → output_dir/images」的顺序解析，
        相对路径正好命中最后一条，于是**下游一行都不用改**。

    返回 `(成功外置的张数, {原始 data URI: 相对路径})`。
    第二个返回值给调用方改写 markdown 用 —— 见 `_parse_sync` 里的说明。
    """
    image_dir = output_dir / "images"
    written = 0
    mapping: dict[str, str] = {}

    for index, item in enumerate(items):
        if item.get("type") not in _VISUAL_TYPES:
            continue
        raw = item.get("img_path")
        if not isinstance(raw, str) or not raw.startswith("data:"):
            continue  # 非内联（已是路径）或为空 —— 保持原样

        match = _DATA_URI_RE.match(raw)
        if not match:
            continue
        payload = match.group("payload")
        mime = (match.group("mime") or "image/jpeg").strip().lower()
        ext = _EXT_BY_MIME.get(mime, ".jpg")
        try:
            blob = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError):
            continue
        if not blob:
            continue

        image_dir.mkdir(parents=True, exist_ok=True)
        page_idx = item.get("page_idx", 0)
        name = f"page_{int(page_idx) + 1}_block_{index}{ext}"
        (image_dir / name).write_bytes(blob)
        rel = f"images/{name}"
        item["img_path"] = rel
        mapping[raw] = rel
        written += 1

    return written, mapping


# ============================================================
# 解析（同步，跑在线程里）
# ============================================================
def _parse_sync(req: ParseRequest) -> dict[str, Any]:
    """
    真正干活的部分。**全部同步**，由调用方丢进 `asyncio.to_thread`。

    一次性完成「解析 → content_list → 图片外置 → 落盘」，
    而不是拆成多个 await 点：中途失败时要留下的产物要么完整要么没有，
    避免出现"content_list 有图但图片文件不存在"这种下游很难排查的中间态。
    """
    # ⚠️ 惰性导入：mineru 的 import 会拉起 torch 等重依赖（数秒）。
    #    放在模块顶层会让 /health 也变慢，并且一旦依赖损坏连健康检查都起不来
    #    —— 健康端点必须永远可用。
    from mineru.parser import parse as mineru_parse
    from mineru.render.content_list import render_content_list

    src = Path(req.file_path)
    if not src.is_file():
        raise FileNotFoundError(
            f"待解析文件不存在：{src}。"
            f"（mineru 容器与 worker/api 必须挂同一个 ./storage，"
            f"否则同一路径在两个容器里指向不同文件）"
        )

    out_dir = Path(req.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ★ `page_range` **只对 PDF 有效**。实测（2026-09-17）：对 pptx 传 "all"
    #   会直接抛 `mineru.errors.InvalidRequestError: Page range is only supported
    #   for PDF files; 'pptx' uses full-document parsing.`
    #   所以这里按扩展名分支：非 PDF 一律传空串（Office 本来就是整篇解析）。
    suffix = src.suffix.lower().lstrip(".")
    page_range = req.page_range if suffix == "pdf" else ""

    started = time.monotonic()
    result = mineru_parse(
        str(src),
        tier=req.tier,              # type: ignore[arg-type]
        ocr_mode=req.ocr_mode,      # type: ignore[arg-type]
        page_range=page_range,
        image_analysis=req.image_analysis,
    )
    items = render_content_list(result.middle_json)
    images_written, image_uri_map = _materialize_images(items, out_dir)

    # ---- 落盘：content_list.json 是下游唯一消费的产物 ----
    content_list_path = out_dir / "content_list.json"
    content_list_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- markdown.md 供人工溯源（带页标记 <!-- page N of M -->）----
    # 刻意**不写** middle_json / model_output：它们对 Office 文档会把
    # 图片 base64 内联进去（实测 3 页就有 40KB），页数一多就是几十 MB，
    # 而本项目没有任何消费方需要它们。
    #
    # ★ 2026-09-17 补修：`result.markdown()` 自身**也会内联 base64** ——
    #   实测 3 页 PPT 的 markdown.md 达 37KB，其中一行 36416 字符就是那张
    #   柱状图的 data URI（即上面外置掉的那张）。所以这里必须用
    #   `_materialize_images` 返回的映射把 markdown 里的 URI 换成相对路径，
    #   否则"图片外置"只在外置了 content_list，markdown 仍是膨胀的。
    #   用**精确串替换**而不是正则/顺序匹配：映射来自原始 URI 本身，
    #   不存在"第几张图对应第几个占位"的脆弱假设。
    markdown_path: Path | None = None
    try:
        markdown_text = result.markdown(add_markers=True)
        for uri, rel in image_uri_map.items():
            if uri in markdown_text:
                markdown_text = markdown_text.replace(uri, rel)
        markdown_path = out_dir / "markdown.md"
        markdown_path.write_text(markdown_text, encoding="utf-8")
    except Exception:  # noqa: BLE001
        # markdown 只是辅助产物，失败不影响入库
        markdown_path = None

    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("type") or "unknown")
        counts[key] = counts.get(key, 0) + 1

    return {
        "status": "ok",
        "markdown_path": str(markdown_path) if markdown_path else None,
        "content_list_path": str(content_list_path),
        "image_dir": str(out_dir / "images"),
        "page_count": len(result.pages),
        "elapsed_sec": round(time.monotonic() - started, 2),
        "tier": req.tier,
        "counts": {**counts, "_images_written": images_written},
    }


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(
    title="mineru-service",
    version="1.0.0",
    description="MinerU 4.0 解析服务的 HTTP 包装（离线入库 Stage 1）",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """
    轻量存活检查。

    `gpu_lock_required` 直接暴露给运维看 —— 它是「这个容器会不会和 model-service
    抢显存」的唯一判据，配合启动日志里的 lock_factory 自检一起定位锁层级问题。

    `mineru_version` 必须暴露：`requirements.txt` 不锁版本，
    升级镜像时"到底装成了哪个大版本"是排查所有解析问题的第一手信息。
    （2.x → 4.0 的架构差异曾让整个 /parse 实现失效，就是靠它发现的。）
    """
    version = "unknown"
    try:
        from importlib.metadata import version as _pkg_version

        version = _pkg_version("mineru")
    except Exception:  # noqa: BLE001
        pass

    return {
        "status": "ok",
        "service": "mineru",
        "mineru_version": version,
        "device": MINERU_DEVICE,
        "gpu_lock_required": GPU_LOCK_REQUIRED,
        "gpu_lock_key": GPU_LOCK_KEY if GPU_LOCK_REQUIRED else None,
        "omp_num_threads": OMP_NUM_THREADS or "unset",
        "parsing": _parse_lock.locked(),
    }


@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest) -> ParseResponse:
    """
    解析文档并返回产物路径。

    ⚠️ 这是**长请求**：CPU 模式下几十页 PPT 解析 4-12 分钟是正常的。
       调用方（`mineru_client`）的超时必须相应放宽（config 里是 1800s），
       设成 60s 会表现成"随机解析失败"。
    """
    async with _parse_lock:
        try:
            async with _gpu_lock_if_needed():
                payload = await asyncio.to_thread(_parse_sync, req)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            # 堆栈必须进日志：MinerU 的内部异常对调用方几乎没有可读性，
            # 只把 str(exc) 透出去会让线上问题无从下手。
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    return ParseResponse(**payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001)
