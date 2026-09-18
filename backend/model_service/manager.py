"""
显存驻留管理器（S2 完整实现）。

它要解决的问题
--------------
8GB 卡上装不下 VLM + reranker + embedder 三个模型同时驻留：

    VLM 生成态          ≈ 6.3GB   （权重 2.4 + 视觉激活 1.4~1.8 + KV cache + context）
    BGE-reranker-large  ≈ 1.3GB
    BGE-large-zh        ≈ 0.9GB
    ─────────────────────────────
    三者同驻            ≈ 8.5GB   → 必 OOM

所以必须做「按需加载 + 卸载式换出」，规则如下：

    keep_resident = ["embedding"]
        0.9GB，换出的代价（每次检索都要重载）大于收益，且它不与 VLM 触发互斥
        （实测运行态 B/D 证明两者共存无压力）。

    exclusive_pairs = [["vlm", "reranker"]]
        二者绝不共存。检索链路里 rerank → 生成是必经顺序，
        所以每轮问答会触发一次换出+换入，代价 6-10s（原生 ext4 下的实测口径）。

关键设计：卸载方式必须是**进程级**
----------------------------------
`del model + torch.cuda.empty_cache()` 是常见但**不够**的做法：

    它能回收的：权重张量占的显存、当前未使用的缓存块
    它回收不了的：CUDA context（0.6~0.9GB / 每进程）
                  cuBLAS / cuDNN 的 workspace

CUDA context 只在**进程退出**时才归还驱动。所以 long-running 的容器里反复
load/unload，每次都会留下一点碎片，最终以「显存明明没用满却 OOM」的形式爆炸。

本项目的做法（unload_mode: process）：模型跑在子进程里，卸载 = kill 子进程，
context 随之彻底归还。代价是下次加载多 1-3s 的进程冷启，完全值得。

两把锁，职责不同（不要合并）
--------------------------
    self._op_lock   进程内 asyncio.Lock。串行**状态变更**（加载/换出/自检），
                    防止两个并发请求同时触发加载 → 两份权重 → 打穿显存。
                    所有模型都要，与显存大小无关。

    self._gpu_lock  来自 lock_factory 的锁。护的是**显存竞争**：
                    inproc 模式 = 容器内 asyncio.Lock（护 VLM ⇄ reranker）
                    redis  模式 = 跨容器锁（MinerU 走 GPU 时它也来抢）
                    只对 _LOCKED_KEYS 生效，embedding 不持锁（见下）。

    获取顺序恒为 `gpu_lock → op_lock`。空闲回收线程只拿 op_lock，
    所以不可能与请求路径形成环。

    为什么 embedding 不持锁：它只有 0.9GB，与 VLM 共存实测无压力。
    不持锁的收益是离线入库的向量化阶段能与在线问答**真正并发** ——
    否则入库的 embedding 批次会把每次问答都堵在锁上，这在双流水线架构里
    是很明显的体验倒退（DESIGN.md §10.2 锁粒度表）。

对应设计文档：DESIGN.md §10.2 / §10.3
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue as _queue
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from app.core.config import get_config, get_env
from app.core.logging import get_logger

logger = get_logger(__name__)

# 保守的显存估算表（GB）。仅用于换出决策，不参与精确核算 ——
# 真实占用随输入长度浮动，所以 max_resident_gb 留了 1.4GB 缓冲。
_ESTIMATED_GB: dict[str, float] = {
    "vlm": 6.3,
    "reranker": 1.3,
    "embedding": 0.9,
}

# 允许的模型 key。config 段的段名与 key 同名（models.vlm / models.embedding / ...），
# 这样 _spec_for 不需要维护一张映射表。
_MODEL_KEYS: tuple[str, ...] = ("embedding", "reranker", "vlm")

# 推理期间必须持有 GPU 锁的模型；embedding 刻意不在其中（理由见模块头）。
_LOCKED_KEYS: frozenset[str] = frozenset({"vlm", "reranker"})

# 默认超时（秒）。会被 configs/config.yaml 的 models.manager.timeouts 覆盖 ——
# 放在这里只是兜底，业务代码里不出现裸数字。
_DEFAULT_TIMEOUTS: dict[str, int] = {
    "load": 180,
    "embed": 120,
    "rerank": 120,
    "caption": 300,
    "generate": 900,
}

# 模型 key → 默认推理超时的档位。vlm 默认按 caption 算，
# S3 的流式生成会显式传 op="generate"（1024 token 慢得多，需要更长超时）。
_TIMEOUT_SLOT_BY_MODEL: dict[str, str] = {
    "embedding": "embed",
    "reranker": "rerank",
    "vlm": "caption",
}


# ============================================================
# 异常
# ============================================================
class ModelLoadError(RuntimeError):
    """
    模型加载失败（权重缺失、显存不足、依赖导入错误……）。

    ⚠️ 与「推理失败」分开成两个类，是因为两者的处置完全不同：
       加载失败 → 环境问题，重试大概率还是失败，但要给用户明确原因
       推理失败 → 单次输入问题（如坏图），重试或跳过即可
    """


class ModelInferenceError(RuntimeError):
    """单次推理失败。"""


class WorkerDeadError(ModelInferenceError):
    """子进程已死（被 OOM killer 干掉、或崩溃）。下次 acquire 会自动重载。"""


# ============================================================
# 子进程句柄
# ============================================================
class WorkerHandle:
    """
    一个模型子进程的句柄：进程 + 双向队列 + 调用串行锁。

    为什么句柄自己还需要一把锁
    --------------------------
    子进程的主循环是**单线程取队列**的，两条请求同时进队列时后到的会排队，
    但**响应队列是共享的** —— 两个协程同时 get() 会互相抢走对方的响应。
    所以句柄级串行化是必须的，不是优化。
    这与 GPU 只能串行推理的事实一致，不构成额外损失。

    ⚠️ 队列操作是**阻塞**的（multiprocessing.Queue 没有异步接口），
       统一用 asyncio.to_thread 包一层，避免卡死事件循环。
       这不是可省的开销：一次 VLM caption 要 3-10s，
       在事件循环里同步等它会让 /health 都失去响应。
    """

    __slots__ = ("key", "proc", "req_q", "resp_q", "loaded_at", "_call_lock", "_closed")

    def __init__(self, key: str, proc: Any, req_q: Any, resp_q: Any) -> None:
        self.key = key
        self.proc = proc
        self.req_q = req_q
        self.resp_q = resp_q
        self.loaded_at: float = time.monotonic()
        self._call_lock = asyncio.Lock()
        self._closed = False

    # ---------------- 状态 ----------------
    @property
    def pid(self) -> int | None:
        """
        ⚠️ 必须容错：`multiprocessing.Process.close()` 之后再访问 `.pid` 会抛
           `ValueError("process object is closed")` —— CPython 的实现里
           `pid` / `is_alive` / `terminate` / `join` 都会先走 `_check_closed()`。
           本项目在 shutdown() 里会 close()，所以任何"关闭后还想看一眼 pid"
           的代码（日志、/health 快照）都必须走这里，不能直接摸 self.proc.pid。
           （2026-09-17 实测踩到：/admin/unload 明明卸成功了却返回 500，
             原因就是关闭后打日志取 pid。）
        """
        try:
            return self.proc.pid
        except (AssertionError, ValueError):
            return None

    @property
    def uptime_sec(self) -> float | None:
        return round(time.monotonic() - self.loaded_at, 1)

    def is_alive(self) -> bool:
        try:
            return bool(self.proc.is_alive())
        except (AssertionError, ValueError):
            # 进程对象已 close()，或进程尚未 start()
            return False

    # ---------------- 调用 ----------------
    def _call_blocking(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """
        投一条请求并等它的响应。**必须在工作线程里调用。**

        收到 `rid is None` 的响应表示子进程的通知消息（如 loaded / 加载失败），
        跳过继续等 —— 这些消息在 acquire 阶段已消费掉，这里只是兜底。
        """
        if self._closed:
            raise WorkerDeadError(f"子进程句柄已关闭: {self.key}")

        request_id = f"{self.key}-{time.monotonic_ns()}"
        try:
            self.req_q.put({"rid": request_id, "payload": payload})
        except (OSError, ValueError) as exc:
            raise WorkerDeadError(
                f"向 {self.key} 子进程投递请求失败（管道已断）: {exc}"
            ) from exc

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelInferenceError(
                    f"{self.key} 推理超时（>{timeout:.0f}s）。"
                    f"若持续发生，检查 models.manager.timeouts 是否过小。"
                )
            try:
                message = self.resp_q.get(block=True, timeout=remaining)
            except _queue.Empty as exc:
                raise ModelInferenceError(
                    f"{self.key} 推理超时（>{timeout:.0f}s）"
                ) from exc
            except (OSError, ValueError) as exc:
                raise WorkerDeadError(f"{self.key} 子进程管道异常: {exc}") from exc

            if not isinstance(message, dict):
                logger.warning("%s 子进程返回了非 dict 消息: %r", self.key, message)
                continue
            if message.get("rid") != request_id:
                # 通知消息或别人的响应 —— 单串行化下不该出现"别人的响应"
                logger.debug(
                    "%s 跳过一条无关消息: event=%s rid=%s",
                    self.key, message.get("event"), message.get("rid"),
                )
                continue
            if not message.get("ok"):
                raise ModelInferenceError(
                    f"{self.key} 推理失败: {message.get('error') or '未知原因'}"
                )
            return dict(message.get("data") or {})

    async def acall(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """投一条推理请求。同一模型的并发调用在此串行化。"""
        async with self._call_lock:
            if not self.is_alive():
                raise WorkerDeadError(
                    f"{self.key} 子进程已退出（pid={self.proc.pid}）。"
                    f"常见原因：容器内存超限被 OOM killer 终止、或模型加载后崩溃。"
                    f"下一次请求会自动重载。"
                )
            return await asyncio.to_thread(self._call_blocking, payload, timeout)

    # ---------------- 关闭 ----------------
    def shutdown(self, grace_sec: float = 5.0) -> None:
        """
        终止子进程并释放队列。**进程退出的这一刻，显存才真正归还驱动** ——
        含 CUDA context、cuBLAS/cuDNN workspace 与全部碎片。
        """
        if self._closed:
            return
        self._closed = True

        if self.is_alive():
            self.proc.terminate()          # SIGTERM
            self.proc.join(timeout=grace_sec)
            if self.is_alive():
                # 子进程卡在 CUDA 调用里时 SIGTERM 可能不生效
                logger.warning("%s 子进程未响应 SIGTERM，改用 SIGKILL", self.key)
                self.proc.kill()
                self.proc.join(timeout=grace_sec)

        # 关队列：不关会泄漏 pipe fd 与 feeder 线程。
        # cancel_join_thread 让父进程不必等子进程把缓冲刷完（子进程已死，等不到）。
        for name, q in (("req_q", self.req_q), ("resp_q", self.resp_q)):
            try:
                q.close()
                q.cancel_join_thread()
            except (OSError, ValueError, AttributeError) as exc:
                logger.debug("关闭 %s.%s 时忽略异常: %s", self.key, name, exc)

        try:
            self.proc.close()
        except (OSError, ValueError) as exc:
            logger.debug("proc.close() 忽略异常: %s", exc)


# ============================================================
# 管理器
# ============================================================
class ModelManager:
    """
    管理 VLM / embedder / reranker 的显存驻留。

    对外只有一个主入口 `infer()`（一次性调用）和一个 `acquire()`
    （需要连续多次调用的场景，如 S3 的流式生成）。
    """

    def __init__(self, gpu_lock: Any | None = None) -> None:
        manager_cfg = get_config().get_path("models.manager", {}) or {}

        self.mode: str = str(manager_cfg.get("mode", "lru_evict"))
        self.max_resident_gb: float = float(manager_cfg.get("max_resident_gb", 5.8))
        self.sys_overhead_gb: float = float(manager_cfg.get("sys_overhead_gb", 1.4))
        self.keep_resident: set[str] = set(manager_cfg.get("keep_resident", ["embedding"]) or [])
        self.evict_lru_order: list[str] = list(
            manager_cfg.get("evict_lru_order", ["vlm", "reranker"]) or []
        )
        self.exclusive_pairs: list[list[str]] = [
            [str(item) for item in pair]
            for pair in (manager_cfg.get("exclusive_pairs", [["vlm", "reranker"]]) or [])
        ]
        # ★ v1.1.3：process = 子进程 kill（真回收 CUDA context）
        self.unload_mode: str = str(manager_cfg.get("unload_mode", "process"))
        self.idle_ttl_sec: int = int(manager_cfg.get("idle_ttl_sec", 180))
        self.reaper_interval_sec: int = int(manager_cfg.get("reaper_interval_sec", 30))
        self.load_timeout_sec: float = float(manager_cfg.get("load_timeout_sec", 180))

        configured_timeouts = manager_cfg.get("timeouts", {}) or {}
        self.timeouts: dict[str, float] = {
            slot: float(configured_timeouts.get(slot, default))
            for slot, default in _DEFAULT_TIMEOUTS.items()
        }

        # 两把锁，职责不同（详见模块头）
        self._op_lock = asyncio.Lock()
        self._gpu_lock = gpu_lock

        self._residents: dict[str, WorkerHandle] = {}
        self._last_used: dict[str, float] = {}
        # 正在被使用的计数。空闲回收线程据此跳过"正在服务中"的模型 ——
        # 否则一次 10 分钟的大文档向量化可能被回收线程在中间卸掉。
        self._inflight: dict[str, int] = {}

        self._reaper_task: asyncio.Task[None] | None = None
        self._closing = False

        logger.info(
            "ModelManager 初始化: mode=%s max_resident=%sGB keep=%s "
            "exclusive=%s unload_mode=%s idle_ttl=%ss",
            self.mode,
            self.max_resident_gb,
            sorted(self.keep_resident),
            self.exclusive_pairs,
            self.unload_mode,
            self.idle_ttl_sec,
        )

    # ============================================================
    # 账目（纯逻辑，可单测）
    # ============================================================
    def current_gb(self) -> float:
        """当前已驻留模型的估算总占用。"""
        return sum(_ESTIMATED_GB.get(key, 0.0) for key in self._residents)

    def is_resident(self, key: str) -> bool:
        return key in self._residents

    def residents(self) -> list[str]:
        return sorted(self._residents)

    def _estimate(self, key: str) -> float:
        return _ESTIMATED_GB.get(key, 0.0)

    def _budget_key(self, incoming: str) -> list[str]:
        """
        按 `evict_lru_order` 算出预算口径下需要换出的模型。

        ⚠️ 这里**不**再减 sys_overhead_gb。
           DESIGN.md §10.3 的伪码写作
           `current + need - sys_overhead_gb > max_resident_gb`，
           但 config.yaml 的注释已说明 max_resident_gb=5.8 是
           「可用 7.2GB − 1.4GB 缓冲」**已经扣过**的净值。
           再减一次等于把缓冲扣两遍（5.8 → 4.4），会让 VLM(6.3) 永远"超预算"，
           使这个阈值退化成噪声。故以 config 的语义为准：max_resident_gb 是净阈值。
           （已记录在 HANDOFF.md，作为与设计文档的一处已知偏差。）
        """
        victims: list[str] = []
        protected = self.keep_resident | {incoming}
        projected = self.current_gb() + self._estimate(incoming)

        for key in self.evict_lru_order:
            if projected <= self.max_resident_gb:
                break
            if key in protected or key not in self._residents:
                continue
            victims.append(key)
            projected -= self._estimate(key)

        # 兜底：evict_lru_order 没覆盖到的 key（未来新增模型时容易漏配），
        # 按 LRU 顺序继续换出，直到回到预算内。不加这一步，
        # 一个新模型一旦不在 evict_lru_order 里就永远换不出去 —— 这是隐蔽的 OOM 源。
        if projected > self.max_resident_gb:
            leftovers = sorted(
                (k for k in self._residents if k not in protected and k not in victims),
                key=lambda k: self._last_used.get(k, 0.0),
            )
            for key in leftovers:
                if projected <= self.max_resident_gb:
                    break
                victims.append(key)
                projected -= self._estimate(key)

        return victims

    # ============================================================
    # 生命周期
    # ============================================================
    async def start(self, warmup: list[str] | None = None) -> None:
        """
        启动空闲回收线程，并按需预热 embedding。

        预热放在**后台任务**里而不是 await 到底，理由：
          embedding 权重 1.3GB，冷启要 2-3s（读盘 + torch 初始化更久）；
          若阻塞 lifespan，容器在这期间 /health 不可达，
          healthcheck 会一直等到 start_period 结束。
          后台加载则让服务立刻可用，第一次真正用到 embedding 时再等它 ——
          对入库流水线来说，那条路径本来就有 UI 进度条。
        """
        self._reaper_task = asyncio.create_task(self._reaper_loop(), name="model-reaper")

        for key in warmup or []:
            if key not in _MODEL_KEYS:
                logger.warning("预热配置里有未知模型 key，已跳过: %s", key)
                continue
            asyncio.create_task(self._warmup(key), name=f"model-warmup-{key}")

    async def shutdown(self) -> None:
        """停回收线程 + 卸掉所有子进程。容器退出前必须走到这里。"""
        self._closing = True

        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None

        async with self._op_lock:
            for key in list(self._residents):
                await self._unload(key, reason="shutdown")

        logger.info("ModelManager 已停止，全部子进程已回收")

    async def _warmup(self, key: str) -> None:
        """尽力而为的预热：失败只记日志，不影响服务可用性。"""
        try:
            await self.prefetch(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("预热 %s 失败（不影响服务启动）: %s", key, exc)

    async def prefetch(self, key: str) -> None:
        """异步预加载（不跑推理）。用于把加载耗时与用户请求重叠。"""
        async with self.acquire(key):
            pass

    # ---------------- 取用 ----------------
    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[WorkerHandle]:
        """
        取模型句柄：需要时加载、必要时先换出。

        heavy 模型（vlm / reranker）在整个使用期间持有 GPU 锁；
        embedding 不持锁（理由见模块头）。
        """
        if key not in _MODEL_KEYS:
            raise ValueError(f"未知的模型 key: {key!r}，只允许 {list(_MODEL_KEYS)}")

        locked = key in _LOCKED_KEYS and self._gpu_lock is not None

        if locked:
            # GpuBusyError 由锁实现抛出（等待超时），交给上层转 HTTP 503
            await self._gpu_lock.__aenter__()
        try:
            async with self._op_lock:
                if not self.is_resident(key):
                    await self._make_room(key)
                    await self._load(key)
                self._last_used[key] = time.monotonic()
                self._inflight[key] = self._inflight.get(key, 0) + 1

            try:
                yield self._residents[key]
            finally:
                self._inflight[key] = max(self._inflight.get(key, 1) - 1, 0)
                self._last_used[key] = time.monotonic()
        finally:
            if locked:
                await self._gpu_lock.__aexit__(None, None, None)

    def _timeout_for(self, key: str, op: str | None) -> float:
        slot = op or _TIMEOUT_SLOT_BY_MODEL.get(key, "embed")
        return float(self.timeouts.get(slot, _DEFAULT_TIMEOUTS.get(slot, 120)))

    async def infer(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        op: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        一次性推理：取模型 → 调一次 → 归还。

        这是 api/worker 侧的唯一常用入口（经由 HTTP 端点转发到这里）。
        """
        resolved_timeout = float(timeout or self._timeout_for(key, op))
        async with self.acquire(key) as handle:
            return await handle.acall(payload, resolved_timeout)

    async def infer_stream(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        op: str | None = None,
        timeout: float | None = None,
        idle_timeout: float = 60.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        流式推理：yield `{"event": "delta", "text": 片段}` 若干次，
        最后 yield `{"event": "done", "text": 全文}`。

        与 infer 的三点差异（都来自"一条请求多条响应"）：
          1. **响应循环在这里而不是 WorkerHandle 里** —— acall 的
             "投一条等一条"模型装不下 delta 流，故直接操作句柄的队列；
          2. **整个流式期间持有 `_call_lock` + GPU 锁** —— 流式生成
             可长达数十秒，此间同模型的其它请求会排队（与 GPU 串行一致）；
          3. **双超时**：首 token 用整体 timeout（含冷启动）；
             之后每收到一个 delta 就把截止线重置为 now+idle_timeout，
             避免"模型还在稳定吐字"却被总时长误杀，也避免"生成挂死"
             永远不超时。

        客户端提前断开时本生成器被 GC/aclose，锁随 acquire 的 finally 释放，
        子进程会把剩余 token 推进无人读取的队列（无害，句柄关闭时一起回收）。
        """
        import _queue as _std_queue

        resolved_timeout = float(timeout or self._timeout_for(key, op))
        async with self.acquire(key) as handle:
            async with handle._call_lock:
                if not handle.is_alive():
                    raise WorkerDeadError(
                        f"{self.key if hasattr(self, 'key') else key} 子进程已退出"
                    )
                request_id = f"{handle.key}-{time.monotonic_ns()}"
                try:
                    await asyncio.to_thread(
                        handle.req_q.put, {"rid": request_id, "payload": payload}
                    )
                except (OSError, ValueError) as exc:
                    raise WorkerDeadError(
                        f"向 {handle.key} 子进程投递流式请求失败: {exc}"
                    ) from exc

                deadline = time.monotonic() + resolved_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelInferenceError(
                            f"{handle.key} 流式推理超时"
                            f"（{'首 token' if deadline <= time.monotonic() + resolved_timeout else '空闲'}"
                            f" >{resolved_timeout if deadline <= time.monotonic() + resolved_timeout else idle_timeout:.0f}s）"
                        )
                    try:
                        message = await asyncio.to_thread(
                            handle.resp_q.get, True, remaining
                        )
                    except _std_queue.Empty as exc:
                        raise ModelInferenceError(
                            f"{handle.key} 流式推理超时（>{remaining:.0f}s 无新 token）"
                        ) from exc
                    except (OSError, ValueError) as exc:
                        raise WorkerDeadError(
                            f"{handle.key} 子进程管道异常: {exc}"
                        ) from exc

                    if not isinstance(message, dict):
                        continue
                    if message.get("rid") != request_id:
                        continue   # 通知消息（loaded 等），跳过
                    if not message.get("ok"):
                        raise ModelInferenceError(
                            f"{handle.key} 流式推理失败: {message.get('error') or '未知原因'}"
                        )

                    event = str(message.get("event") or "")
                    data = dict(message.get("data") or {})

                    if event == "delta":
                        # 有新 token → 重置空闲截止线
                        deadline = time.monotonic() + idle_timeout
                        yield {"event": "delta", "text": str(data.get("text") or "")}
                    elif event == "done":
                        yield {"event": "done", "text": str(data.get("text") or "")}
                        return
                    else:
                        # 兼容：非流式响应（ 子进程旧版 / 未走 stream 分支 ）
                        yield {"event": "done", "text": str(data.get("text") or "")}
                        return

    # ---------------- 加载 / 卸载 ----------------
    async def _make_room(self, incoming: str) -> None:
        """
        为 incoming 腾出显存。调用方必须已持有 `_op_lock`。

        两步：
          1. exclusive_pairs 强制互斥（vlm ⇄ reranker 绝不共存）——
             这一步是**语义约束**，与预算无关：即使显存够也不允许并存，
             因为并存会让 KV cache 没有余量，随机 OOM。
          2. 预算口径换出（`_budget_key`）。
        """
        # 1) 互斥对
        for pair in self.exclusive_pairs:
            if incoming not in pair:
                continue
            for partner in pair:
                if partner != incoming and partner in self._residents:
                    await self._unload(partner, reason=f"与 {incoming} 互斥（exclusive_pairs）")

        # 2) 预算
        for victim in self._budget_key(incoming):
            if victim in self._residents:
                await self._unload(victim, reason="超出 max_resident_gb")

        projected = self.current_gb() + self._estimate(incoming)
        if projected > self.max_resident_gb:
            # 不是异常：max_resident_gb 是**策略阈值**，不是物理上限。
            # 单一模型（VLM 6.3GB）+ 常驻 embedding 本身就超过 5.8，
            # 而这是设计上允许的组合（实测运行态 B/D 成立）。
            logger.info(
                "装下 %s 后预计驻留 %.1fGB，高于策略阈值 %.1fGB"
                "（incoming + keep_resident 的固有占用，按物理显存判定可容纳）",
                incoming, projected, self.max_resident_gb,
            )

    def _spec_for(self, key: str) -> dict[str, Any]:
        """
        组装传给子进程的模型 spec。

        ⚠️ 必须是**纯 dict / list / 基本类型**：
           spawn 出来的子进程用 pickle 反序列化它，若塞进 DotDict
           会连带要求子进程 import app.core.config，白拉一堆依赖。
        """
        section = get_config().get_path(f"models.{key}", {}) or {}
        spec = _to_plain(section)
        if not isinstance(spec, dict):
            raise ModelLoadError(f"models.{key} 配置段格式非法: {type(spec).__name__}")

        spec["key"] = key
        spec["path"] = _resolve_model_path(key, spec)
        return spec

    async def _load(self, key: str) -> WorkerHandle:
        """
        起一个模型子进程并等它就绪。调用方必须已持有 `_op_lock`。

        进程启动方式用 **spawn 而不是 fork**：
          fork 在「多线程父进程」里有已知的 fork-safety 陷阱 ——
          子进程会继承父进程在 fork 瞬间的锁状态（例如 logging 的 handler 锁、
          Queue 的 feeder 线程锁），一旦那把锁当时被别的线程持有，
          子进程可能永久阻塞。而 model-service 主进程就是多线程的
          （uvicorn 事件循环 + asyncio.to_thread 的线程池 + Queue feeder 线程），
          正好落在这个陷阱里。
          代价是 spawn 多 0.5-2s（子进程要重新 import torch）。
          相对于 bitsandbytes 反量化初始化的 5-8s，这个代价可以接受；
          而换来的是「不会随机挂死在加载阶段」——
          那类故障极难定位。（DESIGN.md §10.3 写的是 fork，
          此处按工程安全性取 spawn，属已知偏差。）
        """
        if self.unload_mode != "process":
            raise ModelLoadError(
                f"unload_mode={self.unload_mode} 当前不支持："
                f"inproc 模式无法回收 CUDA context（详见 DESIGN.md §10.3），"
                f"本项目固定使用 'process'。"
            )

        # 残留的死亡句柄先清掉，避免 pid 复用造成误判
        stale = self._residents.get(key)
        if stale is not None:
            if stale.is_alive():
                return stale
            logger.warning("%s 的旧子进程已死亡，先回收再重载", key)
            await self._unload(key, reason="stale")

        spec = self._spec_for(key)
        path = str(spec.get("path") or "")
        logger.info("加载模型 %s: path=%s", key, path)

        # 延迟导入：worker_main 只在子进程里执行，但父进程 import 它是为了
        # 把函数引用交给 spawn（spawn 靠「模块名 + 函数名」重建引用）。
        # 该模块顶层不 import torch，所以父进程不会被拖入 CUDA。
        from model_service.workers import worker_main

        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()
        process = ctx.Process(
            target=worker_main,
            args=(key, spec, request_queue, response_queue),
            name=f"model-worker-{key}",
            daemon=True,   # 父进程退出时一并回收，避免容器停不掉
        )

        started_at = time.monotonic()
        process.start()
        handle = WorkerHandle(key, process, request_queue, response_queue)

        try:
            ready = await asyncio.to_thread(
                response_queue.get, True, float(self.load_timeout_sec)
            )
        except _queue.Empty as exc:
            handle.shutdown()
            raise ModelLoadError(
                f"{key} 加载超时（>{self.load_timeout_sec:.0f}s）。"
                f"常见原因：权重文件不完整（还在下载）、磁盘 IO 慢、"
                f"或 models.manager.load_timeout_sec 过小。path={path}"
            ) from exc
        except (OSError, ValueError) as exc:
            handle.shutdown()
            raise ModelLoadError(f"{key} 加载时队列异常: {exc}") from exc

        if not isinstance(ready, dict) or not ready.get("ok"):
            detail = (ready or {}).get("error") if isinstance(ready, dict) else repr(ready)
            handle.shutdown()
            raise ModelLoadError(f"{key} 加载失败: {detail}")

        self._residents[key] = handle
        self._last_used[key] = time.monotonic()
        self._inflight.setdefault(key, 0)

        logger.info(
            "模型 %s 就绪: pid=%s 耗时=%.1fs 当前驻留=%s(%.1fGB)",
            key, process.pid, time.monotonic() - started_at,
            self.residents(), self.current_gb(),
        )
        return handle

    async def _unload(self, key: str, reason: str = "") -> None:
        """
        卸载模型。调用方必须已持有 `_op_lock`。

        unload_mode=process 时这是**进程级 kill** —— 显存含 CUDA context
        与碎片一并归还，见模块头。
        """
        handle = self._residents.pop(key, None)
        self._last_used.pop(key, None)
        self._inflight.pop(key, None)
        if handle is None:
            return

        # ★ 必须在 shutdown() 之前取：shutdown 会 close() 掉进程对象，
        #   之后读 pid 会抛 ValueError（见 WorkerHandle.pid 的说明）。
        pid = handle.pid
        started_at = time.monotonic()
        await asyncio.to_thread(handle.shutdown)
        logger.info(
            "卸载模型 %s: pid=%s reason=%s 耗时=%.2fs 剩余驻留=%s(%.1fGB)",
            key, pid, reason or "-",
            time.monotonic() - started_at, self.residents(), self.current_gb(),
        )

    async def unload(self, key: str, reason: str = "manual") -> bool:
        """对外的手动换出（/admin/unload 用）。返回是否真的卸了东西。"""
        async with self._op_lock:
            existed = key in self._residents
            await self._unload(key, reason=reason)
            return existed

    async def unload_all(self, reason: str = "manual") -> list[str]:
        """卸掉所有模型（含 keep_resident）。排障用。"""
        async with self._op_lock:
            keys = list(self._residents)
            for key in keys:
                await self._unload(key, reason=reason)
            return keys

    # ---------------- 空闲回收 ----------------
    async def _reaper_loop(self) -> None:
        """
        定期卸载空闲超时的模型，把显存尽快还给系统。

        只拿 `_op_lock`、不拿 `_gpu_lock`：换出不消耗显存，
        没理由为了它去等 MinerU 的解析锁（GPU 模式下那个锁一持有就是几分钟）。
        获取顺序上也不会成环（请求路径是 gpu → op，这里只有 op）。
        """
        logger.info(
            "空闲回收线程启动: interval=%ss ttl=%ss",
            self.reaper_interval_sec, self.idle_ttl_sec,
        )
        while not self._closing:
            try:
                await asyncio.sleep(self.reaper_interval_sec)
                await self._reap_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # 回收线程绝不能死：它挂了就不会再释放显存，最终 OOM。
                logger.exception("空闲回收出错（已忽略，下一轮继续）: %s", exc)

    async def _reap_once(self) -> None:
        now = time.monotonic()
        async with self._op_lock:
            for key in list(self._residents):
                if key in self.keep_resident:
                    continue
                if self._inflight.get(key, 0) > 0:
                    continue          # 正在服务中，不能动
                idle = now - self._last_used.get(key, now)
                if idle < self.idle_ttl_sec:
                    continue
                await self._unload(key, reason=f"空闲 {idle:.0f}s > {self.idle_ttl_sec}s")

    # ---------------- 观测 ----------------
    def snapshot(self) -> dict[str, Any]:
        """
        /health 用的状态快照。排障时能一眼看出谁驻留、谁的子进程死了。

        ⚠️ 这里对 handle 一律走 getattr 兜底，不用直接属性访问：
           /health 是 compose 的 healthcheck 探针，它一旦抛异常，
           容器会被判定为 unhealthy → restart 策略触发 → 反复重启，
           把一个本来只是"某个句柄状态异常"的小问题放大成服务不可用。
           健康端点必须永远能返回。
        """
        now = time.monotonic()
        detail: dict[str, Any] = {}
        for key in _MODEL_KEYS:
            handle = self._residents.get(key)
            last_used = self._last_used.get(key)
            alive: bool | None = None
            pid: int | None = None
            uptime: float | None = None
            if handle is not None:
                # 注意：不能靠 getattr(obj, "pid", None) 兜底 ——
                # getattr 只吞 AttributeError，而 close() 之后访问 pid
                # 抛的是 ValueError，会直接穿透出去把 /health 打挂。
                try:
                    alive = bool(handle.is_alive())
                    pid = handle.pid
                    uptime = handle.uptime_sec
                except Exception:  # noqa: BLE001 —— 健康端点必须永远能返回
                    alive, pid, uptime = None, None, None
            detail[key] = {
                "resident": handle is not None,
                "alive": alive,
                "pid": pid,
                "uptime_sec": uptime,
                "inflight": self._inflight.get(key, 0),
                "idle_sec": round(now - last_used, 1) if last_used else None,
                "estimated_gb": self._estimate(key),
            }

        return {
            "residents": self.residents(),
            "resident_gb": round(self.current_gb(), 2),
            "max_resident_gb": self.max_resident_gb,
            "keep_resident": sorted(self.keep_resident),
            "exclusive_pairs": [list(pair) for pair in self.exclusive_pairs],
            "unload_mode": self.unload_mode,
            "idle_ttl_sec": self.idle_ttl_sec,
            "models": detail,
        }


# ============================================================
# 辅助
# ============================================================
def _to_plain(value: Any) -> Any:
    """DotDict（dict 子类）/ 列表 → 纯内建类型，保证可被 spawn 安全 pickle。"""
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _assert_weights_ready(key: str, directory: str) -> None:
    """
    确认权重目录里真的有**完整**权重文件。

    为什么值得单独做这一步（而不是让 transformers 自己报错）：
      下载中的目录会长这样 ——
          model-00001-of-00002.safetensors.incomplete
          model-00002-of-00002.safetensors.incomplete
      此时目录"存在"，但 transformers 报的是 index 与分片不匹配之类
      绕来绕去的错误，看不出真实原因是"还没下完"。
      这里提前把状态翻译成一句人话，S2 阶段（权重边下边测）尤其有用。

    判据与 scripts/download_models.py 的 check_models 保持一致：
      存在 .safetensors / .bin / .pt 文件即视为就绪（.incomplete 不算）。
    """
    root = Path(directory)
    weights = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".bin", ".pt"}
    ]
    if weights:
        return

    incomplete = [p.name for p in root.rglob("*.incomplete")]
    if incomplete:
        raise ModelLoadError(
            f"{key} 权重仍在下载中（{len(incomplete)} 个未完成分片："
            f"{', '.join(sorted(incomplete)[:3])}…）。等下载完成再试，"
            f"或调 scripts/download_models.py --check 看进度。目录：{directory}"
        )
    raise ModelLoadError(
        f"{key} 权重目录里没有任何权重文件（.safetensors/.bin/.pt）：{directory}。"
        f"请先跑 scripts/download_models.py。"
    )


def _resolve_model_path(key: str, spec: dict[str, Any]) -> str:
    """
    解析权重目录的绝对路径。

    两级回退：
      1. config 里的 path（容器内是 /models/<name>）—— 存在就用它
      2. `$MODELS_ROOT`（.env 注入，默认 /models）+ 子目录名
         （子目录名取 config 的 name，退化时取 path 的末段）

    第 2 级不是冗余：本地开发时 config 里的 /models 并不存在，
    而权重挂在宿主的 E:\\Mutilmodel-RAG\\models。没有这一级，
    model-service 的端点只能在容器里跑，本地没法调试。
    """
    raw = str(spec.get("path") or "").strip()
    if raw and Path(raw).is_dir():
        _assert_weights_ready(key, raw)
        return raw

    root = Path(get_env().models_root)
    # 去重：config 的 name 与 path 末段通常是同一个字符串
    # （bge-large-zh-v1.5 / bge-large-zh-v1.5），
    # 不去重的话报错信息里会出现三遍同样的路径，看起来像 bug。
    candidates: list[str] = []
    for candidate in (str(spec.get("name") or "").strip(), Path(raw).name if raw else ""):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            _assert_weights_ready(key, str(path))
            return str(path)

    tried: list[str] = []
    for item in [raw, *[str(root / c) for c in candidates]]:
        if item and item not in tried:
            tried.append(item)
    raise ModelLoadError(
        f"找不到 {key} 的权重目录。已尝试：{tried}。"
        f"请确认 ./models 已挂载且权重已下载（scripts/download_models.py）。"
    )


__all__ = [
    "ModelInferenceError",
    "ModelLoadError",
    "ModelManager",
    "WorkerDeadError",
    "WorkerHandle",
]
