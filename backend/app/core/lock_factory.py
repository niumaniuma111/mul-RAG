"""
锁层级工厂 —— 把「MinerU 跑在哪」与「锁管到谁」绑成同一个决策。

为什么必须有这个文件
--------------------
进程内 `asyncio.Lock` 的管辖范围**仅限本容器**，管不到 MinerU 容器的显存。所以：

    mineru.device = cpu   → MinerU 不占显存，锁只需护 model-service 内的 VLM ⇄ reranker  → inproc
    mineru.device = cuda  → MinerU 也占卡，必须跨容器互斥                               → redis

如果给人两个独立开关（device 一个、lock_backend 一个），就等于多给一次犯错机会，
而配错的代价是**随机 OOM** —— 这是最难查的一类故障：不是稳定复现，
只在「解析任务与问答恰好撞上」时才炸，日志里看不出来是谁占的显存。

所以：只认 `device` 一个输入，`lock_backend: auto` 是默认值；
显式配置与推导结果不符时**直接拒绝启动**（fail-fast）。
宁可起不来，也不要带着错的锁跑起来 —— 后者的排查成本远高于一次启动失败。

对应设计文档：DESIGN.md §10.2 ｜ HANDOFF.md 坑 3
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.core.config import get_config, resolve_mineru_device
from app.core.redis_lock import RedisGpuLock

logger = logging.getLogger(__name__)

# 合法值
_VALID_BACKENDS = {"auto", "inproc", "redis"}
_VALID_DEVICES = {"cpu", "cuda"}

# GPU 模式下 redis 锁 TTL 的**下限**（秒）。设计文档 §10.2 明确要求。
# ⚠️ 是下限、不是覆盖值：实际 TTL = max(config.lock_ttl_sec, _TTL_GPU_MODE)。
#    用 max 而非直接赋值，是为了让 lock_ttl_sec 这个配置项真的可调
#    （此前是硬编码，写进 config.yaml 也不生效），同时下限不可被误配破掉。
_TTL_GPU_MODE = 1800
# config.yaml 未写 lock_ttl_sec 时的兜底值
_TTL_DEFAULT = 600


class LockBackendMismatchError(RuntimeError):
    """
    锁层级与 MinerU 设备不匹配。

    启动期抛出，进程直接退出。修复方式永远是改 `ingest.mineru.device`
    （或把 `lock_backend` 改回 `auto`），而不是手动指定成另一个后端。
    """


@runtime_checkable
class GpuLockProtocol(Protocol):
    """锁的统一接口。model-service 只依赖这个协议，不关心背后是哪种实现。"""

    async def __aenter__(self) -> Any: ...
    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


class InProcGpuLock:
    """
    进程内 GPU 互斥锁（CPU 模式的锁层级）。

    它要保证的事情只有一件：VLM 与 reranker 不同时驻留。
    换出逻辑本身由 model_service.manager.ModelManager._make_room 负责，
    这把锁是它的并发安全外壳 —— 防止两个并发请求同时触发加载，
    导致两者各持一份权重、把显存打穿。

    注意：Python 3.10+ 的 asyncio.Lock 不在构造时绑定事件循环，
    而是在首次 await 时才绑定，所以可以在这里放心创建。
    """

    scope = "model-service 容器内"

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "InProcGpuLock":
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._lock.release()

    async def current_holder(self) -> str | None:
        """inproc 模式下没有跨进程可见的持有者，仅用于日志。"""
        return "inproc" if self._lock.locked() else None


@dataclass(frozen=True)
class LockPlan:
    """一次锁层级推导的完整结果，供启动日志与排障使用。"""

    backend: str          # inproc / redis
    device: str           # cpu / cuda
    configured: str       # 配置里写的原始值（auto / inproc / redis）
    ttl: int | None       # 仅 redis 模式有值
    key: str | None       # 仅 redis 模式有值
    reason: str           # 推导理由（一句话）

    def log_line(self) -> str:
        if self.backend == "redis":
            return (
                f"[lock_factory] mineru.device={self.device} → lock_backend=redis "
                f"(key={self.key}, ttl={self.ttl}s, 管辖范围=全部容器)"
            )
        return (
            f"[lock_factory] mineru.device={self.device} → lock_backend=inproc "
            f"(管辖范围=model-service 容器内, 仅护 VLM⇄reranker 互斥)"
        )

    def confirm_line(self) -> str:
        if self.backend == "redis":
            return "[lock_factory] ✓ 锁层级与 device 匹配：MinerU 占卡，跨容器互斥已生效"
        return (
            "[lock_factory] ✓ 锁层级与 device 匹配：MinerU 走 CPU 不占显存，"
            "无需跨容器锁"
        )


def resolve_lock_plan(
    device: str | None = None,
    configured: str | None = None,
) -> LockPlan:
    """
    推导锁层级。整个项目只有这一处做这件事。

    :param device: 显式指定 MinerU 设备；不传则按 config.yaml / .env 解析
    :param configured: 显式指定 lock_backend 配置值；不传则读 config.yaml

    :raises LockBackendMismatchError: configured 不是 auto，且与 device 推导结果冲突
    :raises ValueError: device 或 configured 取值非法
    """
    cfg = get_config()

    resolved_device = (device or resolve_mineru_device() or "cpu").strip().lower()
    if resolved_device not in _VALID_DEVICES:
        raise ValueError(
            f"ingest.mineru.device 取值非法: {resolved_device!r}，只允许 {sorted(_VALID_DEVICES)}"
        )

    resolved_configured = str(
        configured or cfg.get_path("models.manager.lock_backend", "auto")
    ).strip().lower()
    if resolved_configured not in _VALID_BACKENDS:
        raise ValueError(
            f"models.manager.lock_backend 取值非法: {resolved_configured!r}，"
            f"只允许 {sorted(_VALID_BACKENDS)}"
        )

    # ---- 推导 ----
    if resolved_configured == "auto":
        backend = "redis" if resolved_device == "cuda" else "inproc"
        reason = f"lock_backend=auto，按 device={resolved_device} 推导"
    else:
        backend = resolved_configured
        if backend != ("redis" if resolved_device == "cuda" else "inproc"):
            expected = "redis" if resolved_device == "cuda" else "inproc"
            raise LockBackendMismatchError(
                f"锁层级与 MinerU 设备不匹配：device={resolved_device} 要求 "
                f"lock_backend={expected}，但配置写的是 {backend}。\n"
                f"这会直接导致随机 OOM（MinerU 与 VLM 同时占卡时无人互斥）。\n"
                f"修复方式：把 models.manager.lock_backend 改回 'auto'，"
                f"或把 ingest.mineru.device 改成与锁层级一致的设备。"
            )
        reason = "lock_backend 显式指定，且与 device 推导一致"

    # ---- 组装 ----
    if backend == "redis":
        key = str(cfg.get_path("models.manager.gpu_lock.key", "lock:gpu:heavy"))
        # TTL：从 config.yaml 读（原来是硬编码，导致 lock_ttl_sec 成了死配置）。
        # 但 GPU 模式下有**下限**，不能只取配置值：
        #   解析是长任务（1~3 分钟起，大文档更久），TTL 若短于解析耗时，
        #   锁会在解析中途过期 → MinerU 与 VLM 同时占卡 → 随机 OOM。
        # 所以取 max(配置值, 1800)：配置可调，但下限不可破。
        base_ttl = int(
            cfg.get_path("models.manager.gpu_lock.lock_ttl_sec", _TTL_DEFAULT)
            or _TTL_DEFAULT
        )
        ttl = max(base_ttl, _TTL_GPU_MODE) if resolved_device == "cuda" else base_ttl
        return LockPlan(backend, resolved_device, resolved_configured, ttl, key, reason)

    return LockPlan(backend, resolved_device, resolved_configured, None, None, reason)


def build_gpu_lock(
    redis_client: Any | None = None,
    device: str | None = None,
    configured: str | None = None,
) -> GpuLockProtocol:
    """
    构造锁实例。调用点**不要**自己判断该用哪种锁，一律走这里。

    :param redis_client: redis.asyncio 客户端。redis 模式必传，inproc 模式忽略。
    """
    plan = resolve_lock_plan(device=device, configured=configured)

    if plan.backend == "inproc":
        return InProcGpuLock()

    if redis_client is None:
        raise ValueError(
            f"锁层级推导为 redis（device={plan.device}），但未传入 redis 客户端。"
            f"调用方需要在启动时建立 Redis 连接池并传进来。"
        )

    gpu_cfg = get_config().get_path("models.manager.gpu_lock", {}) or {}
    return RedisGpuLock(
        redis=redis_client,
        key=plan.key or "lock:gpu:heavy",
        ttl=plan.ttl or _TTL_DEFAULT,
        acquire_timeout=int(gpu_cfg.get("acquire_timeout_sec", 180) or 180),
        renew_interval=int(gpu_cfg.get("renew_interval_sec", 30) or 30),
    )


def preflight(device: str | None = None, configured: str | None = None) -> LockPlan:
    """
    启动期自检：解析锁层级 → 打日志 → 不匹配就抛异常。

    每个需要知道锁层级的容器（api / worker / model-service）都应在
    lifespan 启动阶段调用一次，这样启动日志里必然留下一条明确记录。
    S1 的验收标准之一就是这条日志。

    :raises LockBackendMismatchError: 配置冲突（进程应直接退出）
    """
    plan = resolve_lock_plan(device=device, configured=configured)

    # 用 print 而不是 logger：自检发生在 logging 初始化前后都可能，
    # 而这条信息必须在最原始的容器日志里可见（验收要看它）。
    print(plan.log_line(), flush=True)
    print(plan.confirm_line(), flush=True)
    logger.info(
        "锁层级自检通过: device=%s backend=%s ttl=%s reason=%s",
        plan.device,
        plan.backend,
        plan.ttl,
        plan.reason,
    )
    return plan


__all__ = [
    "GpuLockProtocol",
    "InProcGpuLock",
    "LockBackendMismatchError",
    "LockPlan",
    "build_gpu_lock",
    "preflight",
    "resolve_lock_plan",
]
