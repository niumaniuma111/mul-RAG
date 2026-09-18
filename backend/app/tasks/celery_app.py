"""
Celery 实例与全局配置。

S1 阶段只定义实例 + 配置，**不注册任何任务** —— 任务链在 S2 落地。
这样 worker 容器可以正常启动、连上 broker，让「基础设施能跑通」这条验收
在写业务代码之前就成立。

几个参数是针对本机约束特意调的（来自 configs/config.yaml 的 celery 段），
不是照抄默认值：

    worker_concurrency = 1
        单并发。worker 本身不持有模型，串行由 model-service 内部保证；
        开多并发只会让多个任务同时去抢 model-service，排队更长且更易 OOM。

    worker_prefetch_multiplier = 1
        不预取。默认值 4 会让 worker 提前把 4 个任务拉进内存，
        在 16GB 机器上等于白占几百 MB —— 而本项目任务本身就要吃几个 GB。

    task_soft_time_limit = 2400s
        MinerU 走 CPU 时，一份 50 页 PPT 解析要 4-12 分钟，加上 caption 与
        embedding，单任务 40 分钟的上限是必要的。默认值会误杀。
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_config, get_env

_cfg = get_config()
_env = get_env()
_celery_cfg = _cfg.get_path("celery", {}) or {}

celery_app = Celery(
    "ppt_agentic_rag",
    broker=_env.celery_broker_url,
    backend=_env.celery_result_backend,
    # ★ 用 include 声明任务模块，而不是在文件末尾 import。
    #   任务模块需要 `from app.tasks.celery_app import celery_app`，
    #   若在末尾 import 就会形成循环导入（虽然能侥幸跑通，但一旦
    #   任务模块的导入顺序变化就会出现"任务未注册"的诡异问题）。
    include=["app.tasks.ingest_tasks"],
)

celery_app.conf.update(
    # ---- 序列化 ----
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # ---- 时区 ----
    timezone=_env.tz,
    enable_utc=False,
    # ---- 超时与内存 ----
    task_soft_time_limit=int(_celery_cfg.get("task_soft_time_limit", 2400)),
    task_time_limit=int(_celery_cfg.get("task_time_limit", 3000)),
    worker_max_tasks_per_child=int(_celery_cfg.get("worker_max_tasks_per_child", 20)),
    # ---- 并发 ----
    worker_concurrency=int(_celery_cfg.get("worker_concurrency", 1)),
    task_acks_late=bool(_celery_cfg.get("task_acks_late", True)),
    worker_prefetch_multiplier=int(_celery_cfg.get("worker_prefetch_multiplier", 1)),
    # ---- 结果 ----
    result_expires=86400,
    # ---- 任务路由（S2 起陆续启用）----
    # task_routes={"app.tasks.ingest_tasks.*": {"queue": "ingest"}},
)

# 任务模块通过上面的 `include=` 声明（S2 起：app.tasks.ingest_tasks）。
# 不要在文件末尾 import 它们 —— 会与任务模块的反向导入形成循环。


__all__ = ["celery_app"]
