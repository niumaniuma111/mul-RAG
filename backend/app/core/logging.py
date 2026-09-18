"""
日志初始化。

配置全部来自 `configs/logging.yaml`（dictConfig），代码里不写 handler ——
这样调日志级别、加输出目标都不需要改代码、更不需要重建镜像
（configs 是只读挂载卷，改完重启容器即可）。
"""

from __future__ import annotations

import logging
import logging.config

from app.core.config import load_logging_dict

_configured = False


def setup_logging() -> None:
    """
    幂等初始化 —— 多次调用只生效一次。

    uvicorn / celery 的启动路径不同，两边都会调这个方法，
    幂等是为了避免 handler 被重复挂载导致日志打两遍。
    """
    global _configured
    if _configured:
        return
    logging.config.dictConfig(load_logging_dict())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """统一入口，方便日后换实现（比如从 stdlib 换 structlog）。"""
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
