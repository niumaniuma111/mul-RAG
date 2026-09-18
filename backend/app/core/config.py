"""
应用配置中心（唯一入口）。

职责划分
--------
    .env                  → 密钥与连接串   → EnvSettings（pydantic-settings，字段有类型校验）
    configs/config.yaml   → 超参与容器限额 → DotDict（递归字典，支持点号访问）
    configs/logging.yaml  → 日志配置       → 由 app/core/logging.py 用 dictConfig 读取
    configs/prompts/*.txt → Prompt 模板    → load_prompt()

关于"为什么 config.yaml 不用 pydantic 全量建模"
-----------------------------------------------
DESIGN.md §6 原文是「用 pydantic-settings 加载 config.yaml + .env」。

.env 部分照做（它字段少、且类型错误代价高，值得强校验）；
config.yaml 部分改为 DotDict，理由是本项目超参 200+ 项且随迭代持续增删，
用 pydantic 全量建模意味着每改一处配置就要同步改一次 Python 类，
维护成本高于它带来的类型提示收益。关键数值在使用处显式转换
（例如 int(cfg.retrieval.dense_top_k)），代价可接受。

对应设计文档：DESIGN.md §6
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 路径解析
# ============================================================
def _resolve_config_dir() -> Path:
    """
    按「环境变量 → 容器约定 → 本地开发」三级回退定位 configs 目录。

    容器内代码被 COPY 到 /app，所以没法靠 __file__ 上溯到项目根；
    而本地开发时又希望直接跑 backend/ 下的代码。两条路径都要能用。
    """
    env_dir = os.getenv("CONFIG_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    # 容器内：docker-compose 把 ./configs 挂到 /app/configs
    container_dir = Path("/app/configs")
    if container_dir.is_dir():
        return container_dir

    # 本地开发：backend/app/core/config.py → 上溯 3 级 = 项目根
    return Path(__file__).resolve().parents[3] / "configs"


CONFIG_DIR: Path = _resolve_config_dir()
CONFIG_FILE: Path = CONFIG_DIR / "config.yaml"
LOGGING_FILE: Path = CONFIG_DIR / "logging.yaml"
PROMPTS_DIR: Path = CONFIG_DIR / "prompts"


# ============================================================
# DotDict —— 支持点号访问的递归字典
# ============================================================
class DotDict(dict):
    """
    让 `cfg.ingest.mineru.device` 这种写法成立，同时保留 dict 的全部能力。

    实现要点：只在常规属性查找失败时才走 __getattr__，所以 keys()/items()/
    get() 等 dict 自身的方法不受影响。
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        """递归包装：dict → DotDict，list 内的 dict 也要包。"""
        if isinstance(value, DotDict):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"配置项不存在: {item!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """
        按 'a.b.c' 路径取值，任一层缺失就返回 default（不抛异常）。

        用于读取「可选配置」，例如：
            cfg.get_path("ingest.mineru.device", "cpu")
        """
        current: Any = self
        for part in dotted.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current


# ============================================================
# .env —— 密钥与连接串
# ============================================================
class EnvSettings(BaseSettings):
    """
    只放密钥与连接串。

    这些内容不能进 git，所以走 .env；容器里由 compose 的 env_file 注入成
    环境变量（此时 .env 文件在容器内并不存在，pydantic-settings 会忽略缺失的文件）。
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(os.getenv("ENV_FILE", ".env"))),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- PostgreSQL ----
    postgres_user: str = "pptrag"
    postgres_password: str = "pptrag"
    postgres_db: str = "pptrag"
    # SQLAlchemy 异步引擎（asyncpg 驱动）
    database_url: str = "postgresql+asyncpg://pptrag:pptrag@postgres:5432/pptrag"
    # LangGraph AsyncPostgresSaver（libpq / psycopg3，不能带 +asyncpg）
    checkpoint_database_url: str = "postgresql://pptrag:pptrag@postgres:5432/pptrag"

    # ---- Redis ----
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/3"

    # ---- Qdrant ----
    qdrant_url: str = "http://qdrant:6333"
    qdrant_grpc_port: int = 6334

    # ---- 安全 ----
    jwt_secret: str = "dev-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"

    # ---- 服务地址（容器间用服务名互访）----
    models_root: str = "/models"
    model_service_url: str = "http://model-service:8002"
    mineru_url: str = "http://mineru:8001"

    # ---- MinerU 设备联动开关（config.yaml 里也有一份，这里是 compose 注入的覆盖值）----
    mineru_device: str = ""

    # ---- 运行时 ----
    tz: str = "Asia/Shanghai"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_env() -> EnvSettings:
    """进程内单例。"""
    return EnvSettings()


# ============================================================
# config.yaml —— 超参
# ============================================================
_config_cache: DotDict | None = None


def get_config(reload: bool = False) -> DotDict:
    """
    读取 configs/config.yaml（进程内缓存）。

    reload=True 用于测试或热改配置的场景。
    """
    global _config_cache
    if _config_cache is None or reload:
        if not CONFIG_FILE.is_file():
            raise FileNotFoundError(
                f"找不到配置文件 {CONFIG_FILE}。"
                f"CONFIG_DIR 当前解析为 {CONFIG_DIR}（可用环境变量 CONFIG_DIR 覆盖）。"
            )
        with CONFIG_FILE.open("r", encoding="utf-8") as fp:
            _config_cache = DotDict(yaml.safe_load(fp) or {})
    return _config_cache


def load_logging_dict() -> dict[str, Any]:
    """给 logging.config.dictConfig 用。"""
    with LOGGING_FILE.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_prompt(filename: str) -> str:
    """
    读取 configs/prompts 下的 Prompt 模板。

    外置的目的：调 prompt 不需要重建镜像（configs 是只读挂载卷）。
    """
    path = PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    return path.read_text(encoding="utf-8")


def resolve_mineru_device() -> str:
    """
    解析 MinerU 设备 —— 本项目最容易配错的一处，所以把优先级写死在这里。

    优先级：.env 里的 MINERU_DEVICE（compose 注入，用于临时切换）
            → config.yaml 的 ingest.mineru.device（默认 cpu）

    ⚠️ 这个值同时决定锁层级（见 lock_factory），不要在任何其它地方重复解析它。
    """
    env_device = (get_env().mineru_device or "").strip().lower()
    if env_device in {"cpu", "cuda"}:
        return env_device
    return str(get_config().get_path("ingest.mineru.device", "cpu")).strip().lower()


__all__ = [
    "CONFIG_DIR",
    "CONFIG_FILE",
    "LOGGING_FILE",
    "PROMPTS_DIR",
    "DotDict",
    "EnvSettings",
    "get_config",
    "get_env",
    "load_logging_dict",
    "load_prompt",
    "resolve_mineru_device",
]
