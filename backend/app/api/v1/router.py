"""
API v1 路由聚合。

这里只做挂载，不写业务逻辑。S1 只有 health；S2 加入 auth / files / tasks。

⚠️ 各子路由的 `prefix` 在**各自模块内**定义（如 files.py 里是 `prefix="/files"`），
   这里 `include_router` 不再传 prefix —— 两处都写会导致路径变成
   `/files/files/...`。这是很常见的低级错误，改路由时先看模块头。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, files, health, qa, sessions, tasks

api_router = APIRouter()

# ---- S1：健康检查 ----
api_router.include_router(health.router, tags=["health"])

# ---- S2：认证 + 离线入库 ----
api_router.include_router(auth.router)
api_router.include_router(files.router)
api_router.include_router(tasks.router)

# ---- S3：在线问答 ----
api_router.include_router(qa.router)

# ---- B2：会话历史（只读）----
# ⚠️ 必须在 qa.router **之后**挂载。两者 prefix 同为 `/qa`，但 sessions 的
#    路径是 `/qa/sessions`，与 qa 的 `/qa/ask`、`/qa/resume` 无冲突；
#    放后面是为了保持"S3 主体在前、后续补的读接口在后"的阅读顺序。
api_router.include_router(sessions.router)

__all__ = ["api_router"]
