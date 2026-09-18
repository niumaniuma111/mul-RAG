"""
认证原语：密码哈希 + JWT 签发/校验。

为什么用 bcrypt + python-jose（而不是自己写）
--------------------------------------------
bcrypt 的加盐与 cost 轮数、JWT 的 base64url 编码与签名验签都有细节陷阱，
自己实现极易出错。两个库都在 requirements 里锁定版本。

⚠️ v1.1.8 变更：移除 passlib，直接调用 bcrypt
--------------------------------------------
原实现用 `passlib.context.CryptContext(schemes=["bcrypt"])`。它在本机（实测 2026-09-17）
**必然报错**，且报错点在 passlib 内部，与业务代码无关：

    ValueError: password cannot be longer than 72 bytes,
                truncate manually if necessary

调用链：
    CryptContext.hash → _load_backend_mixin → _finalize_backend_mixin
      → detect_wrap_bug(IDENT_2A) → verify(<超长测试串>, bug_hash)
      → bcrypt.hashpw → ValueError

真因是**版本不兼容**：passlib 1.7.4（2020 年后未再更新）在加载 bcrypt 后端时，
会用一个超过 72 字节的固定测试串做自检（用于探测某个历史 wrap bug）；
而 bcrypt>=4.1 把"超 72 字节"从**静默截断**改成了**抛异常**。
于是 passlib 的自检在 initialize 阶段就炸，任何一次密码哈希都走不到业务逻辑。

两个修法：
    a) 把 bcrypt 钉回 4.0.1（passlib 兼容的最后一版）—— 但等于为了一个
       已停止维护的库而冻结一个安全敏感的依赖，方向反了
    b) **直接用 bcrypt**（本文件的方案）—— bcrypt 本来就是 passlib[bcrypt]
       拉进来的传递依赖，去掉 passlib 反而少一层；代码量也更少

⚠️ bcrypt 的 72 字节密码上限
--------------------------
bcrypt 算法本身只取前 72 字节，超过的部分**不参与运算**。中文 UTF-8 一个字 3 字节，
即 24 个汉字就到顶。bcrypt>=4.1 对超长输入会**抛 ValueError**（不再静默截断），
所以必须在调用前自己截断 —— 见 `_encode_password`。

这不是安全漏洞（校验仍需要正确的前 72 字节），但要在注册接口限制 max_length，
避免用户以为设了很长的密码。见 `app/schemas/auth.py` 的 max_length。

对应设计文档：DESIGN.md §11.1
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_config, get_env
from app.core.exceptions import UnauthorizedError

# bcrypt 的 72 字节硬上限，超出的部分算法本身就会忽略
_BCRYPT_MAX_BYTES = 72
# bcrypt 的 rounds 合法区间（算法限制），用于夹住配置值防止误配
_BCRYPT_ROUNDS_MIN = 4
_BCRYPT_ROUNDS_MAX = 16


def _rounds() -> int:
    """按 config.yaml 的 auth.bcrypt_rounds 取轮数（12 轮约 250ms/次）。"""
    configured = int(get_config().get_path("auth.bcrypt_rounds", 12))
    return max(_BCRYPT_ROUNDS_MIN, min(_BCRYPT_ROUNDS_MAX, configured))


def _encode_password(password: str) -> bytes:
    """
    把密码编码成 UTF-8 并截到 72 字节。

    ⚠️ 必须按**字节**截断，不能按字符 —— 中文一个字 3 字节，按字符截会越界。
    多字节字符被截半会变成非法 UTF-8，所以用 errors="ignore" 丢弃残字节
    （丢掉半个字符不影响：那半个字符本来也不参与 bcrypt 运算）。
    """
    raw = password.encode("utf-8")
    if len(raw) <= _BCRYPT_MAX_BYTES:
        return raw
    return raw[:_BCRYPT_MAX_BYTES].decode("utf-8", errors="ignore").encode("utf-8")


def hash_password(password: str) -> str:
    """生成 bcrypt 哈希（字符串形式，形如 `$2b$12$...`）。"""
    salt = bcrypt.gensalt(rounds=_rounds())
    return bcrypt.hashpw(_encode_password(password), salt).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """
    校验密码。

    任何异常一律当作"校验失败"返回 False，不向上抛：
    库里若有手工插入的脏数据（非 bcrypt 格式），抛 500 会把
    "密码错"这个正常业务结果伪装成服务故障。
    """
    try:
        return bcrypt.checkpw(_encode_password(password), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ============================================================
# JWT
# ============================================================
def create_access_token(user_id: int, username: str) -> tuple[str, int]:
    """
    签发 access token。

    返回 (token, expires_in_seconds)。sub 用 str(user_id) —— JWT 规范要求
    sub 是字符串，塞 int 会让部分解析库报错。
    """
    env = get_env()
    algorithm = str(get_config().get_path("auth.algorithm", env.jwt_algorithm))
    expire_minutes = int(get_config().get_path("auth.access_token_expire_minutes", 1440))
    expires_delta = timedelta(minutes=expire_minutes)

    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    token = jwt.encode(payload, env.jwt_secret, algorithm=algorithm)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    """解码并验签。任何失败都归一成 1002 未认证，不向外泄露具体原因。"""
    env = get_env()
    algorithm = str(get_config().get_path("auth.algorithm", env.jwt_algorithm))
    try:
        return jwt.decode(token, env.jwt_secret, algorithms=[algorithm])
    except JWTError as exc:
        raise UnauthorizedError(f"令牌无效或已过期: {exc}") from exc


__all__ = [
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
