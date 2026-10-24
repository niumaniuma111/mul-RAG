# -*- coding: utf-8 -*-
"""1x1 透明 PNG 与极小文件的测试素材（接口级测试不依赖内容质量）。"""
from __future__ import annotations

import base64

# 1x1 透明 PNG（标准最小 PNG）
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def unique_name(ext: str = "png") -> str:
    import uuid

    return f"api-test-{uuid.uuid4().hex[:10]}.{ext}"
