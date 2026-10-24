"""pytest 全局夹具：环境配置、登录态客户端、测试文档清理。

环境变量（均有默认值，见 README）：
    API_BASE        默认 http://127.0.0.1:8000/api/v1
    TEST_USERNAME   默认 testkit_db107b7f6d（已存在测试账号）
    TEST_PASSWORD   默认 Testkit!Pass_2026
    RUN_QA_SMOKE    置 1 才跑问答链路冒烟（需要 GPU/模型服务）
"""
from __future__ import annotations

import os
import uuid

import pytest

from api_client import ApiClient


def pytest_addoption(parser: pytest.Parser) -> None:  # noqa: ARG001
    pass  # 配置统一走环境变量，保持与 CI 变量注入方式一致


@pytest.fixture(scope="session")
def api_base() -> str:
    return os.environ.get("API_BASE", "http://127.0.0.1:8000/api/v1")


@pytest.fixture(scope="session")
def client(api_base: str) -> ApiClient:
    """主测试账号：登录态 session 级复用。"""
    c = ApiClient(api_base)
    username = os.environ.get("TEST_USERNAME", "testkit_db107b7f6d")
    password = os.environ.get("TEST_PASSWORD", "Testkit!Pass_2026")
    resp = c.login(username, password)
    assert resp.status_code == 200, f"测试账号登录失败，请确认服务已启动: {resp.text[:200]}"
    return c


@pytest.fixture()
def fresh_client(api_base: str) -> ApiClient:
    """未登录的全新客户端（认证类用例自管注册/登录）。"""
    return ApiClient(api_base)


@pytest.fixture()
def new_user(fresh_client: ApiClient) -> ApiClient:
    """注册并登录一个一次性用户（用例结束即弃，不清理——注册不可删）。"""
    suffix = uuid.uuid4().hex[:10]
    username = f"api_test_{suffix}"
    resp = fresh_client.register(username, "Test!Pass_2026", f"{username}@example.com")
    assert resp.status_code == 201, f"注册失败: {resp.text[:200]}"
    assert fresh_client.login(username, "Test!Pass_2026").status_code == 200
    return fresh_client


@pytest.fixture(scope="session")
def created_docs(client: ApiClient) -> list[str]:
    """跟踪用例创建的 doc_id，会话结束时尽力硬删，保持库内干净。"""
    doc_ids: list[str] = []

    yield doc_ids

    for doc_id in list(doc_ids):
        try:
            client.delete(f"/files/{doc_id}", timeout=60)
        except Exception:  # noqa: BLE001, S110 —— 清理尽力而为
            pass
