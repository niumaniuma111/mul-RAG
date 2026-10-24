# -*- coding: utf-8 -*-
"""认证模块接口测试（对应用例 AUTH-001 ~ AUTH-006）。"""
from __future__ import annotations

import uuid

import pytest

from allure_support import allure
from api_client import assert_error, assert_ok

CASE_PREFIX = "AUTH"


@allure.feature("用户认证")
class TestAuthLogin:
    def test_register_then_login_ok(self, fresh_client):
        """AUTH-001 有效注册并登录成功"""
        suffix = uuid.uuid4().hex[:10]
        username = f"{CASE_PREFIX.lower()}_{suffix}"
        assert_ok(fresh_client.register(username, "Test!Pass_2026", f"{username}@example.com"), http=201)
        data = assert_ok(fresh_client.login(username, "Test!Pass_2026"))
        assert data["access_token"] and data["username"] == username

    def test_register_duplicate_username_rejected(self, new_user):
        """AUTH-005 重复用户名注册被拒绝（1001）"""
        resp = new_user.register(new_user.username, "Test!Pass_2026", None)
        assert_error(resp, http=400, code=1001)

    def test_login_wrong_password(self, new_user):
        """AUTH-002 密码错误登录失败（1002，且不区分用户不存在/密码错）"""
        resp = new_client_login(new_user, new_user.username, "Wrong!Pass_000")
        assert_error(resp, http=401, code=1002)

    def test_login_missing_password(self, fresh_client):
        """AUTH-004 密码为空提交被拦截（422 参数校验）"""
        resp = fresh_client.post("/auth/login", json_body={"username": "whatever"}, auth=False)
        assert resp.status_code == 422

    def test_login_missing_username(self, fresh_client):
        """AUTH-003 用户名为空提交被拦截（422 参数校验）"""
        resp = fresh_client.post("/auth/login", json_body={"password": "x"}, auth=False)
        assert resp.status_code == 422

    def test_me_with_token(self, new_user):
        """登录态 GET /auth/me 返回当前用户"""
        data = assert_ok(new_user.get("/auth/me"))
        assert data["username"] == new_user.username

    def test_me_without_token_rejected(self, fresh_client):
        """AUTH-006 未登录访问受保护接口被拒（1002）"""
        assert_error(fresh_client.get("/auth/me", auth=False), http=401, code=1002)

    def test_files_without_token_rejected(self, fresh_client):
        """未登录访问业务接口同样被拒（安全边界统一在 get_current_user）"""
        assert_error(fresh_client.get("/files", auth=False), http=401, code=1002)


def new_client_login(client, username: str, password: str):
    return client.post(
        "/auth/login", json_body={"username": username, "password": password}, auth=False
    )
