"""极简 requests 封装：统一 base_url、JWT 注入与 {code, message, data} 信封断言。

设计对应课程分层法：本文件即「工具层 + 业务层」，
test_*.py 只写「用例层」，不直接碰 requests 细节。
"""
from __future__ import annotations

import json
from typing import Any

import requests


def assert_ok(resp: requests.Response, *, http: int = 200) -> dict[str, Any]:
    """断言业务成功：HTTP 状态码 + 信封 code==0，返回 data。"""
    assert resp.status_code == http, f"HTTP {resp.status_code} != {http}: {resp.text[:300]}"
    body = resp.json()
    assert body.get("code") == 0, f"业务码非 0: {body}"
    return body.get("data") or {}


def assert_error(
    resp: requests.Response, *, http: int, code: int | None = None
) -> dict[str, Any]:
    """断言业务失败：HTTP 状态码 +（可选）信封错误码，返回完整信封。"""
    assert resp.status_code == http, f"HTTP {resp.status_code} != {http}: {resp.text[:300]}"
    body = resp.json()
    assert body.get("code") != 0, f"预期失败但 code==0: {body}"
    if code is not None:
        assert body.get("code") == code, f"业务码 {body.get('code')} != {code}: {body}"
    return body


class ApiClient:
    """带登录态的接口客户端。"""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.token: str | None = None
        self.username: str | None = None

    # ---------- 基础 ----------
    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict | None = None,
        files: dict | None = None,
        data: dict | None = None,
        timeout: float = 30,
        auth: bool = True,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = self._headers() if auth else {}
        return self.session.request(
            method, url, json=json_body, params=params, files=files,
            data=data, headers=headers, timeout=timeout,
        )

    def get(self, path: str, **kw: Any) -> requests.Response:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> requests.Response:
        return self.request("POST", path, **kw)

    def delete(self, path: str, **kw: Any) -> requests.Response:
        return self.request("DELETE", path, **kw)

    # ---------- 业务 ----------
    def register(self, username: str, password: str, email: str | None = None) -> requests.Response:
        return self.post(
            "/auth/register",
            json_body={"username": username, "password": password, "email": email},
            auth=False,
        )

    def login(self, username: str, password: str) -> requests.Response:
        resp = self.post("/auth/login", json_body={"username": username, "password": password}, auth=False)
        if resp.status_code == 200 and resp.json().get("code") == 0:
            self.token = resp.json()["data"]["access_token"]
            self.username = username
        return resp

    def upload(self, filename: str, content: bytes, mime: str = "application/octet-stream") -> requests.Response:
        return self.post(
            "/files/upload",
            files={"file": (filename, content, mime)},
            timeout=60,
        )

    # ---------- SSE ----------
    def ask_sse(self, payload: dict, timeout: float = 180.0):
        """POST /qa/ask，流式返回 (event, data_dict) 序列。"""
        url = f"{self.base_url}/qa/ask"
        resp = self.session.post(url, json=payload, headers=self._headers(), stream=True, timeout=timeout)
        resp.raise_for_status()
        event, data_buf = None, []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw == "":
                if event is not None:
                    yield event, (json.loads(data_buf[0]) if data_buf else {})
                event, data_buf = None, []
                continue
            if raw.startswith("event:"):
                event = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:"):
                data_buf.append(raw.split(":", 1)[1].strip())
