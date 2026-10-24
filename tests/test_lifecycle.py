# -*- coding: utf-8 -*-
"""完整链路 e2e（slow）：png 上传 → 解析 → 版本升级 → 切换 → 硬删除 → 任务历史保留。

对应用例：UPLOAD-003 / UPLOAD-007 / VER-001 / VER-004 / DEL-001 / DEL-002 / DEL-003 / DEL-004。
运行：pytest -m slow（需要完整 docker compose 栈在线，png 解析约 30s/次）。
"""
from __future__ import annotations

import time
import uuid

import pytest

from allure_support import allure
from api_client import assert_error, assert_ok
from sample_data import TINY_PNG

POLL_INTERVAL = 3
POLL_TIMEOUT = 300


def wait_task_done(client, task_id: str) -> dict:
    """轮询任务直至 success/failed，返回最终任务详情。"""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        data = assert_ok(client.get(f"/tasks/{task_id}"))
        if data["status"] in ("success", "failed"):
            return data
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"任务 {task_id} 超时未完成")


@pytest.mark.slow
@allure.feature("全链路")
class TestDocumentLifecycle:
    def test_upload_parse_upgrade_activate_delete(self, client, created_docs):
        # ---- 1. 首次上传（UPLOAD-003）：立即返回 task_id ----
        filename = f"api-test-{uuid.uuid4().hex[:10]}.png"
        resp = client.upload(filename, TINY_PNG, mime="image/png")
        data = assert_ok(resp, http=201)
        doc_id, task_id = data["doc_id"], data["task_id"]
        created_docs.append(doc_id)
        assert data["version"] == 1 and data["is_new_doc"] is True

        # ---- 2. 轮询解析完成（TASK-001）----
        done = wait_task_done(client, task_id)
        assert done["status"] == "success", f"解析失败: {done.get('error_log')}"
        assert (done.get("chunk_count") or 0) >= 1

        # ---- 3. 同名重传触发版本升级（UPLOAD-007 / VER-001）----
        resp2 = client.upload(filename, TINY_PNG, mime="image/png")
        data2 = assert_ok(resp2, http=201)
        assert data2["doc_id"] == doc_id and data2["version"] == 2
        assert data2["is_new_doc"] is False
        done2 = wait_task_done(client, data2["task_id"])
        assert done2["status"] == "success", f"v2 解析失败: {done2.get('error_log')}"

        # ---- 4. 版本列表与详情（VER-005）----
        versions = assert_ok(client.get(f"/files/{doc_id}/versions"))
        assert {v["version"] for v in versions} >= {1, 2}

        # ---- 5. 切回 v1 再切回 v2（VER-004）----
        act = assert_ok(client.post(f"/files/{doc_id}/versions/1/activate"))
        assert act["active_version"] == 1
        act = assert_ok(client.post(f"/files/{doc_id}/versions/2/activate"))
        assert act["active_version"] == 2

        # ---- 6. 硬删除（DEL-001 / DEL-003）----
        del_data = assert_ok(client.delete(f"/files/{doc_id}"))
        assert del_data["deleted_versions"] >= 2
        warnings = del_data.get("warnings") or []
        if warnings:
            if all("Permission denied" in str(w) for w in warnings):
                # BUG-004（已知缺陷）：mineru 容器写入 parsed 的图片属主与 api 用户
                # 不一致，硬删除时磁盘清理 Permission denied，产生残留。
                # 修复后此 xfail 应转为通过。
                pytest.xfail(f"BUG-004 磁盘残留（跨容器权限）: {warnings}")
            pytest.fail(f"删除存在未清干净的载体: {warnings}")

        # ---- 7. 任务历史保留 + 快照文件名（DEL-002，回归 BUG-003）----
        detail = assert_ok(client.get(f"/tasks/{task_id}"))
        assert detail["file_name"] == filename
        assert detail["version_id"] is None

        # ---- 8. 已删文档任务重试被拦（DEL-004）----
        resp = client.post(f"/tasks/{task_id}/retry")
        assert_error(resp, http=400, code=1001)

        # ---- 9. 文档确认不可再访问（DEL-001）----
        assert_error(client.get(f"/files/{doc_id}"), http=400, code=1001)
        created_docs.remove(doc_id)  # 已删除，避免 finalizer 重复删
