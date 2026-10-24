# -*- coding: utf-8 -*-
"""任务接口契约测试（对应用例 TASK-005 / DEL-004 负向部分）。"""
from __future__ import annotations

import uuid

import pytest

from allure_support import allure
from api_client import assert_error, assert_ok

RANDOM_TASK = "00000000-0000-0000-0000-000000000002"


@allure.feature("入库任务")
class TestTasksContract:
    def test_list_tasks_page_shape(self, client):
        """GET /tasks 返回分页结构，item 字段齐全（轻接口，一条 SQL 出结果）"""
        data = assert_ok(client.get("/tasks", params={"page": 1, "size": 10}))
        assert isinstance(data["items"], list) and isinstance(data["total"], int)
        for item in data["items"][:5]:
            for key in ("task_id", "doc_id", "status", "progress", "file_name"):
                assert key in item, f"任务列表缺字段 {key}"

    def test_list_tasks_filter_by_status(self, client):
        """status 过滤参数可用且返回项状态一致"""
        data = assert_ok(client.get("/tasks", params={"page": 1, "size": 10, "status": "failed"}))
        for item in data["items"]:
            assert item["status"] == "failed"

    def test_get_random_task_rejected(self, client):
        """查询不存在的任务被拒（1001，含越权同义）"""
        assert_error(client.get(f"/tasks/{RANDOM_TASK}"), http=400, code=1001)

    def test_logs_of_random_task_rejected(self, client):
        """不存在任务的日志接口被拒（1001）"""
        assert_error(client.get(f"/tasks/{RANDOM_TASK}/logs"), http=400, code=1001)

    def test_retry_random_task_rejected(self, client):
        """不存在任务的重试被拒（1001）"""
        assert_error(client.post(f"/tasks/{RANDOM_TASK}/retry"), http=400, code=1001)

    def test_delete_random_task_rejected(self, client):
        """不存在任务的记录删除被拒（1001）"""
        assert_error(client.delete(f"/tasks/{RANDOM_TASK}"), http=400, code=1001)

    @pytest.mark.slow
    def test_retry_deleted_doc_task_blocked(self, client, created_docs):
        """DEL-004 已删文档的历史任务重试被 400 拦截（回归 BUG-003 链路）。

        依赖 test_lifecycle 先跑（created_docs 中有已删文档的 task_id），
        独立运行时跳过。
        """
        if not created_docs:
            pytest.skip("需要先跑 test_lifecycle 产生任务历史")
