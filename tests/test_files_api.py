# -*- coding: utf-8 -*-
"""文档列表/详情/版本接口的快速契约测试（对应用例 VER-005 / DEL-001 负向部分）。

完整上传-解析-切换链路见 test_lifecycle.py（slow 标记）。
"""
from __future__ import annotations

import uuid

import pytest

from allure_support import allure
from api_client import assert_error, assert_ok

RANDOM_DOC = "00000000-0000-0000-0000-000000000001"


@allure.feature("文档与版本")
class TestFilesContract:
    def test_list_files_page_shape(self, client):
        """GET /files 返回分页结构且 items 为列表"""
        data = assert_ok(client.get("/files", params={"page": 1, "size": 5}))
        assert isinstance(data["items"], list)
        assert isinstance(data["total"], int)
        assert data["page"] == 1 and data["size"] == 5

    def test_list_files_keyword_filter(self, client):
        """keyword 模糊搜索不报错且结果都命中关键词"""
        data = assert_ok(client.get("/files", params={"page": 1, "size": 20, "keyword": "api-test"}))
        for item in data["items"]:
            assert "api-test" in (item.get("file_name") or "")

    def test_detail_of_random_doc_rejected(self, client):
        """查询不存在的 doc_id 被拒（1001），不泄露他人数据"""
        assert_error(client.get(f"/files/{RANDOM_DOC}"), http=400, code=1001)

    def test_versions_of_random_doc_rejected(self, client):
        """不存在的 doc_id 版本列表被拒（1001）"""
        assert_error(client.get(f"/files/{RANDOM_DOC}/versions"), http=400, code=1001)

    def test_activate_random_doc_rejected(self, client):
        """对不存在的 doc 切换版本被拒（1001）"""
        resp = client.post(f"/files/{RANDOM_DOC}/versions/1/activate")
        assert_error(resp, http=400, code=1001)

    def test_delete_random_doc_rejected(self, client):
        """删除不存在的 doc 被拒（1001）"""
        assert_error(client.delete(f"/files/{RANDOM_DOC}"), http=400, code=1001)


@pytest.mark.slow
@allure.feature("文档与版本")
class TestUploadValidation:
    """上传边界（这些用例依赖真实服务，且 png 上传会触发解析，归入 slow）。"""

    def test_upload_unsupported_ext_rejected(self, client):
        """UPLOAD-004 上传 .exe 被拦（400 / 2001）"""
        resp = client.upload("evil.exe", b"MZ\x90\x00fake")
        assert_error(resp, http=400, code=2001)

    def test_upload_without_file_rejected(self, client):
        """缺少 file 字段 → 422 参数校验"""
        resp = client.post("/files/upload", data={})
        assert resp.status_code == 422
