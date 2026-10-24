# -*- coding: utf-8 -*-
"""问答链路 SSE 冒烟（qa 标记，默认不跑）。

只做**事件级**断言：会话建立 → 流式推进 → 收到 final/error 终帧。
不断言答案文本 —— LLM 输出不确定，断言内容必挂；
答案质量由三层标注评测集（tools/testkit）负责，职责分离。

运行：RUN_QA_SMOKE=1 pytest -m qa（需要 GPU 与模型服务在线，单例 40~83s）。
"""
from __future__ import annotations

import os

import pytest

from allure_support import allure

pytestmark = [
    pytest.mark.qa,
    pytest.mark.skipif(
        os.environ.get("RUN_QA_SMOKE") != "1",
        reason="问答链路冒烟需显式设置 RUN_QA_SMOKE=1（依赖 GPU）",
    ),
]


@allure.feature("问答链路")
class TestQaSmoke:
    def test_chat_question_streams_to_final(self, client):
        """QA-002 闲聊类问题：SSE 全事件链可达终帧（不触发检索链路也可回答）"""
        events = list(client.ask_sse({"query": "你好，请用一句话介绍你自己"}, timeout=180))
        names = [e for e, _ in events]

        assert names, "未收到任何 SSE 事件"
        assert names[0] == "session", f"首帧应为 session，实际: {names[:3]}"
        assert names[-1] in ("final", "error"), f"终帧应为 final/error，实际: {names[-1]}"

        # 收到过 token 或 final，说明生成链路打通
        assert "token" in names or names[-1] == "final"

        if names[-1] == "final":
            # final 必须是最后一帧（事件顺序契约：final 之后不再有业务事件）
            assert names.index("final") == len(names) - 1
