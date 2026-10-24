# 接口自动化测试（requests + pytest）

对 FastAPI 后端做黑盒接口回归：跑在**真实的 docker compose 栈**上，
覆盖认证 / 上传校验 / 任务契约 / 版本切换 / 硬删除的**快速回归集**，
以及一条 png 全链路（slow）与问答 SSE 冒烟（qa）。

## 职责边界（与评测集分工）

| 层 | 工具 | 管什么 |
|---|---|---|
| 接口契约/业务流转 | 本目录（pytest + requests） | 状态码、错误码、分页结构、版本流转、删除一致性 |
| 问答质量 | `tools/testkit/` 三层评测集 | 检索 hit/MRR、答案断言、HITL 阈值 |
| SSE/前端渲染 | `tools/s4_smoke.py` + 手工 | 事件链完整性、流式渲染 |

## 运行

```bash
# 前置：docker compose --profile frontend up -d（栈在线）
pip install -r tests/requirements.txt

cd tests
pytest                    # 快速接口回归（默认，约 10s）
pytest -m slow            # png 全链路（上传→解析→版本→删除，约 2min）
RUN_QA_SMOKE=1 pytest -m qa   # 问答 SSE 冒烟（需 GPU）
pytest -m ""              # 全量

# Allure 报告
pytest --alluredir=allure-results
allure serve allure-results
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_BASE` | `http://127.0.0.1:8000/api/v1` | 后端地址 |
| `TEST_USERNAME` / `TEST_PASSWORD` | testkit 测试账号 | 主账号（登录态复用） |
| `RUN_QA_SMOKE` | 未设置 | 置 `1` 启用问答冒烟 |

## 设计要点

- **信封断言统一**：`assert_ok` / `assert_error`（HTTP 码 + 业务码双层校验，
  错误码对照 `app/core/exceptions.py`：1001 参数 / 1002 未认证 / 2001 格式不支持）。
- **登录态注入**：`ApiClient` 统一带 `Bearer`；未认证访问有专项负向用例。
- **数据自清理**：`created_docs` 夹具会话结束时尽力硬删测试文档；
  测试账号内的脏数据不会累积（注册用户不可删，属已知取舍）。
- **事件级 SSE 断言**：问答只断言事件序列（首帧 session → 终帧 final/error），
  不断言答案文本——LLM 输出不确定，质量归评测集管。
- **回归锚点**：`test_lifecycle.py` 第 7/8 步是缺陷 BUG-003（删除连删任务历史）
  的回归用例；用例编号与 `docs/testing/测试用例集.xlsx` 对应。
