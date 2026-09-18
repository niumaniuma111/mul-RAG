"""
S2 端到端入库验收（从宿主机走真实 HTTP 边界，127.0.0.1:8000）。

为什么从宿主机跑而不是 `docker exec` 进容器：
    容器内直连 api:8000 会绕过端口映射与 uvicorn 的对外监听；
    从宿主机打 127.0.0.1:8000 才是前端/外部客户端的真实路径。

覆盖的验收点（对应 HANDOFF「S2 验收」条目）：
    A. 注册/登录拿到 token
    B. 上传 sample.pptx → 返回 doc_id / version=1 / task_id
    C. 轮询任务到 success（含各阶段进度单调推进）
    D. PG：documents.active_version=1、file_versions v1 parse_status=success
    E. Qdrant：v1 出现 text / table / image_caption 三类点，payload 字段齐全
    F. Qdrant：图片点 image_path 是可解析的相对路径，**没有** data: base64 残留
    G. 重传同名文件 → version=2；v2 点 is_active=true，且 v1 **全量** is_active=false
    H. 版本列表里 v1/v2 都在，active_version 切到 2

用法：
    <venv>/Scripts/python.exe e2e_ingest.py
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime

import requests

BASE = "http://127.0.0.1:8000/api/v1"
QDRANT = "http://127.0.0.1:6333"
COLLECTION = "ppt_chunks"
SAMPLE = (
    r"\\wsl.localhost\Ubuntu-22.04\root\projects\ppt-agentic-rag\_e2e_work\sample.pptx"
)

POLL_TIMEOUT_SEC = int(900)
POLL_INTERVAL_SEC = 3

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    line = f"  [{flag}] {name}"
    if detail:
        line += f"\n         {detail}"
    print(line, flush=True)
    return ok


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def api(method: str, path: str, **kw) -> tuple[int, dict]:
    """调 api，返回 (status_code, body)。body 保证是 dict。"""
    url = BASE + path
    kw.setdefault("timeout", 120)
    resp = requests.request(method, url, **kw)
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = {"_raw": resp.text[:600]}
    return resp.status_code, body


def qdrant(path: str, payload: dict | None = None) -> dict:
    url = QDRANT + path
    if payload is None:
        r = requests.get(url, timeout=60)
    else:
        r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def scroll_points(doc_id: str, version: int | None = None) -> list[dict]:
    """拉取该文档（可指定版本）的全部点 payload。"""
    must: list[dict] = [{"key": "doc_id", "match": {"value": doc_id}}]
    if version is not None:
        must.append({"key": "version", "match": {"value": version}})

    points: list[dict] = []
    offset = None
    while True:
        body: dict = {
            "filter": {"must": must},
            "limit": 256,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        data = qdrant(f"/collections/{COLLECTION}/points/scroll", body)
        result = data.get("result") or {}
        batch = result.get("points") or []
        points.extend(batch)
        offset = result.get("next_page_offset")
        if not offset or len(points) >= 5000:
            break
    return [dict(p.get("payload") or {}) for p in points]


def count_active(doc_id: str, version: int) -> tuple[int, int]:
    """返回该版本的 (总点数, is_active=true 点数)。"""
    def _count(extra: list[dict]) -> int:
        must = [{"key": "doc_id", "match": {"value": doc_id}}]
        must.extend(extra)
        data = qdrant(
            f"/collections/{COLLECTION}/points/count",
            {"filter": {"must": must}, "exact": True},
        )
        return int((data.get("result") or {}).get("count") or 0)

    total = _count([{"key": "version", "match": {"value": version}}])
    active = _count(
        [
            {"key": "version", "match": {"value": version}},
            {"key": "is_active", "match": {"value": True}},
        ]
    )
    return total, active


def wait_task(task_id: str) -> tuple[str, dict]:
    """轮询任务直到终态。返回 (status, data)。"""
    deadline = time.time() + POLL_TIMEOUT_SEC
    seen_stages: list[str] = []
    last_progress = -1
    regressions: list[str] = []
    last_data: dict = {}

    while time.time() < deadline:
        code, body = api("GET", f"/tasks/{task_id}")
        if code != 200:
            time.sleep(POLL_INTERVAL_SEC)
            continue
        data = body.get("data") or {}
        last_data = data
        stage = str(data.get("current_stage"))
        progress = int(data.get("progress") or 0)
        if stage not in seen_stages:
            seen_stages.append(stage)
            print(f"         stage -> {stage} ({progress}%)", flush=True)
        if progress < last_progress:
            regressions.append(f"{last_progress}->{progress}@{stage}")
        last_progress = max(last_progress, progress)

        if data.get("status") in ("success", "failed"):
            print(f"         阶段轨迹: {' -> '.join(seen_stages)}", flush=True)
            check("C2 进度单调不回退", not regressions, f"回退点: {regressions}" if regressions else "无回退")
            return str(data.get("status")), data
        time.sleep(POLL_INTERVAL_SEC)

    return "timeout", last_data


def upload(path: str, token: str, filename: str) -> tuple[int, dict]:
    with open(path, "rb") as fp:
        blob = fp.read()
    return api(
        "POST",
        "/files/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "file": (
                filename,
                blob,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        },
    )


def main() -> int:
    ts = datetime.now().strftime("%H%M%S")
    username = f"e2e_{ts}_{uuid.uuid4().hex[:4]}"
    password = "E2ePassw0rd!2026"

    # ---------------- A. 认证 ----------------
    section("A. 注册 / 登录")
    code, body = api(
        "POST", "/auth/register", json={"username": username, "password": password}
    )
    check("A1 注册 201", code == 201, f"status={code} body={json.dumps(body, ensure_ascii=False)[:200]}")

    code, body = api(
        "POST", "/auth/login", json={"username": username, "password": password}
    )
    token = (body.get("data") or {}).get("access_token")
    check("A2 登录 200 且拿到 token", code == 200 and bool(token), f"status={code}")

    headers = {"Authorization": f"Bearer {token}"}

    # ---------------- B. 上传 v1 ----------------
    section("B. 上传 sample.pptx（v1）")
    code, body = upload(SAMPLE, token, "sample.pptx")
    data = body.get("data") or {}
    doc_id = str(data.get("doc_id") or "")
    version = data.get("version")
    task_id = str(data.get("task_id") or "")
    check(
        "B1 上传 201",
        code == 201 and bool(doc_id) and version == 1 and bool(task_id),
        f"status={code} doc_id={doc_id} version={version} task_id={task_id}",
    )
    if not (doc_id and task_id):
        print(f"         上传响应: {json.dumps(body, ensure_ascii=False)[:600]}")
        return summarize()

    # ---------------- C. 轮询 ----------------
    section("C. 入库任务执行")
    status, tdata = wait_task(task_id)
    ok = check(
        "C1 任务 success",
        status == "success",
        f"status={status} stage={tdata.get('current_stage')} chunks={tdata.get('chunk_count')}",
    )
    if not ok:
        print(f"         error_log: {str(tdata.get('error_log'))[:1500]}")
        # 失败也要把已发生的现象落下来，便于定位
        _dump_qdrant(doc_id)
        return summarize()

    check(
        "C3 chunk_count > 0",
        int(tdata.get("chunk_count") or 0) > 0,
        f"chunk_count={tdata.get('chunk_count')}",
    )

    # ---------------- D. PG 侧 ----------------
    section("D. PG 版本状态")
    code, body = api("GET", f"/files/{doc_id}", headers=headers)
    detail = body.get("data") or {}
    versions = detail.get("versions") or []
    v1 = next((v for v in versions if v.get("version") == 1), None)
    check(
        "D1 active_version=1",
        code == 200 and detail.get("active_version") == 1,
        f"status={code} active_version={detail.get('active_version')} current_version={detail.get('current_version')}",
    )
    check(
        "D2 v1 parse_status=success",
        bool(v1) and v1.get("parse_status") == "success",
        f"v1={json.dumps(v1, ensure_ascii=False, default=str)[:300]}",
    )
    check(
        "D3 v1 page_count 已回填",
        bool(v1) and (v1.get("page_count") or 0) > 0,
        f"page_count={v1.get('page_count') if v1 else None}",
    )

    # ---------------- E/F. Qdrant 侧 ----------------
    section("E. Qdrant 点位与 payload")
    v1_points = scroll_points(doc_id, 1)
    check("E1 v1 有点位", len(v1_points) > 0, f"count={len(v1_points)}")

    kinds = Counter(p.get("chunk_type") for p in v1_points)
    check(
        "E2 出现三类内容 (text/table/image_caption)",
        {"text", "table", "image_caption"} <= set(kinds),
        f"分布={dict(kinds)}",
    )

    required = [
        "user_id",
        "doc_id",
        "is_active",
        "version",
        "file_name",
        "chunk_type",
        "chunk_index",
        "page_num",
        "content",
    ]
    missing: dict[str, list[str]] = {}
    for p in v1_points:
        miss = [k for k in required if k not in p or p.get(k) is None]
        if miss:
            missing.setdefault(",".join(miss), []).append(p.get("chunk_index"))
    check(
        "E3 payload 必需字段齐全",
        not missing,
        f"缺失分布={ {k: v[:5] for k, v in missing.items()} }" if missing else f"{len(required)} 个字段全齐",
    )

    active_v1 = sum(1 for p in v1_points if p.get("is_active") is True)
    check(
        "E4 v1 全部 is_active=true",
        active_v1 == len(v1_points),
        f"active={active_v1}/{len(v1_points)}",
    )

    img_points = [p for p in v1_points if p.get("chunk_type") == "image_caption"]
    bad_data_uri = [p for p in img_points if str(p.get("image_path") or "").startswith("data:")]
    check(
        "F1 image_path 无 base64 残留",
        not bad_data_uri,
        f"base64 残留={len(bad_data_uri)}/{len(img_points)}",
    )
    with_path = [p for p in img_points if p.get("image_path")]
    check(
        "F2 图片点带 image_path",
        len(with_path) == len(img_points) and len(img_points) > 0,
        f"带路径={len(with_path)}/{len(img_points)}  例={with_path[0].get('image_path') if with_path else None}",
    )
    captions = [p.get("content") for p in img_points]
    check(
        "F3 image_caption 内容非空",
        all(bool(c) for c in captions) if captions else False,
        f"例={str(captions[0])[:120] if captions else None}",
    )
    table_points = [p for p in v1_points if p.get("chunk_type") == "table"]
    has_html = [p for p in table_points if "<table" in str(p.get("table_html") or "").lower()]
    check(
        "F4 表格保留 HTML 原文 (D6)",
        bool(table_points) and len(has_html) == len(table_points),
        f"含 <table> 的表格点={len(has_html)}/{len(table_points)}",
    )

    # ---------------- G/H. 重传 → v2 ----------------
    section("G. 重传同名文件 → v2 版本切换")
    code, body = upload(SAMPLE, token, "sample.pptx")
    data2 = body.get("data") or {}
    doc2 = str(data2.get("doc_id") or "")
    v2_no = data2.get("version")
    task2 = str(data2.get("task_id") or "")
    check(
        "G1 重传复用同一 doc_id 且 version=2",
        code == 201 and doc2 == doc_id and v2_no == 2,
        f"status={code} doc_id={doc2} version={v2_no} is_new_doc={data2.get('is_new_doc')}",
    )
    if code != 201:
        print(f"         响应: {json.dumps(body, ensure_ascii=False)[:600]}")
        return summarize()

    status2, tdata2 = wait_task(task2)
    if not check(
        "G2 v2 任务 success",
        status2 == "success",
        f"status={status2} chunks={tdata2.get('chunk_count')}",
    ):
        print(f"         error_log: {str(tdata2.get('error_log'))[:1200]}")
        return summarize()

    t_v1, a_v1 = count_active(doc_id, 1)
    t_v2, a_v2 = count_active(doc_id, 2)
    check(
        "G3 v2 点全量 is_active=true",
        t_v2 > 0 and a_v2 == t_v2,
        f"v2 active={a_v2}/{t_v2}",
    )
    check(
        "G4 v1 全量 is_active=false",
        t_v1 > 0 and a_v1 == 0,
        f"v1 active={a_v1}/{t_v1}",
    )

    section("H. 版本列表与 active_version")
    code, body = api("GET", f"/files/{doc_id}", headers=headers)
    detail = body.get("data") or {}
    versions = detail.get("versions") or []
    nums = sorted(int(v.get("version")) for v in versions)
    check(
        "H1 v1/v2 都在且 active_version=2",
        nums == [1, 2] and detail.get("active_version") == 2,
        f"versions={nums} active_version={detail.get('active_version')} current_version={detail.get('current_version')}",
    )
    v1_row = next((v for v in versions if v.get("version") == 1), {})
    check(
        "H2 v1 行 is_active 已下线",
        v1_row.get("is_active") is False,
        f"v1.is_active={v1_row.get('is_active')}",
    )

    return summarize()


def _dump_qdrant(doc_id: str) -> None:
    try:
        pts = scroll_points(doc_id)
        print(f"         [诊断] Qdrant 中该 doc 的点数={len(pts)}", flush=True)
        if pts:
            print(
                f"         [诊断] chunk_type 分布={dict(Counter(p.get('chunk_type') for p in pts))}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"         [诊断] 查询 Qdrant 失败: {exc}", flush=True)


def summarize() -> int:
    section("汇总")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  === {passed}/{total} 通过 ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
