"""model-service 端点探针（S2 验收）。

覆盖四个此前**未真正跑过**的端点：
    /embed          向量化（BGE-large-zh-v1.5，1024 维）
    /rerank         精排（BGE-reranker-large，分数应已过 sigmoid → 0~1）
    /vlm/caption    图片 caption（Qwen2.5-VL-3B 4bit）
    /admin/unload   手动换出（验证进程级卸载）

为什么必须实测而不是"看代码觉得对"
----------------------------------
端点的**返回体键名**是上下游的硬契约（`vectors` / `results[].score` / `caption`），
键名写错不会报错，只会让下游静默拿到空值。这类问题只有真跑一次才看得见。

用法（容器内）：
    docker cp _probe_model_service.py pptrag-model-service:/tmp/ && \
    docker exec pptrag-model-service python /tmp/_probe_model_service.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8002"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "/app/storage/images/e2e_chart.png"

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, f"{label}  {detail}"[:300]))
    print(("  ✓ " if ok else "  ✗ ") + label + (f"   {detail}"[:220] if detail else ""))


def req(method: str, path: str, body: dict | None = None, timeout: int = 600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def jload(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}


print("=" * 72)
print(f"model-service 端点探针   image={IMAGE_PATH}")
print("=" * 72)

# ---------- [1] /health ----------
st, raw = req("GET", "/health")
body = jload(raw)
print(f"\n[1] GET /health → HTTP {st}")
if st == 200:
    print(f"    lock_backend={body.get('lock_backend')}  mineru_device={body.get('mineru_device')}")
    print(f"    residents={body.get('residents')}  resident_gb={body.get('resident_gb')}")
    if body.get("weights"):
        for k, v in body["weights"].items():
            print(f"    weights[{k}] = {json.dumps(v, ensure_ascii=False)[:160]}")
check(st == 200, "[1] /health 200", f"lock_backend={body.get('lock_backend')}")

# ---------- [2] /embed ----------
texts = ["第三季度营业收入下降百分之九点六", "毛利率环比提升两个百分点", "研发费用投入继续增加"]
t0 = time.monotonic()
st, raw = req("POST", "/embed", {"texts": texts})
el = time.monotonic() - t0
body = jload(raw)
vecs = body.get("vectors") or []
dim = body.get("dim") or (len(vecs[0]) if vecs else 0)
check(st == 200, f"[2] /embed 200（{el:.1f}s 冷启）", f"count={len(vecs)} dim={dim}")
check(len(vecs) == len(texts), "[2b] 向量条数一致", f"{len(vecs)}/{len(texts)}")
check(dim == 1024, "[2c] 维度 == 1024（BGE-large-zh-v1.5）", f"dim={dim}")
if vecs:
    import math

    norm = math.sqrt(sum(x * x for x in vecs[0]))
    check(abs(norm - 1.0) < 0.02, "[2d] 已归一化（normalize=True）", f"|v|={norm:.4f}")

# ---------- [3] /rerank ----------
docs = [
    "本季度公司整体营收较上一季度有所回落，海外渠道库存调整尚未结束",
    "毛利率受益于供应链成本下降与产品结构优化，环比提升约两个百分点",
    "公司食堂本周推出了新的菜品，员工反馈良好",
]
t0 = time.monotonic()
st, raw = req("POST", "/rerank", {"query": "营收为什么下降", "documents": docs, "top_n": 3})
el = time.monotonic() - t0
body = jload(raw)
res = body.get("results") or []
check(st == 200, f"[3] /rerank 200（{el:.1f}s 冷启）", f"n={len(res)}")
if res:
    for item in res:
        print(f"      idx={item.get('index')} score={item.get('score')} {str(item.get('document'))[:30]!r}")
    scores = [float(i.get("score", 0)) for i in res]
    check(all(0.0 <= s <= 1.0 for s in scores), "[3b] 分数在 0~1（已过 sigmoid）", str([round(s, 4) for s in scores]))
    check(res[0].get("index") == 0, "[3c] 最相关的是 index=0（营收下降那条）", f"top={res[0].get('index')}")
    check(len(res) <= 3, "[3d] top_n 生效", f"n={len(res)}")

# ---------- [4] /vlm/caption ----------
t0 = time.monotonic()
st, raw = req("POST", "/vlm/caption", {"image_path": IMAGE_PATH, "context": "季度营收趋势页"})
el = time.monotonic() - t0
body = jload(raw)
caption = str(body.get("caption") or "")
check(st == 200, f"[4] /vlm/caption 200（{el:.1f}s）", caption[:160] if caption else raw[:160])
check(bool(caption), "[4b] caption 非空", f"len={len(caption)}")

# ---------- [4c] 越权路径必须 400 ----------
st, raw = req("POST", "/vlm/caption", {"image_path": "/etc/passwd"})
check(st == 400, f"[4c] 路径越界 → 400（实际 {st}）", raw[:140])

# ---------- [5] /admin/unload ----------
st, raw = req("POST", "/admin/unload", {"key": None})
body = jload(raw)
check(st == 200, f"[5] /admin/unload 200（实际 {st}）", raw[:160])
print(f"    unloaded={body.get('unloaded')}  resident_gb={body.get('resident_gb')}")

# ---------- [6] 卸载后 /health ----------
st, raw = req("GET", "/health")
body = jload(raw)
check(st == 200 and not (body.get("residents") or []), "[6] 卸载后 residents 为空", str(body.get("residents")))

print("-" * 72)
passed = sum(1 for ok, _ in results if ok)
print(f"通过 {passed}/{len(results)}")
for ok, label in results:
    if not ok:
        print("  FAILED: " + label)
print("=" * 72)
raise SystemExit(0 if passed == len(results) else 1)
