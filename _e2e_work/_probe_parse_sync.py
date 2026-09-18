"""直接调用 server.py 的 _parse_sync，验证「解析 → 图片外置 → img_path 改写」。

为什么用这种"绕开 HTTP"的方式先测：
    镜像重建要十几分钟，而这里要验证的是**纯逻辑**（图片外置与路径改写）。
    直接 import 被挂载进来的 server.py 就能跑，改一行测一次，不必等镜像。
    验证通过后再重建镜像做真正的 HTTP + Celery 端到端。

用法（容器内，断网）：
    python /work/_probe_parse_sync.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

SERVER_DIR = "/proj/deploy/mineru"
sys.path.insert(0, SERVER_DIR)

PPTX = "/work/sample.pptx"
OUT = "/work/out3"


def main() -> int:
    print("=" * 70)
    try:
        import server
        from server import ParseRequest, _parse_sync
    except Exception:
        print("✗ import server 失败")
        traceback.print_exc()
        return 1

    print(f"[0] server.py 已加载   device={server.MINERU_DEVICE}  "
          f"gpu_lock_required={server.GPU_LOCK_REQUIRED}  "
          f"visual_types={sorted(server._VISUAL_TYPES)}")

    # ---------- 正常解析 ----------
    try:
        res = _parse_sync(ParseRequest(file_path=PPTX, output_dir=OUT))
    except Exception:
        print("✗ _parse_sync 失败")
        traceback.print_exc()
        return 2

    print("\n[1] _parse_sync 返回:")
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # ---------- content_list 的 img_path 必须是相对路径，且文件真的在 ----------
    cl_path = Path(res["content_list_path"])
    items = json.loads(cl_path.read_text(encoding="utf-8"))
    out_dir = Path(OUT)

    print(f"\n[2] content_list 条目 = {len(items)}")
    bad = 0
    for i, it in enumerate(items):
        t = it.get("type")
        raw = it.get("img_path")
        if t in {"image", "chart", "table"}:
            is_data_uri = isinstance(raw, str) and raw.startswith("data:")
            exists = bool(raw) and (out_dir / raw).is_file()
            flag = "✅" if (not is_data_uri and (exists or not raw)) else "❌"
            print(f"  {flag} [{i}] type={t:<6} img_path={str(raw)[:70]!r} exists={exists}")
            if is_data_uri:
                bad += 1
        elif t == "text":
            print(f"  ·  [{i}] type=text   page_idx={it.get('page_idx')}  {str(it.get('text'))[:40]!r}")

    # ---------- 产物树 ----------
    print("\n[3] 产物树:")
    for p in sorted(out_dir.rglob("*")):
        rel = p.relative_to(out_dir)
        if p.is_file():
            print(f"  {p.stat().st_size:>9}  {rel}")

    # ---------- 断言 ----------
    print("\n[4] 断言:")
    checks: list[tuple[bool, str]] = [
        (res["status"] == "ok", "status == ok"),
        (res["page_count"] == 3, f"page_count == 3（实际 {res['page_count']}）"),
        (cl_path.is_file(), "content_list.json 已落盘"),
        (res["image_dir"].endswith("/images"), "image_dir 指向 output_dir/images"),
        (bad == 0, f"没有残留 data URI（实际 {bad} 个）"),
        (res["counts"].get("table", 0) >= 1, "识别到表格"),
        (Path(res["image_dir"]).is_dir() and any(Path(res["image_dir"]).iterdir()), "images/ 非空"),
        (res["tier"] == "flash", "tier 回显正确"),
    ]
    passed = sum(1 for ok, _ in checks if ok)
    for ok, label in checks:
        print(("  ✓ " if ok else "  ✗ ") + label)
    print(f"\n通过 {passed}/{len(checks)}")
    print("=" * 70)
    return 0 if passed == len(checks) else 3


if __name__ == "__main__":
    raise SystemExit(main())
