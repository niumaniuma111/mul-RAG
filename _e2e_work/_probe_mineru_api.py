"""MinerU 4.0 进程内 API 探针：搞清 content_list 的真实形态与图片落盘位置。

为什么必须实测
--------------
`mineru_client.py` 目前按 **2.x** 的约定推断：
    · `content_list.json` 里有 `img_path`，指向**已落盘的图片文件**
本探针要回答三个决定实现方式的问题：

    Q1 进程内 `mineru.parser.parse()` 能否不依赖 `mineru server` 跑通？
    Q2 `render_content_list()` 产出的 `img_path` 是**文件路径**还是 **doclib locator**？
    Q3 `ParseResult.save(FileBasedDataWriter)` 会把图片写到哪、叫什么名字？

用法（容器内，断网）：
    python /work/_probe_mineru_api.py /work/sample.pptx /work/out2
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PPTX = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/sample.pptx")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/work/out2")


def main() -> int:
    print("=" * 70)
    print(f"输入 = {PPTX}")
    print(f"输出 = {OUT}")
    print("=" * 70)

    try:
        from mineru.parser import parse
        from mineru.parser.writer import FileBasedDataWriter
        from mineru.render.content_list import render_content_list
    except Exception:
        print("✗ import 失败")
        traceback.print_exc()
        return 1

    # ---------- Q1：进程内解析 ----------
    try:
        result = parse(str(PPTX), tier="flash")
    except Exception:
        print("✗ parse() 失败")
        traceback.print_exc()
        return 2
    print(f"\n[Q1] ✓ 进程内 parse() 成功（未起 server），pages={len(result.pages)}")

    # ---------- Q2：content_list 形态 ----------
    try:
        items = render_content_list(result.middle_json)
    except Exception:
        print("✗ render_content_list() 失败")
        traceback.print_exc()
        return 3

    print(f"\n[Q2] content_list 条目数 = {len(items)}")
    for i, it in enumerate(items):
        # 只打关键字段，避免刷屏
        slim = {k: (f"<{len(v)} chars>" if isinstance(v, (list, str)) and len(str(v)) > 120 else v)
                for k, v in it.items()}
        print(f"  [{i}] {json.dumps(slim, ensure_ascii=False)[:300]}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "content_list.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  → 已写 {OUT / 'content_list.json'}")

    # ---------- Q3：save() 落盘 ----------
    try:
        result.save(FileBasedDataWriter(str(OUT)))
    except Exception:
        print("✗ save() 失败")
        traceback.print_exc()
        return 4
    print("\n[Q3] ✓ save() 完成")

    # ---------- 产物树 ----------
    print("\n[产物树]")
    for p in sorted(OUT.rglob("*")):
        rel = p.relative_to(OUT)
        if p.is_file():
            print(f"  {p.stat().st_size:>10}  {rel}")
        else:
            print(f"  {'<dir>':>10}  {rel}/")

    # ---------- 关键：img_path 原文 ----------
    print("\n[img_path / table_body 原文]")
    for i, it in enumerate(items):
        if it.get("type") in {"image", "chart", "table"}:
            print(f"  [{i}] type={it.get('type')}  img_path={it.get('img_path')!r}")
            if it.get("type") == "table":
                print(f"        table_body 前 120 字: {str(it.get('table_body'))[:120]!r}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
