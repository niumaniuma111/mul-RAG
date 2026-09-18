"""
模型权重下载（走 ModelScope 国内加速）。

为什么必须走 ModelScope
-----------------------
本项目「全程离线推理」指的是**运行时**不调任何在线大模型 API，但下载权重这一步
必须联网。而 HuggingFace 在国内直连基本拉不动 7.5GB 的模型，ModelScope（魔搭）
是阿里做的镜像站，国内速度稳定 —— 所以本项目所有模型统一从魔搭下。

下载清单
--------
    Qwen2.5-VL-3B-Instruct   ~7.5GB   fp16 原始权重（运行时用 4bit nf4 加载）
    bge-large-zh-v1.5        ~1.3GB   1024 维，检索向量化
    bge-reranker-large       ~2.2GB   精排
    MinerU 权重              ~1.5GB   解析，走 MinerU 自带的下载命令

    ⚠️ 总下载量约 12GB（VLM 权重 7.5GB 是其中大头），下到 WSL 的 ext4 里。
       不要在 /mnt/e 上建这个目录 —— 9P 桥吞吐只有原生 ext4 的 5-6%。

前置
----
    pip install modelscope          # 本脚本是运维工具，不列入 requirements.txt
    export MODELSCOPE_CACHE=~/projects/ppt-agentic-rag/models

用法
----
    cd ~/projects/ppt-agentic-rag
    python scripts/download_models.py                    # 全部下
    python scripts/download_models.py --only vlm         # 只下 VLM
    python scripts/download_models.py --check            # 只检查已有文件，不下载
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# (本地子目录名, ModelScope model id, 大致体积GB)
MODEL_MANIFEST: list[tuple[str, str, float]] = [
    ("Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", 7.5),
    ("bge-large-zh-v1.5", "BAAI/bge-large-zh-v1.5", 1.3),
    ("bge-reranker-large", "BAAI/bge-reranker-large", 2.2),
]

# MinerU 的权重不走 snapshot_download —— 它由 MinerU 自带的命令管理，
# 目录结构也和 HF 格式不同。
MINERU_SUBDIR = "mineru"

# 全局忽略规则。⚠️ 这里是 **fnmatch glob**，不是正则（见 download_one 注释）。
#   *.pth / *.gguf        —— 其他框架的转换格式，用不到
#   *original*            —— 通常是重复的原始权重
#   *.onnx / *.onnx_data  —— ONNX 推理格式。本项目全程 PyTorch（检索走
#                            sentence-transformers 的 SentenceTransformer /
#                            CrossEncoder），从不加载 onnx。而 bge-reranker-large
#                            的 onnx/model.onnx_data 有 **2.24GB**，与 PyTorch
#                            权重完全重复 —— 是整轮下载里最浪费的一段。
BASE_IGNORE_FILE_PATTERNS: list[str] = [
    "*.pth",
    "*.gguf",
    "*original*",
    "*.onnx",
    "*.onnx_data",
    "onnx/*",
]

# 逐模型的追加忽略（glob）。
#   bge-reranker-large：仓库同时提供 model.safetensors 与 pytorch_model.bin
#   （各 2.24GB，内容等价）。sentence-transformers 的 CrossEncoder 优先读
#   safetensors，故 bin 属于纯浪费 —— 2026-09-17 实测两者被**并行**下载，
#   既多占 2.24GB 又让 safetensors 的速率腰斩。这里只保 safetensors。
#   ⚠️ 不放进全局：bge-large-zh-v1.5 等仓库只有 pytorch_model.bin，
#      全局忽略 *.bin 会把它们一起废掉。
EXTRA_IGNORE: dict[str, list[str]] = {
    "bge-reranker-large": ["*.bin"],
}


def human(size_gb: float) -> str:
    return f"{size_gb:.1f}GB"


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**3)


def check_models(root: Path) -> list[tuple[str, bool, float]]:
    """返回 (名称, 是否就绪, 实际体积GB)。"""
    result = []
    for local_name, _, _ in MODEL_MANIFEST:
        path = root / local_name
        # 判据：目录存在且至少有一个权重文件（config.json 不够，那可能只是残骸）
        has_weight = path.exists() and any(
            f.suffix in {".safetensors", ".bin", ".pt"} for f in path.rglob("*")
        )
        result.append((local_name, has_weight, dir_size_gb(path)))
    return result


def ensure_modelscope() -> bool:
    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("✗ 缺少 modelscope 包。请先安装：")
        print("    pip install modelscope")
        return False
    return True


def download_one(local_name: str, model_id: str, root: Path) -> bool:
    from modelscope import snapshot_download

    target = root / local_name
    print(f"\n▶ 下载 {model_id}")
    print(f"  → {target}")
    try:
        snapshot_download(
            model_id=model_id,
            local_dir=str(target),
            # ★★ 语义陷阱（2026-09-17 实测踩坑，务必读）★★
            #   本参数匹配用的是 **fnmatch 通配符（glob）**，不是正则！
            #   依据：modelscope 1.40.1 `utils/repo_utils.py:206,211`
            #         → `fnmatch(path, r) for r in allow_patterns/ignore_patterns`
            #   早期这里写的是 `r".*\.onnx_data$"` 这类正则写法，被当**字面量 glob**，
            #   于是**一条都没生效**：onnx/model.onnx_data（2.24GB）照样下，
            #   还和 model.safetensors 并行抢带宽，单文件速率被腰斩
            #   （实测 834kB/s + 1.16MB/s，而串行时能到 5MB/s）。
            #   写法对照：❌ `.*\.bin$`   ✅ `*.bin`
            #   另一个坑：`*` 在 fnmatch 里**也能匹配 `/`**，故 `*.onnx_data`
            #   可以直接命中 `onnx/model.onnx_data`，无需写 `onnx/*`；
            #   但两个都留着更稳。
            ignore_file_pattern=(
                BASE_IGNORE_FILE_PATTERNS + EXTRA_IGNORE.get(local_name, [])
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 下载失败: {exc}")
        print("  常见原因：网络不通（检查代理）、model id 变更、磁盘空间不足")
        return False

    print(f"✓ 完成，实际体积 {dir_size_gb(target):.2f}GB")
    return True


def download_mineru(root: Path) -> bool:
    """MinerU 权重走它自己的命令（目录结构与 HF 格式不同）。"""
    target = root / MINERU_SUBDIR
    print(f"\n▶ MinerU 权重")
    print(f"  → {target}")

    cli = shutil.which("mineru-models-download")
    if cli is None:
        print("! 未找到 mineru-models-download 命令（本机 Python 环境里没装 MinerU）。")
        print("  两种处理方式，任选其一：")
        print("    a) 在 mineru 容器里下载（推荐，依赖已在镜像内）：")
        print(f"         docker compose exec mineru mineru-models-download -s modelscope -m pipeline")
        print(f"       然后把容器内 ~/.cache/mineru 拷到 {target}")
        print("    b) 直接跳过 —— MinerU 首次解析时会自己下载到缓存目录")
        return False

    target.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MINERU_MODEL_SOURCE": "modelscope"}
    try:
        subprocess.run(
            [cli, "-s", "modelscope", "-m", "pipeline", "-p", str(target)],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        print(f"✗ MinerU 权重下载失败（exit={exc.returncode}）")
        return False
    print("✓ MinerU 权重完成")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型权重")
    parser.add_argument(
        "--root",
        default=os.environ.get("MODELS_ROOT", str(Path.home() / "projects/ppt-agentic-rag/models")),
        help="模型根目录（默认 $MODELS_ROOT 或 ~/projects/ppt-agentic-rag/models）",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in MODEL_MANIFEST] + ["mineru"],
        help="只下载指定的模型",
    )
    parser.add_argument("--check", action="store_true", help="只检查现状，不下载")
    parser.add_argument("--skip-mineru", action="store_true", help="跳过 MinerU 权重")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"模型根目录: {root}")

    # 自检：必须落在 ext4 上，不能是 9P 桥（否则容器重载权重会慢几十倍）
    try:
        stat = os.statvfs(root)
        print(f"可用空间: {stat.f_bavail * stat.f_frsize / (1024**3):.1f}GB")
    except Exception:  # noqa: BLE001
        pass
    print("⚠️ 请确认本目录在 WSL 原生 ext4 上，用 `df -T .` 检查（须显示 ext4，不能是 9p）")
    print("=" * 64)

    # 现状
    print("\n当前状态：")
    total_have = 0.0
    for name, ready, size in check_models(root):
        flag = "✓" if ready else "✗"
        print(f"  {flag} {name:<26} {size:>6.2f}GB")
        total_have += size
    print(f"  合计已下载 {total_have:.2f}GB / 预期约 12GB")

    if args.check:
        return 0

    if not ensure_modelscope():
        return 1

    failures: list[str] = []
    for local_name, model_id, _ in MODEL_MANIFEST:
        if args.only and args.only != local_name:
            continue
        if not download_one(local_name, model_id, root):
            failures.append(local_name)

    if not args.skip_mineru and (not args.only or args.only == "mineru"):
        if not download_mineru(root):
            failures.append("mineru")

    print("\n" + "=" * 64)
    if failures:
        print(f"以下模型未成功：{', '.join(failures)}")
        return 1
    print("全部完成。下一步：docker compose up -d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
