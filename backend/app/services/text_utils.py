"""
中文文本处理工具：分词、切句、token 估算。

为什么把 jieba 初始化放在这里（而不是各服务里各写一份）
------------------------------------------------------
jieba 加载词典是**进程级单例**且首次加载要 1-2 秒。分散初始化会导致：
    · worker 与 api 各自加载一份，内存翻倍
    · 首次调用出现难以解释的延迟毛刺

所以统一走本模块的 `tokenize()`，内部懒加载并缓存。

关于 token 估算的准确性（重要）
-----------------------------
本模块的 `estimate_tokens()` 是**启发式估算**，不是精确 tokenizer：

    BGE-large-zh-v1.5 用的是中文 BERT 词表，**汉字基本 1 字 = 1 token**，
    英文按子词切分（约 1 词 ≈ 1.3 token）。

为什么不用真实 tokenizer：
    chunker 的合并循环里要对每个候选块反复算 token 数，走 HTTP 请求
    模型服务会慢到不可用；而在 api/worker 容器里装一份 tokenizers +
    词表又违背了「api 镜像不含模型依赖」的拆分决策。

因此 `max_tokens=512` 在这里的角色是**兜底硬上限**，不是精确边界 ——
**略微超出（±10%）是可接受的**，因为下游 model-service 有 max_length=512
截断兜底。若要精确，可在 S5 换成一个独立的轻量 tokenizer。

对应设计文档：DESIGN.md §7.3
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from app.core.config import get_config
from app.core.logging import get_logger

logger = get_logger(__name__)

# 默认中文断句标点（会被 config 的 ingest.chunk.text.sentence_split_chars 覆盖）
_DEFAULT_SEPS = ["。", "！", "？", "；", "\n", ".", "!", "?"]

# 中日韩统一表意文字（含扩展 A）—— 用于区分「中文字」与「英文词」
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# ============================================================
# 分词（BM25 稀疏向量预处理）
# ============================================================
@lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    """加载停用词表。文件缺失时退化为空集合（不阻断入库）。"""
    path_str = str(
        get_config().get_path("ingest.jieba.stopwords_path", "/app/configs/stopwords_zh.txt")
    )
    path = Path(path_str)
    if not path.is_file():
        logger.warning("停用词表不存在，退化为空集合: %s", path)
        return frozenset()
    words = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    logger.info("停用词表已加载: %s 条", len(words))
    return frozenset(words)


@lru_cache(maxsize=1)
def _init_jieba() -> bool:
    """
    初始化 jieba：加载自定义词典 + 离线加载模式。

    `jieba.initialize()` 会构建前缀词典缓存（约 1 秒），显式调用它可以把
    这段延迟从「第一个请求」挪到「模块首次使用」，并且能打日志。
    """
    import jieba

    cfg = get_config()
    user_dict = str(cfg.get_path("ingest.jieba.user_dict", "/app/configs/userdict.txt"))
    if Path(user_dict).is_file():
        n = jieba.load_userdict(user_dict)
        logger.info("jieba 自定义词典已加载: %s（%s 条）", user_dict, n)
    else:
        logger.warning("jieba 自定义词典不存在，使用默认词典: %s", user_dict)

    jieba.initialize()
    return True


def tokenize(text: str) -> list[str]:
    """
    BM25 用的分词。规则来自 config.yaml 的 ingest.jieba 段：

        cut_all=false      精确模式
        lowercase=true     转小写（英文归一）
        min_term_len=2     丢弃单字，降噪（"的"、"了" 这类无区分度）
        remove_stopwords=true

    ⚠️ 单字在这里被丢弃是有意的：BM25 的价值在于"区分度"，
       单字中文词（尤其是虚词）会引入大量噪声匹配。
       这也意味着**单个汉字查询词召回率会下降** —— 这是刻意的取舍。
    """
    import jieba

    _init_jieba()
    cfg = get_config()
    cut_all = bool(cfg.get_path("ingest.jieba.cut_all", False))
    min_len = int(cfg.get_path("ingest.jieba.min_term_len", 2))
    lowercase = bool(cfg.get_path("ingest.jieba.lowercase", True))
    drop_stop = bool(cfg.get_path("ingest.jieba.remove_stopwords", True))

    # 全角转半角 + 去掉控制字符，避免 "１００" 与 "100" 被当成不同词
    normalized = unicodedata.normalize("NFKC", text)
    tokens = jieba.lcut(normalized, cut_all=cut_all)

    stops = _load_stopwords() if drop_stop else frozenset()
    result: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if lowercase:
            token = token.lower()
        if not token:
            continue
        if len(token) < min_len:
            continue
        # 纯标点/空白不算词
        if not any(ch.isalnum() or _CJK_RE.match(ch) for ch in token):
            continue
        if token in stops:
            continue
        result.append(token)
    return result


# ============================================================
# 切句
# ============================================================
def split_sentences(text: str) -> list[str]:
    """
    按中文标点切句，保留标点（标点承载语气，去掉会影响 embedding 语义）。

    实现上把分隔符统一替换成「分隔符 + \\x00」再切，这样标点会留在句尾。
    """
    seps = list(
        get_config().get_path("ingest.chunk.text.sentence_split_chars", _DEFAULT_SEPS)
        or _DEFAULT_SEPS
    )
    if not text.strip():
        return []

    pattern = "|".join(re.escape(s) for s in seps)
    parts = re.split(f"({pattern})", text)

    sentences: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if part in seps:            # 当前片段是标点 → 收束成一句
            if buffer.strip():
                sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():              # 结尾没有标点的残句
        sentences.append(buffer.strip())
    return sentences


# ============================================================
# token 估算
# ============================================================
def estimate_tokens(text: str) -> int:
    """
    估算 token 数（见模块头的准确性说明）。

    启发式：
        汉字        → 每个 1 token
        其余字母数字 → 按空格切词，每词 1.3 token（向上取整）
        其它字符    → 忽略
    """
    if not text:
        return 0

    cjk_count = len(_CJK_RE.findall(text))
    # 去掉汉字后剩下的按空白切词
    non_cjk = _CJK_RE.sub(" ", text)
    words = [w for w in non_cjk.split() if any(ch.isalnum() for ch in w)]

    return cjk_count + int(len(words) * 1.3 + 0.999)


__all__ = [
    "estimate_tokens",
    "split_sentences",
    "tokenize",
]
