"""
模型子进程：加载权重 + 处理推理请求。

为什么模型跑在**子进程**里（本项目最关键的一个工程决策）
----------------------------------------------------
8GB 卡上装不下 VLM + reranker + embedder 同时驻留。要在长跑容器里做
"按需加载 + 彻底换出"，唯一的办法是**kill 掉持有模型的进程**：

    del model + torch.cuda.empty_cache()  只能回收权重张量与空闲缓存块，
    回收不了 CUDA context（0.6-0.9GB）、cuBLAS/cuDNN workspace、以及显存碎片。
    这些**只在进程退出时**归还驱动。

后果是长跑容器会以"总空闲够、连续块不够"的形式 OOM —— `nvidia-smi`
显示还有空间，程序却报 OOM，排查极其反直觉。

所以本模块的每个模型都在自己的子进程里；卸载 = terminate()。

通信方式
--------
    主进程 ──req_q──▶ 子进程 ──resp_q──▶ 主进程

用 `multiprocessing.Queue` 传 **JSON 可序列化**的数据（文本进、向量/文本出）。
设计文档里提到 shared memory 传张量 —— 那对"传大张量"才有收益；
本项目的跨进程数据是文本与向量（KB~MB 级），Queue 的 pickle 开销远小于
图片预处理与推理本身，不值得引入 shared memory 的复杂度。
（进程级卸载带来的收益与 shared memory 无关，见上。）

⚠️ 子进程里**不要** import 主进程的任何业务模块（config/logging 除外）：
   会连带拉起 SQLAlchemy、httpx 等一堆用不到的东西，
   而 fork 出来的子进程内存是**共享后再写时复制**的，
   持有这些对象会拖慢 fork 并让内存统计失真。所以配置以 dict 传入。
"""

from __future__ import annotations

import os
import traceback
from typing import Any

# ★ 必须在 import torch 之前设置分配器策略：4bit 量化 + 变长输入下
#   默认分配器容易留下空洞。compose 里也设了一份，这里是双保险
#   （子进程用 spawn/fork 时环境变量继承不完全可靠）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 全程离线：禁止 transformers 去 huggingface 查更新（会拖慢启动且需要网络）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ============================================================
# 各模型的加载
# ============================================================
def _load_embedding(spec: dict[str, Any]) -> Any:
    """
    BGE-large-zh-v1.5（1024 维）。

    用 sentence-transformers 加载：BAAI 官方两种支持方式之一，
    且它自带 pooling 与 normalize，避免手工实现 cls/mean pooling 出错。
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        spec["path"],
        device=spec.get("device", "cuda:0"),
        trust_remote_code=True,
    )
    model.max_seq_length = int(spec.get("max_length", 512))
    model.eval()
    return model


def _load_reranker(spec: dict[str, Any]) -> Any:
    """
    bge-reranker-large。

    ⚠️ v1.1.7 变更：由 FlagEmbedding 改为 sentence-transformers 的 CrossEncoder。
       原因：FlagEmbedding 1.3.3 在 requires_dist 里硬钉 transformers==4.44.2，
       与 Qwen2.5-VL 需要的 >=4.49 数学上不可调和（pip 直接 ResolutionImpossible）。
       CrossEncoder 是 BAAI 官方支持的另一条加载路径，零新增依赖。
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        spec["path"],
        device=spec.get("device", "cuda:0"),
        max_length=int(spec.get("max_length", 512)),
        trust_remote_code=True,
    )
    if bool(spec.get("use_fp16", True)):
        try:
            model.model.half()
        except Exception:  # noqa: BLE001 —— 某些结构不支持直接 half，忽略即可
            pass
    model.model.eval()
    return model


def _load_vlm(spec: dict[str, Any]) -> Any:
    """
    Qwen2.5-VL-3B-Instruct，4bit nf4 量化。

    ⚠️ 两个显存关键点：
        1. `bnb_4bit_use_double_quant` —— 再省约 0.4GB，8GB 卡上是必要的
        2. `max_pixels` 在 processor 侧限制 —— 图片分辨率直接决定视觉 token 数，
           不限制的话一张 4K 截图能顶掉 1GB 显存（见 config 的 max_pixels）
    """
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    quantization_config = None
    if bool(spec.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(spec.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=getattr(
                torch, str(spec.get("bnb_4bit_compute_dtype", "float16"))
            ),
            bnb_4bit_use_double_quant=bool(spec.get("bnb_4bit_use_double_quant", True)),
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        spec["path"],
        quantization_config=quantization_config,
        device_map=spec.get("device_map", "cuda:0"),
        torch_dtype=getattr(torch, str(spec.get("torch_dtype", "float16"))),
        attn_implementation=spec.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=True,      # 逐个 shard 加载，显著降低 CPU 内存峰值
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        spec["path"],
        min_pixels=int(spec.get("min_pixels", 200704)),
        max_pixels=int(spec.get("max_pixels", 1003520)),
    )
    return {"model": model, "processor": processor}


_LOADERS = {
    "embedding": _load_embedding,
    "reranker": _load_reranker,
    "vlm": _load_vlm,
}


# ============================================================
# 各模型的推理
# ============================================================
def _do_embed(handle: Any, payload: dict[str, Any]) -> dict[str, Any]:
    texts = list(payload.get("texts") or [])
    normalize = bool(payload.get("normalize", True))
    batch_size = int(payload.get("batch_size") or 0) or 16

    vectors = handle.encode(
        texts,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return {"vectors": [v.tolist() for v in vectors], "dim": len(vectors[0]) if len(vectors) else 0}


def _do_rerank(handle: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """
    精排。返回 sigmoid 后的分数（0~1）—— config 里的 score_threshold=0.35
    就是与这个尺度比较，**不是**与原始 logits 比。
    """
    import math

    query = str(payload.get("query") or "")
    documents = list(payload.get("documents") or [])
    top_n = int(payload.get("top_n") or len(documents))

    if not documents:
        return {"results": []}

    pairs = [(query, doc) for doc in documents]
    raw_scores = handle.predict(pairs, show_progress_bar=False)

    results = []
    for index, raw in enumerate(raw_scores):
        score = float(raw)
        # bge-reranker 输出的是 logits，转成概率才有可解释的阈值
        prob = 1.0 / (1.0 + math.exp(-score)) if -700 < score < 700 else (0.0 if score < 0 else 1.0)
        results.append({"index": index, "score": prob, "document": documents[index]})

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"results": results[:top_n]}


def _build_vlm_inputs(
    handle: dict[str, Any],
    prompt: str,
    image_path: str | None,
    max_pixels: int | None = None,
    system_prompt: str | None = None,
) -> tuple[Any, Any]:
    """
    把「可选的 system 指令 + 可选的图片 + 文本指令」组装成 Qwen2.5-VL 的输入张量。

    ⚠️ system 用独立的 system 角色消息，而不是拼进 user 文本：
       Qwen2.5-VL 的 chat template 对 system 角色有专门处理，
       拼进 user 会削弱指令权重（caption 的"只描述可见事实"这类约束容易被忽略）。
    """
    content: list[dict[str, Any]] = []
    if image_path:
        image_item: dict[str, Any] = {"type": "image", "image": f"file://{image_path}"}
        if max_pixels:
            # ★ 逐请求覆盖分辨率上限：caption 阶段可以比对话阶段更激进地压
            image_item["max_pixels"] = max_pixels
        content.append(image_item)
    content.append({"type": "text", "text": prompt})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": content})

    from qwen_vl_utils import process_vision_info

    text = handle["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = handle["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(handle["model"].device), handle["processor"]


def _do_vlm(
    handle: dict[str, Any],
    payload: dict[str, Any],
    spec: dict[str, Any],
    response_queue: Any = None,
    rid: str | None = None,
) -> dict[str, Any]:
    """
    VLM 生成（caption 与 generate 共用同一条路径）。

    ★ S3 扩展：当 `payload["stream"]=True` 且调用方传入了 response_queue 时，
      切换为**流式模式** —— 用 TextIteratorStreamer 边生成边往响应队列推
      `{"event":"delta","data":{"text":片段}}`，结束后推 `{"event":"done"}`。
      返回值仍是完整文本（供内部兼容），但主进程按事件消费。
    """
    import threading

    import torch
    from transformers import TextIteratorStreamer

    prompt = str(payload.get("prompt") or "")
    image_path = payload.get("image_path") or None
    system_prompt = payload.get("system_prompt") or None
    max_new_tokens = int(payload.get("max_new_tokens") or spec.get("max_new_tokens", 512))
    max_pixels = payload.get("max_pixels")
    stream = bool(payload.get("stream")) and response_queue is not None

    inputs, processor = _build_vlm_inputs(handle, prompt, image_path, max_pixels, system_prompt)

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=float(payload.get("temperature", spec.get("temperature", 0.2))),
        top_p=float(payload.get("top_p", spec.get("top_p", 0.8))),
        repetition_penalty=float(
            payload.get("repetition_penalty", spec.get("repetition_penalty", 1.05))
        ),
    )

    if not stream:
        with torch.inference_mode():
            generated = handle["model"].generate(**inputs, **gen_kwargs)

        # 只取新生成的部分（去掉 prompt）
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated, strict=False)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = text[0].strip() if text else ""

        del inputs, generated
        torch.cuda.empty_cache()      # 及时归还本次激活占用，压低峰值
        return {"text": result}

    # ---------- 流式模式 ----------
    # TextIteratorStreamer 在**后台线程**里跑 generate，本线程迭代它吐 token：
    # 这是对 transformers 流式的官方姿势 —— generate() 本身是阻塞调用，
    # 不开线程的话 streamer 队列永远没人填。
    tokenizer = getattr(processor, "tokenizer", processor)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    def _generate() -> None:
        with torch.inference_mode():
            handle["model"].generate(**inputs, **gen_kwargs, streamer=streamer)

    gen_thread = threading.Thread(target=_generate, name="vlm-stream-gen", daemon=True)
    gen_thread.start()

    pieces: list[str] = []
    for piece in streamer:                      # 阻塞迭代，直到生成结束
        if not piece:
            continue
        pieces.append(piece)
        # 每个片段独立入队：主进程按 event 增量消费并经 SSE 转发给前端
        response_queue.put({"rid": rid, "ok": True, "event": "delta", "data": {"text": piece}})

    gen_thread.join(timeout=30)                 # streamer 结束后 generate 也应返回
    torch.cuda.empty_cache()

    full_text = "".join(pieces).strip()
    response_queue.put({"rid": rid, "ok": True, "event": "done", "data": {"text": full_text}})
    return {"text": full_text}


_HANDLERS = {
    "embedding": lambda h, p, s: _do_embed(h, p),
    "reranker": lambda h, p, s: _do_rerank(h, p),
    "vlm": _do_vlm,
}


# ============================================================
# 子进程主循环
# ============================================================
def worker_main(
    key: str,
    spec: dict[str, Any],
    request_queue: Any,
    response_queue: Any,
) -> None:
    """
    子进程入口：加载模型 → 循环取请求 → 推理 → 回结果。

    ⚠️ 加载失败也要**回一条错误响应**再退出，不能静默死掉 ——
       否则主进程会一直等 timeout（默认 180s），
       用户看到的是"卡了 3 分钟然后报超时"，而不是"模型加载失败"。
    """
    try:
        loader = _LOADERS.get(key)
        if loader is None:
            raise ValueError(f"未知的模型 key: {key}")
        handle = loader(spec)
    except Exception as exc:  # noqa: BLE001
        response_queue.put(
            {
                "rid": None,
                "ok": False,
                "error": f"模型加载失败({key}): {exc}\n{traceback.format_exc()[:2000]}",
            }
        )
        return

    # 加载成功 → 通知主进程（主进程据此判断"模型可用"）
    response_queue.put({"rid": None, "ok": True, "event": "loaded", "key": key})

    handler = _HANDLERS[key]
    while True:
        item = request_queue.get()
        if item is None:            # 终止信号
            break
        rid = item.get("rid")
        payload = item.get("payload") or {}
        try:
            # ★ 流式分支：handler 自己往 response_queue 推 delta/done 事件，
            #   返回值仅作内部兼容，这里不再额外回包（否则主进程会收到重复响应）。
            if bool(payload.get("stream")):
                handler(handle, payload, spec, response_queue=response_queue, rid=rid)
            else:
                data = handler(handle, payload, spec)
                response_queue.put({"rid": rid, "ok": True, "data": data})
        except Exception as exc:  # noqa: BLE001
            response_queue.put(
                {
                    "rid": rid,
                    "ok": False,
                    "event": "error",
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:2000]}",
                }
            )

    # 子进程退出 —— 这一步才是"显存真正归还"的时刻
    try:
        import torch

        del handle
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


__all__ = ["worker_main"]
