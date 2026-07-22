#!/usr/bin/env python3
"""
独立 tokenize benchmark — 可直接复制到任意机器运行

用法：
  # 基线
  python bench_tokenize_standalone.py --model /path/to/model

  # tcmalloc
  LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4 \
  python bench_tokenize_standalone.py --model /path/to/model

  # 还原 vLLM 单请求场景（如 vLLM offline）
  python bench_tokenize_standalone.py --model /path/to/model --vllm
"""

import argparse
import os
import time
from statistics import median

import numpy as np

N_WARMUP = 5
N_REPEAT = 100


def build_fake_prompt(tokenizer, n_items=200):
    """用随机 SID 构造一条 prompt，不依赖真实数据文件"""
    rng = np.random.default_rng(42)
    sids = []
    for _ in range(n_items):
        c0, c1, c2 = rng.integers(0, 8192, size=3)
        sids.append(f'<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>')
    messages = [
        {"role": "system", "content": "你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。请根据历史序列推荐后续视频。"},
        {"role": "user", "content": ''.join(sids)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def bench(fn, n_warmup, n_repeat):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型路径（本地）")
    parser.add_argument("--n-items", type=int, default=200, help="历史序列长度（默认200）")
    parser.add_argument("--batch-size", type=int, default=16, help="batch 大小（默认16）")
    parser.add_argument("--vllm", action="store_true",
                        help="用 vLLM TokenizerPool 包装（还原 vLLM 场景）")
    parser.add_argument("--parallelism", default="true",
                        choices=["true", "false"], help="TOKENIZERS_PARALLELISM")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = args.parallelism
    # rayon 线程数设为 batch_size：线程数超过任务数只会带来调度开销
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.batch_size))

    preload = os.environ.get("LD_PRELOAD", "")
    malloc_tag = "tcmalloc" if "tcmalloc" in preload else "glibc"

    # ── 加载 tokenizer ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.vllm:
        from vllm.tokenizers.hf import maybe_make_thread_pool
        tokenizer = maybe_make_thread_pool(tokenizer, copies=1)

    # ── 构造 prompts ──────────────────────────────────────────────────────────
    # vllm 模式下 tokenizer 是 TokenizerPool，用原始 tokenizer 构造 prompt
    raw = AutoTokenizer.from_pretrained(args.model) if args.vllm else tokenizer
    prompts = [build_fake_prompt(raw, args.n_items) for _ in range(args.batch_size)]
    n_tok = len(raw.encode(prompts[0]))

    mode_tag = "vLLM-TokenizerPool" if args.vllm else "直接调用"
    print(f"\n{'='*62}")
    print(f"  malloc={malloc_tag}  parallelism={args.parallelism}  mode={mode_tag}")
    print(f"  n_items={args.n_items}  token数≈{n_tok}  batch_size={args.batch_size}")
    print(f"{'='*62}")

    # ── 验证 tcmalloc 是否真正加载 ────────────────────────────────────────────
    if preload:
        pid = os.getpid()
        maps = open(f"/proc/{pid}/maps").read()
        loaded = any("tcmalloc" in line for line in maps.splitlines())
        print(f"  LD_PRELOAD tcmalloc 加载验证: {'OK' if loaded else 'FAILED — 实际仍使用 glibc'}")

    # ── 单请求 encode（还原 vLLM 逐条处理的场景）────────────────────────────
    idx = [0]
    def single_encode():
        p = prompts[idx[0] % len(prompts)]
        idx[0] += 1
        tokenizer([p])

    times_single = bench(single_encode, N_WARMUP, N_REPEAT)
    p50_s = median(times_single)
    p95_s = float(np.percentile(times_single, 95))
    print(f"\n  单请求 encode:  p50={p50_s:.2f}ms  p95={p95_s:.2f}ms")

    # ── batch encode（并行场景）───────────────────────────────────────────────
    def batch_encode():
        tokenizer(prompts, padding=True)

    times_batch = bench(batch_encode, N_WARMUP, N_REPEAT)
    p50_b = median(times_batch)
    p95_b = float(np.percentile(times_batch, 95))
    throughput = 1000 / p50_b * args.batch_size
    print(f"  batch={args.batch_size} encode: p50={p50_b:.2f}ms  p95={p95_b:.2f}ms  "
          f"吞吐={throughput:.0f} req/s")

    print(f"\n  [RESULT] malloc={malloc_tag} single={p50_s:.2f}ms batch={p50_b:.2f}ms "
          f"throughput={throughput:.0f}req/s")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
