#!/usr/bin/env python3
"""
非 LLM overhead 基准测试

测量推理 pipeline 中与 LLM 无关的各阶段时延分布（p50/p95/p99）：
  - Stage 2a: PID -> SID 转换
  - Stage 2b: apply_chat_template
  - Stage 3 : tokenizer.encode
  - Stage 6 : tokenizer.decode
  - Stage 7 : SID 正则解析
  - Stage 8 : SID -> PID lookup

每个阶段重复 N 次（默认200），输出时延分布。
支持打印框架版本信息，便于对比不同环境。

用法:
  python scripts/bench_nonllm.py
  python scripts/bench_nonllm.py --n 500 --model OpenOneRec/OneRec-1.7B
  python scripts/bench_nonllm.py --hist-len 100   # 限制历史序列长度
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
CODEBOOK_SIZE = 8192
DATA_DIR = str(Path(__file__).parent.parent / "raw_data" / "onerec_data")


def make_sid(c0, c1, c2):
    return SID_FORMAT.format(c0=c0, c1=c1, c2=c2)


def parse_sids(text):
    return [(int(a), int(b), int(c))
            for a, b, c in re.findall(r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>', text)]


def sid_to_key(c0, c1, c2):
    return str(c0 * CODEBOOK_SIZE * CODEBOOK_SIZE + c1 * CODEBOOK_SIZE + c2)


def stats(times_ms):
    a = np.array(times_ms)
    return {
        "min":  float(np.min(a)),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
        "max":  float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def print_stats(label, times_ms):
    s = stats(times_ms)
    print(f"  {label:<36} "
          f"p50={s['p50']:6.2f}ms  p95={s['p95']:6.2f}ms  "
          f"p99={s['p99']:6.2f}ms  mean={s['mean']:6.2f}ms  "
          f"min={s['min']:6.2f}ms  max={s['max']:6.2f}ms")


def print_env_info():
    """打印当前环境的关键包版本。"""
    print("\n[env]")
    pkgs = ["transformers", "tokenizers", "torch", "vllm", "sglang"]
    for pkg in pkgs:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg:<16} {ver}")
        except ImportError:
            print(f"  {pkg:<16} (not installed)")
    print(f"  python           {sys.version.split()[0]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="非 LLM overhead 基准测试")
    parser.add_argument("--model", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--n", type=int, default=200,
                        help="每个阶段的重复次数（默认200）")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="使用 video_test.parquet 第几条样本")
    parser.add_argument("--hist-len", type=int, default=None,
                        help="截断历史序列到此长度（None=全量）")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="扫描不同历史长度下的 encode 时延")
    parser.add_argument("--batch", action="store_true",
                        help="测试 batch tokenize（串行 vs batch 接口 vs 并行）")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="batch 大小（默认16）")
    args = parser.parse_args()

    print_env_info()

    # ── 加载基础数据 ──────────────────────────────────────────────────────────
    print("[data] 加载 pid2sid ...")
    t0 = time.time()
    df_p2s = pd.read_parquet(os.path.join(args.data_dir, "video_ad_pid2sid.parquet"))
    pid2sid = dict(zip(df_p2s['pid'], df_p2s['sid']))
    print(f"       {len(pid2sid):,} 条  {time.time()-t0:.2f}s")

    print("[data] 加载 sid2pid ...")
    t0 = time.time()
    with open(os.path.join(args.data_dir, "benchmark_data/sid2pid.json")) as f:
        sid2pid = json.load(f)
    print(f"       {len(sid2pid):,} 条  {time.time()-t0:.2f}s")

    path = os.path.join(args.data_dir, "benchmark_data/video/video_test.parquet")
    df_test = pd.read_parquet(path)
    row = df_test.iloc[args.sample_index]
    hist_pids = row['hist_pid'].tolist()
    if args.hist_len:
        hist_pids = hist_pids[:args.hist_len]
    print(f"[data] 样本 #{args.sample_index}: 历史长度={len(hist_pids)} PID\n")

    # ── 加载 tokenizer ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    print(f"[init] 加载 tokenizer: {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[init] 加载完成  {time.time()-t0:.2f}s\n")

    # ── 预先构建一次，用于后续解码 benchmark ─────────────────────────────────
    # Stage 2a: PID -> SID（在 benchmark 外预热一次）
    sid_strs = []
    for pid in hist_pids:
        sid = pid2sid.get(pid)
        if sid is None:
            continue
        sid_strs.append(make_sid(int(sid[0]), int(sid[1]), int(sid[2])))
    history_text = ''.join(sid_strs)
    messages = [
        {"role": "system",
         "content": "你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。请根据历史序列推荐后续视频。"},
        {"role": "user", "content": history_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    input_ids = inputs.input_ids
    n_input_tokens = input_ids.shape[1]

    # 构造一段假的"模型输出"，包含几个 SID，用于 Stage 6/7 benchmark
    fake_output_text = ''.join(make_sid(i * 100 % 8192, i * 200 % 8192, i * 300 % 8192)
                               for i in range(10))
    fake_output_ids = tokenizer([fake_output_text], return_tensors="pt").input_ids[0]

    # Stage 8 的 lookup 数据
    sids_for_lookup = [(i * 100 % 8192, i * 200 % 8192, i * 300 % 8192) for i in range(10)]

    N = args.n
    print(f"[bench] 每阶段重复 {N} 次")
    print(f"        历史长度={len(hist_pids)} PID / {len(sid_strs)} SID")
    print(f"        prompt 字符长度={len(prompt_text)}  token 数={n_input_tokens}\n")
    print(f"{'='*90}")

    # ── Stage 2a: PID -> SID 转换 ─────────────────────────────────────────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        sid_strs_local = []
        for pid in hist_pids:
            sid = pid2sid.get(pid)
            if sid is not None:
                sid_strs_local.append(make_sid(int(sid[0]), int(sid[1]), int(sid[2])))
        _ = ''.join(sid_strs_local)
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 2a  PID->SID ({len(hist_pids)} pids)", times)

    # ── Stage 2b: apply_chat_template ────────────────────────────────────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 2b  apply_chat_template", times)

    # ── Stage 3: tokenizer([...], return_tensors="pt") — batch 包装方式 ────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = tokenizer([prompt_text], return_tensors="pt")
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 3a  tokenizer([...], return_tensors=pt) ({n_input_tokens} tok)", times)

    # ── Stage 3b: tokenizer.encode() — 绕过 batch 包装 ───────────────────────
    import torch
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        ids = tokenizer.encode(prompt_text)
        _ = torch.tensor([ids])
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 3b  tokenizer.encode() + torch.tensor ({n_input_tokens} tok)", times)

    # ── Stage 3c: tokenizer.encode() 纯 encode，不转 tensor ──────────────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = tokenizer.encode(prompt_text)
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 3c  tokenizer.encode() only ({n_input_tokens} tok)", times)

    # ── Stage 6: tokenizer.decode ─────────────────────────────────────────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = tokenizer.decode(fake_output_ids, skip_special_tokens=False)
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 6   tokenizer.decode ({len(fake_output_ids)} tokens)", times)

    # ── Stage 7: SID 正则解析 ─────────────────────────────────────────────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = parse_sids(fake_output_text)
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 7   parse_sids ({len(sids_for_lookup)} sids)", times)

    # ── Stage 8: SID -> PID lookup ────────────────────────────────────────────
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        for c0, c1, c2 in sids_for_lookup:
            key = sid_to_key(c0, c1, c2)
            entries = sid2pid.get(key, [])
            if entries:
                _ = max(entries, key=lambda e: e.get("count", 0))
        times.append((time.perf_counter() - t0) * 1000)
    print_stats(f"Stage 8   sid2pid lookup ({len(sids_for_lookup)} sids)", times)

    print(f"{'='*90}\n")

    # ── Sweep: 不同历史长度下的 encode 时延 ──────────────────────────────────
    if args.sweep:
        all_hist_pids = df_test.iloc[args.sample_index]['hist_pid'].tolist()
        max_len = len(all_hist_pids)
        sweep_lens = [10, 25, 50, 100, 150, 200, 300, 400]
        sweep_lens = [l for l in sweep_lens if l < max_len] + [max_len]

        print(f"{'='*70}")
        print(f"  Sweep: encode 时延 vs 历史长度  (N={N})")
        print(f"{'='*70}")
        print(f"  {'hist_len':>10}  {'n_tokens':>10}  {'p50':>8}  {'p95':>8}  {'mean':>8}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

        for hlen in sweep_lens:
            pids = all_hist_pids[:hlen]
            sids = []
            for pid in pids:
                sid = pid2sid.get(pid)
                if sid is not None:
                    sids.append(make_sid(int(sid[0]), int(sid[1]), int(sid[2])))
            hist_text = ''.join(sids)
            msgs = [
                {"role": "system", "content": "你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。请根据历史序列推荐后续视频。"},
                {"role": "user", "content": hist_text},
            ]
            pt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            n_tok = len(tokenizer.encode(pt))

            # 预热
            for _ in range(5):
                tokenizer.encode(pt)

            times = []
            for _ in range(N):
                t0 = time.perf_counter()
                tokenizer.encode(pt)
                times.append((time.perf_counter() - t0) * 1000)
            s = stats(times)
            print(f"  {hlen:>10}  {n_tok:>10}  {s['p50']:>7.2f}ms  {s['p95']:>7.2f}ms  {s['mean']:>7.2f}ms")

        print(f"{'='*70}\n")

    # ── Batch: 串行 vs batch 接口 vs enable_parallelism ──────────────────────
    if args.batch:
        B = args.batch_size
        # 取 B 个不同样本的 prompt（不足则复用）
        total_samples = len(df_test)
        batch_prompts = []
        for i in range(B):
            row = df_test.iloc[i % total_samples]
            pids = row['hist_pid'].tolist()
            if args.hist_len:
                pids = pids[:args.hist_len]
            sids = []
            for pid in pids:
                sid = pid2sid.get(pid)
                if sid is not None:
                    sids.append(make_sid(int(sid[0]), int(sid[1]), int(sid[2])))
            msgs = [
                {"role": "system", "content": "你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。请根据历史序列推荐后续视频。"},
                {"role": "user", "content": ''.join(sids)},
            ]
            batch_prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            ))

        token_counts = [len(tokenizer.encode(p)) for p in batch_prompts]
        print(f"{'='*80}")
        print(f"  Batch tokenize  (batch_size={B}, N={N})")
        print(f"  token 数范围: {min(token_counts)}~{max(token_counts)}, "
              f"均值={sum(token_counts)/len(token_counts):.0f}")
        print(f"{'='*80}")

        par_env = os.environ.get("TOKENIZERS_PARALLELISM", "true")

        def run_serial(prompts):
            for p in prompts:
                tokenizer.encode(p)

        def run_batch(prompts):
            tokenizer(prompts, return_tensors="pt", padding=True)

        def run_batch_no_tensor(prompts):
            tokenizer(prompts, padding=True)

        # 预热
        for _ in range(3):
            run_serial(batch_prompts)
            run_batch(batch_prompts)

        # 1) 串行 encode × B
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            run_serial(batch_prompts)
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  串行 encode ×{B}:                "
              f"总p50={s['p50']:7.2f}ms  均摊={s['p50']/B:5.2f}ms/req  "
              f"吞吐={B/s['p50']*1000:6.0f} req/s")

        # 2) batch 接口（return_tensors="pt" + padding）
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            run_batch(batch_prompts)
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  batch(return_tensors=pt, padding):  "
              f"总p50={s['p50']:7.2f}ms  均摊={s['p50']/B:5.2f}ms/req  "
              f"吞吐={B/s['p50']*1000:6.0f} req/s")

        # 3) batch 接口（不转 tensor，只 padding）
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            run_batch_no_tensor(batch_prompts)
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  batch(no tensor, padding):          "
              f"总p50={s['p50']:7.2f}ms  均摊={s['p50']/B:5.2f}ms/req  "
              f"吞吐={B/s['p50']*1000:6.0f} req/s")

        print(f"  (TOKENIZERS_PARALLELISM={par_env})")

        # ── tensor 转换方式对比 ─────────────────────────────────────────────
        print(f"\n  --- tensor 转换方式对比 (tokenize 已完成，只测转换) ---")
        enc_cache = tokenizer(batch_prompts, padding=True)

        import torch

        # 预热
        for _ in range(5):
            torch.tensor(enc_cache["input_ids"])
            np.array(enc_cache["input_ids"], dtype=np.int64)
            enc_cache.convert_to_tensors("pt", prepend_batch_axis=False)

        # A: torch.tensor(list of list)
        times = []
        for _ in range(N):
            enc = tokenizer(batch_prompts, padding=True)
            t0 = time.perf_counter()
            _ = torch.tensor(enc["input_ids"])
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  A torch.tensor(list):               p50={s['p50']:6.2f}ms  p95={s['p95']:6.2f}ms")

        # B: np.array + torch.from_numpy（零拷贝）
        times = []
        for _ in range(N):
            enc = tokenizer(batch_prompts, padding=True)
            t0 = time.perf_counter()
            arr = np.array(enc["input_ids"], dtype=np.int64)
            _ = torch.from_numpy(arr)
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  B np.array + from_numpy:            p50={s['p50']:6.2f}ms  p95={s['p95']:6.2f}ms")

        # C: BatchEncoding.convert_to_tensors
        times = []
        for _ in range(N):
            enc = tokenizer(batch_prompts, padding=True)
            t0 = time.perf_counter()
            _ = enc.convert_to_tensors("pt", prepend_batch_axis=False)
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  C BatchEncoding.convert_to_tensors: p50={s['p50']:6.2f}ms  p95={s['p95']:6.2f}ms")

        # D: 合计最优路径（tokenize + B）
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            enc = tokenizer(batch_prompts, padding=True)
            arr = np.array(enc["input_ids"], dtype=np.int64)
            _ = torch.from_numpy(arr)
            times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        print(f"  D tokenize + np.array + from_numpy: p50={s['p50']:6.2f}ms  均摊={s['p50']/B:.2f}ms/req  "
              f"吞吐={B/s['p50']*1000:.0f} req/s")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
