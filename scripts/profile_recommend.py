#!/usr/bin/env python
"""OpenOneRec 推荐推理 torch.profiler 性能分析 (CPU)

用 torch.profiler 抓取 model.generate 的执行 trace:
  - 记录算子级 CPU 耗时、调用栈、内存分配
  - 输出 Chrome trace JSON (可在 chrome://tracing 可视化)
  - 输出文本版性能统计 (top 算子、调用栈)

用法:
  python scripts/profile_recommend.py
  python scripts/profile_recommend.py --model OpenOneRec/OneRec-1.7B --rec-tokens 60
  python scripts/profile_recommend.py --burn-in 3 --profile 10   # 跳过前3步, profile 10步

可视化:
  1. 用 Chrome 打开 chrome://tracing
  2. Load 输出的 .json.gz 文件
"""

import argparse
import os
import gzip
import json
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_recommend import (
    SID_FORMAT,
    make_sid,
    parse_sids,
)


def build_video_rec_prompt(tokenizer, enable_thinking=False):
    """构建视频推荐 prompt (与 run_recommend 一致)"""
    history = ''.join([
        make_sid(340, 6566, 5603),
        make_sid(102, 3400, 1289),
        make_sid(7821, 1200, 4456),
        make_sid(340, 6566, 5603),
        make_sid(560, 7800, 2341),
    ])
    messages = [
        {
            "role": "system",
            "content": "你是一个智能推荐助手，能够根据用户的浏览历史预测用户可能感兴趣的下一个内容。",
        },
        {
            "role": "user",
            "content": f"根据以下用户浏览记录，请预测用户接下来可能观看的内容：\n{history}",
        },
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer([text], return_tensors="pt")


def main():
    parser = argparse.ArgumentParser(
        description="OpenOneRec 推荐推理 torch.profiler 性能分析 (CPU)"
    )
    parser.add_argument("--model", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--rec-tokens", type=int, default=60,
                        help="生成 token 数 (默认60)")
    parser.add_argument("--think", action="store_true",
                        help="启用思考模式")
    parser.add_argument("--torch-dtype", default="float32",
                        choices=["float32", "bfloat16", "auto"])
    parser.add_argument("--burn-in", type=int, default=0,
                        help="profile 前跳过的预热步数 (不记录)")
    parser.add_argument("--output-dir", default="./trace_output",
                        help="trace 输出目录")
    parser.add_argument("--record-shapes", action="store_true",
                        help="记录算子输入 shape (trace 更大但信息更多)")
    parser.add_argument("--with-stack", action="store_true",
                        help="记录 Python 调用栈 (可用 torch.profiler 可视化)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[init] 模型: {args.model} | dtype: {args.torch_dtype} | device: cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=args.torch_dtype, device_map="cpu",
    )
    model.eval()
    print(f"[init] 模型加载完成")

    inputs = build_video_rec_prompt(tokenizer, enable_thinking=args.think)
    print(f"[init] prompt token 数: {inputs.input_ids.shape[1]}")
    print(f"[init] 生成配置: max_new_tokens={args.rec_tokens}, "
          f"burn_in={args.burn_in}, think={args.think}\n")

    # ---- 预热 (不记录) ----
    for i in range(args.burn_in):
        t0 = time.time()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=args.rec_tokens,
                do_sample=True,
                top_p=0.95,
                top_k=20,
                temperature=0.75,
            )
        dt = time.time() - t0
        print(f"[warmup {i+1}/{args.burn_in}] {dt:.2f}s")

    # ---- Profile ----
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_name = f"trace_{args.model.replace('/', '_')}_{args.rec_tokens}tok"

    print("=" * 60)
    print("  开始 Profiling ...")
    print("=" * 60)

    t0 = time.time()
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.rec_tokens,
                do_sample=True,
                top_p=0.95,
                top_k=20,
                temperature=0.75,
            )
        prof.step()

    dt = time.time() - t0
    gen_ids = output[0][len(inputs.input_ids[0]):]
    n_tokens = len(gen_ids)
    print(f"\n[done] 生成 {n_tokens} tokens, 总耗时 {dt:.2f}s "
          f"({n_tokens / max(dt, 1e-6):.2f} tok/s)")

    # ---- 输出 1: Chrome trace JSON (gzip) ----
    trace_json_path = os.path.join(args.output_dir, f"{trace_name}.json.gz")
    prof.export_chrome_trace(trace_json_path)
    print(f"\n[trace] Chrome trace: {trace_json_path}")
    print(f"        用 Chrome 打开 chrome://tracing → Load 该文件")

    # ---- 输出 2: 文本版 top 算子统计 ----
    stats_path = os.path.join(args.output_dir, f"{trace_name}_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Device: cpu | dtype: {args.torch_dtype}\n")
        f.write(f"Prompt tokens: {inputs.input_ids.shape[1]}\n")
        f.write(f"Generated tokens: {n_tokens}\n")
        f.write(f"Total time: {dt:.2f}s ({n_tokens / max(dt, 1e-6):.2f} tok/s)\n")
        f.write("=" * 80 + "\n\n")

        f.write("[Top 20 CPU 算子 (按自身耗时)]\n")
        f.write(str(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=20
        )))
        f.write("\n\n")

        f.write("[Top 20 CPU 算子 (按总耗时)]\n")
        f.write(str(prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=20
        )))
        f.write("\n\n")

        f.write("[Top 20 CPU 算子 (按 CPU 内存分配)]\n")
        f.write(str(prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=20
        )))
        f.write("\n\n")

        f.write("[Top 20 CPU 算子 (按调用次数)]\n")
        f.write(str(prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=20
        )))

    print(f"[trace] 统计报告: {stats_path}")

    # ---- 控制台摘要 ----
    print("\n" + "=" * 60)
    print("  Top 15 CPU 算子 (按自身耗时)")
    print("=" * 60)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=15
    ))

    # ---- 解析推荐结果 ----
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    sids = parse_sids(raw)
    print(f"\n[result] 推荐了 {len(sids)} 个视频:")
    for i, (c0, c1, c2) in enumerate(sids, 1):
        print(f"  {i}. {make_sid(c0, c1, c2)}")

    print(f"\n{'=' * 60}")
    print(f"  Profiling 完成")
    print(f"  Chrome trace: {trace_json_path}")
    print(f"  统计报告:     {stats_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
