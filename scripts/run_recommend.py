#!/usr/bin/env python
"""OpenOneRec 推荐任务演示 (CPU)

演示 OneRec 模型的推荐能力:
  1. 视频推荐 — 根据用户浏览历史(itemic tokens)预测下一个视频
  2. 标签预测 — 判断用户是否会长时观看某视频 (是/否)
  3. 物品理解 — 根据视频SID生成文字描述 (可选 --understand)

物品以 itemic tokens 表示: <|sid_begin|><s_a_X><s_b_Y><s_c_Z><|sid_end|>
用户浏览历史为多个 SID 的直接拼接(无分隔符)。

用法:
  python scripts/run_recommend.py
  python scripts/run_recommend.py --think            # 推荐任务使用思考模式
  python scripts/run_recommend.py --understand        # 额外演示物品理解
  python scripts/run_recommend.py --model OpenOneRec/OneRec-8B
  python scripts/run_recommend.py --real-data         # 使用真实测试集 SID
"""

import argparse
import json
import os
import re
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModelForCausalLM, AutoTokenizer

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
THINK_END_ID = 151668
CODEBOOK_SIZE = 8192


def make_sid(c0, c1, c2):
    return SID_FORMAT.format(c0=c0, c1=c1, c2=c2)


def parse_sids(text):
    """从模型输出文本中解析所有 SID 三元组"""
    pattern = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'
    return [(int(a), int(b), int(c)) for a, b, c in re.findall(pattern, text)]


def sid_to_key(c0, c1, c2):
    """将 SID 三元组转为 sid2pid.json 的 key"""
    return str(c0 * CODEBOOK_SIZE * CODEBOOK_SIZE + c1 * CODEBOOK_SIZE + c2)


def lookup_pid(sid_tuple, sid2pid):
    """SID 三元组 -> PID (取 count 最大的)"""
    key = sid_to_key(*sid_tuple)
    entries = sid2pid.get(key)
    if not entries:
        return None
    best = max(entries, key=lambda e: e.get("count", 0))
    return best["pid"]


def load_sid2pid(data_dir):
    """加载 sid2pid.json"""
    path = os.path.join(data_dir, "benchmark_data", "sid2pid.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_real_video_sample(data_dir, index=0):
    """从 video_test.parquet 加载一条真实样本"""
    import pandas as pd
    path = os.path.join(data_dir, "benchmark_data", "video", "video_test.parquet")
    df = pd.read_parquet(path)
    row = df.iloc[index]
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    messages = _convert_messages(messages)
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    return messages, meta


def load_real_label_pred_sample(data_dir, index=0):
    """从 label_pred_test.parquet 加载一条真实样本"""
    import pandas as pd
    path = os.path.join(data_dir, "benchmark_data", "label_pred", "label_pred_test.parquet")
    df = pd.read_parquet(path)
    row = df.iloc[index]
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    messages = _convert_messages(messages)
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    return messages, meta


def _convert_messages(messages):
    """将 content 从 list 格式转为纯字符串: [{"type":"text","text":"..."}] -> "..." """
    converted = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content
                          if isinstance(item, dict) and item.get("type") == "text"]
            content = "".join(text_parts)
        converted.append({"role": msg["role"], "content": content})
    return converted


def split_thinking(output_ids, tokenizer):
    """将生成 token 分为 thinking 和 content 两部分"""
    try:
        idx = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        idx = 0
    thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
    return thinking, content


def run_video_rec(model, tokenizer, max_new_tokens=60, enable_thinking=False,
                  real_sample=None, sid2pid=None):
    """视频推荐: 根据浏览历史预测下一个视频"""
    print("\n" + "=" * 60)
    print("  任务 1: 视频推荐 (Video Recommendation)")
    print("=" * 60)

    if real_sample:
        messages, meta = real_sample
        ground_truth_pids = meta.get("answer_pid", [])
        ground_truth_sids = parse_sids(meta.get("answer", ""))
        user_content = messages[-1]["content"]
        history_sids = parse_sids(user_content)
        print(f"  [真实数据] 样本 #{meta.get('uuid', '?')}, "
              f"uid={meta.get('uid', '?')}")
    else:
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
        history_sids = parse_sids(history)
        ground_truth_pids = None
        ground_truth_sids = None

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"\n  用户浏览历史 ({len(history_sids)}个视频的 itemic tokens):")
    for sid in history_sids[:10]:
        print(f"    - {make_sid(*sid)}")
    if len(history_sids) > 10:
        print(f"    - ... (共 {len(history_sids)} 个)")
    mode = "thinking" if enable_thinking else "no_think"
    print(f"\n  生成中 [{mode}, max_new_tokens={max_new_tokens}] ...")

    t0 = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=20,
        temperature=0.75,
    )
    dt = time.time() - t0
    gen_ids = output[0][len(inputs.input_ids[0]):]
    n_tokens = len(gen_ids)
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)

    print(f"  完成: {n_tokens} tokens, {dt:.1f}s ({n_tokens / max(dt, 1e-6):.2f} tok/s)")

    sids = parse_sids(raw)
    if sids:
        print(f"\n  >>> 推荐了 {len(sids)} 个视频:")
        for i, sid_tuple in enumerate(sids, 1):
            pid_str = ""
            if sid2pid:
                pid = lookup_pid(sid_tuple, sid2pid)
                pid_str = f"  -> pid={pid}" if pid else "  -> pid=未找到"
            print(f"      {i}. {make_sid(*sid_tuple)}{pid_str}")
    else:
        clean = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"\n  >>> 未解析出 SID, 原始输出:")
        print(f"      {clean[:300]}")
        if not enable_thinking:
            print(f"  (提示: 可尝试 --think 启用思考模式)")

    if ground_truth_pids:
        print(f"\n  [Ground Truth] 真实下一个视频 PID (前10个):")
        for i, pid in enumerate(ground_truth_pids[:10], 1):
            print(f"      {i}. pid={pid}")
        if sids and sid2pid:
            predicted_pids = set()
            for sid_tuple in sids:
                pid = lookup_pid(sid_tuple, sid2pid)
                if pid:
                    predicted_pids.add(pid)
            gt_set = set(ground_truth_pids)
            hits = predicted_pids & gt_set
            print(f"\n  [评估] 预测 PID 命中: {len(hits)}/{len(gt_set)} "
                  f"(预测 {len(predicted_pids)} 个, GT {len(gt_set)} 个)")
            if hits:
                print(f"      命中 PID: {hits}")

    return sids


def run_label_pred(model, tokenizer, max_new_tokens=1, real_sample=None):
    """标签预测: 判断用户是否会长时观看某视频"""
    print("\n" + "=" * 60)
    print("  任务 2: 标签预测 (Label Prediction)")
    print("=" * 60)

    if real_sample:
        messages, meta = real_sample
        user_content = messages[-1]["content"]
        history_sids = parse_sids(user_content)
        gt_label = meta.get("label", meta.get("answer", "?"))
        print(f"  [真实数据] uid={meta.get('uid', '?')}, "
              f"GT label={gt_label}")
    else:
        history = ''.join([
            make_sid(340, 6566, 5603),
            make_sid(102, 3400, 1289),
            make_sid(7821, 1200, 4456),
        ])
        candidate = make_sid(560, 7800, 2341)
        messages = [
            {
                "role": "system",
                "content": "你是一个内容推荐专家，擅长分析用户的互动模式，预测用户的内容偏好。",
            },
            {
                "role": "user",
                "content": f"用户长时观看过以下内容：{history}\n请判断用户是否会长时观看视频{candidate}？",
            },
        ]
        history_sids = parse_sids(history)
        gt_label = None

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"\n  用户历史: {len(history_sids)}个视频")
    print(f"  生成中 [greedy, max_new_tokens={max_new_tokens}] ...")

    t0 = time.time()
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    dt = time.time() - t0
    gen_ids = output[0][len(inputs.input_ids[0]):]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    print(f"  完成: {dt:.1f}s")
    will_watch = "是" in answer
    print(f"\n  >>> 预测: 用户{'会' if will_watch else '不会'}长时观看该视频")
    print(f"      (模型输出: {answer})")

    if gt_label is not None:
        gt_str = "是" if str(gt_label) in ("1", "是") else "否"
        correct = (will_watch and gt_str == "是") or (not will_watch and gt_str == "否")
        print(f"      (Ground Truth: {gt_str}, {'正确' if correct else '错误'})")

    return answer


def run_item_understand(model, tokenizer, sid_codes, max_new_tokens=200):
    """物品理解: 根据视频SID生成文字描述"""
    print("\n" + "=" * 60)
    print("  任务 3: 物品理解 (Item Understanding)")
    print("=" * 60)

    sid = make_sid(*sid_codes)
    messages = [
        {
            "role": "system",
            "content": "你是一名视频描述生成器，请根据下面的视频token生成视频描述。",
        },
        {
            "role": "user",
            "content": f"请描述 {sid} 的内容",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"\n  视频SID: {sid}")
    print(f"  生成中 [thinking, max_new_tokens={max_new_tokens}] ...")

    t0 = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=20,
        temperature=0.75,
    )
    dt = time.time() - t0
    gen_ids = output[0][len(inputs.input_ids[0]):]
    n_tokens = len(gen_ids)

    thinking, content = split_thinking(gen_ids.tolist(), tokenizer)

    print(f"  完成: {n_tokens} tokens, {dt:.1f}s ({n_tokens / max(dt, 1e-6):.2f} tok/s)")
    if thinking:
        print(f"\n  [思考过程]")
        print(f"    {thinking[:500]}")
    print(f"\n  >>> 视频描述:")
    print(f"    {content[:500]}")


def main():
    parser = argparse.ArgumentParser(description="OpenOneRec 推荐任务演示 (CPU)")
    parser.add_argument("--model", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--rec-tokens", type=int, default=60,
                        help="视频推荐最大生成token数 (默认60, 约10个推荐)")
    parser.add_argument("--label-tokens", type=int, default=1,
                        help="标签预测最大生成token数 (默认1, 是/否)")
    parser.add_argument("--think", action="store_true",
                        help="推荐任务使用思考模式 (更慢但可能更准)")
    parser.add_argument("--understand", action="store_true",
                        help="额外演示物品理解任务")
    parser.add_argument("--understand-tokens", type=int, default=200)
    parser.add_argument("--torch-dtype", default="auto",
                        choices=["float32", "bfloat16", "auto"],
                        help="模型精度 (GPU默认auto, CPU建议float32)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="运行设备 (默认auto: 有GPU用cuda, 否则cpu)")
    parser.add_argument("--real-data", action="store_true",
                        help="使用 RecIF 测试集真实 SID (需先下载数据集)")
    parser.add_argument("--data-dir", default="raw_data/onerec_data",
                        help="RecIF 数据集目录 (默认 raw_data/onerec_data)")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="使用测试集第几条样本 (默认0)")
    args = parser.parse_args()

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch_dtype = args.torch_dtype
    if torch_dtype == "auto":
        torch_dtype = "bfloat16" if device == "cuda" else "float32"

    print(f"[init] 模型: {args.model} | device: {device} | dtype: {torch_dtype}")
    if args.real_data:
        print(f"[init] 真实数据模式: {args.data_dir}, 样本 #{args.sample_index}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device,
    )
    print("[init] 模型加载完成\n")

    sid2pid = None
    video_sample = None
    label_sample = None

    if args.real_data:
        print("[data] 加载 sid2pid 映射 ...")
        sid2pid = load_sid2pid(args.data_dir)
        print(f"[data] sid2pid: {len(sid2pid)} 条映射")

        print(f"[data] 加载 video_test 样本 #{args.sample_index} ...")
        video_sample = load_real_video_sample(args.data_dir, args.sample_index)

        print(f"[data] 加载 label_pred_test 样本 #{args.sample_index} ...")
        label_sample = load_real_label_pred_sample(args.data_dir, args.sample_index)
        print()

    rec_sids = run_video_rec(
        model, tokenizer,
        max_new_tokens=args.rec_tokens,
        enable_thinking=args.think,
        real_sample=video_sample,
        sid2pid=sid2pid,
    )

    run_label_pred(
        model, tokenizer,
        max_new_tokens=args.label_tokens,
        real_sample=label_sample,
    )

    if args.understand:
        target_sid = rec_sids[0] if rec_sids else (340, 6566, 5603)
        run_item_understand(
            model, tokenizer,
            sid_codes=target_sid,
            max_new_tokens=args.understand_tokens,
        )

    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
