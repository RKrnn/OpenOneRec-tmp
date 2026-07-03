"""OpenOneRec README Quick Start Demo (CPU adaptation)

基于 README 中的 Quick Start 示例，适配纯 CPU aarch64 环境:
  - 使用 hf-mirror.com 镜像下载模型 (HuggingFace 直连不可达)
  - device_map="cpu" (无 GPU/NPU)
  - torch_dtype=float32 (aarch64 CPU 对 bfloat16 支持有限)
  - 默认使用 OneRec-1.7B (8B 在 CPU 上推理极慢)
  - max_new_tokens 可通过命令行参数控制 (默认 256, 便于快速验证)

用法:
  python scripts/run_demo.py
  python scripts/run_demo.py --model OpenOneRec/OneRec-8B --max-new-tokens 512
"""

import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="OpenOneRec README Demo (CPU)")
    parser.add_argument(
        "--model",
        default="OpenOneRec/OneRec-1.7B",
        help="HuggingFace 模型名 (默认 OneRec-1.7B; 可选 OneRec-8B)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", default="float32",
                        choices=["float32", "bfloat16", "auto"])
    args = parser.parse_args()

    model_name = args.model
    print(f"[demo] 模型: {model_name}")
    print(f"[demo] dtype: {args.torch_dtype} | device: cpu | max_new_tokens: {args.max_new_tokens}")

    print("[demo] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[demo] 加载模型 (CPU) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=args.torch_dtype,
        device_map="cpu",
    )
    print(f"[demo] 模型加载完成, 耗时 {time.time() - t0:.1f}s")

    prompt = (
        "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，"
        "帮我总结一下这个视频讲述了什么内容"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("[demo] 开始生成 ...")
    t0 = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        top_p=0.95,
        top_k=20,
        temperature=0.75,
        do_sample=True,
    )
    gen_time = time.time() - t0
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    n_new = len(output_ids)
    print(f"[demo] 生成完成, 耗时 {gen_time:.1f}s, {n_new} tokens ({n_new/gen_time:.2f} tok/s)")
    print("=" * 60)
    print("thinking content:")
    print(thinking_content)
    print("-" * 60)
    print("content:")
    print(content)
    print("=" * 60)


if __name__ == "__main__":
    main()
