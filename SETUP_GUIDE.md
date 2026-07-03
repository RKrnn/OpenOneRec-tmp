# OpenOneRec 实验环境搭建指南

从零开始搭建 OpenOneRec 推荐推理实验环境，包括数据下载、模型加载、推理测试和性能分析。

## 1. 环境准备

### 1.1 克隆仓库

```bash
git clone https://github.com/Kuaishou-OneRec/OpenOneRec.git
cd OpenOneRec
```

### 1.2 创建 Conda 环境

```bash
conda create -n onerec python=3.10
conda activate onerec

pip install torch==2.5.1 transformers==4.52.0
pip install pandas pyarrow huggingface_hub datasets
```

> GPU 环境请根据 CUDA 版本安装对应 PyTorch，例如：
> `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

## 2. 下载数据集

RecIF 数据集是 gated dataset，需要先在 HuggingFace 上申请访问权限：

1. 访问 https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF
2. 登录并点击 "Request access"，等待审批通过
3. 在 https://huggingface.co/settings/tokens 创建 Access Token (Read 权限)

### 2.1 下载完整数据集

```bash
export HF_TOKEN=<your_huggingface_token>

# 国内用户使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载完整 RecIF 数据集 (约 8.3GB)
hf download OpenOneRec/OpenOneRec-RecIF \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/onerec_data
```

### 2.2 仅下载映射文件 (最小化下载)

如果只需要 SID <-> PID 映射，不需测试集：

```bash
hf download OpenOneRec/OpenOneRec-RecIF \
    --repo-type dataset \
    --token $HF_TOKEN \
    --local-dir ./raw_data/onerec_data \
    --include "benchmark_data/sid2pid.json" \
    --include "benchmark_data/sid2iid.json" \
    --include "video_ad_pid2sid.parquet" \
    --include "product_pid2sid.parquet"
```

### 2.3 数据集文件说明

| 文件 | 大小 | 用途 |
|------|------|------|
| `video_ad_pid2sid.parquet` | 154M | PID -> SID 正向映射 (视频/广告域, 1588万条) |
| `product_pid2sid.parquet` | 20M | PID -> SID 正向映射 (商品域) |
| `benchmark_data/sid2pid.json` | 166M | SID -> PID 反向映射 (152万条) |
| `benchmark_data/sid2iid.json` | 201M | SID -> IID 反向映射 (商品域) |
| `benchmark_data/video/video_test.parquet` | 387M | 视频推荐测试集 (38781条) |
| `benchmark_data/label_pred/label_pred_test.parquet` | 174M | 标签预测测试集 (346190条) |
| `benchmark_data/ad/ad_test.parquet` | 76M | 广告推荐测试集 |
| `benchmark_data/product/product_test.parquet` | 93M | 商品推荐测试集 |
| `benchmark_data/interactive/interactive_test.parquet` | 12M | 交互推荐测试集 |
| `benchmark_data/label_cond/label_cond_test.parquet` | 23M | 标签条件推荐测试集 |
| `benchmark_data/item_understand/item_understand_test.parquet` | 374K | 物品理解测试集 |
| `benchmark_data/rec_reason/rec_reason_test.parquet` | 9.5M | 推荐理由测试集 |
| `onerec_bench_release.parquet` | 1.7G | 完整原始数据 |
| `pid2caption.parquet` | 5.5G | 视频描述数据 |

## 3. 运行推理

### 3.1 模型说明

| 模型 | 参数量 | 适用场景 |
|------|--------|---------|
| `OpenOneRec/OneRec-1.7B` | 1.7B | CPU 可跑 (float32/bfloat16, ~0.3 tok/s) |
| `OpenOneRec/OneRec-8B` | 8B | 需 GPU (建议 >= 16GB 显存) |

### 3.2 示例数据模式 (无需下载数据集)

```bash
# CPU, 1.7B 模型
python scripts/run_recommend.py --torch-dtype float32

# GPU, 8B 模型
python scripts/run_recommend.py --model OpenOneRec/OneRec-8B --device cuda

# 启用思考模式
python scripts/run_recommend.py --model OpenOneRec/OneRec-8B --device cuda --think

# 额外演示物品理解任务
python scripts/run_recommend.py --model OpenOneRec/OneRec-8B --device cuda --understand
```

### 3.3 真实数据模式 (需先下载数据集)

使用 RecIF 测试集中的真实 SID，推理后反查 PID 并对比 Ground Truth：

```bash
# GPU, 8B 模型, 真实数据
python scripts/run_recommend.py \
    --real-data \
    --model OpenOneRec/OneRec-8B \
    --device cuda \
    --torch-dtype bfloat16

# 指定测试集样本索引
python scripts/run_recommend.py \
    --real-data \
    --model OpenOneRec/OneRec-8B \
    --device cuda \
    --sample-index 42

# 思考模式 + 物品理解
python scripts/run_recommend.py \
    --real-data \
    --model OpenOneRec/OneRec-8B \
    --device cuda \
    --think \
    --understand
```

### 3.4 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `OpenOneRec/OneRec-1.7B` | HuggingFace 模型名 |
| `--device` | `auto` | 运行设备: auto/cpu/cuda |
| `--torch-dtype` | `auto` | 精度: auto/float32/bfloat16 (auto: GPU=bfloat16, CPU=float32) |
| `--rec-tokens` | `60` | 视频推荐最大生成 token 数 (60 token ≈ 10 个推荐) |
| `--label-tokens` | `1` | 标签预测最大生成 token 数 |
| `--think` | `False` | 推荐任务启用思考模式 |
| `--understand` | `False` | 额外演示物品理解任务 |
| `--understand-tokens` | `200` | 物品理解最大生成 token 数 |
| `--real-data` | `False` | 使用 RecIF 测试集真实 SID |
| `--data-dir` | `raw_data/onerec_data` | 数据集目录 |
| `--sample-index` | `0` | 测试集样本索引 |

## 4. 性能分析 (Profiling)

使用 `torch.profiler` 抓取推理 trace，生成 Chrome trace 文件和统计报告：

```bash
# 基本 profiling (CPU)
python scripts/profile_recommend.py --torch-dtype bfloat16 --rec-tokens 20

# GPU profiling
python scripts/profile_recommend.py --model OpenOneRec/OneRec-8B --device cuda --rec-tokens 60

# 带预热 + 记录 shape + 调用栈
python scripts/profile_recommend.py \
    --model OpenOneRec/OneRec-8B \
    --device cuda \
    --rec-tokens 60 \
    --burn-in 3 \
    --record-shapes \
    --with-stack
```

输出文件：
- `trace_output/trace_*.json.gz` — Chrome trace (用 `chrome://tracing` 打开可视化)
- `trace_output/trace_*_stats.txt` — 文本版 top 算子统计

## 5. SID 与 PID 映射说明

### SID 格式

```
<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>
```

其中 `c0`, `c1`, `c2` 是残差 K-means 量化的三层 codebook 索引 (各 0-8191)。

### SID -> PID 反查

`sid2pid.json` 的 key 格式为 `c0 * 8192^2 + c1 * 8192 + c2` 的整数字符串：

```python
import json

with open("raw_data/onerec_data/benchmark_data/sid2pid.json") as f:
    sid2pid = json.load(f)

c0, c1, c2 = 2398, 1901, 5357
key = str(c0 * 8192 * 8192 + c1 * 8192 + c2)
entries = sid2pid[key]  # [{"pid": 9445630, "count": 4}]
# 一个 SID 可能对应多个 PID (量化有损), 取 count 最大的
best_pid = max(entries, key=lambda e: e["count"])["pid"]
```

### Tokenizer 词表结构

```
Embedding 表: (176384, 2048)
  [0, 151643)         — Qwen3 原始文本 token
  [151643, 151669)    — 特殊 token (think, no_think 等)
  [151669, 159861)    — <s_a_0> ~ <s_a_8191>  (codebook layer 0)
  [159861, 168053)    — <s_b_0> ~ <s_b_8191>  (codebook layer 1)
  [168053, 176245)    — <s_c_0> ~ <s_c_8191>  (codebook layer 2)
  [176245, 176247)    — <|sid_begin|>, <|sid_end|>
```

推理时 codebook token 与普通文本 token 共享同一张 embedding 表，无特殊处理。

## 6. 快速验证清单

```bash
# 1. 验证环境
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

# 2. 验证 tokenizer
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('OpenOneRec/OneRec-1.7B')
print(f'vocab size: {len(t)}')
print(t.encode('<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>', add_special_tokens=False))
"

# 3. 示例数据快速测试 (CPU, 约2分钟)
python scripts/run_recommend.py --torch-dtype bfloat16 --rec-tokens 20

# 4. 真实数据测试 (需先下载数据集, GPU 推荐)
python scripts/run_recommend.py --real-data --model OpenOneRec/OneRec-8B --device cuda
```
