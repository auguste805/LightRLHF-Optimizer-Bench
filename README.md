# LightRLHF-Optimizer-Bench 🚀

[English](#english) | [中文](#中文)

<a name="english"></a>
## English

### 🎯 Project Objective
This repository provides a lightweight, highly reproducible framework for training **Reward Models (RM)** in the RLHF pipeline. It is specifically designed to allow researchers to explore preference learning dynamics using standard baseline optimization algorithms (e.g., **AdamW** and **SGD**) on consumer-grade GPUs.

By using **GPT-2 Small** combined with **LoRA**, this project lowers the compute barrier, making it a perfect tool for understanding the core of model alignment.

### 🧠 Model Architecture & Objective
- **Base model:** `gpt2` (GPT-2 Small, ~124M parameters). The code loads `transformers.GPT2Model` (no LM head).
- **Reward Model Head:** A scalar value head (`nn.Linear(hidden_size, 1)`). For each sequence, it takes the **last non-pad token** hidden state (based on `attention_mask`) and maps it to a reward scalar.
- **LoRA Fine-tuning:** Enabled by default. Only LoRA adapters are trained in the GPT-2 backbone; the **value head is trained fully**. Target modules: `c_attn`, `c_fc`, `c_proj` (Task type: `FEATURE_EXTRACTION`).
- **Training Objective:** Bradley-Terry pairwise ranking loss on preference pairs:
  $$\mathcal{L} = -\log\sigma(r_w - r_l)$$
  where $r_w$ is the reward for the **chosen** response and $r_l$ is the reward for the **rejected** response.

### 📊 Dataset & Processing
- **Dataset:** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **Splits:** `train` for training, `test` for validation.
- **Each sample contains** two texts: `chosen` and `rejected`.
- **Tokenization:** Uses `GPT2Tokenizer` with `pad_token = eos_token` and `padding_side = "right"`. Variable-length sequences are padded dynamically in `collate_fn`. Truncates/pads to `--max-length`.

### ⚙️ Optimizer & LR Scheduler
- **Supported Optimizers:** `adam`, `adamw`, `sgd`, `rmsprop`.
  - *Adam/AdamW:* `betas=(0.9, 0.999)`, `eps=1e-8`.
  - *SGD:* default `momentum=0.9` (Nesterov=False).
- **LR Scheduler (in `benchmark.py`):** Cosine schedule with warmup (`get_cosine_schedule_with_warmup`). Minimum LR is clamped to `0.1 * peak_lr` after the warmup phase. Current LR is logged.

### 🛠️ Usage
**1. Clone and Setup Environment**
```bash
git clone [https://github.com/auguste805/LightRLHF-Optimizer-Bench.git](https://github.com/auguste805/LightRLHF-Optimizer-Bench.git)
cd LightRLHF-Optimizer-Bench
pip install -r requirements.txt
```

**2. Run Training**
You can run the standard baseline training:
```bash
# Baseline: AdamW (example with typical hyperparameters)
python train.py --optim-name adamw --lr 5e-5 --lora-r 8
```

### 📈 Results
We benchmarked the training dynamics using the AdamW optimizer. The following figures show the training loss and validation accuracy over 20,000 steps, demonstrating the convergence of the model on the preference learning task.

#### Train Loss (Snapshot at step 20000)
The chart shows the raw train loss and its Exponential Moving Average (EMA, $\alpha=0.1$). We observe a smooth descent, indicating stable training.

![AdamW Train Loss](image_8.png)

#### Validation Accuracy (Snapshot at step 20000)
The chart shows the validation accuracy over steps. The best val accuracy achieved is approximately 0.5858, which is a significant improvement over random chance (0.5), showing that the RM has successfully learned human preferences.

![AdamW Val Accuracy](image_7.png)

---

<a name="chinese"></a>
## 中文

### 🎯 实验目的
本项目提供了一个轻量化、高可复现的 RLHF **奖励模型 (RM)** 训练框架。其核心目标是在消费级 GPU 上，使用标准基准优化算法（如 **AdamW** 和 **SGD**）深入探索偏好学习任务中的收敛动力学。

通过结合 **GPT-2 Small** 与 **LoRA** 技术，本项目降低了模型对齐核心环节的计算门槛，是一个理想的学习与研究工具。

### 🧠 模型架构与优化目标
- **基础模型:** `gpt2` (约 124M 参数)。代码加载 `transformers.GPT2Model`（不包含 LM Head）。
- **奖励价值头 (Value Head):** 采用标量线性层 (`nn.Linear(hidden_size, 1)`)。提取每个序列中**最后一个非填充 Token** 的隐藏状态（基于 `attention_mask`），将其映射为奖励得分。
- **LoRA 微调:** 默认开启。仅更新 GPT-2 骨干网络中的 LoRA 权重，而**价值头 (Value Head) 进行全参训练**。LoRA 注入模块：`c_attn`, `c_fc`, `c_proj` (任务类型: `FEATURE_EXTRACTION`)。
- **损失函数:** 基于偏好数据对的 Bradley-Terry 排序损失：
  $$\mathcal{L} = -\log\sigma(r_w - r_l)$$
  其中 $r_w$ 为人类**偏好 (chosen)** 回答的得分，$r_l$ 为**拒绝 (rejected)** 回答的得分。

### 📊 数据集与预处理
- **数据集:** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **数据划分:** `train` 用于训练，`test` 用于验证。
- **样本包含**两个文本：`chosen` (偏好) 和 `rejected` (拒绝)。
- **分词与批处理 (Tokenization):** 使用 `GPT2Tokenizer`，设定 `pad_token = eos_token` 及右侧填充 (`padding_side = "right"`)。在 `collate_fn` 中完成动态填充与 chosen/rejected 张量的构建。根据 `--max-length` 进行截断/填充。

### ⚙️ 优化器与学习率调度
- **支持的优化器 (`optimizer.py`):** `adam`, `adamw`, `sgd`, `rmsprop`。
  - *Adam/AdamW:* 默认 `betas=(0.9, 0.999)`, `eps=1e-8`。
  - *SGD:* 默认 `momentum=0.9` (不开启 Nesterov)。
- **学习率调度 (`benchmark.py`):** 采用带 Warmup 的余弦退火策略 (Cosine schedule with warmup)。退火结束后的最低学习率限制为峰值学习率的 10%。当前学习率会被记录日志。

### 🛠️ 使用方法
**1. 克隆代码与环境配置**
```bash
git clone [https://github.com/auguste805/LightRLHF-Optimizer-Bench.git](https://github.com/auguste805/LightRLHF-Optimizer-Bench.git)
cd LightRLHF-Optimizer-Bench
pip install -r requirements.txt
```

**2. 启动训练**
你可以运行标准基准测试：
```bash
# 基准测试：AdamW（典型超参数示例）
python train.py --optim-name adamw --lr 5e-5 --lora-r 8
```

### 📈 实验结果
我们使用 AdamW 优化器对训练动态进行了基准测试。下图展示了 20,000 步的训练损失和验证准确率，证明了模型在偏好学习任务上的收敛。

#### 训练损失 (20000 步快照)
图表展示了原始训练损失及其指数移动平均 (EMA, $\alpha=0.1$)。我们观察到平滑下降，表明训练稳定。

![AdamW Train Loss](image_8.png)
*(注意：请确保 image_8.png 文件已存在于您的仓库根目录下)*

#### 验证准确率 (20000 步快照)
图表展示了随时间步变化的验证准确率。实现的最佳验证准确率约为 0.5858，相对于随机概率 (0.5) 有显著提高，表明奖励模型已成功学习人类偏好。

![AdamW Val Accuracy](image_7.png)
*(注意：请确保 image_7.png 文件已存在于您的仓库根目录下)*
