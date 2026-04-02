# LightRLHF-Optimizer-Bench 🚀

[English](#english) | [中文](#中文)

## English

### 🎯 Project Objective
This repository provides a lightweight, highly reproducible framework for training **Reward Models (RM)** in the RLHF pipeline. It is specifically designed to benchmark custom optimization algorithms against standard baselines (AdamW) on consumer-grade GPUs.

By using **GPT-2 Small** combined with **LoRA**, this project allows researchers to explore the dynamics of preference learning and non-convex optimization without requiring industrial-scale compute resources.

### 📊 Model & Dataset
- **Base Model:** `gpt2` (124M parameters)
- **Adaptation:** LoRA (Low-Rank Adaptation) via `peft`
- **Dataset:** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (Helpful and Harmless RLHF dataset)
- **Task:** Sequence Classification (Ranking loss between `chosen` and `rejected` responses)

### 🛠️ Usage
**1. Clone and Setup Environment**
```bash
git clone https://github.com/auguste805/LightRLHF-Optimizer-Bench.git
cd LightRLHF-Optimizer-Bench
pip install -r requirements.txt
```

**2. Run Training**
You can toggle between different optimizers and hyperparameters:
```bash
# Baseline: AdamW
python train.py --optimizer adamw --lr 5e-5 --lora_rank 8
```

### ⚙️ Parameter Tuning
| Parameter | Description | Recommended Range |
| :--- | :--- | :--- |
| `--lr` | Learning rate | `5e-6` to `1e-4` |
| `--lora_rank` | Rank of LoRA update matrices | `4`, `8`, `16` |
| `--batch_size` | Training batch size | `4` to `16` (depends on VRAM) |
| `--optimizer` | Optimization algorithm | `adamw`, `sgd`, `rmsprop` |

---

## 中文

### 🎯 实验目的
本项目提供了一个轻量化、高可复现的 RLHF **奖励模型 (RM)** 训练框架。其核心目标是在消费级 GPU 上，针对自定义优化算法与标准基准算法（AdamW）在偏好学习任务中的表现进行 Benchmark 评测。

通过结合 **GPT-2 Small** 与 **LoRA** 技术，本项目使研究者能够在不需要大规模计算集群的情况下，深入探索非凸随机优化在模型对齐阶段的收敛动力学。

### 📊 模型与数据集
- **基础模型：** `gpt2` (124M 参数)
- **微调技术：** LoRA (低秩自适应)
- **数据集：** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (包含 chosen/rejected 对的偏好数据集)
- **任务类型：** 序列分类（基于排序损失 Ranking Loss）

### 🛠️ 使用方法
**1. 克隆代码与环境配置**
```bash
git clone https://github.com/auguste805/LightRLHF-Optimizer-Bench.git
cd LightRLHF-Optimizer-Bench
pip install -r requirements.txt
```

**2. 启动训练**
你可以自由切换不同的优化器与超参数：
```bash
# 基准测试：AdamW
python train.py --optimizer adamw --lr 5e-5 --lora_rank 8
```

### ⚙️ 参数调整
| 参数 | 描述 | 建议范围 |
| :--- | :--- | :--- |
| `--lr` | 学习率 | `5e-6` 至 `1e-4` |
| `--lora_rank` | LoRA 的秩 (Rank) | `4`, `8`, `16` |
| `--batch_size` | 训练批大小 | `4` 至 `16` (视显存而定) |
| `--optimizer` | 优化算法 | `adamw`, `adaspid`, `ropid` |

## 🧠 Citation
If you find this repository or our optimizers useful, please consider citing our work:
```bibtex
@article{jian2026adaspid,
  title={AdaPID: Adaptive Momentum Gradient Method based on PID Controller for Non-Convex Stochastic Optimization in Deep Learning},
  author={Jian, Ailun and [Other Authors]},
  journal={Under Review},
  year={2026}
}
```
