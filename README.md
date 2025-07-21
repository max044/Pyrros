# Pyrros 🔥

**Pyrros** is an open-source collection of ready-to-use training scripts and techniques for large language models (LLMs), built with efficiency, modularity, and clarity in mind.

It combines:
- 🧠 Research-grade training methods (SFT, DPO, PPO, GRPO…)
- ⚡ Memory-optimized execution (QLoRA, gradient checkpointing, Triton)
- 🧩 A clean, Composer-powered structure you can hack or extend
- 🖥️ Easy multi-GPU support, Lightning-style

## 🚀 Goals

- Make training LLMs accessible, reproducible, and fast.
- Provide production-ready and research-friendly codebases.
- Stay clean, no magic, no overengineering.

## 📦 Status

> The project is under early development. First training scripts for Qwen + GRPO + QLoRA coming soon.

## 🛠️ Installation for development

```bash
git clone https://github.com/max044/Pyrros.git
cd Pyrros
uv sync
```

## Install Pyrros

**with uv**
```bash
uv init
uv venv --python 3.10
source .venv/bin/activate
uv add Pyrros or uv add "Pyrros @ git+https://github.com/max044/Pyrros.git"
```