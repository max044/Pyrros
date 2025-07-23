# Pyrros 🔥

**Pyrros** is an open-source collection of ready-to-use training scripts and techniques for large language models (LLMs), built with efficiency, modularity, and clarity in mind.

It will combines:
- 🧠 Research-grade training methods (SFT, DPO, PPO, GRPO…)
- ⚡ Memory-optimized execution (QLoRA, gradient checkpointing, Triton)
- 🧩 A clean, Composer-powered structure you can hack or extend
- 🖥️ Easy multi-GPU support, Lightning-style

## 🚀 Goals

- Make training LLMs accessible, reproducible, and fast.
- Provide production-ready and research-friendly codebases.
- Stay clean, no magic, no overengineering.

## 📦 Status

> The project is under early development. Feel free to contribute or suggest features!

## Install Pyrros

**with uv**
```bash
uv init
uv venv --python 3.10
source .venv/bin/activate
```
<!-- ```bash
uv add Pyrros
```
or -->
```bash
uv add "Pyrros @ git+https://github.com/max044/Pyrros.git"
```

## Usage
```bash
pyrros --help
```
This will show you the available commands and options.

## 🧪 Example: GRPO + QLoRA on Qwen

```bash
pyrros add grpo
python3 train_grpo_qwen.py
```

# Development

## 🛠️ Installation for development

```bash
git clone https://github.com/max044/Pyrros.git
cd Pyrros
uv sync
```

## 🧪 Running tests

```bash
python3 -m pytest -m unit
python3 -m pytest -m smoke
```