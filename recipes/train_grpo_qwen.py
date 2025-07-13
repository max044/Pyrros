#!/usr/bin/env python
"""
Train GRPO sur un dataset prompt-only avec MosaicML Composer.

Usage :
    python recipes/train_grpo_qwen.py \
        --model_name Qwen/Qwen1.5-0.5B \
        --data_path data/my_prompts.jsonl \
        --output_dir runs/grpo_qwen \
        --batch_size 4 \
        --max_duration 1ep \
        --use_qlora \
        --bnb4bit
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from composer import Trainer
from composer.loggers import WandBLogger, TensorboardLogger

# ───── Pyrros core ────────────────────────────────────────────────
from pyrros.algorithms.grpo_sampler import GRPOSampler
from pyrros.modules.model import load_model
from pyrros.models.grpo_model import GRPOModel
from pyrros.modules.dataset import load_dataset
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("recipe")


def make_collate_fn(tokenizer, max_len: int = 512):
    def _collate(batch):
        # Utilise le template de chat du tokenizer
        enc = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            padding="longest",
            padding_side="left",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_dict=True,
        )
        return {
            "input_ids": enc.input_ids,
            "labels":    enc.input_ids.clone(),
            "attention_mask": enc.attention_mask,
        }
    return _collate

def no_reward(completion_ids: Sequence[str], **kwargs) -> torch.Tensor:
    return torch.zeros(len(completion_ids))


def parse_args():
    p = argparse.ArgumentParser(description="Train GRPO with Composer")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B")
    p.add_argument("--data_path", type=str, required=True, help="JSONL prompts")
    p.add_argument("--output_dir", type=str, default="runs/grpo_qwen")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_duration", type=str, default="1ep")
    p.add_argument("--lr", type=float, default=5e-5)
    # QLoRA / quantisation
    p.add_argument("--use_qlora", action="store_true")
    p.add_argument("--bnb4bit", action="store_true")
    # Generation parameters
    p.add_argument("--num_samples", type=int, default=4, help="K replies / prompt")
    p.add_argument("--max_new_tokens", type=int, default=128)
    # WandB / TB
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--tensorboard", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Modèle + tokenizer
    model, tokenizer = load_model(
        args.model_name,
        use_qlora=args.use_qlora,
        bnb4bit=args.bnb4bit,
        gradient_checkpointing=args.use_qlora,
    )

    # 2. Dataset & DataLoader
    dataset = load_dataset(
        args.data_path,
        tokenizer=tokenizer,
        seq_len=512,
        text_field="prompt",
        max_samples=None,
    )
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    # 3. Optimiseur
    optimiser = AdamW(model.parameters(), lr=args.lr)

    # 4. Loggers
    loggers = []
    if args.wandb:
        loggers.append(WandBLogger())
    if args.tensorboard:
        loggers.append(TensorboardLogger())

    # 5. Algorithme GRPO
    grpo_sampler = GRPOSampler(
        old_model=None,
        ref_model=model,
        tokenizer=tokenizer,
        reward_fns=[no_reward],
        G=args.num_samples,
    )

    # 6. Trainer Composer
    trainer = Trainer(
        model=model,
        train_dataloader=dl,
        max_duration=args.max_duration,
        algorithms=[grpo_sampler],
        optimizers=optimiser,
        loggers=loggers,
        run_name="grpo_qwen",
        save_folder=args.output_dir,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
