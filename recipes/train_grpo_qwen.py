#!/usr/bin/env python
"""
Train GRPO on a prompt-only dataset with MosaicML Composer.

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
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from composer import Trainer
from composer.loggers import WandBLogger, TensorboardLogger

# ───── Pyrros core ────────────────────────────────────────────────
from pyrros.trainers.grpo_trainer.grpo_loss import GRPOLossAlgorithm
from pyrros.trainers.grpo_trainer.grpo_generation import GRPOGenerationCallback
from pyrros.modules.model import load_model
from pyrros.modules.rewards import no_reward
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("recipe")


# ------------------------------------------------------------------#
# 1. Dataset JSONL (« prompt-only »)                                 #
# ------------------------------------------------------------------#
class JsonlPromptDataset(Dataset):
    """
    Attend un fichier JSONL contenant au moins la clé « prompt ».
    Chaque ligne exemple :
    {"prompt": "Write a short story about..."}
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: List[Dict[str, str]] = [
            json.loads(l) for l in self.path.open()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["prompt"]


def make_collate_fn(tokenizer, max_len: int = 512):
    def _collate(batch_prompts):
        enc = tokenizer(
            batch_prompts,
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        # Labels = input_ids  (profite de GRPOLossAlgorithm pour la vraie loss)
        return {
            "input_ids": enc.input_ids,
            "labels": enc.input_ids.clone(),
            "attention_mask": enc.attention_mask,
        }

    return _collate


# ------------------------------------------------------------------#
# 2. CLI & hyper-paramètres                                         #
# ------------------------------------------------------------------#
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


# ------------------------------------------------------------------#
# 3. Main recipe                                                    #
# ------------------------------------------------------------------#
def main():
    args = parse_args()

    # 3.1  Model + tokenizer
    model, tokenizer = load_model(
        args.model_name,
        use_qlora=args.use_qlora,
        bnb4bit=args.bnb4bit,
        gradient_checkpointing=args.use_qlora,
    )

    # 3.2  Dataset & DataLoader
    dataset = JsonlPromptDataset(args.data_path)
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    # 3.3  Optimiser
    optimiser = AdamW(model.parameters(), lr=args.lr)

    # 3.4  Loggers
    loggers = []
    if args.wandb:
        loggers.append(
            WandBLogger(
                project="pyrros-grpo",
                name=Path(args.output_dir).name,
            )
        )
    if args.tensorboard:
        loggers.append(TensorboardLogger(log_dir=args.output_dir))

    # 3.5  GRPO components
    algo = GRPOLossAlgorithm(beta=0.1, kl_target=0.05)
    gen_cb = GRPOGenerationCallback(
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        reward_fn=no_reward,  # ↩️  remplace par ton vrai reward modèle
    )

    # 3.6  Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dl,
        max_duration=args.max_duration,
        algorithms=[algo],
        callbacks=[gen_cb],
        precision="bf16" if torch.cuda.is_available() else "fp32",
        loggers=loggers,
        optimizers=optimiser,
        run_name=Path(args.output_dir).name,
        save_folder=args.output_dir,
    )

    # 3.7  FIT !
    trainer.fit()
    trainer.close()  # force flush loggers


if __name__ == "__main__":
    main()
