import string
from typing import Sequence
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from composer import Trainer
import copy
from transformers import PreTrainedTokenizer

# ───────── Pyrros imports ────────────────────────────────
from registry.algorithms.grpo.load_ref_model import LoadRefModel
from registry.algorithms.grpo.grpo_sampler import GRPOSampler
from pyrros.utils.model_utils import load_model

from registry.models.grpo.grpo_model import GRPOModel
from pyrros.utils.reward_utils import RewardFunction

# ────────────────────────────────────────────────────────


class NoReward(RewardFunction):
    """Stub reward function that returns zero rewards."""

    def __call__(
        self,
        completions: Sequence[str],
        completions_ids: Sequence[str],
        prompts: Sequence[dict],
        answers: Sequence[str],
    ) -> torch.Tensor:
        return torch.zeros(len(completions), dtype=torch.float32)


# ---------- 1. Dataset « prompt-only » -------------------
class FakePromptDataset(Dataset):
    CHARS = string.ascii_lowercase + " "

    def __init__(self, length=4, min_tokens=5, max_tokens=12):
        self.len = length
        self.min, self.max = min_tokens, max_tokens

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        prompt = f"n.{idx} J'aime le chocolat et toi ?"
        return [
            {"role": "system", "content": "tu es un gentil assistant"},
            {"role": "user", "content": prompt},
        ]


# ---------- 2. Collate -----------------------------------
def make_collate_fn(tokenizer: PreTrainedTokenizer, max_len: int = 32):
    def _collate(batch):

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
            "attention_mask": enc.attention_mask,
            "prompts": ["" for _ in batch],
            "answers": ["" for _ in batch],
        }

    return _collate


# ---------- 3. Test --------------------------------------
@pytest.mark.parametrize("device", ["mps"])
@pytest.mark.smoke
def test_grpo_smoke(device: str):
    """Boucle GRPO complète sur 2 batches, 100 % CPU."""
    # (a) Policy courante π_θ + tokenizer  (wrapper Composer)
    model, tokenizer = load_model(
        "Qwen/Qwen1.5-0.5B",
        pretrained=False,
        dtype=torch.float32,
    )
    model = GRPOModel(model=model, tokenizer=tokenizer, num_iterations=2)

    # (b) DataLoader factice
    ds = FakePromptDataset(length=4)
    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_len=32),
    )

    # (c) Algorithme
    load_ref_model = LoadRefModel(ref_model_name="Qwen/Qwen1.5-0.5B", device=device)

    grpo_sampler = GRPOSampler(
        tokenizer=tokenizer,
        reward_fns=[NoReward(), NoReward()],  # stub rewards
        G=2,  # 2 samples per prompt
        num_iterations=2,  # 2 iterations
        generation_kwargs={
            "max_new_tokens": 64,
            "max_length": 128
        }
    )

    # (d) Trainer Composer – 2 batches
    trainer = Trainer(
        model=model,
        train_dataloader=dl,
        max_duration="1ba",
        algorithms=[load_ref_model, grpo_sampler],
        device=device,
        precision="fp32",
        optimizers=optim.SGD(model.parameters(), lr=1e-3),
        loggers=[],
    )

    trainer.fit()  # Test réussi si aucune exception n’est levée
