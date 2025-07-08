"""
Implémentation GRPO *full Composer*.

⚠️  Dépendances :
    pip install composer==0.31.*  torch==2.7.0  # (versions testées)
"""

from __future__ import annotations

import math
import types
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from pyrros.trainers.base import BaseTrainer

# === Composer ===
from composer import Trainer as _ComposerTrainer  # type: ignore
from composer.models import ComposerModel  # type: ignore
from composer.optim import DecoupledAdamW  # type: ignore

__all__ = ["GRPOComposerTrainer"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _calc_logpi_on_generated(
    logits: torch.Tensor,
    seq: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Calcule log π(a|s) pour *chaque séquence* en ne gardant que les tokens
    post-prompt (i.e. réellement « choisis » par l’agent).

    Args:
        logits: [B, L, V]
        seq:    [B, L] (prompt + generated)
        prompt_lens: [B] taille du prompt (int) par sample.

    Returns:
        log_π par séquence – shape [B]
    """
    # Alignement logits[t] ↔ target token[t]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, L-1, V]
    tgt_ids = seq[:, 1:]  # targets = seq décalé
    token_logp = log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    # Masque: on met 1 sur les positions générées, 0 sur le prompt
    seq_len_minus1 = token_logp.size(1)
    arange = torch.arange(seq_len_minus1, device=seq.device).unsqueeze(0)  # [1, L-1]
    mask = arange >= (prompt_lens.unsqueeze(1))  # [B, L-1] (True = généré)
    selected = token_logp.masked_fill(~mask, 0.0)
    return selected.sum(dim=1)  # [B]


@torch.no_grad()
def _generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: torch.Tensor,
    max_new_tokens: int,
    gen_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Génère `max_new_tokens` pour chaque prompt (shape [B, L_prompt]).
    Retourne les *séquences complètes* (prompt + generated).
    """
    gen_kwargs = gen_kwargs or {}
    outputs = model.generate(
        input_ids=prompts,
        attention_mask=prompts.ne(tokenizer.pad_token_id),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    return outputs


# --------------------------------------------------------------------------- #
# Composer Model                                                              #
# --------------------------------------------------------------------------- #
class _GRPOComposerModel(ComposerModel):  # noqa: D101
    def __init__(
        self,
        policy: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        baseline_momentum: float = 0.9,
        max_new_tokens: int = 32,
        gen_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.baseline_momentum = baseline_momentum
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = gen_kwargs or {}

        # Baseline mobile (scalaire) pour réduire la variance
        self.register_buffer("_baseline", torch.tensor(0.0), persistent=False)

    # ===== Composer API ===== #
    def forward(self, batch: Dict[str, torch.Tensor]):  # noqa: D401
        """Simple passthrough : logits sur la *séquence complète*."""
        return self.policy(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits

    def loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]):  # noqa: D401
        """
        REINFORCE :  L = −(R − b) · log π.

        `outputs` = logits forward pass sur (prompt + generated).
        """
        seq = batch["input_ids"]
        prompt_lens = batch["prompt_len"]  # [B]
        rewards = batch["reward"]  # [B]

        log_pi = _calc_logpi_on_generated(outputs, seq, prompt_lens)  # [B]
        adv = rewards - self._baseline  # [B]

        # MaJ baseline (EMA)
        self._baseline.mul_(self.baseline_momentum).add_(
            rewards.mean() * (1 - self.baseline_momentum)
        )

        return -(adv * log_pi).mean()

    def get_metrics(self, is_train: bool = False):  # noqa: D401
        return {}


# --------------------------------------------------------------------------- #
# Data pipeline                                                               #
# --------------------------------------------------------------------------- #
class _PromptDataset(torch.utils.data.IterableDataset):
    """Itère sans fin sur des prompts HF déjà tokenisés."""

    def __init__(self, token_ids: torch.Tensor, batch_size: int):
        super().__init__()
        self.token_ids = token_ids  # [N, L] tensor CPU
        self.batch_size = batch_size

    def __iter__(self):
        N = self.token_ids.size(0)
        idx = torch.randint(0, N, size=(self.batch_size,))
        while True:
            yield {"input_ids": self.token_ids[idx]}


def _build_dataloader(
    prompts: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> DataLoader:
    ds = _PromptDataset(prompts, batch_size)
    pad_id = tokenizer.pad_token_id

    def _collate(batch: List[Dict]):  # batch list -> dict
        input_ids = torch.stack([b["input_ids"] for b in batch])  # [B, L]
        attention_mask = input_ids.ne(pad_id).long()
        prompt_len = attention_mask.sum(dim=1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": prompt_len,
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=_collate)


# --------------------------------------------------------------------------- #
# Public trainer                                                              #
# --------------------------------------------------------------------------- #
class GRPOComposerTrainer(BaseTrainer):  # noqa: D101
    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        prompt_dataset: Iterable[str] | torch.Tensor,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        batch_size: int = 4,
        max_new_tokens: int = 32,
        lr: float = 2e-5,
        max_steps: int = 1_000,
        device: str | torch.device = "cuda",
    ) -> None:
        if isinstance(prompt_dataset, torch.Tensor):
            prompt_tensor = prompt_dataset  # déjà tokenisé
        else:
            # Liste de str → tokenisation fixe longueur 1ère phase (simple)
            enc = tokenizer(
                list(prompt_dataset),
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            prompt_tensor = enc.input_ids  # CPU

        dataloader = _build_dataloader(prompt_tensor, tokenizer, batch_size)

        composer_model = _GRPOComposerModel(
            policy=policy_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            max_new_tokens=max_new_tokens,
        )

        optim = DecoupledAdamW(composer_model.parameters(), lr=lr)

        self._trainer = _ComposerTrainer(
            model=composer_model,
            train_dataloader=dataloader,
            max_duration=f"{max_steps}ba",
            precision="FP32",
            optimizers=optim,
            profiler=None,
            device=device,
        )

    # ---- BaseTrainer API ---- #
    def train(self) -> None:  # noqa: D401
        self._trainer.fit()

    def save_checkpoint(self, path: str) -> None:  # noqa: D401
        self._trainer.save_checkpoint(path)

    def eval(self, *args, **kwargs):  # noqa: D401
        return self._trainer.eval(*args, **kwargs)
