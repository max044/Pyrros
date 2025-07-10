import random
import string
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from composer import Trainer

# ────────────────── Pyrros imports ──────────────────
from pyrros.trainers.grpo_trainer.grpo_loss import GRPOLossAlgorithm
from pyrros.trainers.grpo_trainer.grpo_generation import GRPOGenerationCallback
from pyrros.modules.model import load_model
from pyrros.modules.rewards import no_reward
# ─────────────────────────────────────────────────────


# ---------- 1. Dataset « prompt-only » ----------
class FakePromptDataset(Dataset):
    """Génère des chaînes aléatoires type 'aaaa bb ccc'."""
    CHARS = string.ascii_lowercase + " "

    def __init__(self, length=4, min_tokens=5, max_tokens=12):
        self.length = length
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def __len__(self):
        return self.length

    def _rand_prompt(self) -> str:
        n = random.randint(self.min_tokens, self.max_tokens)
        return "".join(random.choice(self.CHARS) for _ in range(n))

    def __getitem__(self, idx):
        return {"prompt": self._rand_prompt()}


# ---------- 2. Collate utilisant le tokenizer HF ----------
def make_collate_fn(tokenizer, max_len: int = 32):
    """
    Retourne une fonction collate qui :
      • tokenise les prompts,
      • tronque à `max_len`,
      • pad à la longueur max du batch.
    """
    def _collate(batch):
        prompts = [b["prompt"] for b in batch]
        enc = tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        # Pour un smoke-test, on peut prendre labels = input_ids
        return {
            "input_ids": enc.input_ids,   # [B, L']
            "labels":    enc.input_ids.clone(),
            "attention_mask": enc.attention_mask,
        }

    return _collate


# ---------- 3. Test PyTest ----------
@pytest.mark.cpu
def test_grpo_smoke():
    """Boucle GRPO complète sur 1 batch (CPU / MPS, <30 s)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # (a) Modèle aléatoire + vrai tokenizer
    model, tokenizer = load_model(
        "Qwen/Qwen1.5-0.5B",
        pretrained=False,
        device=device,
        dtype=torch.float32,
    )

    # (b) DataLoader factice
    fake_ds  = FakePromptDataset(length=4)
    train_dl = DataLoader(
        fake_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_len=32),
    )

    # (c) Trainer Composer (1 batch)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        max_duration="1ba",
        algorithms=[GRPOLossAlgorithm(beta=0.1, kl_target=0.05)],
        callbacks=[GRPOGenerationCallback(num_samples=1, reward_fn=no_reward)],
        precision="fp32",
        loggers=[],
        optimizers=optim.SGD(model.parameters(), lr=1e-3),
    )

    trainer.fit()  # Le test réussit si aucune exception n'est levée
