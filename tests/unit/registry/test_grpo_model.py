import torch
import pytest
from types import SimpleNamespace
from registry.models.grpo.grpo_model import GRPOModel


class _TinyLM(torch.nn.Module):
    """Minimal causal-LM returning constant logits for fast tests."""
    def __init__(self, vocab_size: int = 10):
        super().__init__()
        # Required by HuggingFaceModel: config.vocab_size + model.device
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        self.device = torch.device("cpu")
        self.embed = torch.nn.Embedding(vocab_size, 4)
        self.lm_head = torch.nn.Linear(4, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        return {"logits": logits}


class _DummyTokenizer:
    """Minimal tokenizer with required attrs and __len__."""
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    def __len__(self):
        return 1


@pytest.fixture
def tiny_batch():
    ids = torch.tensor([[1, 3, 4, 2]])          # <bos> x x <eos>
    mask = (ids != 0).long()
    return {
        "sequence_ids": ids,
        "attention_mask": mask,
        "logprobs_ref": torch.zeros_like(ids[:, 1:], dtype=torch.float32),
        "advantages": torch.ones(ids.size(0), dtype=torch.float32),
        "completion_mask": torch.ones_like(ids[:, 1:], dtype=torch.bool),
        "labels": ids,
    }


@pytest.fixture
def grpo_model():
    model = _TinyLM()
    tokenizer = _DummyTokenizer()
    return GRPOModel(model=model, tokenizer=tokenizer)


@pytest.mark.unit
def test_compute_log_probs_shape(grpo_model, tiny_batch):
    lp = grpo_model.compute_log_probs(
        tiny_batch["sequence_ids"],
        tiny_batch["attention_mask"]
    )
    assert lp.shape == (1, tiny_batch["sequence_ids"].size(1) - 1)


@pytest.mark.unit
def test_kl_non_negative(grpo_model, tiny_batch):
    lp = grpo_model.compute_log_probs(
        tiny_batch["sequence_ids"],
        tiny_batch["attention_mask"]
    )
    kl = grpo_model._approximate_kl_divergence(lp, lp, tiny_batch["completion_mask"])
    assert torch.allclose(kl, torch.zeros_like(kl))


@pytest.mark.unit
def test_loss_scalar_finite_scalar(grpo_model, tiny_batch):
    out = grpo_model.forward(tiny_batch)
    loss = grpo_model.loss(out, tiny_batch)
    # Should be a single finite scalar
    assert loss.numel() == 1
    assert torch.isfinite(loss).all()