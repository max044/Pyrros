import torch
import pytest
from types import SimpleNamespace
from registry.algorithms.grpo.grpo_sampler import GRPOSampler
from pyrros.utils.reward_utils import RewardFunction
from composer.core import Event


class _ConstReward(RewardFunction):
    def __init__(self, value):
        self.value = value
    def __call__(self, **kwargs):
        bs = len(kwargs["completions"])
        return [self.value] * bs


class _DummyTok:
    pad_token_id = 0
    eos_token = ""  # ensure .endswith works
    def batch_decode(self, seq, skip_special_tokens=True):
        return ["dummy"] * seq.size(0)


class _DummyLogger:
    def log_metrics(self, metrics: dict):
        pass


@pytest.fixture
def sampler():
    tok = _DummyTok()
    return GRPOSampler(
        tokenizer=tok,
        reward_fns=[_ConstReward(1), _ConstReward(0.5)],
        G=2,
        num_iterations=2,
        generation_kwargs={
            "max_new_tokens": 64,
            "max_length": 128
        }
    )


@pytest.mark.unit
def test_compute_advantages_unit_variance(sampler):
    t = torch.tensor([1., 2., 3.])
    g = sampler._compute_advantages(t)
    assert torch.allclose(g.mean(), torch.tensor(0.), atol=1e-6)
    assert pytest.approx(g.std().item(), rel=1e-4) == 1.0


@pytest.mark.unit
def test_compute_rewards_sum(sampler):
    seq_ids = torch.tensor([[1., 2., 3.]])
    inputs = {}
    inputs["input_ids"] = torch.tensor([[1., 2.]])
    inputs["prompts"] = ""
    inputs["answers"] = ""
    summed = sampler._compute_rewards(
        completions=["a", "b"],
        seq_ids=seq_ids,
        inputs=inputs,
        logger=_DummyLogger(),
    )
    assert torch.allclose(summed, torch.tensor([1.5, 1.5]))


@pytest.mark.unit
def test_apply_produces_expected_keys(sampler):
    class _M(torch.nn.Module):
        config = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2)
        device = torch.device("cpu")
        def generate(self, input_ids, attention_mask=None, generation_config=None):
            pad = torch.full((input_ids.size(0), 1), 3, dtype=input_ids.dtype)
            return SimpleNamespace(sequences=torch.cat([input_ids, pad], dim=1))

    model = _M()
    state = SimpleNamespace()
    state.batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "prompts": [""],
        "answers": [""],
    }
    state.model = model
    state.ref_model = SimpleNamespace(
        model=model,
        compute_log_probs=lambda seq_ids, attn: torch.zeros_like(seq_ids[:,1:], dtype=torch.float32)
    )

    sampler.apply(Event.BEFORE_FORWARD, state, _DummyLogger())

    for key in ["sequence_ids", "logprobs_ref", "advantages", "completion_mask"]:
        assert key in state.batch
