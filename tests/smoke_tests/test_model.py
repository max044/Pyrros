"""Unit tests for pyrros.modules.model"""

import pytest
import torch
from transformers import AutoTokenizer

from pyrros.utils.model_utils import load_model

_TINY_MODEL = "Qwen/Qwen3-0.6B"


def test_load_model_basic():
    """Model loads and returns objects on CPU by default."""
    model, tok = load_model(_TINY_MODEL, device_map={"": "cpu"})
    assert model is not None and tok is not None


@pytest.mark.parametrize("use_qlora", [True])
def test_load_model_qlora(use_qlora):
    """QLoRA wrapper attaches PEFT adapters when available."""
    peft = pytest.importorskip("peft")  # skip if PEFT not installed
    model, _ = load_model(
        _TINY_MODEL,
        use_qlora=use_qlora,
    )
    # PEFT wraps model into PeftModel
    from peft import PeftModel

    assert isinstance(model, PeftModel), "Model should be wrapped by PEFT when use_qlora=True"


@pytest.mark.parametrize("bnb4bit", [True])
def test_load_model_4bit(bnb4bit):
    """4‑bit NF4 quantisation loads when bitsandbytes is available."""
    bnb = pytest.importorskip("bitsandbytes")  # noqa: F841 – imported for skip only
    model, _ = load_model(
        _TINY_MODEL,
        bnb4bit=bnb4bit,
        device_map={"": "cpu"},
    )
    # bitsandbytes linear layers inherit from bnb.nn.Linear4bit – sample check on first linear layer
    import bitsandbytes as bnb  # type: ignore

    linear_found = False
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            linear_found = True
            break
    assert linear_found, "No 4‑bit Linear layers found – quantisation may have failed"
