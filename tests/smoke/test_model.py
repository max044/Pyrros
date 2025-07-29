"""Unit tests for pyrros.modules.model"""

import pytest
import torch
from transformers import AutoTokenizer

from pyrros.utils.model_utils import load_model

_TINY_MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.smoke
def test_load_model_basic():
    """Model loads and returns objects."""
    model, tok = load_model(_TINY_MODEL)
    assert model is not None and tok is not None


@pytest.mark.parametrize("use_qlora", [True])
@pytest.mark.smoke
def test_load_model_qlora(use_qlora):
    """QLoRA wrapper attaches PEFT adapters when available."""
    peft = pytest.importorskip("peft")  # skip if PEFT not installed
    model, _ = load_model(
        _TINY_MODEL,
        use_qlora=use_qlora,
    )
    # PEFT wraps model into PeftModel
    from peft import PeftModel

    assert isinstance(
        model, PeftModel
    ), "Model should be wrapped by PEFT when use_qlora=True"


@pytest.mark.parametrize("bnb4bit", [True])
@pytest.mark.smoke
def test_load_model_4bit(bnb4bit):
    """4-bit NF4 quantization loads when bitsandbytes is available."""
    bnb = pytest.importorskip("bitsandbytes")  # noqa: F841 – imported for skip only
    model, _ = load_model(
        _TINY_MODEL,
        bnb4bit=bnb4bit,
    )
    # bitsandbytes linear layers inherit from bnb.nn.Linear4bit – sample check on first linear layer
    import bitsandbytes as bnb  # type: ignore

    linear_found = False
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            linear_found = True
            break
    assert linear_found, "No 4-bit Linear layers found – quantization may have failed"
