import pytest
from pyrros.utils.model_utils import load_model

_TINY = "Qwen/Qwen3-0.6B"


@pytest.mark.unit
def test_load_pretrained_false(monkeypatch):
    from transformers import AutoModelForCausalLM
    monkeypatch.setattr(AutoModelForCausalLM, "from_config", lambda cfg, **kw: object())
    model, tok = load_model(_TINY, pretrained=False)
    assert model is not None
    assert hasattr(tok, "pad_token_id")


@pytest.mark.unit
@pytest.mark.skipif(pytest.importorskip("bitsandbytes", reason="bnb absent") is None, reason="bitsandbytes not installed")
def test_4bit_layers_present(monkeypatch):
    import bitsandbytes as bnb
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: bnb.nn.Linear4bit(4,4)
    )
    model, _ = load_model(_TINY, bnb4bit=True, device_map={"": "cpu"})
    assert any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())


@pytest.mark.unit
@pytest.mark.skipif(pytest.importorskip("peft", reason="peft absent") is None, reason="peft not installed")
def test_qlora_wrap(monkeypatch):
    from peft import PeftModel
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: __import__("torch").nn.Linear(4,4)
    )
    model, _ = load_model(_TINY, use_qlora=True)
    assert isinstance(model, PeftModel)
