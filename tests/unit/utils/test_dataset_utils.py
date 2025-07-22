from pathlib import Path
import pytest, torch
from datasets import Dataset as HFDataset
from pyrros.utils.dataset_utils import load_dataset
from transformers import AutoTokenizer


@pytest.fixture
def local_ds(tmp_path: Path):
    data = HFDataset.from_dict({"text": ["alpha", "beta", "gamma"]})
    p = tmp_path / "hf"
    data.save_to_disk(p)
    return str(p)


@pytest.mark.unit
def test_local_tokenised(local_ds):
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset(local_ds, tokenizer=tok, seq_len=8)
    assert len(ds) == 3
    assert isinstance(ds[0]["input_ids"], torch.Tensor)
    assert ds[0]["input_ids"].shape[0] == 8


@pytest.mark.unit
def test_max_samples(local_ds):
    ds = load_dataset(local_ds, tokenizer=None, max_samples=2)
    assert len(ds) == 2
