"""Unit tests for pyrros.modules.dataset"""

from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from pyrros.utils.dataset_utils import load_dataset
import pytest

def _create_dummy_dataset(tmp_path: Path):
    """Create a small HF dataset on disk and return its path."""
    data = HFDataset.from_dict({"text": ["Hello world", "Testing 123", "Voilà"]})
    local_dir = tmp_path / "dummy_ds"
    data.save_to_disk(local_dir)
    return str(local_dir)


@pytest.mark.smoke
def test_load_dataset_tokenise(tmp_path):
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds_path = _create_dummy_dataset(tmp_path)

    tokenised = load_dataset(ds_path, tokenizer=tok, seq_len=16, max_samples=2)
    assert len(tokenised) == 2, "Should return exactly max_samples elements"

    sample = tokenised[0]
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert (
        sample["input_ids"].shape[0] == 16
    ), "Sequence length should be padded/truncated to seq_len"
