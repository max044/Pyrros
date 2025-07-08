'''Dataset loading & tokenisation helpers.'''

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

from datasets import load_dataset as hf_load_dataset, load_from_disk, Dataset, IterableDataset, DatasetDict

logger = logging.getLogger(__name__)

__all__: list[str] = [
    'load_dataset',
]


def load_dataset(
    name_or_path: str,
    *,
    split: str = 'train',
    streaming: bool = False,
    max_samples: Optional[int] = None,
    tokenizer=None,
    seq_len: int = 2048,
    text_field: str = 'text',
    seed: int = 42,
    loader_kwargs: Dict[str, Any] | None = None,
) -> Union[Dataset, IterableDataset]:
    '''Load a dataset (local folder or HF Hub) and tokenise it with *tokenizer*.

    If *tokenizer* is ``None``, returns the raw dataset / stream.
    '''

    loader_kwargs = loader_kwargs or {}

    # 1) Fetch dataset ------------------------------------------------------
    if Path(name_or_path).expanduser().exists():
        logger.info("Loading local dataset from '%s'", name_or_path)
        data = load_from_disk(name_or_path, **loader_kwargs)
    else:
        logger.info("Fetching dataset '%s' from the HF Hub", name_or_path)
        data = hf_load_dataset(name_or_path, split=None if streaming else split, streaming=streaming, **loader_kwargs)

    # 2) Ensure split availability ----------------------------------------
    if isinstance(data, DatasetDict):
        if split not in data:
            data = data['train'].train_test_split(test_size=0.1, seed=seed)
        data = data[split]

    # 3) Subsample for quick runs -----------------------------------------
    if max_samples is not None and not streaming:
        logger.info('Sub‑selecting first %d samples.', max_samples)
        data = data.select(range(max_samples))

    # 4) Early return if no tokenizer -------------------------------------
    if tokenizer is None:
        logger.warning('No tokenizer provided – returning raw dataset.')
        return data

    # 5) Tokenisation ------------------------------------------------------
    def _tok(batch: Dict[str, Any]):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=seq_len,
            padding='max_length',
        )

    tokenised = data.map(
        _tok,
        batched=True,
        remove_columns=[text_field],
        num_proc=None if streaming else 4,
    )

    tokenised.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    logger.info('Dataset ready: %s samples (streaming=%s).', len(tokenised) if not streaming else '∞', streaming)
    return tokenised
