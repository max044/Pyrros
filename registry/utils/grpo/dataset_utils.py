from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from datasets import load_dataset
import transformers


class GRPOGSM8KDataset(Dataset):
    """Charge et prépare GSM8K pour GRPO."""

    def __init__(
        self,
        tokenizer: Union[
            transformers.PreTrainedTokenizer,
            transformers.PreTrainedTokenizerFast,
        ],
        system_prompt: str,
        user_prompt: str,
        max_length: int,
        split="train",
    ):
        ds = load_dataset("openai/gsm8k", "main")[split]
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_length = max_length
        self.samples = ds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_prompt.format(question=ex["question"]),
            },
        ]
        answers = ex["answer"]
        tok = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "prompts": messages,
            "answers": answers,
        }


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": [x["prompts"] for x in batch],
        "answers": [x["answers"] for x in batch],
    }


class RepeatBatchSampler(BatchSampler):
    """Repeats each batch `mu` times."""

    def __init__(self, base_sampler, batch_size, mu, drop_last=False):
        super().__init__(base_sampler, batch_size, drop_last)
        self.mu = mu

    def __iter__(self):
        for batch in super().__iter__():
            for _ in range(self.mu):
                yield batch

    def __len__(self):
        return super().__len__() * self.mu


def get_grpo_dataloader(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    batch_size: int,
    mu: int,
    max_length: int,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = GRPOGSM8KDataset(tokenizer, system_prompt, user_prompt, max_length, split)
    sampler = RandomSampler(dataset) if shuffle else None
    batch_sampler = RepeatBatchSampler(sampler, batch_size, mu, drop_last=False)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
