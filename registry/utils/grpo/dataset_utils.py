from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import transformers

from composer.utils import dist
from torch.utils.data.distributed import DistributedSampler

class GRPODataset(Dataset):
    """Load and prepare a dataset for GRPO"""

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


from torch.utils.data import Sampler
import torch
from typing import Optional, Sized, List
import math

class RepeatSampler(Sampler):
    """
    Sampler compatible DDP, qui répète les indices d’un dataset.
    Inspiré de Hugging Face.

    Chaque élément est répété `mini_repeat_count` fois (ex: pour GRPO: `mu`).
    Le tout est répété `repeat_count` fois au total (ex: nombre d'itérations globales).

    Si utilisé avec DDP, `torch.utils.data.DistributedSampler` doit wrapper ce sampler.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 42
        self.num_samples = len(data_source)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        for _ in range(self.repeat_count):
            indices = list(range(self.num_samples))
            if self.shuffle:
                indices = torch.randperm(self.num_samples, generator=g).tolist()

            # Répétition des indices par exemple
            for idx in indices:
                for _ in range(self.mini_repeat_count):
                    yield idx

    def __len__(self):
        return self.num_samples * self.mini_repeat_count * self.repeat_count


def get_grpo_dataloader(
    tokenizer,
    batch_size: int,
    mu: int,
    max_length: int,
    system_prompt: str = None,
    user_prompt: str = None,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 0,
    repeat_count: int = 1,
):
    dataset = GRPODataset(tokenizer, system_prompt, user_prompt, max_length, split)

    base_sampler = RepeatSampler(
        data_source=dataset,
        mini_repeat_count=mu,
        repeat_count=repeat_count,
        shuffle=shuffle,
        seed=dist.get_global_rank(),
    )

    if dist.get_world_size() > 1:
        sampler = DistributedSampler(
            base_sampler,
            num_replicas=dist.get_world_size(),
            rank=dist.get_global_rank(),
            shuffle=False,
        )
    else:
        sampler = base_sampler

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )