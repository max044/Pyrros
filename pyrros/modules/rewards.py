"""Simple reward functions for RL-style fine-tuning.

For MVP we provide:
- **`length_bonus`** : encourages longer answers.
- **`keyword_match`** : +1 reward if a target keyword appears.
- **`no_reward`**     : always returns 0 (baseline debugging).
"""
from __future__ import annotations

from typing import Sequence, Callable, List

__all__ = [
    "length_bonus",
    "keyword_match",
    "no_reward",
]


def length_bonus(responses: Sequence[str], bonus_per_token: float = 0.01) -> List[float]:
    return [len(r.split()) * bonus_per_token for r in responses]


def keyword_match(responses: Sequence[str], keyword: str) -> List[float]:
    kw_lower = keyword.lower()
    return [1.0 if kw_lower in r.lower() else 0.0 for r in responses]


def no_reward(responses: Sequence[str]) -> List[float]:
    return [0.0 for _ in responses]
