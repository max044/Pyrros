from typing import Sequence, Callable, List, Any, Union

from peft import PeftModel
import torch
import torch.nn.functional as F
from composer.core import Algorithm, Event, State

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from pyrros.models.grpo_model import GRPOModel


class GRPOSampler(Algorithm):
    def __init__(self,
                 old_model: GRPOModel,
                 ref_model: GRPOModel,
                 tokenizer: PreTrainedTokenizer,
                 reward_fns, G):
        self.old_model = old_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.G = G

    def match(self, event, state):
        return event == Event.BEFORE_FORWARD
    
    def _sample_group(self, input_ids, attention_mask) -> tuple[torch.Tensor, torch.Tensor]:
        old_log_probs = []
        ref_log_probs = []

        with torch.no_grad():
            old_log_probs = self.old_model.compute_log_probs(input_ids, attention_mask)
            ref_log_probs = self.ref_model.compute_log_probs(input_ids, attention_mask)

        return old_log_probs, ref_log_probs

    def _compute_rewards(self, completions: List[str]) -> torch.Tensor:
        list_of_rewards = [reward_fn(completions) for reward_fn in self.reward_fns]
        return torch.stack(list_of_rewards, dim=1).sum(dim=1)

    
    def _group_advantages(self, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (rewards - rewards.mean()) / (rewards.std() + eps)



    def apply(self, event, state, logger):
        input_ids: torch.Tensor = state.batch["input_ids"]
        attention_mask: torch.Tensor = state.batch["attention_mask"]

        # duplicate prompt num_rollouts times
        input_ids = input_ids.repeat(self.G, 1)
        attention_mask = attention_mask.repeat(self.G, 1)

        # 1. generate completions
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        generation_config = GenerationConfig(
            do_sample=True,
            top_p=1.0,
            temperature=0.6,
            max_length=128,
            pad_token_id=pad_token_id,
        )
        sequence_ids = self.old_model.generate(input_ids, generation_config=generation_config)
        completions = self.tokenizer.batch_decode(
            sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )


        completion_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        completion_mask[:, input_ids.shape[1] :] = True
        completion_mask[sequence_ids == pad_token_id] = False
        completion_mask = completion_mask[:, 1:]


        # 2. compute rewards and advantages
        rewards = self._compute_rewards(completions)
        advantages = self._group_advantages(rewards)
        advantages = advantages.to(input_ids.device)

        # 3. compute log-probabilities
        attention_mask = sequence_ids != pad_token_id
        logp_old, logp_ref = self._sample_group(sequence_ids, attention_mask)
        
        state.batch = {
            "input_ids": input_ids,
            "labels": state.batch["labels"],
            "attention_mask": attention_mask,

            "sequence_ids": sequence_ids,  # (B,G,L)
            "logprobs_old": logp_old,  # (B,G,V)
            "logprobs_ref": logp_ref,  # (B,G,V)
            "advantages": advantages,  # (B,G)
            "completion_mask": completion_mask,  # (B,G,L)
        }
