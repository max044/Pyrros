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
        # self.old_model = old_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.G = G

    def match(self, event, state):
        return event == Event.BEFORE_FORWARD

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
            output_logits=True,
            return_dict_in_generate=True,
        )
        # old_model_output = self.old_model.generate(input_ids, generation_config=generation_config)
        old_model_output = state.model.generate(input_ids, generation_config=generation_config)
        
        sequence_ids = old_model_output.sequences

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
        with torch.no_grad():
            logp_ref = self.ref_model.compute_log_probs(sequence_ids, attention_mask, gen_mask=completion_mask)
        
            logits = torch.stack(old_model_output.logits, dim=1)
            # 2. log-probas π_old sur les tokens générés
            logp_old = torch.log_softmax(logits, dim=-1)
            gen_ids  = sequence_ids[:, -logits.size(1):]             # tokens générés
            logp_old = logp_old.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, L_gen)

        
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
