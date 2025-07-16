from typing import List

import torch
from composer.core import Algorithm, Event

from transformers import GenerationConfig, PreTrainedTokenizer

class GRPOSampler(Algorithm):
    def __init__(self, tokenizer: PreTrainedTokenizer, reward_fns, G, generation_kwargs: dict = None):
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.G = G
        self.generation_kwargs = generation_kwargs or {}

    def match(self, event, state):
        return event == Event.BEFORE_FORWARD

    def _compute_rewards(self, completions: List[str], completions_ids, prompts, answers,) -> torch.Tensor:
        list_of_rewards = [
            reward_fn(
                completions=completions,
                completions_ids=completions_ids,
                prompts=prompts,
                answers=answers,
            ) for reward_fn in self.reward_fns]
        # Convert list of rewards to tensor
        list_of_rewards = [
            torch.tensor(rewards, dtype=torch.float32) for rewards in list_of_rewards
        ]

        return torch.stack(list_of_rewards, dim=1).sum(dim=1)


    def _group_advantages(self, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (rewards - rewards.mean()) / (rewards.std() + eps)



    def apply(self, event, state, logger):
        ref_model = state.ref_model
        input_ids: torch.Tensor = state.batch["input_ids"]
        attention_mask: torch.Tensor = state.batch["attention_mask"]
        prompts = state.batch["prompts"]
        answers = state.batch["answers"]


        # duplicate prompt num_rollouts times
        input_ids = input_ids.repeat(self.G, 1)
        attention_mask = attention_mask.repeat(self.G, 1)
        prompts = prompts * self.G
        answers = answers * self.G

        generation_config = GenerationConfig(
            pad_token_id=state.model.config.pad_token_id,
            bos_token_id=state.model.config.bos_token_id,
            eos_token_id=state.model.config.eos_token_id,
            do_sample=True,
            return_dict_in_generate=True,
            # output_logits=True,

            **self.generation_kwargs,
        )


        # old_model_output = self.old_model.generate(input_ids, generation_config=generation_config)
        state.model.eval()
        old_model_output = state.model.generate(input_ids, generation_config=generation_config)
        state.model.train()

        sequence_ids = old_model_output.sequences

        completions = self.tokenizer.batch_decode(
            sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )


        completion_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        completion_mask[:, input_ids.shape[1] :] = True
        completion_mask[sequence_ids == self.tokenizer.pad_token_id] = False
        completion_mask = completion_mask[:, 1:]


        # 2. compute rewards and advantages
        rewards = self._compute_rewards(
            completions=completions,
            completions_ids=sequence_ids[:, input_ids.shape[1]:],
            prompts=prompts,
            answers=answers,
        )
        advantages = self._group_advantages(rewards)
        advantages = advantages.to(input_ids.device)

        # 3. compute log-probabilities
        attention_mask = sequence_ids != self.tokenizer.pad_token_id
        with torch.no_grad():
            logp_ref = ref_model.compute_log_probs(sequence_ids, attention_mask)

            # logits = torch.stack(old_model_output.logits, dim=1)
            # # 2. log-probas π_old sur les tokens générés
            # logp_old = torch.log_softmax(logits, dim=-1)
            # gen_ids  = sequence_ids[:, -logits.size(1):]             # tokens générés
            # logp_old = logp_old.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, L_gen)

            # B, total_len = sequence_ids.shape
            # L_gen = old_model_output.logits[0].size(1)   # ou logits.size(1) après stack
            # print("sequence_ids:", sequence_ids.shape)
            # print("logits:", logits.shape)
            # print("completion_mask:", completion_mask.shape)
            # print("— prompt_len:", input_ids.shape[1], "gen_len:", L_gen)
            # print("true par batch:", completion_mask.sum(dim=1))


        state.batch = {
            "input_ids": input_ids,
            "labels": state.batch["labels"],
            "attention_mask": attention_mask,

            "sequence_ids": sequence_ids,  # (B,G,L)
            # "logprobs_old": logp_old,  # (B,G,V)
            "logprobs_ref": logp_ref,  # (B,G,V)
            "advantages": advantages,  # (B,G)
            "completion_mask": completion_mask,  # (B,G,L)
        }
