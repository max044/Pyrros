from typing import List
from composer import Logger, State
import torch
from composer.core import Algorithm, Event
from transformers import GenerationConfig, PreTrainedTokenizer

class GRPOSampler(Algorithm):
    def __init__(self, tokenizer: PreTrainedTokenizer, reward_fns, G, generation_kwargs: dict = None):
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.G = G
        self.generation_kwargs = generation_kwargs or {}

    def match(self, event: Event, state: State):
        return event == Event.BEFORE_FORWARD

    def _compute_rewards(
        self,
        completions: List[str],
        completions_ids,
        prompts,
        answers,
        logger: Logger,
    ) -> torch.Tensor:
        
        # 1. Call each reward function
        list_of_rewards = [
            reward_fn(
                completions=completions,
                completions_ids=completions_ids,
                prompts=prompts,
                answers=answers,
            )
            for reward_fn in self.reward_fns
        ]

        # 2. Logging (mean + std) for each reward function
        for reward_fn, rewards in zip(self.reward_fns, list_of_rewards):
            rewards_np = torch.as_tensor(rewards, dtype=torch.float32)
            logger.log_metrics({
                f"rewards/{reward_fn.__class__.__name__}/mean": rewards_np.mean().item(),
                f"rewards/{reward_fn.__class__.__name__}/std": rewards_np.std().item(),
            })

        # 3. Convert rewards to torch tensors
        list_of_rewards = [
            torch.as_tensor(r, dtype=torch.float32) for r in list_of_rewards
        ]

        # 4. Sum rewards
        return torch.stack(list_of_rewards, dim=1).sum(dim=1)


    def _group_advantages(self, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    def apply(self, event: Event, state: State, logger: Logger):
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
            **self.generation_kwargs,
        )


        state.model.eval()
        old_model_output = state.model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config)
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
            logger=logger,
        )
        advantages = self._group_advantages(rewards)
        advantages = advantages.to(input_ids.device)

        # 3. compute log-probabilities
        attention_mask = sequence_ids != self.tokenizer.pad_token_id
        with torch.no_grad():
            sequence_ids = sequence_ids.to(ref_model.model.device)
            attention_mask = attention_mask.to(ref_model.model.device)
            logp_ref = ref_model.compute_log_probs(sequence_ids, attention_mask)

        terminated = [c for c in completions if c.endswith(self.tokenizer.eos_token)]
        terminated_lengths = [len(c) for c in terminated]
        all_lengths = [len(c) for c in completions]

        logger.log_metrics({
            "rewards/reward_mean": rewards.mean().item(),
            "rewards/reward_std": rewards.std().item(),
            "completions/max_length": max(all_lengths),
            "completions/mean_length": sum(all_lengths) / len(all_lengths),
            "completions/min_length": min(all_lengths),

            # Safe: only if there is at least one terminated completion
            "completions/max_terminated_length": max(terminated_lengths) if terminated_lengths else 0,
            "completions/mean_terminated_length": sum(terminated_lengths) / len(terminated_lengths) if terminated_lengths else 0,
            "completions/min_terminated_length": min(terminated_lengths) if terminated_lengths else 0,
        })

        state.batch = {
            "input_ids": input_ids,
            "labels": state.batch["labels"],
            "attention_mask": attention_mask,

            "sequence_ids": sequence_ids,
            "logprobs_ref": logp_ref,
            "advantages": advantages,
            "completion_mask": completion_mask,
        }
