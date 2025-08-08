from typing import List, Optional
from composer import Logger, State
from composer.core import Algorithm, Event
from transformers import GenerationConfig, PreTrainedTokenizer
import torch

from pyrros.utils.model_utils import unwrap_ddp

class GRPOSampler(Algorithm):
    """
    Composer algorithm for GRPO: generates rollouts, computes rewards,
    advantages, and prepares training batch fields.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_fns: List,
        G: int,
        num_iterations: int,
        generation_kwargs: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.G = G
        self.num_iterations = num_iterations
        self._mu_iterations = 0
        self.generation_kwargs = generation_kwargs or {}
        self._cached_batch = None

    def match(self, event: Event, state: State) -> bool:
        return event == Event.BEFORE_FORWARD

    def apply(self, event: Event, state: State, logger: Logger):
        # Reset counter at cycle start
        if self._mu_iterations == 0:
            batch = state.batch
            inputs = self._expand_batch(batch)
            seq_ids, mask, texts = self._generate(inputs, state)
            rewards = self._compute_rewards(texts, seq_ids, inputs, logger)
            advantages = self._compute_advantages(rewards)
            attention_mask = seq_ids != self.tokenizer.pad_token_id
            logp_ref, logp_old = self._compute_logprobs(seq_ids, attention_mask, state)
            # TODO: compute logprobs for old model if num_iterations > 1
            state.batch = self._build_state_batch(inputs, seq_ids, logp_ref, logp_old, advantages, mask)
            self._cached_batch = state.batch
        else:
            # reuse last generated batch
            state.batch = self._cached_batch

        self._mu_iterations = (self._mu_iterations + 1) % self.num_iterations

    def _expand_batch(self, batch: dict) -> dict:
        input_ids = batch["input_ids"].repeat(self.G, 1)
        attention_mask = batch["attention_mask"].repeat(self.G, 1)
        prompts = batch["prompts"] * self.G
        answers = batch["answers"] * self.G
        return {"input_ids": input_ids, "attention_mask": attention_mask, "prompts": prompts, "answers": answers}

    def _generate(self, inputs: dict, state: State):
        with unwrap_ddp(state.model) as model:
            gen_conf = GenerationConfig(
                pad_token_id=model.config.pad_token_id,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                return_dict_in_generate=True,
                **self.generation_kwargs,
            )
            model.eval()
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"], attention_mask=inputs["attention_mask"], generation_config=gen_conf
                )
        state.model.train()

        seq = output.sequences
        start_idx = inputs["input_ids"].shape[1]
        mask = torch.zeros_like(seq, dtype=torch.bool)
        mask[:, start_idx:] = True
        mask[seq == self.tokenizer.pad_token_id] = False
        mask = mask[:, 1:]
        texts = self.tokenizer.batch_decode(seq[:, start_idx:], skip_special_tokens=True)
        return seq, mask, texts

    def _compute_rewards(
        self,
        completions: List[str],
        seq_ids: torch.Tensor,
        inputs: dict,
        logger: Logger,
    ) -> torch.Tensor:
        rewards_per_fn = []
        for fn in self.reward_fns:
            r = fn(
                completions=completions,
                completions_ids=seq_ids[:, inputs["input_ids"].shape[1]:],
                prompts=inputs["prompts"],
                answers=inputs["answers"],
            )
            tensor_r = torch.as_tensor(r, dtype=torch.float32)
            rewards_per_fn.append(tensor_r)
            # detailed logging per reward fn
            logger.log_metrics({
                f"rewards/{fn.__class__.__name__}/mean": tensor_r.mean().item(),
                f"rewards/{fn.__class__.__name__}/std": tensor_r.std().item(),
            })
        total_rewards = torch.stack(rewards_per_fn, dim=1).sum(dim=1)
        # aggregated reward metrics
        logger.log_metrics({
            "rewards/total_mean": total_rewards.mean().item(),
            "rewards/total_std": total_rewards.std().item(),
        })
        # completion length metrics
        all_lengths = [len(c) for c in completions]
        terminated = [c for c in completions if c.endswith(self.tokenizer.eos_token)]
        terminated_lengths = [len(c) for c in terminated]
        logger.log_metrics({
            "rewards/reward_mean": total_rewards.mean().item(),
            "rewards/reward_std": total_rewards.std().item(),
            "completions/max_length": max(all_lengths),
            "completions/mean_length": sum(all_lengths) / len(all_lengths),
            "completions/min_length": min(all_lengths),
            "completions/max_terminated_length": max(terminated_lengths) if terminated_lengths else 0,
            "completions/mean_terminated_length": sum(terminated_lengths) / len(terminated_lengths) if terminated_lengths else 0,
            "completions/min_terminated_length": min(terminated_lengths) if terminated_lengths else 0,
        })
        return total_rewards

    def _compute_advantages(self, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mean, std = rewards.mean(), rewards.std()
        return (rewards - mean) / (std + eps)

    def _compute_logprobs(
        self,
        seq_ids: torch.Tensor,
        mask: torch.Tensor,
        state: State,
    ) -> torch.Tensor:
        with unwrap_ddp(state.model) as current_model:
            ref_model = state.ref_model
            
            seq_ids = seq_ids.to(current_model.model.device)
            mask = mask.to(current_model.model.device)

            with torch.no_grad():
                logp_ref = ref_model.compute_log_probs(seq_ids, mask)
                logp_old = current_model.compute_log_probs(seq_ids, mask)

        return logp_ref, logp_old

    def _build_state_batch(
        self,
        inputs: dict,
        seq_ids: torch.Tensor,
        logp_ref: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        attention_mask = seq_ids != self.tokenizer.pad_token_id
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": attention_mask,
            "sequence_ids": seq_ids,
            "logprobs_ref": logp_ref,
            "logprobs_old": logp_old,
            "advantages": advantages.to(inputs["input_ids"].device),
            "completion_mask": mask,
        }
