from typing import Union
from composer.models import HuggingFaceModel
import torch, torch.nn.functional as F


class GRPOModel(HuggingFaceModel):
    """
    Composer wrapper for a causal LM supporting GRPO loss and KL penalty.
    """

    def __init__(self, model, tokenizer, num_iterations, epsilon=0.2, beta=0.02):
        """
        Initialize with clipping epsilon and KL weight beta.
        """

        super().__init__(model=model, tokenizer=tokenizer)
        
        self._mu_iterations = 0
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, batch):
        """
        Compute log-probabilities of generated token sequences.

        Returns:
            Tensor of shape (batch_size * G, sequence_length).
        """

        if self._mu_iterations == 0:
            # First iteration, compute log-probs
            sequence_ids = batch["sequence_ids"]
            attention_mask = batch["attention_mask"]
            log_probs = self.compute_log_probs(sequence_ids, attention_mask)
            return log_probs
        else:
            # Reuse cached log-probs from previous iteration
            return batch["logprobs_old"]

    def loss(self, outputs, batch):
        """
        Compute GRPO loss: clipped policy gradient surrogate plus KL regularization.

        Args:
            outputs: Log-probs from forward().
            batch: Dict containing `logprobs_ref`, `advantages`, and `completion_mask`.

        Returns:
            A tensor of per-example loss values, aggregated over tokens.
        """

        log_probs = outputs
        logprobs_ref = batch["logprobs_ref"]
        advantages = batch["advantages"]
        completion_mask = batch["completion_mask"]

        kl = self._approximate_kl_divergence(log_probs, logprobs_ref, completion_mask)

        if self.num_iterations > 1:
            # For multiple iterations, we use the cached log-probs from the previous iteration
            logprobs_old = batch["logprobs_old"]
        else:
            # For single iteration, we use the current log-probs as old log-probs
            # This is the default behavior for the first iteration.
            # In practice, this would be replaced with the cached log-probs from the previous iteration
            logprobs_old = log_probs.detach()

        # Compute the loss
        ratio = (log_probs - logprobs_old).exp()
        surrogate_loss = ratio * advantages.unsqueeze(-1)
        surrogate_loss_clipped = torch.clamp(
            ratio, 1 - self.epsilon, 1 + self.epsilon
        ) * advantages.unsqueeze(-1)
        loss = -torch.min(surrogate_loss, surrogate_loss_clipped) + self.beta * kl

        if completion_mask is not None:
            loss = (loss * completion_mask).sum() / completion_mask.sum(axis=-1).clamp(
                min=1.0
            )
        else:
            loss = loss.mean(axis=-1)

        self._mu_iterations = (self._mu_iterations + 1) % self.num_iterations

        return loss

    def _approximate_kl_divergence(
        self,
        logp: torch.Tensor,
        logp_ref: torch.Tensor,
        completion_mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Compute element-wise approximate KL divergence: exp(Δ) - Δ - 1.

        Args:
            logp: Current model log-probs.
            logp_ref: Reference model log-probs.
            completion_mask: Mask for only generated tokens.

        Returns:
            A tensor of KL divergences per token.
        """

        log_ratio = logp_ref.float() - logp.float()
        if completion_mask is not None:
            log_ratio = log_ratio * completion_mask

        return log_ratio.exp() - log_ratio - 1

    def compute_log_probs(
        self, sequence_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate log-probabilities of each token in `sequence_ids` given `attention_mask`.

        Returns:
            A tensor of shape (batch_size * G, sequence_length - 1).
        """

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

        sequence_ids = sequence_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        position_ids = position_ids.to(self.model.device)

        output = self.model.forward(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output["logits"]

        logits = logits[:, :-1].to(torch.float32)
        output_ids = sequence_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

        return log_probs
