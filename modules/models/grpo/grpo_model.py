from typing import Union
from composer.models import HuggingFaceModel
import torch, torch.nn.functional as F

class GRPOModel(HuggingFaceModel):
    def __init__(self, model, tokenizer, epsilon=0.2, beta=0.02):
        super().__init__(model=model, tokenizer=tokenizer)
        self.epsilon, self.beta = epsilon, beta

    def forward(self, batch):
        sequence_ids = batch["sequence_ids"]
        attention_mask = batch["attention_mask"]
        log_probs = self.compute_log_probs(sequence_ids, attention_mask)
        return log_probs

    def loss(self, outputs, batch):
        log_probs = outputs
        logprobs_ref = batch["logprobs_ref"]
        advantages = batch["advantages"]
        completion_mask = batch["completion_mask"]

        kl = self._approximate_kl_divergence(log_probs, logprobs_ref, completion_mask)

        logprobs_old = log_probs.detach()

        # Compute the loss
        ratio = (log_probs - logprobs_old).exp()
        surrogate_loss = ratio * advantages.unsqueeze(-1)
        surrogate_loss_clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.unsqueeze(-1)
        loss = -torch.min(surrogate_loss, surrogate_loss_clipped) + self.beta * kl

        if completion_mask is not None:
            loss = (loss * completion_mask).sum() / completion_mask.sum(axis=-1)
        else:
            loss = loss.mean(axis=-1)

        return loss


    def _approximate_kl_divergence(self, logp: torch.Tensor, logp_ref: torch.Tensor, completion_mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        log_ratio = logp_ref.float() - logp.float()
        if completion_mask is not None:
            log_ratio = log_ratio * completion_mask

        return log_ratio.exp() - log_ratio - 1

    def compute_log_probs(
        self,
        sequence_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log-probabilities of the generated tokens.
        Args:
            sequence_ids (torch.Tensor): The IDs of the generated tokens.
            attention_mask (torch.Tensor): The attention mask.
        Returns:
            torch.Tensor: The log-probabilities of the generated tokens.
        """
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

        # place ids on model device
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
