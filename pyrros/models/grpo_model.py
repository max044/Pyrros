from typing import Union
from composer.models import HuggingFaceModel
from peft import PeftModel
import torch, torch.nn.functional as F
import transformers

class GRPOModel(HuggingFaceModel):
    def __init__(self, model, tokenizer, epsilon=0.2, beta=0.02):
        super().__init__(model=model, tokenizer=tokenizer)
        self.epsilon, self.beta = epsilon, beta

    def forward(self, batch):
        sequence_ids = batch["sequence_ids"]
        attention_mask = batch["attention_mask"]
        completion_mask = batch["completion_mask"]
        log_probs = self.compute_log_probs(sequence_ids, attention_mask, gen_mask=completion_mask)
        return log_probs

    def loss(self, outputs, batch):
        log_probs = outputs
        logprobs_old = batch["logprobs_old"]
        logprobs_ref = batch["logprobs_ref"]
        advantages = batch["advantages"]
        completion_mask = batch["completion_mask"]
        completion_mask = completion_mask[:, -logprobs_old.size(1):]

        kl = self._approximate_kl_divergence(log_probs, logprobs_ref, completion_mask)
        
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
        attention_mask: torch.Tensor,
        gen_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calcule les log-probabilités des tokens générés par le modèle.
        Args:
            sequence_ids (torch.Tensor): Les IDs de séquence des tokens générés.
            attention_mask (torch.Tensor): Le masque d'attention.
        Returns:
            torch.Tensor: Les log-probabilités des tokens générés.
        """
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
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

        if gen_mask is not None:
            # shaped (B, L_total-1) → on compresse en (B, L_gen)
            log_probs = log_probs.masked_select(gen_mask).view(log_probs.size(0), -1)


        return log_probs
