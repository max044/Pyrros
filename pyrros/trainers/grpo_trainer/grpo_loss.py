from __future__ import annotations
from composer.core import Algorithm, Event
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

class GRPOLossAlgorithm(Algorithm):
    def __init__(
        self,
        reference_model,        # π_ref (gelé)  ➜ à passer depuis la recipe
        eps: float = 0.2,       # clip ε
        beta: float = 0.01,     # poids KL
    ):
        self.reference = reference_model.eval().requires_grad_(False)
        self.eps = eps
        self.beta = beta

    def match(self, event, state):
        return event == Event.BEFORE_LOSS

    def apply(self, event, state, logger):
        # ---- sorties sous π_θ (policy current) ----------------------------
        logits = state.outputs.logits if hasattr(state.outputs, "logits") else state.outputs[0]  # [B*G, L, V]
        logp_new = F.log_softmax(logits, dim=-1)                           # idem

        # ---- infos “old” préparées par le callback ------------------------
        logp_old = state.batch["logp_old"]                                 # [B*G, L]
        adv      = state.batch["advantages"]                               # [B*G, L]
        labels   = state.batch["labels"]                                   # [B*G, L]

        # gather log-probs pour les tokens choisis
        logp_new_tokens = logp_new.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B*G, L]

        # ---- ratio + clipping --------------------------------------------
        ratio = torch.exp(logp_new_tokens - logp_old)                      # [B*G, L]
        ratio_clipped = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
        policy_loss = -(torch.minimum(ratio * adv, ratio_clipped * adv)).mean()

        # ---- KL with reference (estimator Eq.4) ---------------------------
        with torch.no_grad():
            ref_logits = self.reference(labels)["logits"] if callable(self.reference) else \
                         self.reference.model(labels)["logits"]             # [B*G, L, V]
            logp_ref = F.log_softmax(ref_logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)

        kl_est = (torch.exp(logp_ref - logp_new_tokens)           # π_ref / π_new
                  - (logp_ref - logp_new_tokens) - 1.0).mean()

        state.loss = policy_loss + self.beta * kl_est
