from __future__ import annotations
from composer.core import Callback, Event
import torch
from torch import no_grad
from torch.nn.functional import log_softmax

class GRPOGenerationCallback(Callback):
    """
    • Génére G réponses / prompt avec π_old (state.model est muté en amont).
    • Calcule les rewards puis l’avantage normalisé dans le groupe.
    • Stocke dans `state.batch` :
        - "logp_old" : log-probs sous π_old   (shape [B*G, L])
        - "advantages": Â_i,t                (shape [B*G, L])
        - "rewards": r_i                     (shape [B*G])
    """
    def __init__(
        self,
        num_samples: int = 4,
        max_new_tokens: int = 128,
        reward_fn=None,
    ):
        self.G = num_samples
        self.max_new_tokens = max_new_tokens
        self.reward_fn = reward_fn

    def run_event(self, event, state, logger):
        if event != Event.BATCH_START:
            return

        model_old = state.model                # π_old
        prompts   = state.batch["input_ids"]    # [B, L_prompt]

        B = prompts.size(0)
        prompts_dup = prompts.repeat_interleave(self.G, 0)

        with no_grad():
            out = model_old.generate(
                prompts_dup,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,           # pour récupérer les logits token-par-token
            )

        # ----- 1. reconstruit log-probs sous π_old --------------------------
        # `out.scores` : list[T] de tensors [B*G, V] (logits du token t)
        logits_seq = torch.stack(out.scores, dim=1)             # [B*G, L_new, V]
        logp_old   = log_softmax(logits_seq, dim=-1)            # mêmes dims
        # gather prob of chosen tokens
        gen_ids = out.sequences[:, prompts.size(1):]            # [B*G, L_new]
        logp_old = logp_old.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)  # [B*G, L_new]

        # ----- 2. rewards & advantages -------------------------------------
        rewards = self.reward_fn(out.sequences)                 # tensor [B*G]
        rewards = rewards.to(logp_old.device, dtype=torch.float32)

        rewards_group = rewards.view(B, self.G)                 # [B, G]
        mean, std = rewards_group.mean(1, keepdim=True), rewards_group.std(1, keepdim=True).clamp_min(1e-6)
        advantages = ((rewards_group - mean) / std).repeat_interleave(self.G, 0)  # [B*G]
        # broadcast sur chaque token de la réponse
        advantages = advantages.unsqueeze(-1).expand_as(logp_old)                # [B*G, L_new]

        # ----- 3. pousse dans state.batch -----------------------------------
        state.batch["logp_old"]   = logp_old.detach()
        state.batch["advantages"] = advantages.detach()
        state.batch["rewards"]    = rewards.detach()
        # Séquences générées comme labels (teacher forcing)
        state.batch["labels"]     = gen_ids
        state.batch["input_ids"]  = torch.cat([prompts_dup, gen_ids], dim=1)
