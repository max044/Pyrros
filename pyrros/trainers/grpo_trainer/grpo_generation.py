from composer.core import Callback, Event
from torch import no_grad

class GRPOGenerationCallback(Callback):
    def __init__(self, num_samples: int = 4, max_new_tokens: int = 128,
                 reward_fn=None):
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.reward_fn = reward_fn                      # fonction externe

    def run_event(self, event, state, logger):
        if event != Event.BATCH_START:
            return

        model = state.model
        prompts = state.batch["input_ids"]              # shape [B, L]

        with no_grad():
            # 1. génère K réponses par prompt
            generations = model.generate(
                prompts.repeat_interleave(self.num_samples, 0),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )

        # 2. calcule le reward (arbitraire : modèle de récompense, heuristique…)
        rewards = self.reward_fn(generations)           # shape [B*K]

        # 3. replie moyennes et range dans le batch pour que l’algo les lise
        B = prompts.size(0)
        rewards = rewards.view(B, self.num_samples).mean(1)  # [B]
        state.batch["rewards"] = rewards.to(prompts.device)
