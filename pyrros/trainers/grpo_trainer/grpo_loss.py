from composer.core import Algorithm, Event
import torch.nn.functional as F
import torch

class GRPOLossAlgorithm(Algorithm):
    def __init__(self, beta: float = 0.1, kl_target: float = 0.05):
        self.beta = beta
        self.kl_target = kl_target

    def match(self, event, state):
        return event == Event.BEFORE_LOSS               # se déclenche une fois / batch

    def apply(self, event, state, logger):
        print("state: ", state)
        outputs = state.outputs
        if torch.is_tensor(outputs):               # cas rare : déjà un Tensor
            logits = outputs
        elif hasattr(outputs, "logits"):           # cas HF return_dict=True
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):   # return_dict=False → tuple
            logits = outputs[0]                    # logits en première position
        else:
            raise TypeError(f"Unexpected output type: {type(outputs)}")

        labels = state.batch["labels"]                  # [B, L]
        rewards = state.batch["rewards"]                # [B]

        # log-prob des actions
        logp = -F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        ).view(labels.size())

        # moyenne sur la séquence -> [B]
        logp = logp.mean(1)

        # KL vs. modèle de référence (option : passer en argu.)
        kl = (logp.detach() - logp).mean()              # simplifié —> KL approx.

        # loss GRPO (négative pour maximiser le reward)
        loss = -(rewards * logp).mean() + self.beta * kl

        # ajustement automatique de beta si tu veux suivre kl_target
        if self.kl_target is not None:
            self.beta *= torch.exp(kl - self.kl_target)

        state.loss = loss
