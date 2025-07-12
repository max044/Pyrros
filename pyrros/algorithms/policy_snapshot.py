from copy import deepcopy
from composer.core import Algorithm, Event

class PolicySnapshot(Algorithm):
    def __init__(self, sampler_algo):
        self.sampler_name = sampler_algo.__class__.__name__
        # self.freq = every_steps
        # self._step = 0

    def match(self, event, state):
        return event == Event.BATCH_START

    def apply(self, event, state, logger):
        for algo in state.algorithms:
            if algo.__class__.__name__ == self.sampler_name:
                algo.old_model = deepcopy(state.model).eval().requires_grad_(False).to(state.device.name)
                break