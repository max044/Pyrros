from copy import deepcopy
from composer.core import Algorithm, Event

class PolicySnapshot(Algorithm):
    def __init__(self, sampler_algo_name: str, every_steps=100):
        self.sampler_name = sampler_algo_name
        self.freq = every_steps
        self._step = 0

    def match(self, event, state):
        return event == Event.BATCH_END

    def apply(self, event, state, logger):
        # self._step += 1
        # if self._step % self.freq == 0:
        #     sampler = state.algorithms_by_name[self.sampler_name]
        #     sampler.policy_old.load_state_dict(deepcopy(state.model.state_dict()))
        
        sampler = state.algorithms_by_name[self.sampler_name]
        sampler.policy_old.load_state_dict(deepcopy(state.model.state_dict()))
