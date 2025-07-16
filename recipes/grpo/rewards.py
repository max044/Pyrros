import re
from pyrros.utils.reward_utils import RewardFunction

class FormatReward(RewardFunction):
    def __call__(self, **kwargs) -> list[float]:
        completions = kwargs.get("completions", [])
        pattern = r"^<think>.*?</think>.*?$"
        return [1.0 if re.match(pattern, c) else 0.0 for c in completions]

class MathAnswerReward(RewardFunction):
    def __call__(self, **kwargs) -> list[float]:
        completions = kwargs.get("completions", [])
        answers = kwargs.get("answers", [])
        matches = [re.search(r"\\boxed\{(.*?)\}", c) for c in completions]
        gts = [re.search(r"#### (\d+)", a) for a in answers]
        return [1.0 if m and g and m.group(1) == g.group(1) else 0.0 for m, g in zip(matches, gts)]
