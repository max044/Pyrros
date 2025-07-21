import re
from pyrros.utils.reward_utils import RewardFunction


class FormatReward(RewardFunction):
    """
    Reward 1.0 if the completion matches the pattern `<think>…</think>…`.
    """

    def __call__(self, **kwargs) -> list[float]:
        """
        Check each completion against the `<think>…</think>` regex.

        Returns:
            1.0 for formatted completions, else 0.0.
        """

        completions = kwargs.get("completions", [])
        pattern = r"^<think>.*?</think>.*?$"

        return [1.0 if re.match(pattern, c) else 0.0 for c in completions]


class MathAnswerReward(RewardFunction):
    """
    Reward 1.0 if the numeric result in \\boxed{…} matches the ground-truth answer.
    """

    def __call__(self, **kwargs) -> list[float]:
        """
        Extract boxed answer from completion and compare to the answer key.

        Returns:
            1.0 for correct numerical answers, else 0.0.
        """

        completions = kwargs.get("completions", [])
        answers = kwargs.get("answers", [])
        matches = [re.search(r"\\boxed\{(.*?)\}", c) for c in completions]
        gts = [re.search(r"#### (\d+)", a) for a in answers]

        return [
            1.0 if m and g and m.group(1) == g.group(1) else 0.0
            for m, g in zip(matches, gts)
        ]
