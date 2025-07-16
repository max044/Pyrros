from abc import ABC, abstractmethod

class RewardFunction(ABC):
    """
    Base class for GRPO reward functions.

    The following variables may be provided via kwargs:
      - completions (List[str])
      - completions_ids (Tensor or List[List[int]])
      - prompts (List[str] or List[dict])
      - answers (List[str])
      - others…

    Subclasses can use any subset of them.
    """

    @abstractmethod
    def __call__(
        self,
        *,
        completions=None,
        completions_ids=None,
        prompts=None,
        answers=None,
        **kwargs
    ) -> list[float]:
        """
        Computes the reward for each example in the batch.

        Returns:
            A list of floats, one per example.
        """
        pass
