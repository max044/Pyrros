# file to define rewards for GRPO fine-tuning with GSM8K dataset

import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>.*?$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_math_output(completions, answer, **kwargs):
    """Reward function that checks if the completion matches the ground truth."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completion_contents]
    gts = [re.search(r"#### (\d+)", gt) for gt in answer]

    contents = [match.group(1) if match else "" for match in matches]
    answer = [gt.group(1) if gt else "" for gt in gts]

    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, answer)]

if __name__ == "__main__":
    # Example usage
    completions = [[{"content": "<think> 2 + 2 = 4 </think>The result is \\boxed{4}"}],
                   [{"content": "<think> 3 + 3 = 6 </think>The result is \\boxed{6}"}]]
    answers = ["the answer of 2+2 is\n#### 4", "#### 6"]

    print(format_reward_func(completions))
    print(reward_math_output(completions, answers))