import pytest, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyrros.trainers.grpo_composer import GRPOComposerTrainer
from pyrros.modules.rewards import no_reward

def test_composer_trainer_smoke():
    tok = AutoTokenizer.from_pretrained("Qwen/qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/qwen3-0.6B")
    prompts = ["Hello world!"] * 16
    tr = GRPOComposerTrainer(
        policy_model=model,
        tokenizer=tok,
        prompt_dataset=prompts,
        reward_fn=no_reward,
        batch_size=2,
        max_steps=10,
        device="cpu",
    )
    tr.train()
