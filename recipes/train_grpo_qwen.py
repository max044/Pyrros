from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset  # seulement pour récupérer des prompts

from pyrros.trainers.grpo_composer import GRPOComposerTrainer
from pyrros.modules.rewards import no_reward  # ou ta reward perso

tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B", torch_dtype="bfloat16"
)

# On récupère des lignes Wikipedia en guise de prompts
wiki = load_dataset("wikitext", "wikitext-2-v1", split="train[:500]")
prompts = wiki["text"]

trainer = GRPOComposerTrainer(
    policy_model=model,
    tokenizer=tok,
    prompt_dataset=prompts,
    reward_fn=no_reward,
    batch_size=8,
    lr=2e-5,
    max_steps=2_000,
    device="cuda",
)

trainer.train()
