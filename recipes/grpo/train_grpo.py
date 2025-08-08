import torch
from torch import optim

from composer import Trainer
from composer.core import DataSpec, Precision
from composer.loggers import TensorboardLogger

from pyrros.utils.model_utils import load_model

from registry.models.grpo.grpo_model import GRPOModel
from registry.algorithms.grpo.grpo_sampler import GRPOSampler
from registry.algorithms.grpo.load_ref_model import LoadRefModel
from registry.utils.grpo.dataset_utils import get_grpo_dataloader

from .rewards import FormatReward, MathAnswerReward


# — configuration —
WEIGHT_PATH     = "LiquidAI/LFM2-350M"
GSM8K_SPLIT     = "train"
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You will be given a question and you need to provide a detailed answer. "
    "Use step-by-step reasoning to arrive at the answer. "
    "If the question is ambiguous, ask for clarification. "
    "Always provide the final answer at the end of your response."
)
USER_PROMPT = (
    "Please answer the following question: {question} "
    "Use step-by-step reasoning to arrive at the answer. "
    "If the question is ambiguous, ask for clarification. "
    "Always provide the final answer at the end of your response. "
    "Use <think>...</think> tags to indicate your reasoning steps. "
    "If the answer is a number, use \\boxed{{answer}} to format it."
)
BATCH_SIZE      = 2
MU              = 5
MAX_LENGTH      = 256
LR              = 2e-4
DEVICE          = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# — model —
model, tokenizer = load_model(WEIGHT_PATH, pretrained=True, dtype=torch.bfloat16, use_qlora=True, lora_r=16, lora_alpha=32)
model = GRPOModel(model=model, tokenizer=tokenizer, num_iterations=MU)

# — dataloader —
dataloader = get_grpo_dataloader(
    tokenizer=tokenizer,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    batch_size=BATCH_SIZE,
    mu=MU,
    max_length=MAX_LENGTH,
    split=GSM8K_SPLIT,
)

# — algorithms —
load_ref    = LoadRefModel(ref_model_name=WEIGHT_PATH, device=DEVICE)
grpo_sampler = GRPOSampler(
    tokenizer=tokenizer,
    reward_fns=[FormatReward(), MathAnswerReward()],
    G=2,
    num_iterations=MU,
    generation_kwargs={
        "max_new_tokens": 128,
        "max_length": MAX_LENGTH,
        "top_p": 1.0,
        "top_k": 50,
        "temperature": 0.6,
        "do_sample": True,
    },
)

# — trainer —
trainer = Trainer(
    model=model,
    train_dataloader=DataSpec(
        dataloader,
        get_num_samples_in_batch=lambda b: b["input_ids"].size(0),
    ),
    device_train_microbatch_size=BATCH_SIZE,
    max_duration="1ep",
    algorithms=[load_ref, grpo_sampler],
    device=DEVICE,
    precision=Precision.AMP_BF16,
    optimizers=optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01),
    save_folder="grpo_output",
    save_interval="50ba",
    save_num_checkpoints_to_keep=5,
    loggers=[TensorboardLogger(flush_interval=1)],
)

trainer.fit()
trainer.close()
