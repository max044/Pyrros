import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from pyrros.utils.model_utils import load_model
from registry.algorithms.grpo.load_ref_model import LoadRefModel
from registry.algorithms.grpo.grpo_sampler import GRPOSampler
from registry.models.grpo.grpo_model import GRPOModel
from .rewards import FormatReward, MathAnswerReward
from composer import Trainer
from torch import optim
from composer.core import DataSpec
from composer.loggers import TensorboardLogger


WEIGHT_PATH = "LiquidAI/LFM2-350M"
GSM8K_DATASET_PATH = "openai/gsm8k"

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


os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


model, tokenizer = load_model(
    WEIGHT_PATH,
    pretrained=True,
    dtype=torch.float16,
    use_qlora=True,
)

model = GRPOModel(model=model, tokenizer=tokenizer)

dataset = load_dataset(GSM8K_DATASET_PATH, "main")
train_ds = dataset["train"]
# test_ds = dataset["test"]

def prepare_example(ex):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT.format(question=ex['question'])}]
    answers = ex['answer']
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding="max_length",
        truncation=True,
        return_dict=True,
        return_tensors="pt",
        max_length=512,
    )

    return {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
        "prompts": messages,
        "answers": answers,
    }

tokenized = train_ds.map(
    lambda ex: prepare_example(ex),
    batched=False,
)

def collate_fn(examples):
    input_ids = torch.tensor([ex['input_ids'] for ex in examples], dtype=torch.long)
    attention_mask = torch.tensor([ex['attention_mask'] for ex in examples], dtype=torch.long)
    labels = input_ids.clone()

    prompts = [ex['prompts'] for ex in examples]
    answers = [ex['answers'] for ex in examples]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'prompts': prompts,
        'answers': answers,
    }

dataloader = DataLoader(
    tokenized,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

load_ref = LoadRefModel(ref_model_name=WEIGHT_PATH)

grpo_sampler = GRPOSampler(
    tokenizer=tokenizer,
    reward_fns=[FormatReward(), MathAnswerReward()],
    G=2,
    generation_kwargs={
        'max_new_tokens': 1024,
        'top_p': 1.0,
        'temperature': 0.6,
    },
)

train_data_spec = DataSpec(
    dataloader,
    get_num_samples_in_batch=lambda batch: batch['input_ids'].shape[0],
)

trainer = Trainer(
    model=model,
    train_dataloader=train_data_spec,
    max_duration="1ep",
    algorithms=[load_ref, grpo_sampler],
    device=device,
    precision="fp32",
    optimizers=optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01),
    save_folder="./output-grpo",
    save_num_checkpoints_to_keep=5,
    loggers=[TensorboardLogger(flush_interval=1)],
)

trainer.fit()
trainer.close()
