from beam import Volume, Image, function
from registry.models.grpo.grpo_model import GRPOModel
from registry.utils.grpo.dataset_utils import get_grpo_dataloader

# Le chemin de montage sur le volume Beam
MOUNT_PATH = "./qwen-ft"
WEIGHT_PATH = "./qwen-ft/weights"
GSM8K_DATASET_PATH = "./qwen-ft/data/"

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


@function(
    volumes=[Volume(name="qwen-ft", mount_path=MOUNT_PATH)],
    image=Image()
    .add_commands(
        [
            'pip install "Pyrros @ git+https://github.com/max044/Pyrros.git"',
            "pyrros add grpo",
        ]
    ),
    gpu="A100-40",
    cpu=4,
)
def qwen_fine_tune_grpo():
    import os
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    from pyrros.utils.model_utils import load_model
    from registry.algorithms.grpo.load_ref_model import LoadRefModel
    from registry.algorithms.grpo.grpo_sampler import GRPOSampler
    from .rewards import FormatReward, MathAnswerReward
    from composer import Trainer
    from torch import optim
    from composer.core import DataSpec
    from composer.loggers import TensorboardLogger

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    BATCH_SIZE      = 2
    MU              = 5
    MAX_LENGTH      = 1024
    LR              = 2e-4
    DEVICE          = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # — model —
    model, tokenizer = load_model(WEIGHT_PATH, pretrained=True, dtype=torch.float32)
    model = GRPOModel(model=model, tokenizer=tokenizer)

    # — dataloader —
    dataloader = get_grpo_dataloader(
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        batch_size=BATCH_SIZE,
        mu=MU,
        max_length=MAX_LENGTH,
        split="train",
    )

    # — algorithms —
    load_ref    = LoadRefModel(ref_model_name=WEIGHT_PATH, device=DEVICE)
    grpo_sampler = GRPOSampler(
        tokenizer=tokenizer,
        reward_fns=[FormatReward(), MathAnswerReward()],
        G=2,
        num_iterations=MU,
        generation_kwargs={
            "max_new_tokens": 512,
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
        precision="fp32",
        optimizers=optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01),
        save_folder="grpo_output",
        save_num_checkpoints_to_keep=5,
        loggers=[TensorboardLogger(log_dir=f"{MOUNT_PATH}/tensorboard_logs", flush_interval=1)],
    )

    trainer.fit()
    trainer.close()


#     device = (
#         "gpu"
#         if torch.cuda.is_available()
#         else "mps" if torch.backends.mps.is_available() else "cpu"
#     )

#     model, tokenizer = load_model(
#         WEIGHT_PATH,
#         pretrained=True,
#         dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
#         use_qlora=True,
#         lora_r=8,
#         lora_alpha=16,
#     )
#     model = GRPOModel(model=model, tokenizer=tokenizer)

#     dataset = load_from_disk(GSM8K_DATASET_PATH)
#     train_ds = dataset["train"]
#     # test_ds = dataset["test"]
#     print(train_ds)

#     def prepare_example(ex):
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": USER_PROMPT.format(question=ex["question"])},
#         ]
#         answers = ex["answer"]
#         tokenized = tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=True,
#             padding="max_length",
#             truncation=True,
#             return_dict=True,
#             return_tensors="pt",
#             max_length=512,
#         )

#         return {
#             "input_ids": tokenized["input_ids"].squeeze(0),
#             "attention_mask": tokenized["attention_mask"].squeeze(0),
#             "prompts": messages,
#             "answers": answers,
#         }

#     tokenized = train_ds.map(
#         lambda ex: prepare_example(ex),
#         batched=False,
#     )

#     def collate_fn(examples):
#         input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
#         attention_mask = torch.tensor(
#             [ex["attention_mask"] for ex in examples], dtype=torch.long
#         )

#         prompts = [ex["prompts"] for ex in examples]
#         answers = [ex["answers"] for ex in examples]

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "prompts": prompts,
#             "answers": answers,
#         }

#     dataloader = DataLoader(
#         tokenized,
#         batch_size=2,
#         shuffle=True,
#         collate_fn=collate_fn,
#     )

#     load_ref = LoadRefModel(ref_model_name=WEIGHT_PATH, device=device)

#     grpo_sampler = GRPOSampler(
#         tokenizer=tokenizer,
#         reward_fns=[FormatReward(), MathAnswerReward()],
#         G=2,
#         num_iterations=2,
#         generation_kwargs={
#             "max_new_tokens": 1024,
#             "top_p": 1.0,
#             "temperature": 0.6,
#         },
#     )

#     train_data_spec = DataSpec(
#         dataloader,
#         get_num_samples_in_batch=lambda batch: batch["input_ids"].shape[0],
#     )

#     trainer = Trainer(
#         model=model,
#         train_dataloader=train_data_spec,
#         max_duration="1ep",
#         algorithms=[load_ref, grpo_sampler],
#         device=device,
#         precision="fp32",
#         optimizers=optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01),
#         save_folder="./output-grpo",
#         save_num_checkpoints_to_keep=5,
#         loggers=[
#             TensorboardLogger(
#                 log_dir=f"{MOUNT_PATH}/tensorboard_logs", flush_interval=1
#             )
#         ],
#     )

#     trainer.fit()
#     trainer.close()


if __name__ == "__main__":
    qwen_fine_tune_grpo.remote()
