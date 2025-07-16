from beam import Volume, Image, function

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
    .with_secrets(["GITHUB_TOKEN"])
    .add_commands([
        # "pip install -U --upgrade uv",
        "pip install  git+https://max044:${GITHUB_TOKEN}@github.com/max044/Pyrros.git",
        # "uv sync",
        # "uv run  "
    ]),
    gpu="RTX4090",
    cpu=4,
    secrets=["GITHUB_TOKEN"]
)
def qwen_fine_tune_grpo():
    import os
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from pyrros.modules.model import load_model
    from pyrros.algorithms.grpo.load_ref_model import LoadRefModel
    from pyrros.algorithms.grpo.grpo_sampler import GRPOSampler
    from .rewards import format_reward_func, reward_math_output
    from composer import Trainer
    from torch import optim
    from composer.core import DataSpec
    from composer.loggers import TensorboardLogger


    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = "cuda"

    # Charger le modèle et le tokenizer via Pyrros (QLoRA ou sans PEFT)
    model, tokenizer = load_model(
        WEIGHT_PATH,
        pretrained=True,
        device=device,
        dtype=torch.float16,
        use_qlora=True,
    )

    # Charger le dataset GSM8K
    dataset = load_dataset(GSM8K_DATASET_PATH, name="main")
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
        # ex['input_ids'] et ex['attention_mask'] sont déjà des listes d'entiers
        input_ids = torch.tensor([ex['input_ids'] for ex in examples], dtype=torch.long)
        print(f"Input IDs shape: {input_ids.shape}")
        attention_mask = torch.tensor([ex['attention_mask'] for ex in examples], dtype=torch.long)
        labels = input_ids.clone()

        # gardez vos prompts/completions pour les reward functions
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

    # Définir les algorithmes GRPO
    load_ref = LoadRefModel(ref_model_name=WEIGHT_PATH)


    grpo_sampler = GRPOSampler(
        tokenizer=tokenizer,
        reward_fns=[format_reward_func, reward_math_output],
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

    # Composer Trainer
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
        loggers=[TensorboardLogger(log_dir=f"{MOUNT_PATH}/tensorboard_logs", flush_interval=1)],
    )

    # Lancer l'entraînement
    trainer.fit()
    trainer.close()



if __name__ == "__main__":
    qwen_fine_tune_grpo.remote()
