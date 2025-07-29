from beam import function, Volume, Image, env

if env.is_remote():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

VOLUME_PATH = "./qwen-ft"


@function(
    image=Image()
    .add_python_packages(
        ["huggingface_hub", "datasets", "huggingface_hub[hf-transfer]"]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    memory="32Gi",
    cpu=4,
    volumes=[Volume(name="qwen-ft", mount_path=VOLUME_PATH)],
)
def upload():
    snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir=f"{VOLUME_PATH}/weights")

    dataset = load_dataset("openai/gsm8k", name="default")
    dataset.save_to_disk(f"{VOLUME_PATH}/data")
    print("Files uploaded successfully")


if __name__ == "__main__":
    upload()
