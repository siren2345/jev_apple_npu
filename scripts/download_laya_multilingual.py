"""Download the Apache-2.0 multilingual Laya decision checkpoint locally."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "convaiinnovations/laya-multilingual"
LOCAL_DIR = ROOT / "models" / "laya-multilingual"


def main() -> None:
    print(
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            allow_patterns=["*.json", "*.safetensors", "tokenizer/*", "encoder/*", "*.txt", "LICENSE", "README.md"],
        )
    )


if __name__ == "__main__":
    main()
