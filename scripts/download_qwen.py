"""Download the initial local LLM checkpoint from Hugging Face.

The checkpoint is intentionally kept outside Git.  It is a text-only,
decoder-only model small enough to repeatedly convert and test on this Mac.
"""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"


def main() -> None:
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer.*",
            "merges.txt",
            "vocab.json",
            "LICENSE",
            "README.md",
        ],
    )
    print(path)


if __name__ == "__main__":
    main()
