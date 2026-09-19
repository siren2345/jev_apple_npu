"""Download the small multilingual semantic encoder used by the Core ML API."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    destination = ROOT / "models" / "multilingual-minilm"
    snapshot_download(
        MODEL_ID,
        local_dir=destination,
        ignore_patterns=["*.onnx", "*.safetensors.index.json", "openvino/*"],
    )
    print(f"saved={destination}")


if __name__ == "__main__":
    main()
