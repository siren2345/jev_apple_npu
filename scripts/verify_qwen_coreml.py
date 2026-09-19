"""Compare the saved Qwen Core ML prefill model with the PyTorch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_qwen_prefill import LastLogits


def main() -> None:
    model_dir = ROOT / "models" / "qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LastLogits(AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()).eval()
    example = tokenizer("State: A payment was charged twice.\nQuestion: Is this billing?\nAnswer:", return_tensors="pt").input_ids.to(torch.int32)
    input_ids = torch.zeros((1, 128), dtype=torch.int32)
    input_ids[:, : example.shape[1]] = example
    index = torch.tensor([example.shape[1] - 1], dtype=torch.int32)
    with torch.inference_mode():
        expected = model(input_ids, index).float().numpy()
    coreml = ct.models.MLModel(str(ROOT / "artifacts" / "qwen2.5-0.5b-prefill-128.mlpackage"))
    received = coreml.predict({"input_ids": input_ids.numpy(), "last_token_index": index.numpy()})["next_token_logits"]
    print(f"max_absolute_error={np.max(np.abs(expected - received.astype(np.float32))):.6f}")
    print(f"same_top_token={int(np.argmax(expected)) == int(np.argmax(received))}")


if __name__ == "__main__":
    main()
