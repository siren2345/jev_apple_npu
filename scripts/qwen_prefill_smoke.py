"""Prove that local Qwen can rank Jev-style choices without decoding text."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_coreml.qwen_prefill import score_continuations


def main() -> None:
    model_dir = ROOT / "models" / "qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    prompt = (
        "State: I was charged twice for my subscription and need a refund.\n"
        "Question: Which department should handle this?\n"
        "Answer:"
    )
    options = [" billing", " technical", " sales"]
    scores = score_continuations(model, tokenizer, prompt, options)
    probabilities = torch.softmax(torch.tensor(scores), dim=0).tolist()
    for option, score, probability in zip(options, scores, probabilities, strict=True):
        print(f"{option.strip()} score={score:.4f} probability={probability:.4f}")


if __name__ == "__main__":
    main()
