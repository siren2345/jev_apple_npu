"""Convert Qwen2.5-0.5B's text prefill graph to a Core ML package.

The exported graph returns only the final-position vocabulary logits. It never
uses ``generate`` and does not expose a text-decoding loop. A later export will
add a stateful KV-cache candidate graph once this baseline is verified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


class LastLogits(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, last_token_index: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
        sequence_length = input_ids.shape[1]
        causal = torch.triu(
            torch.full((sequence_length, sequence_length), torch.finfo(torch.float16).min, device=input_ids.device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        logits = self.model(
            input_ids=input_ids,
            position_ids=positions,
            attention_mask={"full_attention": causal},
            use_cache=False,
        ).logits
        index = last_token_index.to(torch.int64).view(1, 1, 1).expand(1, 1, logits.shape[-1])
        return logits.gather(1, index).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run macOS Core ML Runtime and compare the logits")
    args = parser.parse_args()
    model_dir = ROOT / "models" / "qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Export in fp32 because coremltools' SDPA decomposition expects its scale
    # constant to have the same dtype as Q/K/V. The emitted ML Program is then
    # compressed to fp16 through ``compute_precision`` below.
    source = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    source.config.use_cache = False
    model = LastLogits(source).eval()
    example = tokenizer("State: A payment was charged twice.\nQuestion: Is this billing?\nAnswer:", return_tensors="pt").input_ids.to(torch.int32)
    if example.shape[1] > 128:
        raise RuntimeError("The fixed prefill sequence unexpectedly exceeds 128 tokens")
    padded = torch.zeros((1, 128), dtype=torch.int32)
    padded[:, : example.shape[1]] = example
    last_token_index = torch.tensor([example.shape[1] - 1], dtype=torch.int32)

    exported = torch.export.export(model, (padded, last_token_index)).run_decompositions({})
    package = ROOT / "artifacts" / "qwen2.5-0.5b-prefill-128.mlpackage"
    mlmodel = ct.convert(
        exported,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(name="input_ids", shape=padded.shape, dtype=np.int32),
            ct.TensorType(name="last_token_index", shape=last_token_index.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="next_token_logits", dtype=np.float16)],
    )
    mlmodel.save(str(package))
    print(f"saved={package}")
    print(f"input_tokens={example.shape[1]}")
    if args.verify:
        with torch.inference_mode():
            expected = model(padded, last_token_index).float().numpy()
        received = mlmodel.predict(
            {"input_ids": padded.numpy(), "last_token_index": last_token_index.numpy()}
        )["next_token_logits"]
        max_error = float(np.max(np.abs(expected - received.astype(np.float32))))
        same_token = int(np.argmax(expected)) == int(np.argmax(received))
        print(f"max_absolute_error={max_error:.6f}")
        print(f"same_top_token={same_token}")
        if not same_token:
            raise SystemExit("Core ML changed the highest-probability token")


if __name__ == "__main__":
    main()
