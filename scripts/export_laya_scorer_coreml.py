"""Export the final learned Laya option scorer as a Core ML package."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
import laya

ROOT = Path(__file__).resolve().parents[1]
OPTIONS, HIDDEN = 20, 768


class Scorer(nn.Module):
    def __init__(self, scorer: nn.Module) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(self, marker_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.scorer(marker_hidden_states).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    agent = laya.load(str(ROOT / "models" / "laya-multilingual"), device="cpu")
    inputs = (torch.zeros((args.batch_size, OPTIONS, HIDDEN), dtype=torch.float32),)
    exported = torch.export.export(Scorer(agent.model.scorer).eval(), inputs).run_decompositions({})
    converted = ct.convert(
        exported,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS26,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="marker_hidden_states", shape=inputs[0].shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="option_logits", dtype=np.float16)],
    )
    output = ROOT / "artifacts" / f"laya-multilingual-scorer-b{args.batch_size}-o20.mlpackage"
    converted.save(str(output))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
