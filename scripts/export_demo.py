"""Export a prefill-only Transformer scorer to Core ML and verify it on macOS."""

from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_coreml.model import PrefillCandidateScorer


def main() -> None:
    torch.manual_seed(7)
    model = PrefillCandidateScorer().eval()
    input_ids = torch.tensor(
        [[1, 44, 86, 9, 2, 0, 0, 0], [1, 44, 86, 10, 2, 0, 0, 0]], dtype=torch.int32
    )
    attention_mask = (input_ids != 0).to(torch.int32)

    exported = torch.export.export(model, (input_ids, attention_mask)).run_decompositions({})
    package = ROOT / "artifacts" / "prefill_candidate_scorer.mlpackage"
    package.parent.mkdir(exist_ok=True)
    mlmodel = ct.convert(
        exported,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="candidate_logits", dtype=np.float32)],
    )
    mlmodel.save(str(package))

    with torch.no_grad():
        expected = model(input_ids, attention_mask).numpy()
    received = mlmodel.predict(
        {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
    )["candidate_logits"]
    error = float(np.max(np.abs(expected - received)))
    print(f"saved={package}")
    print(f"pytorch={expected.tolist()}")
    print(f"coreml={received.tolist()}")
    print(f"max_absolute_error={error:.7f}")
    if error > 1e-4:
        raise SystemExit("Core ML output differs from PyTorch beyond tolerance")


if __name__ == "__main__":
    main()
