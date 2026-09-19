"""Export the standard multilingual MiniLM encoder to a Core ML package."""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from transformers import AutoModel
import argparse


ROOT = Path(__file__).resolve().parents[1]
BATCH = 8
SEQUENCE = 128


class Encoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("position_ids", torch.arange(SEQUENCE, dtype=torch.long).unsqueeze(0), persistent=False)
        self.register_buffer("token_type_ids", torch.zeros((BATCH, SEQUENCE), dtype=torch.long), persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Transformers 5's convenience forward path creates position/type ids
        # through tensor-to-Python ``int`` conversions that Core ML Tools 9
        # cannot lower.  This is the mathematically identical fixed-shape BERT
        # encoder path, with those two constant tensors made explicit.
        ids = input_ids.long()
        embeddings = self.model.embeddings.word_embeddings(ids)
        embeddings = embeddings + self.model.embeddings.token_type_embeddings(
            self.token_type_ids
        )
        embeddings = embeddings + self.model.embeddings.position_embeddings(self.position_ids)
        hidden = self.model.embeddings.LayerNorm(embeddings)
        mask = (1.0 - attention_mask.to(hidden.dtype))[:, None, None, :] * -10000.0
        return self.model.encoder(hidden, attention_mask=mask)[0]


def main() -> None:
    global BATCH
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("batch size must be at least 2")
    source = ROOT / "models" / "multilingual-minilm"
    # Encoder has fixed buffers because fixed Core ML shapes make Neural Engine
    # scheduling predictable.  Instantiate a class whose constants agree with
    # the requested batch shape.
    BATCH = args.batch_size
    model = Encoder(AutoModel.from_pretrained(source, local_files_only=True, attn_implementation="eager")).eval()
    inputs = (
        torch.zeros((BATCH, SEQUENCE), dtype=torch.int32),
        torch.ones((BATCH, SEQUENCE), dtype=torch.int32),
    )
    # ``torch.export`` retains fixed tensor shape expressions without the
    # TorchScript tensor-to-Python-int nodes emitted by BERT attention.
    traced = torch.export.export(model, inputs).run_decompositions({})
    package = ROOT / "artifacts" / f"multilingual-minilm-b{BATCH}-s128.mlpackage"
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(name="input_ids", shape=inputs[0].shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=inputs[1].shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="last_hidden_state", dtype=np.float16)],
    )
    converted.save(str(package))
    print(f"saved={package}")


if __name__ == "__main__":
    main()
