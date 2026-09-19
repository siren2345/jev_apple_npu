"""Export the multilingual non-autoregressive Laya decision model to Core ML.

One Core ML call scores a fixed batch of rendered questions.  This is the
production-oriented path: it has no decoder loop and no KV cache.
"""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
import laya
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.mil import Builder as mb


ROOT = Path(__file__).resolve().parents[1]
BATCH = 8
SEQUENCE = 256
OPTIONS = 20


@register_torch_op
def new_ones(context, node):
    """Lower mmBERT's scalar ``x.new_ones([], dtype=bool)`` helper.

    The static graph uses this only as a broadcastable boolean ``True`` while
    preparing its attention mask. Core ML Tools 9 lacks this PyTorch alias.
    """
    context.add(mb.const(val=True, name=node.name))


class LayaExportWrapper(nn.Module):
    """Cast Core ML's int32 marker input to PyTorch's gather index dtype."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, attention_mask, marker_pos.to(torch.int64), marker_mask, qtype)


def main() -> None:
    agent = laya.load(str(ROOT / "models" / "laya-multilingual"), device="cpu")
    model = LayaExportWrapper(agent.model.eval()).eval()
    inputs = (
        torch.full((BATCH, SEQUENCE), agent.tok.unk_token_id, dtype=torch.int32),
        torch.ones((BATCH, SEQUENCE), dtype=torch.int32),
        torch.zeros((BATCH, OPTIONS), dtype=torch.int32),
        torch.zeros((BATCH, OPTIONS), dtype=torch.bool),
        torch.zeros((BATCH,), dtype=torch.int32),
    )
    # mmBERT's attention-mask helper emits `aten.new_ones` in torch.export,
    # which coremltools does not lower. TorchScript preserves an equivalent
    # graph with supported tensor constructors for this static-shape export.
    exported = torch.jit.trace(model, inputs, strict=False, check_trace=False)
    package = ROOT / "artifacts" / "laya-multilingual-b8-s256-o20.mlpackage"
    converted = ct.convert(
        exported,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(name="input_ids", shape=inputs[0].shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=inputs[1].shape, dtype=np.int32),
            ct.TensorType(name="marker_pos", shape=inputs[2].shape, dtype=np.int32),
            ct.TensorType(name="marker_mask", shape=inputs[3].shape, dtype=np.bool_),
            ct.TensorType(name="qtype", shape=inputs[4].shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="option_logits", dtype=np.float16),
            ct.TensorType(name="action_logits", dtype=np.float16),
        ],
    )
    converted.save(str(package))
    print(f"saved={package}")


if __name__ == "__main__":
    main()
