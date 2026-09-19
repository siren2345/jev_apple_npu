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
import argparse
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs, _get_kwinputs


ROOT = Path(__file__).resolve().parents[1]
BATCH = 2
SEQUENCE = 128
OPTIONS = 20


@register_torch_op
def new_ones(context, node):
    """Lower mmBERT's scalar ``x.new_ones([], dtype=bool)`` helper.

    The static graph uses this only as a broadcastable boolean ``True`` while
    preparing its attention mask. Core ML Tools 9 lacks this PyTorch alias.
    """
    context.add(mb.const(val=True, name=node.name))


@register_torch_op(override=True)
def one_hot(context, node):
    """Work around Core ML Tools 9 dropping int64->int32 before one_hot."""
    inputs = _get_inputs(context, node, expected=(1, 2))
    labels = mb.cast(x=inputs[0], dtype="int32")
    num_classes = inputs[1] if len(inputs) > 1 else -1
    num_classes = _get_kwinputs(context, node, "num_classes", default=[num_classes])[0]
    if hasattr(num_classes, "val"):
        num_classes = num_classes.val
    context.add(mb.one_hot(indices=labels, one_hot_vector_size=num_classes, name=node.name))


class LayaExportWrapper(nn.Module):
    """Fixed-shape ModernBERT graph with Laya's original decision head.

    Hugging Face's generic ModernBERT forward dynamically creates full and
    sliding masks, and dynamically regenerates RoPE values.  Those helpers
    contain Python-shape operations Core ML cannot lower.  For an exported
    static shape, the masks and RoPE tables are constants, so this wrapper
    performs the same encoder calculation without those dynamic helpers.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("position_ids", torch.arange(SEQUENCE, dtype=torch.long).unsqueeze(0), persistent=False)
        positions = torch.arange(SEQUENCE, dtype=torch.float32)
        for attention_type in ("full_attention", "sliding_attention"):
            inv_freq = getattr(model.encoder.rotary_emb, f"{attention_type}_inv_freq").detach().float()
            scaling = getattr(model.encoder.rotary_emb, f"{attention_type}_attention_scaling")
            freqs = torch.outer(positions, inv_freq)
            angles = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(f"{attention_type}_cos", (angles.cos() * scaling).unsqueeze(0), persistent=False)
            self.register_buffer(f"{attention_type}_sin", (angles.sin() * scaling).unsqueeze(0), persistent=False)
        distance = torch.arange(SEQUENCE)[:, None] - torch.arange(SEQUENCE)[None, :]
        local = int(model.encoder.config.sliding_window)
        self.register_buffer("sliding_bias", (distance.abs() > local).to(torch.float32)[None, None] * -10000.0, persistent=False)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder = self.model.encoder
        h = encoder.embeddings(input_ids=input_ids.long())
        padding_bias = (1.0 - attention_mask.to(h.dtype))[:, None, None, :] * -10000.0
        position_embeddings = {
            "full_attention": (self.full_attention_cos.to(h.dtype), self.full_attention_sin.to(h.dtype)),
            "sliding_attention": (self.sliding_attention_cos.to(h.dtype), self.sliding_attention_sin.to(h.dtype)),
        }
        attention_masks = {
            "full_attention": padding_bias,
            "sliding_attention": padding_bias + self.sliding_bias.to(h.dtype),
        }
        for layer in encoder.layers:
            attention_type = layer.attention_type
            h = layer(h, attention_mask=attention_masks[attention_type], position_embeddings=position_embeddings[attention_type])
        h = encoder.final_norm(h)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(input_ids, attention_mask)
        h = h + self.model.type_emb(qtype.long())[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        # Core ML Tools 9 incorrectly lowers PyTorch's rank-3 gather index as
        # fp32.  A static-size one-hot selector is equivalent and lowers to a
        # matrix multiplication with integer indices preserved.
        selector = torch.nn.functional.one_hot(
            marker_pos.to(torch.int64).clamp(min=0), num_classes=SEQUENCE
        ).to(h.dtype)
        markers = torch.matmul(selector, h)
        logits = self.model.scorer(markers).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
        probabilities = torch.softmax(logits.detach(), -1)
        count = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(-1) / torch.log(count)
        top2 = probabilities.topk(2, -1).values
        features = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, count / 255.0], -1)
        action_logits = self.model.act_head(torch.cat([h[:, 0].float(), features], -1))
        return logits, action_logits


class LayaEncoderExportWrapper(LayaExportWrapper):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)


class LayaHeadExportWrapper(nn.Module):
    """Laya's two learned decision layers with Core ML-friendly attention.

    This emits all post-head hidden states. Marker selection and the final
    scorer are deliberately outside this probe: Core ML Tools 9 lowers their
    dynamic indices incorrectly, while the expensive contextual head can still
    execute on the Neural Engine.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states + self.model.type_emb(qtype.long())[:, None, :]
        padding_bias = (1.0 - attention_mask.to(h.dtype))[:, None, None, :] * -10000.0
        for layer in self.model.head.layers:
            residual = h
            x = torch.nn.functional.layer_norm(h, (768,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
            qkv = torch.nn.functional.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            qkv = qkv.view(BATCH, SEQUENCE, 3, 12, 64)
            query, key, value = qkv.unbind(dim=2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * 0.125 + padding_bias, dim=-1)
            attended = torch.matmul(weights, value).transpose(1, 2).reshape(BATCH, SEQUENCE, 768)
            h = residual + torch.nn.functional.linear(attended, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
            residual = h
            x = torch.nn.functional.layer_norm(h, (768,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
            x = torch.relu(torch.nn.functional.linear(x, layer.linear1.weight, layer.linear1.bias))
            h = residual + torch.nn.functional.linear(x, layer.linear2.weight, layer.linear2.bias)
        return h


def main() -> None:
    global BATCH, SEQUENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE)
    args = parser.parse_args()
    if args.encoder_only and args.head_only:
        raise ValueError("choose one export mode")
    if args.batch_size < 1 or args.sequence_length < 1:
        raise ValueError("batch size and sequence length must be positive")
    BATCH, SEQUENCE = args.batch_size, args.sequence_length
    agent = laya.load(str(ROOT / "models" / "laya-multilingual"), device="cpu")
    if args.encoder_only:
        model = LayaEncoderExportWrapper(agent.model.eval()).eval()
        inputs = (
            torch.full((BATCH, SEQUENCE), agent.tok.unk_token_id, dtype=torch.int32),
            torch.ones((BATCH, SEQUENCE), dtype=torch.int32),
        )
    elif args.head_only:
        model = LayaHeadExportWrapper(agent.model.eval()).eval()
        inputs = (
            torch.zeros((BATCH, SEQUENCE, 768), dtype=torch.float32),
            torch.ones((BATCH, SEQUENCE), dtype=torch.int32),
            torch.zeros((BATCH,), dtype=torch.int32),
        )
    else:
        model = LayaExportWrapper(agent.model.eval()).eval()
        inputs = (
            torch.full((BATCH, SEQUENCE), agent.tok.unk_token_id, dtype=torch.int32),
            torch.ones((BATCH, SEQUENCE), dtype=torch.int32),
            torch.zeros((BATCH, OPTIONS), dtype=torch.int32),
            torch.zeros((BATCH, OPTIONS), dtype=torch.bool),
            torch.zeros((BATCH,), dtype=torch.int32),
        )
    exported = torch.export.export(model, inputs).run_decompositions({})
    package = ROOT / "artifacts" / (
        f"laya-multilingual-encoder-b{BATCH}-s{SEQUENCE}.mlpackage" if args.encoder_only else
        f"laya-multilingual-head-hidden-b{BATCH}-s{SEQUENCE}.mlpackage" if args.head_only else
        f"laya-multilingual-b{BATCH}-s{SEQUENCE}-o20-macos26.mlpackage"
    )
    tensor_inputs = [
        ct.TensorType(name="hidden_states" if args.head_only else "input_ids", shape=inputs[0].shape, dtype=np.float16 if args.head_only else np.int32),
        ct.TensorType(name="attention_mask", shape=inputs[1].shape, dtype=np.int32),
    ]
    tensor_outputs = [ct.TensorType(name="last_hidden_state", dtype=np.float16)] if args.encoder_only else [
        ct.TensorType(name="option_logits", dtype=np.float16),
        ct.TensorType(name="action_logits", dtype=np.float16),
    ]
    if args.head_only:
        tensor_outputs = [ct.TensorType(name="decision_hidden_states", dtype=np.float16)]
    converted = ct.convert(
        exported,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS26,
        compute_precision=ct.precision.FLOAT16,
        inputs=tensor_inputs if args.encoder_only else tensor_inputs + ([
            ct.TensorType(name="qtype", shape=inputs[2].shape, dtype=np.int32),
        ] if args.head_only else [
            ct.TensorType(name="marker_pos", shape=inputs[2].shape, dtype=np.int32),
            ct.TensorType(name="marker_mask", shape=inputs[3].shape, dtype=np.int32 if args.head_only else np.bool_),
            ct.TensorType(name="qtype", shape=inputs[4].shape, dtype=np.int32),
        ]),
        outputs=tensor_outputs,
    )
    converted.save(str(package))
    print(f"saved={package}")


if __name__ == "__main__":
    main()
