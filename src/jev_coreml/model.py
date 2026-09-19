"""A small prefill-only Transformer candidate scorer.

This model receives one rendered sequence per candidate. It performs no token
generation: a masked pooled hidden state is projected to a scalar candidate
logit. Production weights will replace this fixed-size proof model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PrefillCandidateScorer(nn.Module):
    def __init__(self, *, vocab_size: int = 512, hidden_size: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(128, hidden_size)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return one unnormalised score per candidate sequence.

        input_ids has shape (candidates, sequence_length); attention_mask is
        1 for valid tokens and 0 for padding.  Softmax stays outside Core ML so
        the same scores serve choice, score, and noul response renderers.
        """
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        # Keeping the sequence length fixed makes this first Core ML proof
        # portable.  Padding is ignored by masked mean pooling.  The trained
        # production graph will additionally use the mask inside attention.
        x = self.encoder(x)
        weights = attention_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.score(self.final_norm(pooled)).squeeze(-1)
