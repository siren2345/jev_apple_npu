"""No-decode candidate scoring with a local Qwen causal language model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


@torch.inference_mode()
def score_continuations(model: torch.nn.Module, tokenizer: object, prompt: str, options: Sequence[str]) -> list[float]:
    """Return conditional log-probabilities without calling ``generate``.

    Each option is teacher-forced after the same prompt. The loop is a reference
    implementation; the Core ML runtime will replace repeated prefixes with a
    stateful KV-cache prefill plus a batched candidate pass.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    scores: list[float] = []
    for option in options:
        option_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        tokens = torch.cat((prompt_ids, option_ids), dim=1)
        logits = model(input_ids=tokens, use_cache=False).logits[:, :-1]
        targets = tokens[:, 1:]
        start = prompt_ids.shape[1] - 1
        token_log_probs = F.log_softmax(logits[:, start:], dim=-1)
        chosen = token_log_probs.gather(2, targets[:, start:].unsqueeze(-1)).squeeze(-1)
        scores.append(float(chosen.sum().cpu()))
    return scores
