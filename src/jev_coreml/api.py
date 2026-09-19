"""Jev-compatible request parsing and response rendering for the local model."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .semantic import CoreMLSemanticJev


def _state_text(state: object) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ": "))


def _criteria(raw: object, question_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stable answer keys and their natural-language criteria."""
    if isinstance(raw, Mapping):
        keys = tuple(str(key) for key in raw)
        return keys, tuple(str(raw[key]) for key in raw)
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        labels = tuple(str(value) for value in raw)
        return labels, labels
    if question_type == "noul":
        return (
            ("true", "false"),
            ("The proposition in the instruction is true for this state.", "The proposition in the instruction is false for this state."),
        )
    if question_type == "score":
        labels = tuple(str(value) for value in range(2, 11))
        return labels, labels
    raise ValueError("choice questions require a non-empty criteria mapping")


class LocalJev:
    """Typed local decision model with a Jev-compatible ``system_one`` API."""

    def __init__(self, semantic: CoreMLSemanticJev) -> None:
        self.semantic = semantic

    def system_one(self, state: object, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        text = _state_text(state)
        rendered: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
        for key, spec in questions.items():
            kind = str(spec["type"])
            if kind not in {"choice", "score", "noul"}:
                raise ValueError(f"{key}: unsupported question type {kind!r}")
            instruction = str(spec.get("instructions", spec.get("question", ""))).strip()
            if not instruction:
                raise ValueError(f"{key}: instructions are required")
            labels, descriptions = _criteria(spec.get("criteria"), kind)
            if len(labels) < 2:
                raise ValueError(f"{key}: at least two criteria are required")
            rendered.append((key, kind, labels, descriptions))
        rankings = self.semantic.rank_many(text, [(str(questions[key].get("instructions", questions[key].get("question", ""))).strip(), descriptions) for key, _, _, descriptions in rendered])
        answers: dict[str, Any] = {}
        for (key, kind, labels, descriptions), (_, probabilities) in zip(rendered, rankings, strict=True):
            by_label = dict(zip(labels, map(float, probabilities), strict=True))
            winner = int(probabilities.argmax())
            if kind == "choice":
                answers[key] = {
                    "type": "choice",
                    "choice": labels[winner],
                    "probabilities": by_label,
                    "confidence": float(probabilities[winner]),
                }
            elif kind == "noul":
                # Public Jev schemas conventionally use true/false criterion
                # keys.  If callers choose different labels, the first one is
                # explicitly the affirmative branch.
                answers[key] = {
                    "type": "noul",
                    "noul": float(probabilities[0]),
                    "confidence": float(max(probabilities)),
                }
            else:
                levels = np.arange(1, len(labels) + 1, dtype=np.float32)
                answers[key] = {
                    "type": "score",
                    "score": float(np.dot(levels, probabilities)),
                    "probabilities": by_label,
                    "confidence": float(probabilities[winner]),
                }
        return {"model": "jev-coreml-minilm-v0", "answers": answers}
