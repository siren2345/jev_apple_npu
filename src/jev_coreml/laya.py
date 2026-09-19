"""Core ML runtime for the Apache-2.0 Laya multilingual checkpoint.

The exported Laya graph is split into encoder, decision head, and option
scorer packages so that all Transformer work can execute on Apple Neural
Engine. The small marker gather remains NumPy-side due to a Core ML Tools 9
integer-gather lowering issue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
from laya.common import QTYPES, build_sequence, confidence_from_probs, render_options
from transformers import AutoTokenizer


class CoreMLLayaAgent:
    """Laya-compatible typed-decision API backed by three Core ML packages."""

    max_options = 20

    def __init__(
        self,
        root: str | Path,
        *,
        batch_size: int = 2,
        sequence_length: int = 128,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
    ) -> None:
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive")
        self.root = Path(root)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        checkpoint = self.root / "models" / "laya-multilingual"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
        config = json.loads((checkpoint / "rl_agent_config.json").read_text())
        self.head_max_len = int(config["head_max_len"])
        self.temperature = np.asarray(config["temperature"], dtype=np.float32)
        artifacts = self.root / "artifacts"
        self.encoder = ct.models.MLModel(
            str(artifacts / f"laya-multilingual-encoder-b{batch_size}-s{sequence_length}.mlpackage"),
            compute_units=compute_units,
        )
        self.head = ct.models.MLModel(
            str(artifacts / f"laya-multilingual-head-hidden-b{batch_size}-s{sequence_length}.mlpackage"),
            compute_units=compute_units,
        )
        self.scorer = ct.models.MLModel(
            str(artifacts / f"laya-multilingual-scorer-b{batch_size}-o20.mlpackage"),
            compute_units=compute_units,
        )

    @staticmethod
    def _internal(question: dict[str, Any]) -> dict[str, Any]:
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict):
                raise ValueError("choice criteria must be a mapping")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("score criteria must contain 2 to 10 ordered levels")
        elif kind == "noul":
            criteria = criteria or {}
        else:
            raise ValueError(f"unsupported question type: {kind}")
        return {"t": kind, "ins": str(question["instructions"]), "crit": criteria}

    def _run_batch(
        self, state: object, pending: list[tuple[str, dict[str, Any]]]
    ) -> tuple[list[tuple[str, dict[str, Any], np.ndarray]], int]:
        ids = np.full((self.batch_size, self.sequence_length), self.tokenizer.pad_token_id, np.int32)
        attention = np.zeros((self.batch_size, self.sequence_length), np.int32)
        positions = np.zeros((self.batch_size, self.max_options), np.intp)
        marker_mask = np.zeros((self.batch_size, self.max_options), bool)
        qtype = np.zeros((self.batch_size,), np.int32)
        actual: list[tuple[str, dict[str, Any]]] = []
        input_tokens = 0
        for row, (key, question) in enumerate(pending):
            internal = self._internal(question)
            sequence, markers = build_sequence(
                self.tokenizer, state, internal, max_len=self.sequence_length, head_max_len=self.head_max_len
            )
            if len(markers) > self.max_options:
                raise ValueError(f"{key}: at most {self.max_options} options are supported")
            ids[row, : len(sequence)] = sequence
            attention[row, : len(sequence)] = 1
            positions[row, : len(markers)] = markers
            marker_mask[row, : len(markers)] = True
            qtype[row] = QTYPES[internal["t"]]
            input_tokens += len(sequence)
            actual.append((key, internal))
        encoded = self.encoder.predict({"input_ids": ids, "attention_mask": attention})["last_hidden_state"]
        headed = self.head.predict(
            {"hidden_states": encoded.astype(np.float16), "attention_mask": attention, "qtype": qtype}
        )["decision_hidden_states"]
        selected = np.take_along_axis(
            headed, np.broadcast_to(positions[..., None], (self.batch_size, self.max_options, 768)), axis=1
        )
        logits = self.scorer.predict({"marker_hidden_states": selected.astype(np.float16)})["option_logits"]
        return [(key, item, logits[row, marker_mask[row]]) for row, (key, item) in enumerate(actual)], input_tokens

    def predict(self, state: object, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        input_tokens = 0
        pending = list(questions.items())
        for offset in range(0, len(pending), self.batch_size):
            rows, tokens = self._run_batch(state, pending[offset : offset + self.batch_size])
            input_tokens += tokens
            for key, question, logits in rows:
                options = render_options(question)
                scaled = logits / self.temperature[QTYPES[question["t"]]]
                probabilities = np.exp(scaled - scaled.max())
                probabilities /= probabilities.sum()
                confidence = round(confidence_from_probs(probabilities, len(options)), 4)
                if question["t"] == "choice":
                    labels = list(question["crit"].keys())
                    answers[key] = {"type": "choice", "choice": labels[int(probabilities.argmax())], "probabilities": {label: round(float(value), 4) for label, value in zip(labels, probabilities)}, "confidence": confidence}
                elif question["t"] == "score":
                    answers[key] = {"type": "score", "score": round(float((np.arange(len(options)) * probabilities).sum()), 4), "probabilities": [round(float(value), 4) for value in probabilities], "confidence": confidence}
                else:
                    answers[key] = {"type": "noul", "noul": round(float(probabilities[1]), 4), "confidence": confidence}
        return {"model": "laya-multilingual-coreml-split", "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def load_laya(root: str | Path | None = None, *, batch_size: int = 2, sequence_length: int = 128) -> CoreMLLayaAgent:
    """Load a locally exported Laya Core ML package set."""
    project = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return CoreMLLayaAgent(project, batch_size=batch_size, sequence_length=sequence_length)
