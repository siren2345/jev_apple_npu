"""A Core ML-backed, Jev-shaped semantic decision API.

The encoder is a multilingual MiniLM Transformer exported to Core ML.  It is a
useful zero-shot baseline: each answer candidate is embedded alongside the
state and question, then ranked by cosine similarity.  This is deliberately
not presented as a calibrated Jev replacement; calibration and a trained
cross-encoder head are the next quality stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


QuestionType = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class Question:
    key: str
    type: QuestionType
    question: str
    options: tuple[str, ...] = ()


def _softmax(values: np.ndarray, temperature: float = 0.10) -> np.ndarray:
    scaled = values / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


class CoreMLSemanticJev:
    """Jev-like ``predict(state, questions)`` backed by Core ML.

    The exported models have fixed batch=8 and batch=32, each at 128 tokens.
    The wrapper selects the smaller batch for short requests and packs larger
    candidate sets into the 32-row model.
    """

    batch_size = 8
    large_batch_size = 32
    sequence_length = 128

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        compute_units: str = "cpu_and_ne",
        large_model_path: str | Path | None = None,
    ) -> None:
        unit_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }
        if compute_units not in unit_map:
            raise ValueError(f"unknown compute_units: {compute_units}")
        self.compute_unit = unit_map[compute_units]
        self.model = ct.models.MLModel(str(model_path), compute_units=self.compute_unit)
        self.large_model_path = Path(large_model_path) if large_model_path else None
        self.large_model: ct.models.MLModel | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    def _predict_batch(self, model: ct.models.MLModel, texts: list[str], batch_size: int) -> np.ndarray:
        padded = texts + [""] * (batch_size - len(texts))
        tokens = self.tokenizer(
            padded,
            padding="max_length",
            truncation=True,
            max_length=self.sequence_length,
            return_tensors="np",
        )
        result = model.predict(
            {
                "input_ids": tokens["input_ids"].astype(np.int32),
                "attention_mask": tokens["attention_mask"].astype(np.int32),
            }
        )
        hidden = np.asarray(result["last_hidden_state"], dtype=np.float32)
        mask = tokens["attention_mask"].astype(np.float32)[..., None]
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled[: len(texts)]

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        if len(texts) <= self.batch_size:
            return self._predict_batch(self.model, texts, self.batch_size)
        if self.large_model_path is None:
            selected_model, selected_batch = self.model, self.batch_size
        else:
            if self.large_model is None:
                self.large_model = ct.models.MLModel(str(self.large_model_path), compute_units=self.compute_unit)
            selected_model, selected_batch = self.large_model, self.large_batch_size
        for start in range(0, len(texts), selected_batch):
            vectors.append(self._predict_batch(selected_model, texts[start : start + selected_batch], selected_batch))
        return np.concatenate(vectors, axis=0)

    def _rank(self, state: str, question: str, options: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        return self.rank_many(state, [(question, options)])[0]

    def rank_many(self, state: str, groups: list[tuple[str, tuple[str, ...]]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Rank all questions for a state using densely packed Core ML batches.

        A Jev request normally has many independent questions on the same
        state.  Packing all queries and their options before invoking Core ML
        avoids one Neural Engine dispatch per question.
        """
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for question, options in groups:
            if not options:
                raise ValueError("options must not be empty")
            start = len(texts)
            texts.append(f"{state}\n\n質問: {question}")
            texts.extend(f"質問: {question}\n回答: {option}" for option in options)
            spans.append((start, len(texts)))
        vectors = self._embed(texts)
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for start, end in spans:
            similarities = vectors[start + 1 : end] @ vectors[start]
            results.append((similarities, _softmax(similarities)))
        return results

    def predict(self, state: str, questions: list[Question]) -> dict[str, Any]:
        rendered: list[tuple[Question, tuple[str, ...]]] = []
        for item in questions:
            if item.type == "noul":
                options = ("はい", "いいえ")
            elif item.type == "score":
                options = item.options or tuple(str(value) for value in range(2, 11))
            else:
                options = item.options
            if not options:
                raise ValueError(f"choice question {item.key!r} needs options")
            rendered.append((item, options))
        answers: dict[str, Any] = {}
        rankings = self.rank_many(state, [(item.question, options) for item, options in rendered])
        for (item, options), (_, probabilities) in zip(rendered, rankings, strict=True):
            best = int(probabilities.argmax())
            if item.type == "choice":
                answers[item.key] = {
                    "type": "choice",
                    "choice": options[best],
                    "probabilities": dict(zip(options, map(float, probabilities), strict=True)),
                    "confidence": float(probabilities[best]),
                }
            elif item.type == "score":
                answers[item.key] = {
                    "type": "score",
                    "score": options[best],
                    "probabilities": dict(zip(options, map(float, probabilities), strict=True)),
                    "confidence": float(probabilities[best]),
                }
            else:
                answers[item.key] = {
                    "type": "noul",
                    "noul": float(probabilities[0]),
                    "confidence": float(max(probabilities)),
                }
        return {"model": "coreml-multilingual-minilm", "answers": answers}


def load(root: str | Path | None = None, compute_units: str = "cpu_and_ne") -> CoreMLSemanticJev:
    """Load the local exported model with the same shape as ``laya.load``."""
    project = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return CoreMLSemanticJev(
        project / "artifacts" / "multilingual-minilm-b8-s128.mlpackage",
        project / "models" / "multilingual-minilm",
        compute_units,
        project / "artifacts" / "multilingual-minilm-b32-s128.mlpackage",
    )
