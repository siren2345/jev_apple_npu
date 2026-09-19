"""Benchmark Core ML compute-unit choices on the actual target Mac.

Run after `export_qwen_prefill.py`.  This deliberately measures the compiled
Core ML model, not PyTorch/MPS, because the runtime may partition the graph
between CPU, GPU, and the Neural Engine differently for each configuration.
"""

from __future__ import annotations

import statistics
import time
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts" / "qwen2.5-0.5b-prefill-128.mlpackage"
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
CONFIGURATIONS = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def inputs() -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokens = tokenizer("State: A payment was charged twice.\nQuestion: Is this billing?\nAnswer:", return_tensors="np").input_ids.astype(np.int32)
    ids = np.zeros((1, 128), dtype=np.int32)
    ids[:, : tokens.shape[1]] = tokens
    return {"input_ids": ids, "last_token_index": np.array([tokens.shape[1] - 1], dtype=np.int32)}


def main() -> None:
    request = inputs()
    results: dict[str, dict[str, object]] = {}
    for name, unit in CONFIGURATIONS.items():
        model = ct.models.MLModel(str(PACKAGE), compute_units=unit)
        model.predict(request)  # warm-up / specialization
        times: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            model.predict(request)
            times.append((time.perf_counter() - started) * 1000)
        results[name] = {"median_ms": round(statistics.median(times), 1), "samples_ms": [round(x, 1) for x in times]}
        print(f"{name}: median_ms={results[name]['median_ms']} samples_ms={results[name]['samples_ms']}", flush=True)
    destination = ROOT / "artifacts" / "qwen2.5-0.5b-prefill-128.benchmark.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved={destination}", flush=True)


if __name__ == "__main__":
    main()
