"""Benchmark the exported encoder on every Core ML compute-unit choice."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from jev_coreml.semantic import Question, load


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    question = Question("department", "choice", "この問い合わせをどの担当へ送りますか？", ("請求・返金", "技術サポート", "営業"))
    result = {}
    for unit in ("all", "cpu_and_ne", "cpu_and_gpu", "cpu_only"):
        agent = load(compute_units=unit)
        agent.predict("クレジットカードで二重に請求されました。返金してほしいです。", [question])
        samples = []
        for _ in range(5):
            started = perf_counter()
            agent.predict("クレジットカードで二重に請求されました。返金してほしいです。", [question])
            samples.append((perf_counter() - started) * 1000)
        result[unit] = {"median_ms": sorted(samples)[len(samples) // 2], "samples_ms": samples}
    output = ROOT / "artifacts" / "multilingual-minilm-hybrid.benchmark.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
