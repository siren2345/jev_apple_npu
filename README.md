# jev_apple_npu

macOS で動く、Core ML ベースのローカル型付き判定 API です。Jev の request / response 形式を参考に、`choice`、`score`、`noul` を返します。TypeSafe AI や Jev の公式実装ではありません。

## 現在の到達点

- `choice`、`score`、`noul` を返す HTTP 契約を対象にする。
- 生成（decode）を使わず、state・質問・候補を一回の Transformer forward で採点する。
- まず小型の attention モデルを `.mlpackage` に変換し、macOS の Core ML Runtime で PyTorch と数値一致を検証する。
- 実用のゼロショット基線には Hugging Face の `Qwen/Qwen2.5-0.5B-Instruct` を使う。テキスト専用の decoder-only Transformer で、Jev 形式の候補尤度採点に使う。
- `convaiinnovations/laya-multilingual` の Apache-2.0 重みを、encoder・decision head・scorer の3つの固定形状 Core ML graph に分割して実行できる。22層の ModernBERT/mmBERT と2層の Laya decision head は `CPU_AND_NE` で動く。Core ML Tools 9 の整数 gather 不具合を避ける marker 選択だけは小さな NumPy 操作として残している。
- 最初に使える Core ML 実装として、`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` を固定形状（8文・32文 × 128 token）の `.mlpackage` に変換した。`jev_coreml.load().predict()` は候補数に応じて小さい方のバッチを選び、`choice` / `score` / `noul` の JSON 形状を返す。

詳細な調査と次の実装段階は [docs/research.md](docs/research.md) に記録しています。

## 対象環境

- Apple Silicon を搭載した macOS 15 以降
- Python 3.12
- Core ML Tools 9

CI は macOS 上で API 契約と wheel ビルドを確認する。重みのダウンロードと Core ML 変換はローカル実行に限定する。

## 実行

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/download_multilingual_minilm.py
.venv/bin/python scripts/export_multilingual_minilm.py
.venv/bin/python scripts/export_multilingual_minilm.py --batch-size 32
PYTHONPATH=src .venv/bin/python -m pytest -q
```

モデル重みと Core ML パッケージは Git に含めません。初回セットアップで Hugging Face から取得し、ローカルで変換します。

## Laya multilingual を Core ML で使う

Jevに近い判定品質が必要なら、Laya multilingual の学習済み decision head を使える。変換は固定形状 `batch=2` / `128 token` を既定とし、Apple Neural Engine を使う。

```sh
.venv/bin/python scripts/download_laya_multilingual.py
PYTHONPATH=src .venv/bin/python scripts/export_laya_coreml.py --encoder-only
PYTHONPATH=src .venv/bin/python scripts/export_laya_coreml.py --head-only
PYTHONPATH=src .venv/bin/python scripts/export_laya_scorer_coreml.py
```

```python
from jev_coreml import load_laya

agent = load_laya()
result = agent.predict(
    {"body": "二重に請求されました。重複分を返金してください。"},
    {
        "department": {
            "type": "choice",
            "instructions": "この問い合わせを担当すべき部門はどれですか？",
            "criteria": {"billing": "請求、支払い、返金", "technical": "障害、バグ", "sales": "新規契約、見積もり"},
        },
        "refund_requested": {"type": "noul", "instructions": "利用者は返金を求めていますか？"},
    },
)
print(result["answers"])
```

Mac mini M4 のウォームアップ後の Core ML 実測では、2設問・128 token の encoder + decision head + scorer が中央値 **9.96 ms** だった。tokenizer と prompt rendering は含まない。8設問・128 token は 43.27 ms で、2設問ずつ4回よりわずかに遅かったため、既定は小バッチにしている。

ローカル API を起動する場合は次を実行する。

```sh
PYTHONPATH=src .venv/bin/python scripts/serve.py
```

`http://127.0.0.1:8787/v1/systemone` は Jev と同じ `state` と `questions` の形を受ける。

```sh
curl -X POST http://127.0.0.1:8787/v1/systemone \
  -H 'content-type: application/json' \
  --data '{
    "state": {"ticket": "I was charged twice and need the duplicate refunded today."},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket.",
        "criteria": {
          "billing": "Charges, refunds, invoices, subscription changes.",
          "technical": "Bugs, failed integrations, anything not working.",
          "sales": "Pricing, plan comparisons, plans they do not have."
        }
      },
      "threatens_churn": {
        "type": "noul",
        "instructions": "The customer threatens to cancel, refund, charge back, or leave.",
        "criteria": {
          "true": "The customer threatens to cancel, request a refund, charge back, or leave.",
          "false": "The customer does not threaten to cancel, request a refund, charge back, or leave."
        }
      }
    }
  }'
```

Python API は次の形です。

```python
from jev_coreml import Question, load

agent = load(compute_units="cpu_and_ne")
result = agent.predict(
    "クレジットカードで二重に請求されました。返金してほしいです。",
    [Question("department", "choice", "この問い合わせをどの担当へ送りますか？", ("請求・返金", "技術サポート", "営業"))],
)
print(result["answers"]["department"]["choice"])
```

この初版は事前学習済みの文ベクトルによるゼロショット順位付けです。`choice` はすぐ試せますが、`score` と `noul` の確率は校正済みの意思決定確率ではありません。運用で自動実行する前には、対象の state・質問・正解を使って閾値と校正を検証する必要があります。
