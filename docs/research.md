# Jev と Core ML の調査メモ

調査日: 2026-09-19

## Jev の公開契約

Jev は state と独立した質問群を受け、次の型付き結果を返す。

| 質問型 | 結果 |
| --- | --- |
| `choice` | 候補キー、候補ごとの確率、confidence |
| `score` | 段階ごとの確率、期待値 score、confidence |
| `noul` | yes の確率 |

各質問は同じ state に対して独立して評価される。`choice` は最大 255 候補、`score` は 2–10 段階である。

公開仕様: https://docs.typesafe.ai/introduction

## decode を使わない設計

可能であり、Jev の用途に適している。候補 `c` のスコアは、生成を開始せずに次で求められる。

```
score(c) = log P(candidate tokens | rendered(state, instruction))
P(choice=c) = softmax(score(c))
```

prompt の prefill は一度だけ行い、その KV cache を各候補へ複製する。候補トークンの logits を一括 forward し、正解候補を decode せず採点する。`choice` は候補キー、`score` は各レベル番号、`noul` は `yes` / `no` を候補にする。

この方式の公開実装である open-jev は、Gemma 3 4B と MLX で同じ方式を実装している。

- https://github.com/daseinlabs/open-jev
- https://typesafe.ai/blog/introducing-system-one-models-and-jev

## 重要な限界

prefill は入力から特徴・尤度を得る計算であり、教師信号を作るものではない。事前学習済みモデルの尤度を softmax にした値はゼロショットの順位付けには使えるが、確率校正は保証されない。公開された open-jev の比較でも、ゼロショット Gemma は Jev と異なる判断と過度に鋭い確率を返している。

したがって二段階にする。

1. **ゼロショット基線**: 事前学習済み decoder Transformer の prefill + 候補尤度で完全な API 契約を実装する。
2. **校正段階**: 各タスクの state・質問・候補・正解を集め、frozen feature の小型 cross-attention head と temperature scaling を学習する。精度、ECE、state をシャッフルした対照試験を測る。

Jev 自体の重み、アーキテクチャ、RLCD の再現用データは公開されていない。そのため API 契約の互換性は目標にできるが、Jev と同一の意味的性能・確率値は主張しない。

## Core ML に載せる形

Core ML には tokenizer を含めず、アプリ側で tokenizer を実行する。モデル入出力は固定長の token IDs / mask（または埋め込み）と候補ごとの logit になる。候補を batch 次元に並べれば、`choice` / `score` / `noul` の正規化と JSON 整形は Swift 側で行える。

Core ML は PyTorch/MPS の実行方式とは別物として扱う。モデル選定と最適化は、対象 Mac で `all`、`cpuAndNeuralEngine`、`cpuAndGPU`、`cpuOnly` の各 compute unit を測り、最速の設定を採用する。初期 export は静的な 128-token shape に固定し、可変長へ進む場合も有限の `EnumeratedShapes`（例: 128/256/512）だけを許す。これにより Runtime が各形状を specialization でき、Neural Engine に載る可能性を残す。

モデル圧縮も実測で選ぶ。M4 では Apple の推奨上、INT4 per-block weight quantization は Mac GPU で効きやすく、W8A8 は Neural Engine の latency に有利になり得る。モデルサイズだけで方式を決めず、候補順位・確率校正・常駐メモリ・prefill latency を比較する。

Core ML Tools は PyTorch の `torch.export` から ML Program への変換をサポートする。SDPA を使う Transformer は対応 OS を指定すれば最適化された attention op に変換できる。最初の `scripts/export_demo.py` はその変換と macOS Runtime 予測の数値一致を検証する。

- https://apple.github.io/coremltools/docs-guides/source/model-exporting.html
- https://developer.apple.com/videos/play/wwdc2024/10159/

## 選定した初期バックボーン

初期実装は `Qwen/Qwen2.5-0.5B-Instruct` を使用する。テキスト専用の decoder-only Transformer であり、日本語を含む多言語を扱える。0.5B なので、16 GB RAM の Mac mini でも重み取得（953 MB）と fp16 Core ML package（944 MB）を繰り返し生成できる。

`scripts/export_qwen_prefill.py` は、固定長 128 トークンの prompt を受け、実トークン末尾だけの vocabulary logits を返す graph を出力する。Hugging Face の自動位置計算には Core ML Tools が変換できない `diff` 演算が含まれるため、位置 ID と因果マスクを wrapper で明示している。`scripts/verify_qwen_coreml.py` により PyTorch と Core ML の top-1 token 一致を確認した。fp16 量子化で logits の絶対差は生じるため、候補採点時は順位と確率校正を別々に評価する。

### M4 Mac mini 実測（2026-09-19）

固定長128 token、0.5B Qwen prefill、ウォームアップ後3回の中央値。`CPU_AND_NE` が最速だったため、現段階の既定 compute unit はこれにする。`ALL` はこのグラフでは最速ではなかった。候補バッチ化・KV cache・量子化を導入するたびに再測定する。

| Compute Unit | median |
| --- | ---: |
| CPU + Neural Engine | 23.3 ms |
| CPU + GPU | 59.5 ms |
| CPU only | 73.9 ms |
| all | 139.6 ms |

生データ: `artifacts/qwen2.5-0.5b-prefill-128.benchmark.json`

モデルカード: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

## Laya を参照実装にする

指定された Laya は、このプロジェクトが目指す利用体験の直接の参照になる。`state` と `questions` を一回の forward で受け、選択肢ごとの marker hidden state を decision head で採点して `choice` / `score` / `noul` を返す。日本語の二重請求・返金例はローカル `laya-multilingual` で正常に実行できた。

ただし同 checkpoint は mmBERT/ModernBERT の attention mask 内で `aten.new_ones` や tensor-to-int 変換を使う。Core ML Tools 9 の PyTorch frontend では未対応で、Torch Export と TorchScript の両方で変換が停止した。Core ML 前提の製品版では、Laya の request/response、候補 marker head、temperature calibration、評価方法を採用し、Core ML 変換を最初に通せる標準 BERT 系 encoder へ置換する。

これは性能上の後退を避けるための判断である。互換性のために CPU fallback を混ぜると、Core ML の compute unit に応じたレイテンシ・メモリ特性を評価できなくなる。

## 公開されている質問例

| 領域 | 型 | state の例 | 判定 |
| --- | --- | --- | --- |
| 問い合わせ振り分け | choice | 二重請求・返金希望 | billing / bug / account |
| 緊急度 | noul | 支払い機能が数日壊れている | 今すぐ人へエスカレーションするか |
| 不満度 | score | 同じ統合障害へ三回問い合わせた | calm / frustrated / very angry |
| コード評価 | choice | shell=True でユーザー入力をコマンドへ埋め込む | supported / violated / unknown |
| 会話評価 | noul | 会話履歴と最後のユーザーメッセージ | ユーザーは前の応答に反論しているか |
| ツール選択 | choice | DOM と可能な操作 | 次に実行する操作 |
| 検索の再順位付け | noul | クエリと文書候補 | 文書は質問への根拠になるか |
| モデレーション | noul + score | 投稿テキスト | 各ポリシー違反と深刻度 |
| リアルタイム制御 | choice | ゲーム盤面と合法手 | 次の移動 |
| ソフトウェア issue | choice + noul | issue の title / body | bug / feature / other と脆弱性の有無 |

出典:

- https://jevtypesafeai.com/how-to-use
- https://langfuse.com/blog/2026-09-18-using-typesafes-jev-for-evals
- https://blog.tumf.dev/posts/diary/2026/9/19/prompt-iterator-jev-verdict/
- https://systemonemodels.org/examples/projects/
- https://gist.github.com/mikehostetler/2a3c779a83168ec4f2ddf02a2deace09
# 実装更新: Core ML で使える最初の多言語ベースライン

`convaiinnovations/laya-multilingual` は API とデータ形式の良い参照だが、同モデルの ModernBERT/mmBERT graph は Core ML Tools 9 の `aten.new_ones` とテンソルから Python `int` への変換で止まった。従って、Laya の checkpoint を無理に変換する方針は採らない。

最初の実行可能な Core ML runtime には `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` を採用した。標準 BERT の 12-layer / 384 hidden encoder で、日本語を含む多言語の文類似度事前学習済みである。Transformer 5 の通常 forward は Core ML Tools 9 が下げられない `int` graph node を作るため、position id と token-type id を固定定数にした同値の BERT encoder path を `torch.export` で変換した。`torch.jit.trace` はこの経路では不適切だった。

生成物は `artifacts/multilingual-minilm-b8-s128.mlpackage` と `artifacts/multilingual-minilm-b32-s128.mlpackage`。入力は `int32` の `input_ids` と `attention_mask`、出力は `last_hidden_state`。Python の `jev_coreml.load().predict()` は 8 文以内では小バッチを使い、超える場合にのみ32文バッチを lazy-load する。同じ state の質問と候補を詰めて Core ML 呼び出しを共有し、mean-pool したベクトルを cosine ranking して Jev の `choice` / `score` / `noul` 形へ整形する。

これは使えるゼロショット基線であって、Jev の確率校正済み decision model ではない。特に yes/no と順序 score は対照例を含む学習・校正データが必要である。次段階では公開質問例を正規化し、Laya または Qwen の prefill 尤度を教師として Core ML 互換 BERT cross-encoder head を蒸留する。
