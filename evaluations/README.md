# Evaluations

本ディレクトリは、ハイブリッド検索サンドボックスの評価基盤です。クエリ＋関連ラベル（`data.json`）、ランキング収集ランナー（`run_eval.py`）、メトリクス計算（`calc_metrics.py`）をまとめ、再現可能なベンチマークを保管します。

## データ (`data.json`)

- スキーマ

```jsonc
{
  "version": "2025-12-28",
  "queries": [
    {
      "slug": "cat_general",
      "query": "猫",
      "notes": "猫に関する一般的なクエリ",
      "relevant_doc_ids": {
        "seed2": [1, 11, ...]
      }
    }
  ]
}
```

- `relevant_doc_ids` は docset（例: `documents.source = 'seed2'`）ごとに relevant doc_id リストを持ちます。
- クエリ数は 22 件を収録済み。カテゴリは動物/食/都市/技術などをバランスよく配置しています。

## ランキング収集 (`run_eval.py`)

Ollama でクエリ埋め込みを生成し、PGroonga／pgvector／RRF ハイブリッドを一括評価します。

```bash
python evaluations/run_eval.py \
  --docset seed2 \
  --models "snowflake-arctic-embed:22m,nomic-embed-text:v1.5" \
  --modes text,vector,hybrid \
  --rrf-weights 1:1 2:1 1:2 \
  --outdir evaluations/out/2025-12-28T12-00-00
```

- 出力: `rankings.jsonl`（各クエリ×条件1行）、`run_config.json`。
- JSONL レコード例

```json
{
  "query_slug": "cat_general",
  "mode": "hybrid",
  "model": "nomic-embed-text:v1.5",
  "docset": "seed2",
  "rrf": {"k": 60, "weights": {"text": 1.0, "vector": 1.0}},
  "results": [{"rank": 1, "document_id": 11, ...}, ...]
}
```

## メトリクス (`calc_metrics.py`)

`run_eval.py` の JSONL と `data.json` を用いて recall@k / MRR / nDCG@k を算出します。

```bash
python evaluations/calc_metrics.py \
  --data evaluations/data.json \
  --rankings evaluations/out/2025-12-28T12-00-00/rankings.jsonl \
  --k 3 5 10 \
  --per-query
```

- `metrics.json` (summary + 任意で per-query)、`metrics.md` (Markdown 表) を保存します。

## 推奨フロー

1. `make reset` : DB を clean + init
2. `make embed_all` : 5 モデル分の埋め込みを再計算
3. `make eval` : 既定条件でランキング収集 → メトリクス集計

これらの make ターゲットはリポジトリルートの `Makefile` に定義されています。必要に応じて `MODELS=...` や `EVAL_ARGS=...` を上書きし、部分的な再評価やクエリ絞り込みを行ってください。
