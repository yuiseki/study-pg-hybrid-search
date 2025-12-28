# Evaluation Notes

このディレクトリには、手動でラベル付けした検索実験の結果を再現できるように、クエリと関連判断をまとめた JSON (`data.json`) とメトリクス計算スクリプト (`calc_metrics.py`) を置いています。

## 手順概要

1. `make clean && make start` で DB を初期化し、100件の seed ドキュメントを投入します。
2. `scripts/embed_documents.py` を使って、以下5モデルのベクトルを再計算します。`--batch-size 16` 程度で構いません。
   - `snowflake-arctic-embed:22m`
   - `nomic-embed-text:v1.5`
   - `embeddinggemma:300m`
   - `snowflake-arctic-embed2:568m`
   - `qwen3-embedding:4b`
3. `evaluations/data.json` のクエリをそれぞれ `scripts/search_vector.py` で実行すると、`rankings` セクションと同じ上位10件が得られます。
4. 私が判断した relevant ドキュメント ID は `relevance` セクションにモデル別で記載されています（ここが最重要ラベルです）。
5. `./evaluations/calc_metrics.py` を実行すると、recall@k および MRR を再計算できます。

```bash
./evaluations/calc_metrics.py --k 3 5 10
```

## `data.json` の構造

```json
{
  "<slug>": {
    "query": "クエリ文字列",
    "rankings": {
      "モデル名": [doc_id, ...]
    },
    "relevance": {
      "モデル名": [relevant_doc_id, ...]
    }
  }
}
```

- `rankings` は `scripts/search_vector.py <query> --model <name> --limit 10` の出力順を記録しています。
- `relevance` は私が付けたラベルです。空配列は「relevant なし」を意味します。
- `calc_metrics.py` はこの JSON から recall@k と MRR を計算し、CLI に一覧表示します。

## 収録済みクエリ

| Slug | クエリ | 備考 |
| --- | --- | --- |
| `cat_seed2` | 猫 | 猫関連文書の拾い方を確認 |
| `meal_seed2` | 食事 | 寿司/コーヒー系の関連性 |
| `catphrase_seed2` | 吾輩は猫である。名前はまだ無い。 | 文学フレーズのマッチング |
| `search_seed2` | 探し物 | ハイブリッド検索メモが対象 |
| `embed_seed2` | 埋め込みベクトル | 埋め込み/AIノート系の反応 |

必要に応じて、新しいクエリを追加し `data.json` を更新してください。ラベルを付けたら `calc_metrics.py` で再計算し、`evaluations/notes.md` などにメモを残す運用を想定しています。
