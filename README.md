# study-pg-hybrid-search

PostgreSQL 17 上で PGroonga と pgvector を組み合わせ、全文検索とベクトル検索をハイブリッドに統合するサンドボックスです。Ollama の埋め込み API を前提にしており、手元で簡単に再現・検証できます。

## 概要
- ベースイメージは `pgvector/pgvector:0.8.1-pg17-trixie`。Dockerfile で PGroonga を追加インストールしています。
- `docker-compose` 一発で Postgres を起動し、`db/init` 配下の SQL で拡張・スキーマ・サンプルデータを自動投入します。
- ドキュメント本文 (`documents`) とモデル別ベクトル (`document_embeddings`) を分離し、`embedding_models` で各モデルの次元を一元管理します。
- `scripts/` 以下にはテキスト検索・ベクトル検索・ハイブリッド検索・ドキュメント追加・一括埋め込みの Python CLI を用意しています。
- Python 側の依存は `psycopg` と `requests` のみなので、`pip install -r requirements.txt` だけで利用可能です。

## ディレクトリ構成
```
.
├── Dockerfile                # Postgres + PGroonga + pgvector イメージ
├── docker-compose.yml        # 単一 DB サービス (port 5434)
├── Makefile                  # 起動/停止/psql などのラッパー
├── db/
│   ├── data/                # 永続化ボリューム (初回は空)
│   └── init/                # 001_extensions.sql 等を自動適用
├── scripts/                 # Python CLI 群
├── sql/                     # psql から使うサンプルクエリ
└── requirements.txt         # Python 依存
```

## データモデルと halfvec インデックス
- `documents` : ドキュメント本体。`content` は title+body の generated column。PGroonga で `TokenUnigram` インデックスを張っています。
- `embedding_models` : モデル名と次元を登録するテーブル。初期状態で以下5種類（384〜2560次元）を投入しており、`dims <= 4000` であれば追加入力可能です。
  - `snowflake-arctic-embed:22m` (384)
  - `nomic-embed-text:v1.5` (768)
  - `embeddinggemma:300m` (768)
  - `snowflake-arctic-embed2:568m` (1024)
  - `qwen3-embedding:4b` (2560)
- `document_embeddings` : 各ドキュメント × モデルのベクトルを格納。列型は `vector`（次元未指定）のまま保持し、検索時だけ `embedding::halfvec(dims)` にキャストして half 精度統一で HNSW を使います。
- モデルごとに `document_embeddings` へ部分インデックス（`USING hnsw ((embedding::halfvec(dims)) halfvec_cosine_ops)`）を貼っているため、近似検索条件を完全に揃えた比較が可能です。

## 事前準備
1. Docker / Docker Compose v2
2. `make` (任意、無くても `docker compose` で操作可能)
3. `mkdir -p db/data` で空ディレクトリを用意（クリーン時に自動で維持されます）
4. Ollama (ローカルで `ollama serve` し、比較したい埋め込みモデルを `ollama pull nomic-embed-text:v1.5` などで取得しておくこと)
5. Python 3.11 以上（CLI を実行する場合）

## 起動と停止
```bash
# 初回ビルドと起動（バックグラウンド）
make start

# ログ追跡
make logs

# 停止
make stop

# ボリュームと data/ を掃除
make clean
```
- Postgres は `localhost:5434` で待ち受けます。DSN 例: `postgresql://postgres:postgres@localhost:5434/sandbox`
- `make psql` でコンテナ内の `psql` に入れます。

## Python クライアントのセットアップ
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- 環境変数 `DATABASE_URL`（省略時は上記 DSN）と `OLLAMA_URL`（既定 `http://localhost:11434`）を必要に応じて設定してください。

## サンプル操作

### 1. テキスト検索 (PGroonga)
```bash
python scripts/search_text.py 猫 --literal
```
- PGroonga の `&@~` 演算子で `documents.content` を全文検索します。
- `--literal` を付けるとクエリ文字列をダブルクォートで囲み、PGroonga 独自構文を意識せず検索できます。

### 2. ベクトル検索 (pgvector + halfvec 統一)
```bash
python scripts/search_vector.py "猫と本" --model nomic-embed-text:v1.5
```
- `embedding_models` に登録された次元を取得し、Ollama の結果と突き合わせてから `document_embeddings` を検索します。
- SQL 上は `embedding::halfvec(dims)` とクエリベクトル `::halfvec(dims)` にキャストし、HNSW で距離ソートします。
- `--model` を `qwen3-embedding:4b` に切り替えると 2560 次元の halfvec サーチに自動で変わります。

### 3. ハイブリッド検索 (PGroonga + pgvector)
```bash
python scripts/search_hybrid.py 猫 --model nomic-embed-text:v1.5 --text-k 40 --vec-k 40 --rrf-k 60
```
- テキスト側は PGroonga、ベクトル側は `document_embeddings` を halfvec で検索し、`Reciprocal Rank Fusion` で統合します。
- `--literal` を付けるとテキスト検索部分のみリテラル扱いにできます。

### 4. ドキュメント追加 + 即時埋め込み
```bash
python scripts/add_document.py --title "猫カフェ" \
  --body "猫カフェは癒やしスポット" \
  --source blog \
  --model embeddinggemma:300m
```
- `documents` に 1 行 INSERT した後、指定モデルで生成したベクトルを `document_embeddings` に UPSERT します。
- `embedding_models` に登録されていないモデルを使うとエラーになります。新モデルは後述の手順で追加してください。
- `--embed-field` を `body` にすると本文のみで埋め込みを作成します。

### 5. 既存ドキュメントの一括埋め込み/更新
```bash
python scripts/embed_documents.py --model nomic-embed-text:v1.5 --batch-size 8
```
- `document_embeddings` に同じモデルの行が無い、もしくは `--force` 指定時には全件再計算します。
- バッチごとに Ollama へ投げ、得られた次元が `embedding_models.dims` と一致しない場合は即座に失敗させます。

## モデルを追加したい場合
1. psql で `embedding_models` に (name, dims) を INSERT します。例: `INSERT INTO embedding_models VALUES ('my-model', 1536);`
2. `sql/20_create_vector_index.sql` を参考に、該当モデル専用の halfvec HNSW インデックスを作成します。
3. Ollama で `ollama pull my-model` を実行し、`add_document.py` や `embed_documents.py` の `--model my-model` で試します。
   - dims が 4000 を超えるモデルは halfvec の制約上サポート外です。

## SQL ユーティリティ
- `sql/10_search_text.sql` : `
\set q '猫'
\i sql/10_search_text.sql
` のように psql から読み込むと、PGroonga 検索結果を即座に確認できます。
- `sql/20_create_vector_index.sql` : 埋め込みモデルごとの HNSW インデックス作成と `ANALYZE` をまとめたスクリプトです。モデル名や次元が異なる場合は修正して実行してください。

## 再初期化と検証のすすめ
- スキーマを更新した際は `make clean && make start` で `db/data` を空にし、`db/init` を再適用すると確実です。
- HNSW インデックス作成後は `ANALYZE document_embeddings;` を忘れずに。`sql/20_create_vector_index.sql` を実行するとまとめて処理できます。
- モデル間の比較で recall を確認したい場合、`ORDER BY embedding::vector <-> :qvec` の exact search を同じクエリで走らせ、近似結果との差分をチェックするのが定石です。

## E2Eテスト
- `tests/e2e` にモック Ollama サーバーと `python -m unittest tests.e2e.test_workflow` で実行できる E2E テストを用意しています。
- テストは `make clean && make start` + モックサーバー立ち上げ→ `add_document.py` でのINSERT→ `embed_documents.py` の再埋め込み→ `search_vector.py` / `search_hybrid.py` の実行を通しで検証します。
- 実データを消すため、実行前に `db/data` をバックアップしてください。

## トラブルシューティングのヒント
- Ollama が起動していない／モデル未 pull の場合、埋め込み API 呼び出しで `ConnectionError` になります。`ollama pull nomic-embed-text:v1.5` を忘れずに。
- `Model 'xxx' is not registered in embedding_models` と出たら、`embedding_models` に該当モデルを INSERT し、必要なら HNSW インデックスも作成してください。
- 既存の `db/data` が残っていると `db/init` の SQL は再適用されません。再構築したい場合は `make clean` でデータディレクトリを空にしてください。
- `pgroonga` の tokenizer を変更したい場合は `db/init/010_schema.sql` の `WITH (tokenizer='TokenUnigram')` を編集し、DB を再初期化します。

以上で、ローカル環境でハイブリッド検索の仕組みを試すための準備が整います。検証やチューニングのたびに `scripts/` の CLI を組み合わせて挙動を比べてみてください。
