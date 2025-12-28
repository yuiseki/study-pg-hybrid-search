# study-pg-hybrid-search

PostgreSQL 17 上で PGroonga と pgvector を組み合わせ、全文検索とベクトル検索をハイブリッドに統合するサンドボックスです。Ollama の埋め込み API を前提にしており、手元で簡単に再現・検証できます。

## 概要
- ベースイメージは `pgvector/pgvector:0.8.1-pg17-trixie`。Dockerfile で PGroonga を追加インストールしています。
- `docker-compose` 一発で Postgres を起動し、`db/init` 配下の SQL で拡張・スキーマ・サンプルデータを自動投入します。
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

## 事前準備
1. Docker / Docker Compose v2
2. `make` (任意、無くても `docker compose` で操作可能)
3. Ollama (ローカルで `ollama serve` し、`nomic-embed-text:v1.5` を pull 済みであること)
4. Python 3.11 以上（CLI を実行する場合）

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

### 2. ベクトル検索 (pgvector)
```bash
python scripts/search_vector.py "猫と本"
```
- Ollama にクエリを埋め込んで 768 次元ベクトルを生成し、`embedding_model` が一致する行をコサイン距離で並べ替えて表示します。

### 3. ハイブリッド検索 (RRF)
```bash
python scripts/search_hybrid.py 猫 --text-k 40 --vec-k 40 --rrf-k 60
```
- PGroonga と pgvector の上位候補を `Reciprocal Rank Fusion` でスコア統合し、双方のランク・スコア・類似度を確認できます。

### 4. ドキュメント追加
```bash
python scripts/add_document.py --title "猫カフェ" --body "猫カフェは癒やしスポット" --source blog
```
- `--embed-field` を `body` にすると本文のみで埋め込みを作成します。

### 5. 既存ドキュメントの一括埋め込み
```bash
python scripts/embed_documents.py --model nomic-embed-text:v1.5 --batch-size 8
```
- 既に同じモデルで埋め込まれている行はスキップされます。強制再計算したい場合は `--force` を付けます。

## SQL ユーティリティ
- `sql/10_search_text.sql` : `
\set q '猫'
\i sql/10_search_text.sql
` のように psql から読み込むと、PGroonga 検索結果を即座に確認できます。
- `sql/20_create_vector_index.sql` : 埋め込みモデルごとの HNSW インデックス作成と `ANALYZE` をまとめたスクリプトです。モデル名や次元が異なる場合は修正して実行してください。

## トラブルシューティングのヒント
- Ollama が起動していない／モデル未 pull の場合、埋め込み API 呼び出しで `ConnectionError` になります。`ollama pull nomic-embed-text:v1.5` を忘れずに。
- 既存の `db/data` が残っていると `db/init` の SQL は再適用されません。再構築したい場合は `make clean` でデータディレクトリを空にしてください。
- `pgroonga` の tokenizer を変更したい場合は `db/init/010_schema.sql` の `WITH (tokenizer='TokenUnigram')` を編集し、DB を再初期化します。

以上で、ローカル環境でハイブリッド検索の仕組みを試すための準備が整います。検証やチューニングのたびに `scripts/` の CLI を組み合わせて挙動を比べてみてください。
