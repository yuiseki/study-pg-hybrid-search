CREATE TABLE documents (
  id              bigserial PRIMARY KEY,
  source          text,
  title           text,
  body            text NOT NULL,
  content         text GENERATED ALWAYS AS (coalesce(title, '') || E'\n' || body) STORED,
  embedding_model text,
  embedding       vector,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- 日本語の1文字（例: 「本」「猫」）でも引けるように TokenUnigram
CREATE INDEX documents_content_pgroonga_idx
  ON documents
  USING pgroonga (content)
  WITH (tokenizer='TokenUnigram');

-- 例: Ollamaで nomic-embed-text (768次元) を使う場合のANN(HNSW)インデックス
-- embeddingは vector（次元なし）で保持し、キャスト＋部分インデックスで固定次元を切る
CREATE INDEX documents_embedding_hnsw_nomic_768_idx
  ON documents
  USING hnsw ((embedding::vector(768)) vector_cosine_ops)
  WHERE embedding_model = 'nomic-embed-text' AND embedding IS NOT NULL;
