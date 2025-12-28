CREATE TABLE documents (
  id         bigserial PRIMARY KEY,
  source     text,
  title      text,
  body       text NOT NULL,
  content    text GENERATED ALWAYS AS (coalesce(title, '') || E'\n' || body) STORED,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX documents_content_pgroonga_idx
  ON documents
  USING pgroonga (content)
  WITH (tokenizer = 'TokenUnigram');

CREATE TABLE embedding_models (
  name text PRIMARY KEY,
  dims int  NOT NULL CHECK (dims > 0 AND dims <= 4000)
);

INSERT INTO embedding_models (name, dims) VALUES
  ('snowflake-arctic-embed:22m',  384),
  ('nomic-embed-text:v1.5',       768),
  ('embeddinggemma:300m',         768),
  ('snowflake-arctic-embed2:568m',1024),
  ('qwen3-embedding:4b',         2560)
ON CONFLICT (name) DO UPDATE SET dims = EXCLUDED.dims;

CREATE TABLE document_embeddings (
  document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  model       text   NOT NULL REFERENCES embedding_models(name),
  dims        int    NOT NULL,
  embedding   vector NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (document_id, model),
  CHECK (vector_dims(embedding) = dims)
);

CREATE INDEX document_embeddings_model_idx
  ON document_embeddings (model, document_id);

-- モデルごとに halfvec + HNSW の部分インデックスを貼る
CREATE INDEX document_embeddings_hnsw_arctic22m_384_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops)
  WHERE model = 'snowflake-arctic-embed:22m';

CREATE INDEX document_embeddings_hnsw_nomic_v15_768_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
  WHERE model = 'nomic-embed-text:v1.5';

CREATE INDEX document_embeddings_hnsw_embeddinggemma_768_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
  WHERE model = 'embeddinggemma:300m';

CREATE INDEX document_embeddings_hnsw_arctic2_1024_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
  WHERE model = 'snowflake-arctic-embed2:568m';

CREATE INDEX document_embeddings_hnsw_qwen4b_2560_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(2560)) halfvec_cosine_ops)
  WHERE model = 'qwen3-embedding:4b';
