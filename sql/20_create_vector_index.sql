-- document_embeddings テーブルに halfvec + HNSW インデックスを張るサンプル

CREATE INDEX IF NOT EXISTS document_embeddings_hnsw_arctic22m_384_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops)
  WHERE model = 'snowflake-arctic-embed:22m';

CREATE INDEX IF NOT EXISTS document_embeddings_hnsw_nomic_v15_768_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
  WHERE model = 'nomic-embed-text:v1.5';

CREATE INDEX IF NOT EXISTS document_embeddings_hnsw_embeddinggemma_768_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
  WHERE model = 'embeddinggemma:300m';

CREATE INDEX IF NOT EXISTS document_embeddings_hnsw_arctic2_1024_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
  WHERE model = 'snowflake-arctic-embed2:568m';

CREATE INDEX IF NOT EXISTS document_embeddings_hnsw_qwen4b_2560_half_idx
  ON document_embeddings
  USING hnsw ((embedding::halfvec(2560)) halfvec_cosine_ops)
  WHERE model = 'qwen3-embedding:4b';

ANALYZE document_embeddings;
