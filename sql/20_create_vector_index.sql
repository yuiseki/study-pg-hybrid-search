-- nomic-embed-text:v1.5 (768d) 用のHNSW（コサイン距離）
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_nomic_v15_768_idx
  ON documents
  USING hnsw ((embedding::vector(768)) vector_cosine_ops)
  WHERE embedding_model = 'nomic-embed-text:v1.5' AND embedding IS NOT NULL;

ANALYZE documents;
