#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import psycopg
import requests


def to_pgvector_literal(vec: List[float]) -> str:
    # pgvector は文字列リテラル "[1,2,3]" 形式で渡せる
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def ollama_embed_batch(ollama_url: str, model: str, texts: List[str]) -> List[List[float]]:
    # Ollama: POST /api/embed
    # input: string または string[]
    r = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    return data["embeddings"]


UPSERT_SQL = """
INSERT INTO document_embeddings (document_id, model, dims, embedding)
VALUES (%(document_id)s, %(model)s, %(dims)s, %(embedding)s)
ON CONFLICT (document_id, model)
DO UPDATE SET
  embedding = EXCLUDED.embedding,
  dims = EXCLUDED.dims,
  created_at = now();
"""


def fetch_model_dims(conn: psycopg.Connection, model: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT dims FROM embedding_models WHERE name = %s", (model,))
        row = cur.fetchone()

    if row is None:
        raise SystemExit(
            f"Model '{model}' is not registered in embedding_models. "
            "Insert it with its dimension before embedding documents."
        )

    return int(row[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed documents.content with Ollama and store in pgvector.")
    ap.add_argument("--model", default="nomic-embed-text:v1.5", help="Ollama embedding model name")
    ap.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434)",
    )
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/sandbox"),
        help="PostgreSQL DSN (or set DATABASE_URL)",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if embedding already exists for the same embedding_model",
    )
    args = ap.parse_args()

    limit_sql = "" if args.limit <= 0 else "LIMIT %(limit)s"

    select_sql = f"""
        SELECT d.id, d.content
        FROM documents d
        LEFT JOIN document_embeddings e
          ON e.document_id = d.id AND e.model = %(model)s
        WHERE %(force)s OR e.document_id IS NULL OR e.dims IS DISTINCT FROM %(dims)s
        ORDER BY d.id
        {limit_sql}
    """

    with psycopg.connect(args.dsn) as conn:
        dims = fetch_model_dims(conn, args.model)

        with conn.cursor() as cur:
            params = {"model": args.model, "limit": args.limit, "force": args.force, "dims": dims}
            cur.execute(select_sql, params)
            rows: List[Tuple[int, str]] = cur.fetchall()

    if not rows:
        print("No documents to embed. (Already embedded?)")
        return

    print(f"Embedding {len(rows)} documents with model={args.model} dims={dims}")

    # バッチでOllamaに投げて、順次UPDATE
    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i : i + args.batch_size]
                ids = [doc_id for doc_id, _ in chunk]
                texts = [content for _, content in chunk]

                embs = ollama_embed_batch(args.ollama_url, args.model, texts)
                if len(embs) != len(ids):
                    raise RuntimeError(f"Ollama returned {len(embs)} embeddings for {len(ids)} inputs")

                # 次元チェック（全て同じであることを確認）
                dim = len(embs[0])
                if dim != dims:
                    raise RuntimeError(
                        f"Model {args.model} expected {dims} dims, but got {dim} dims from Ollama"
                    )
                if any(len(e) != dim for e in embs):
                    raise RuntimeError("Embedding dimensions are not consistent within a batch")

                for doc_id, emb in zip(ids, embs):
                    cur.execute(
                        UPSERT_SQL,
                        {
                            "document_id": doc_id,
                            "model": args.model,
                            "dims": dims,
                            "embedding": to_pgvector_literal(emb),
                        },
                    )

                conn.commit()
                print(f"  updated {min(i + args.batch_size, len(rows))}/{len(rows)} (dim={dim})")

    print("Done.")


if __name__ == "__main__":
    main()
