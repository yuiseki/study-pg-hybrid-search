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

    # どの行を埋めるか：
    # - forceなし: embeddingがNULL or embedding_modelが違う
    # - forceあり : 全件
    where = "TRUE" if args.force else "(embedding IS NULL OR embedding_model IS DISTINCT FROM %(model)s)"
    limit_sql = "" if args.limit <= 0 else "LIMIT %(limit)s"

    select_sql = f"""
        SELECT id, content
        FROM documents
        WHERE {where}
        ORDER BY id
        {limit_sql}
    """

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            params = {"model": args.model, "limit": args.limit}
            cur.execute(select_sql, params)
            rows: List[Tuple[int, str]] = cur.fetchall()

    if not rows:
        print("No documents to embed. (Already embedded?)")
        return

    print(f"Embedding {len(rows)} documents with model={args.model}")

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
                if any(len(e) != dim for e in embs):
                    raise RuntimeError("Embedding dimensions are not consistent within a batch")

                for doc_id, emb in zip(ids, embs):
                    cur.execute(
                        """
                        UPDATE documents
                        SET embedding_model = %s,
                            embedding = %s
                        WHERE id = %s
                        """,
                        (args.model, to_pgvector_literal(emb), doc_id),
                    )

                conn.commit()
                print(f"  updated {min(i + args.batch_size, len(rows))}/{len(rows)} (dim={dim})")

    print("Done.")


if __name__ == "__main__":
    main()
