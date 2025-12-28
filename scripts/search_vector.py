#!/usr/bin/env python3
import argparse
import os
from typing import List, Any, Tuple

import psycopg
import requests


def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def ollama_embed_one(ollama_url: str, model: str, text: str) -> List[float]:
    r = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data["embeddings"][0]


SQL = """
SELECT
  id,
  title,
  left(content, 80) AS snippet,
  (embedding::vector(768) <=> %(qvec)s::vector(768)) AS cosine_distance,
  (1 - (embedding::vector(768) <=> %(qvec)s::vector(768))) AS cosine_similarity
FROM documents
WHERE embedding_model = %(model)s AND embedding IS NOT NULL
ORDER BY embedding::vector(768) <=> %(qvec)s::vector(768)
LIMIT %(limit)s;
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Vector search (pgvector) with Ollama embeddings")
    ap.add_argument("query", help="query text")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--model", default="nomic-embed-text:v1.5")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/sandbox"),
    )
    args = ap.parse_args()

    qvec = ollama_embed_one(args.ollama_url, args.model, args.query)
    if len(qvec) != 768:
        raise SystemExit(f"Expected 768-dim embedding, got {len(qvec)}")

    qvec_lit = to_pgvector_literal(qvec)

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL, {"qvec": qvec_lit, "model": args.model, "limit": args.limit})
            rows: List[Tuple[Any, ...]] = cur.fetchall()

    print("id | title | cos_dist | cos_sim | snippet")
    print("-" * 120)
    for (id_, title, snippet, dist, sim) in rows:
        print(f"{id_} | {title} | {dist:.6f} | {sim:.6f} | {snippet}")


if __name__ == "__main__":
    main()
