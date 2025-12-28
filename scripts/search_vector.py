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


SQL_TEMPLATE = """
SELECT
  d.id,
  d.title,
  left(d.content, 80) AS snippet,
  (e.embedding::halfvec({dims}) <=> %(qvec)s::halfvec({dims})) AS cosine_distance,
  (1 - (e.embedding::halfvec({dims}) <=> %(qvec)s::halfvec({dims}))) AS cosine_similarity
FROM document_embeddings e
JOIN documents d ON d.id = e.document_id
WHERE e.model = %(model)s
ORDER BY e.embedding::halfvec({dims}) <=> %(qvec)s::halfvec({dims})
LIMIT %(limit)s;
"""


def fetch_model_dims(conn: psycopg.Connection, model: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT dims FROM embedding_models WHERE name = %s", (model,))
        row = cur.fetchone()

    if row is None:
        raise SystemExit(
            f"Model '{model}' is not registered in embedding_models. "
            "Insert it with its dimension before running searches."
        )

    return int(row[0])


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

    with psycopg.connect(args.dsn) as conn:
        dims = fetch_model_dims(conn, args.model)

        qvec = ollama_embed_one(args.ollama_url, args.model, args.query)
        if len(qvec) != dims:
            raise SystemExit(f"Expected {dims}-dim embedding, got {len(qvec)}")

        qvec_lit = to_pgvector_literal(qvec)

        sql = SQL_TEMPLATE.format(dims=dims)

        with conn.cursor() as cur:
            cur.execute(sql, {"qvec": qvec_lit, "model": args.model, "limit": args.limit})
            rows: List[Tuple[Any, ...]] = cur.fetchall()

    print("id | title | cos_dist | cos_sim | snippet")
    print("-" * 120)
    for (id_, title, snippet, dist, sim) in rows:
        print(f"{id_} | {title} | {dist:.6f} | {sim:.6f} | {snippet}")


if __name__ == "__main__":
    main()
