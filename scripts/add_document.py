#!/usr/bin/env python3
import argparse
import os
from typing import List, Any

import psycopg
import requests


MODEL_DEFAULT = "nomic-embed-text:v1.5"


def to_pgvector_literal(vec: List[float]) -> str:
    # pgvector には "[...]" 形式で渡せます
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def ollama_embed_one(ollama_url: str, model: str, text: str) -> List[float]:
    # Ollama: POST /api/embed で input に文字列を渡す
    r = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": text},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data["embeddings"][0]


INSERT_DOCUMENT_SQL = """
INSERT INTO documents (source, title, body)
VALUES (%(source)s, %(title)s, %(body)s)
RETURNING id;
"""

UPSERT_EMBEDDING_SQL = """
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
            "Insert it with its dimension before adding documents."
        )

    return int(row[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Add a document with embedding (Ollama + pgvector).")
    ap.add_argument("--title", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--source", default="cli")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/sandbox"),
        help="PostgreSQL DSN (or set DATABASE_URL). Default: localhost sandbox",
    )
    ap.add_argument(
        "--embed-field",
        choices=["content", "body"],
        default="content",
        help="What to embed. 'content' = title + '\\n' + body (recommended), 'body' = body only.",
    )
    args = ap.parse_args()

    # スキーマの generated column content と合わせるなら content を埋め込むのが自然です
    text_for_embedding = args.body if args.embed_field == "body" else f"{args.title}\n{args.body}"

    with psycopg.connect(args.dsn) as conn:
        dims = fetch_model_dims(conn, args.model)

        emb = ollama_embed_one(args.ollama_url, args.model, text_for_embedding)

        if len(emb) != dims:
            raise SystemExit(
                f"Expected {dims}-dim embedding for {args.model}, got {len(emb)}"
            )

        emb_lit = to_pgvector_literal(emb)

        with conn.cursor() as cur:
            cur.execute(
                INSERT_DOCUMENT_SQL,
                {
                    "source": args.source,
                    "title": args.title,
                    "body": args.body,
                },
            )
            new_id: Any = cur.fetchone()[0]

            cur.execute(
                UPSERT_EMBEDDING_SQL,
                {
                    "document_id": new_id,
                    "model": args.model,
                    "dims": dims,
                    "embedding": emb_lit,
                },
            )

    print(f"Inserted document id={new_id} with embedding model={args.model} dims={dims}")


if __name__ == "__main__":
    main()
