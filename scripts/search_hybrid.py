#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List, Tuple

import psycopg
import requests


def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def as_literal_query(q: str) -> str:
    # PGroongaのクエリ構文を意識せず「文字列として」検索したい用（"で囲う）
    q = q.replace('"', '\\"')
    return f'"{q}"'


def ollama_embed_one(ollama_url: str, model: str, text: str) -> List[float]:
    r = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]


HYBRID_SQL = """
WITH
text AS (
  SELECT
    id,
    row_number() OVER (ORDER BY pgroonga_score(tableoid, ctid) DESC) AS r_text,
    pgroonga_score(tableoid, ctid) AS s_text
  FROM documents
  WHERE content &@~ %(q)s
  ORDER BY pgroonga_score(tableoid, ctid) DESC
  LIMIT %(text_k)s
),
vec AS (
  SELECT
    id,
    row_number() OVER (ORDER BY embedding::vector(768) <=> %(qvec)s::vector(768)) AS r_vec,
    (embedding::vector(768) <=> %(qvec)s::vector(768)) AS d_vec
  FROM documents
  WHERE embedding_model = %(model)s AND embedding IS NOT NULL
  ORDER BY embedding::vector(768) <=> %(qvec)s::vector(768)
  LIMIT %(vec_k)s
),
fused AS (
  SELECT
    COALESCE(text.id, vec.id) AS id,
    COALESCE(1.0 / (%(rrf_k)s + text.r_text), 0.0) +
    COALESCE(1.0 / (%(rrf_k)s + vec.r_vec),  0.0) AS rrf,
    text.r_text, vec.r_vec, text.s_text, vec.d_vec
  FROM text
  FULL OUTER JOIN vec USING (id)
)
SELECT
  d.id,
  d.title,
  left(d.content, 80) AS snippet,
  round(f.rrf::numeric, 6) AS rrf,
  f.r_text, f.r_vec,
  f.s_text,
  f.d_vec,
  CASE WHEN f.d_vec IS NULL THEN NULL ELSE (1 - f.d_vec) END AS cos_sim
FROM fused f
JOIN documents d USING (id)
ORDER BY f.rrf DESC
LIMIT %(limit)s;
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid search: PGroonga text + pgvector, fused by RRF")
    ap.add_argument("query", help="query text")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--text-k", type=int, default=50, help="candidates from PGroonga")
    ap.add_argument("--vec-k", type=int, default=50, help="candidates from pgvector")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant k")
    ap.add_argument("--model", default="nomic-embed-text:v1.5")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/sandbox"),
    )
    ap.add_argument("--literal", action="store_true", help="treat query as literal for PGroonga (&@~)")
    args = ap.parse_args()

    q_text = as_literal_query(args.query) if args.literal else args.query

    qvec = ollama_embed_one(args.ollama_url, args.model, args.query)
    if len(qvec) != 768:
        raise SystemExit(f"Expected 768-dim embedding, got {len(qvec)}")
    qvec_lit = to_pgvector_literal(qvec)

    params: Dict[str, Any] = {
        "q": q_text,
        "qvec": qvec_lit,
        "model": args.model,
        "limit": args.limit,
        "text_k": args.text_k,
        "vec_k": args.vec_k,
        "rrf_k": args.rrf_k,
    }

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(HYBRID_SQL, params)
            rows: List[Tuple[Any, ...]] = cur.fetchall()

    print("id | title | rrf | r_text | r_vec | s_text | d_vec | cos_sim | snippet")
    print("-" * 140)
    for (id_, title, snippet, rrf, r_text, r_vec, s_text, d_vec, cos_sim) in rows:
        print(
            f"{id_} | {title} | {rrf} | {r_text} | {r_vec} | {s_text} | {d_vec} | {cos_sim} | {snippet}"
        )


if __name__ == "__main__":
    main()
