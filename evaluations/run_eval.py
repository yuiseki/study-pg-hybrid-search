#!/usr/bin/env python3
"""Evaluate text/vector/hybrid search rankings and store them as JSON Lines."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
import requests


DEFAULT_MODELS = [
    "snowflake-arctic-embed:22m",
    "nomic-embed-text:v1.5",
    "embeddinggemma:300m",
    "snowflake-arctic-embed2:568m",
    "qwen3-embedding:4b",
]


TEXT_SQL = """
SELECT
  id,
  title,
  pgroonga_score(tableoid, ctid) AS score
FROM documents
WHERE source = %(docset)s
  AND content &@~ %(q)s
ORDER BY score DESC
LIMIT %(limit)s;
"""


VECTOR_SQL_TEMPLATE = """
SELECT
  d.id,
  d.title,
  (e.embedding::halfvec({dims}) <=> %(qvec)s::halfvec({dims})) AS distance,
  (1 - (e.embedding::halfvec({dims}) <=> %(qvec)s::halfvec({dims}))) AS similarity
FROM document_embeddings e
JOIN documents d ON d.id = e.document_id
WHERE e.model = %(model)s
  AND d.source = %(docset)s
ORDER BY distance ASC
LIMIT %(limit)s;
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation searches and store rankings as JSONL")
    parser.add_argument("--data", default="evaluations/data.json", help="path to queries/relevance JSON")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/sandbox"),
        help="PostgreSQL DSN",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL",
    )
    parser.add_argument("--docset", default="seed2", help="documents.source value to target")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma separated embedding model list (default: all predefined)",
    )
    parser.add_argument(
        "--modes",
        default="text,vector,hybrid",
        help="comma separated modes to run (subset of text,vector,hybrid)",
    )
    parser.add_argument("--limit", type=int, default=20, help="number of rows to store per ranking")
    parser.add_argument("--text-limit", type=int, default=50, help="candidate rows from PGroonga for text/hybrid")
    parser.add_argument("--vector-limit", type=int, default=50, help="candidate rows from pgvector for vector/hybrid")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant k")
    parser.add_argument(
        "--rrf-weights",
        nargs="*",
        default=["1:1"],
        help="weight pairs for hybrid mode (e.g. 1:1 2:1 1:2 or comma separated)",
    )
    parser.add_argument("--k", nargs="*", type=int, default=[3, 5, 10], help="k values for downstream metrics")
    parser.add_argument("--only", default="", help="comma separated query slugs to evaluate (others skipped)")
    parser.add_argument("--outdir", default="", help="output directory (default: evaluations/out/<timestamp>)")
    parser.add_argument("--literal-text", action="store_true", help="quote PGroonga query as literal string")
    parser.add_argument("--verbose", action="store_true", help="print verbose progress logs")
    return parser.parse_args()


def literalize_pg_query(query: str) -> str:
    escaped = query.replace('"', '\"')
    return f'"{escaped}"'


def ensure_modes(modes_arg: str) -> List[str]:
    modes = [m.strip().lower() for m in modes_arg.split(",") if m.strip()]
    valid = {"text", "vector", "hybrid"}
    for mode in modes:
        if mode not in valid:
            raise SystemExit(f"Unsupported mode '{mode}'. Choose from text, vector, hybrid.")
    if not modes:
        raise SystemExit("At least one mode must be specified.")
    return modes


def parse_models(models_arg: str) -> List[str]:
    models = [m.strip() for m in models_arg.split(",") if m.strip()]
    if not models:
        raise SystemExit("At least one embedding model must be specified.")
    return models


def parse_rrf_weights(weight_args: Sequence[str]) -> List[Tuple[float, float]]:
    tokens: List[str] = []
    for val in weight_args:
        tokens.extend([chunk.strip() for chunk in val.split(",") if chunk.strip()])
    weights: List[Tuple[float, float]] = []
    for token in tokens:
        if ":" not in token:
            raise SystemExit(f"Invalid weight format '{token}'. Use like 1:1 or 2.0:1.0")
        left, right = token.split(":", 1)
        try:
            weights.append((float(left), float(right)))
        except ValueError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Weights must be numeric values: '{token}'") from exc
    if not weights:
        weights = [(1.0, 1.0)]
    return weights


def load_queries(path: Path) -> Tuple[str, List[Dict[str, object]]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    version = str(data.get("version", "unknown"))
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        raise SystemExit("'queries' must be a list in evaluations/data.json")
    return version, queries


def select_queries(
    queries: Sequence[Dict[str, object]],
    slug_filter: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    if not slug_filter:
        return list(queries)
    allowed = {slug for slug in slug_filter if slug}
    selected = [q for q in queries if q.get("slug") in allowed]
    missing = allowed - {q.get("slug") for q in selected}
    if missing:
        raise SystemExit(f"Unknown query slug(s): {', '.join(sorted(missing))}")
    return selected


def to_pgvector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def fetch_available_models(conn: psycopg.Connection) -> Dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT name, dims FROM embedding_models ORDER BY name")
        rows = cur.fetchall()
    return {str(name): int(dims) for name, dims in rows}


def get_model_dims(conn: psycopg.Connection, model: str, cache: Dict[str, int]) -> int:
    if model not in cache:
        raise SystemExit(f"Model '{model}' is not registered in embedding_models.")
    return cache[model]


def embed_query(
    ollama_url: str,
    model: str,
    text: str,
    dims: int,
    cache: Dict[Tuple[str, str], List[float]],
) -> List[float]:
    key = (model, text)
    if key in cache:
        return cache[key]

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": text},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    vector = payload["embeddings"][0]
    if len(vector) != dims:
        raise SystemExit(f"Model {model} expected {dims} dims but got {len(vector)}")
    cache[key] = vector
    return vector


def run_text_search(
    conn: psycopg.Connection,
    query: str,
    docset: str,
    limit: int,
) -> List[Dict[str, object]]:
    with conn.cursor() as cur:
        cur.execute(TEXT_SQL, {"q": query, "docset": docset, "limit": limit})
        rows = cur.fetchall()
    return [
        {
            "rank": idx,
            "document_id": int(doc_id),
            "title": title,
            "score": float(score),
        }
        for idx, (doc_id, title, score) in enumerate(rows, start=1)
    ]


def run_vector_search(
    conn: psycopg.Connection,
    query_vec_literal: str,
    model: str,
    docset: str,
    dims: int,
    limit: int,
) -> List[Dict[str, object]]:
    sql = VECTOR_SQL_TEMPLATE.format(dims=dims)
    params = {"qvec": query_vec_literal, "model": model, "docset": docset, "limit": limit}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "rank": idx,
            "document_id": int(doc_id),
            "title": title,
            "distance": float(distance),
            "similarity": float(similarity),
        }
        for idx, (doc_id, title, distance, similarity) in enumerate(rows, start=1)
    ]


def fuse_rrf(
    text_results: List[Dict[str, object]],
    vector_results: List[Dict[str, object]],
    weight_text: float,
    weight_vector: float,
    rrf_k: int,
    limit: int,
) -> List[Dict[str, object]]:
    text_lookup = {
        item["document_id"]: {
            "rank": item["rank"],
            "score": item.get("score"),
            "title": item.get("title"),
        }
        for item in text_results
    }
    vector_lookup = {
        item["document_id"]: {
            "rank": item["rank"],
            "distance": item.get("distance"),
            "similarity": item.get("similarity"),
            "title": item.get("title"),
        }
        for item in vector_results
    }

    doc_ids = set(text_lookup) | set(vector_lookup)
    fused: List[Tuple[int, float, Dict[str, object]]] = []
    for doc_id in doc_ids:
        text_part = text_lookup.get(doc_id)
        vec_part = vector_lookup.get(doc_id)
        score = 0.0
        if text_part:
            score += weight_text / (rrf_k + text_part["rank"])
        if vec_part:
            score += weight_vector / (rrf_k + vec_part["rank"])
        fused.append(
            (
                doc_id,
                score,
                {
                    "title": (text_part or vec_part or {}).get("title"),
                    "text_rank": text_part.get("rank") if text_part else None,
                    "text_score": text_part.get("score") if text_part else None,
                    "vector_rank": vec_part.get("rank") if vec_part else None,
                    "vector_distance": vec_part.get("distance") if vec_part else None,
                    "vector_similarity": vec_part.get("similarity") if vec_part else None,
                },
            )
        )

    fused.sort(key=lambda item: item[1], reverse=True)

    results: List[Dict[str, object]] = []
    for idx, (doc_id, score, extra) in enumerate(fused[:limit], start=1):
        result = {
            "rank": idx,
            "document_id": doc_id,
            "title": extra.get("title"),
            "rrf_score": round(score, 8),
            "text_rank": extra.get("text_rank"),
            "text_score": extra.get("text_score"),
            "vector_rank": extra.get("vector_rank"),
            "vector_distance": extra.get("vector_distance"),
            "vector_similarity": extra.get("vector_similarity"),
        }
        results.append(result)
    return results


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_run_config(path: Path, config: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    modes = ensure_modes(args.modes)
    models = parse_models(args.models)
    weights = parse_rrf_weights(args.rrf_weights)
    slug_filter = [slug.strip() for slug in args.only.split(",") if slug.strip()]

    outdir = Path(args.outdir) if args.outdir else Path("evaluations/out") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    outdir.mkdir(parents=True, exist_ok=True)
    rankings_path = outdir / "rankings.jsonl"
    config_path = outdir / "run_config.json"

    data_version, queries = load_queries(Path(args.data))
    selected_queries = select_queries(queries, slug_filter)
    if not selected_queries:
        raise SystemExit("No queries selected for evaluation.")

    literal_flag = args.literal_text

    with psycopg.connect(args.dsn) as conn:
        model_dims = fetch_available_models(conn)
        missing_models = [m for m in models if m not in model_dims]
        if missing_models:
            raise SystemExit(
                "Missing dims for models: " + ", ".join(missing_models) + ". Register them in embedding_models first."
            )

        embedding_cache: Dict[Tuple[str, str], List[float]] = {}
        jsonl_records: List[Dict[str, object]] = []

        for query_payload in selected_queries:
            slug = str(query_payload.get("slug"))
            query_text = str(query_payload.get("query"))
            pg_query_text = literalize_pg_query(query_text) if literal_flag else query_text

            if args.verbose:
                print(f"[query] {slug}: '{query_text}'")

            text_results: List[Dict[str, object]] = []
            if "text" in modes or "hybrid" in modes:
                text_results = run_text_search(conn, pg_query_text, args.docset, args.text_limit)
                if args.verbose:
                    print(f"  text hits={len(text_results)}")
                if "text" in modes:
                    jsonl_records.append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "query_slug": slug,
                            "query": query_text,
                            "docset": args.docset,
                            "mode": "text",
                            "model": None,
                            "rrf": None,
                            "params": {
                                "limit": args.limit,
                                "text_limit": args.text_limit,
                                "vector_limit": args.vector_limit,
                                "rrf_k": args.rrf_k,
                            },
                            "text_query": pg_query_text,
                            "results": text_results[: args.limit],
                        }
                    )

            if "vector" in modes or "hybrid" in modes:
                for model in models:
                    dims = get_model_dims(conn, model, model_dims)
                    vector = embed_query(args.ollama_url, model, query_text, dims, embedding_cache)
                    qvec_literal = to_pgvector_literal(vector)
                    vector_results = run_vector_search(conn, qvec_literal, model, args.docset, dims, args.vector_limit)
                    if args.verbose:
                        print(f"  vector[{model}] hits={len(vector_results)}")

                    if "vector" in modes:
                        jsonl_records.append(
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "query_slug": slug,
                                "query": query_text,
                                "docset": args.docset,
                                "mode": "vector",
                                "model": model,
                                "rrf": None,
                                "params": {
                                    "limit": args.limit,
                                    "text_limit": args.text_limit,
                                    "vector_limit": args.vector_limit,
                                    "rrf_k": args.rrf_k,
                                    "dims": dims,
                                },
                                "results": vector_results[: args.limit],
                            }
                        )

                    if "hybrid" in modes:
                        if not text_results:
                            # ensure PGroonga results exist for hybrid fusion candidates
                            text_results = run_text_search(conn, pg_query_text, args.docset, args.text_limit)
                        for weight_text, weight_vector in weights:
                            fused_results = fuse_rrf(
                                text_results,
                                vector_results,
                                weight_text,
                                weight_vector,
                                args.rrf_k,
                                args.limit,
                            )
                            jsonl_records.append(
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "query_slug": slug,
                                    "query": query_text,
                                    "docset": args.docset,
                                    "mode": "hybrid",
                                    "model": model,
                                    "rrf": {
                                        "k": args.rrf_k,
                                        "weights": {"text": weight_text, "vector": weight_vector},
                                    },
                                    "params": {
                                        "limit": args.limit,
                                        "text_limit": args.text_limit,
                                        "vector_limit": args.vector_limit,
                                        "dims": dims,
                                    },
                                    "results": fused_results,
                                }
                            )

        write_jsonl(rankings_path, jsonl_records)

    run_config = {
        "generated_at": datetime.utcnow().isoformat(),
        "data_version": data_version,
        "docset": args.docset,
        "modes": modes,
        "models": models,
        "rrf_weights": weights,
        "rrf_k": args.rrf_k,
        "limit": args.limit,
        "text_limit": args.text_limit,
        "vector_limit": args.vector_limit,
        "k_values": args.k,
        "queries": [payload.get("slug") for payload in selected_queries],
        "output": {
            "dir": str(outdir),
            "rankings": str(rankings_path),
        },
    }
    write_run_config(config_path, run_config)

    print(f"Saved rankings to {rankings_path}")
    print(f"Saved run config to {config_path}")


if __name__ == "__main__":
    main()
