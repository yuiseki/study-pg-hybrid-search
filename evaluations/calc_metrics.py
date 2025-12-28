#!/usr/bin/env python3
"""Compute recall@k, MRR, and nDCG@k from evaluation rankings JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class QueryInfo:
    slug: str
    query: str
    relevant_ids: Dict[str, List[int]]


@dataclass
class Ranking:
    slug: str
    docset: str
    mode: str
    model: Optional[str]
    rrf_weights: Optional[Dict[str, float]]
    results: List[int]
    metadata: Dict[str, object] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate recall/MRR/nDCG from rankings JSONL")
    parser.add_argument("--data", default="evaluations/data.json", help="query + relevance JSON path")
    parser.add_argument("--rankings", required=True, help="rankings JSONL path from run_eval")
    parser.add_argument("--k", nargs="*", type=int, default=[3, 5, 10], help="k values for recall/nDCG")
    parser.add_argument("--per-query", action="store_true", help="emit per-query metrics table")
    parser.add_argument("--outdir", default="", help="directory to save metrics.json/.md (defaults next to rankings)")
    return parser.parse_args()


def load_queries(path: Path) -> Dict[str, QueryInfo]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    version = payload.get("version")
    queries: Dict[str, QueryInfo] = {}
    for entry in payload.get("queries", []):
        slug = entry["slug"]
        rel_ids = entry.get("relevant_doc_ids", {})
        if not isinstance(rel_ids, dict):
            raise SystemExit("relevant_doc_ids must be an object keyed by docset")
        queries[slug] = QueryInfo(slug=slug, query=entry.get("query", ""), relevant_ids=rel_ids)
    if not queries:
        raise SystemExit("queries list is empty in data.json")
    return queries


def load_rankings(rankings_path: Path) -> List[Ranking]:
    rankings: List[Ranking] = []
    with rankings_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            slug = payload["query_slug"]
            docset = payload.get("docset", "seed2")
            mode = payload["mode"]
            model = payload.get("model")
            rrf = None
            if payload.get("rrf"):
                weights = payload["rrf"].get("weights", {})
                rrf = {"text": float(weights.get("text", 1.0)), "vector": float(weights.get("vector", 1.0))}
            doc_ids = [int(row["document_id"]) for row in payload.get("results", [])]
            rankings.append(
                Ranking(
                    slug=slug,
                    docset=docset,
                    mode=mode,
                    model=model,
                    rrf_weights=rrf,
                    results=doc_ids,
                    metadata={
                        "timestamp": payload.get("timestamp"),
                        "params": payload.get("params", {}),
                    },
                )
            )
    if not rankings:
        raise SystemExit("rankings JSONL is empty")
    return rankings


def recall_at_k(results: Sequence[int], relevant: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in results[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr(results: Sequence[int], relevant: Sequence[int]) -> float:
    rel_set = set(relevant)
    for idx, doc_id in enumerate(results, start=1):
        if doc_id in rel_set:
            return 1.0 / idx
    return 0.0


def dcg_at_k(results: Sequence[int], relevant: Sequence[int], k: int) -> float:
    rel_set = set(relevant)
    score = 0.0
    for idx, doc_id in enumerate(results[:k], start=1):
        if doc_id in rel_set:
            denom = math.log2(idx + 1)
            score += 1.0 / denom
    return score


def ndcg_at_k(results: Sequence[int], relevant: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    ideal_order = relevant[:k]
    ideal = dcg_at_k(ideal_order, ideal_order, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(results, relevant, k) / ideal


def summarize_rankings(queries: Dict[str, QueryInfo], rankings: List[Ranking], k_values: Sequence[int]):
    summary: Dict[str, Dict[str, float]] = {}
    per_query: Dict[str, Dict[str, float]] = {}

    grouped: Dict[str, List[Ranking]] = defaultdict(list)
    for ranking in rankings:
        key = ranking_key(ranking)
        grouped[key].append(ranking)

    for key, runs in grouped.items():
        values: Dict[str, List[float]] = defaultdict(list)
        query_metrics: Dict[str, Dict[str, float]] = {}
        for ranking in runs:
            info = queries.get(ranking.slug)
            if not info:
                continue
            relevant = info.relevant_ids.get(ranking.docset, [])
            res = ranking.results
            query_result = {
                "MRR": mrr(res, relevant),
            }
            for k in k_values:
                query_result[f"recall@{k}"] = recall_at_k(res, relevant, k)
                query_result[f"nDCG@{k}"] = ndcg_at_k(res, relevant, k)
            per_query_key = f"{key}|{ranking.slug}"
            per_query[per_query_key] = query_result

            for metric, value in query_result.items():
                values[metric].append(value)

        summary[key] = {metric: sum(vals) / len(vals) for metric, vals in values.items()}

    return summary, per_query


def ranking_key(ranking: Ranking) -> str:
    parts = [ranking.docset, ranking.mode]
    if ranking.model:
        parts.append(ranking.model)
    if ranking.rrf_weights:
        parts.append(f"text={ranking.rrf_weights['text']}:vector={ranking.rrf_weights['vector']}")
    return "|".join(parts)


def format_markdown(summary: Dict[str, Dict[str, float]], k_values: Sequence[int]) -> str:
    headers = ["Condition", "MRR", *[f"recall@{k}" for k in k_values], *[f"nDCG@{k}" for k in k_values]]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for key, metrics in sorted(summary.items()):
        row = [key, f"{metrics.get('MRR', 0):.3f}"]
        for k in k_values:
            row.append(f"{metrics.get(f'recall@{k}', 0):.3f}")
        for k in k_values:
            row.append(f"{metrics.get(f'nDCG@{k}', 0):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rankings_path = Path(args.rankings)
    outdir = Path(args.outdir) if args.outdir else rankings_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(Path(args.data))
    rankings = load_rankings(rankings_path)

    summary, per_query = summarize_rankings(queries, rankings, args.k)

    metrics_json_path = outdir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_query": per_query if args.per_query else {}}, f, indent=2, ensure_ascii=False)

    metrics_md_path = outdir / "metrics.md"
    metrics_md_path.write_text(format_markdown(summary, args.k), encoding="utf-8")

    print(f"Saved metrics to {metrics_json_path}")
    print(f"Saved table to {metrics_md_path}")


if __name__ == "__main__":
    main()
