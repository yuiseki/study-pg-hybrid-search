#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_data(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(ranking: List[int], relevance: List[int], ks: List[int]):
    rel_set = set(relevance)
    metrics = {}
    if not rel_set:
        for k in ks:
            metrics[f"recall@{k}"] = 0.0
        metrics["MRR"] = 0.0
        return metrics

    ranks = [idx + 1 for idx, doc_id in enumerate(ranking) if doc_id in rel_set]
    metrics["MRR"] = 1.0 / ranks[0] if ranks else 0.0
    for k in ks:
        top = ranking[:k]
        hits = sum(1 for doc_id in top if doc_id in rel_set)
        metrics[f"recall@{k}"] = hits / len(rel_set)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute recall@k and MRR from evaluations/data.json")
    parser.add_argument("--data", default="evaluations/data.json", help="Path to data JSON")
    parser.add_argument("--k", nargs="*", type=int, default=[3, 5, 10], help="k values for recall")
    args = parser.parse_args()

    data = load_data(Path(args.data))

    for slug, payload in data.items():
        print(f"\n== {slug} ({payload['query']}) ==")
        print("model\trecall@" + "\trecall@".join(map(str, args.k)) + "\tMRR")
        for model, ranking in payload["rankings"].items():
            relevance = payload["relevance"].get(model, [])
            metrics = compute_metrics(ranking, relevance, args.k)
            rec_values = [f"{metrics[f'recall@{k}']:.2f}" for k in args.k]
            print("\t".join([model, *rec_values, f"{metrics['MRR']:.2f}"]))


if __name__ == "__main__":
    main()
