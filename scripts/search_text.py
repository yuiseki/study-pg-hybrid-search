#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List, Tuple

import psycopg


SQL = """
SELECT
  id,
  title,
  left(content, 80) AS snippet,
  pgroonga_score(tableoid, ctid) AS score
FROM documents
WHERE content &@~ %(q)s
ORDER BY score DESC
LIMIT %(limit)s;
"""


def as_literal_query(q: str) -> str:
    """
    PGroongaのクエリ構文を意識せず「文字列として」検索したい場合の簡易ラップ。
    - ダブルクォートで囲う
    - 内部の " を \" にエスケープ
    例: 猫 -> "猫"
    """
    q = q.replace('"', '\\"')
    return f'"{q}"'


def main() -> None:
    ap = argparse.ArgumentParser(description="PGroonga text search (documents.content)")
    ap.add_argument("query", help="Search query (PGroonga query syntax by default)")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/sandbox"),
        help="PostgreSQL DSN (or set DATABASE_URL). Default: localhost sandbox",
    )
    ap.add_argument(
        "--literal",
        action="store_true",
        help="Treat query as a literal string (quote it) to avoid PGroonga query syntax surprises",
    )
    args = ap.parse_args()

    q = as_literal_query(args.query) if args.literal else args.query

    params: Dict[str, Any] = {"q": q, "limit": args.limit}

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL, params)
            rows: List[Tuple[Any, ...]] = cur.fetchall()

    # simple pretty print
    print("id | title | score | snippet")
    print("-" * 120)
    for (id_, title, snippet, score) in rows:
        print(f"{id_} | {title} | {score} | {snippet}")


if __name__ == "__main__":
    main()
