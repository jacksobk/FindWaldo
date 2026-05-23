#!/usr/bin/env python3
"""
Re-normalize all embeddings in the index in place.

Why: An earlier version of the ingest pipeline indexed un-normalized
embeddings from Jina, but the index uses dot_product similarity which
requires unit-length vectors. Searches fail intermittently with:

    The [dot_product] similarity can only be used with unit-length vectors.

Fix: an _update_by_query with a painless script that L2-normalizes every
embedding field. Idempotent — running it twice produces the same result.

Usage:
    python3 scripts/normalize_embeddings.py
    python3 scripts/normalize_embeddings.py --dry-run
"""
import argparse
import json
import os
import sys
from urllib import request, error
from pathlib import Path


def env_or_die(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        # Try reading from backend/.env
        env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
        print(f"error: {key} not set in environment or backend/.env", file=sys.stderr)
        sys.exit(1)
    return val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="wheres-waldo-tiles")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just count documents that would be updated")
    args = ap.parse_args()

    es_url = env_or_die("ES_URL").rstrip("/")
    api_key = env_or_die("ES_API_KEY")

    # Painless script: L2-normalize ctx._source.embedding in place.
    # We compute the squared sum, take sqrt, and divide each element.
    # NO idempotency guard — always renormalize. Why: Elasticsearch's
    # dot_product check is stricter than any tolerance we'd pick. A vector
    # with norm 0.999999998 may pass our "already normalized" check but be
    # rejected by ES. Renormalizing is cheap (~3ms per doc); the savings
    # from the guard aren't worth the failure mode.
    script = """
        if (ctx._source.embedding == null) { return; }
        double sum = 0.0;
        for (double v : ctx._source.embedding) { sum += v * v; }
        double norm = Math.sqrt(sum);
        if (norm < 1e-12) { return; }
        for (int i = 0; i < ctx._source.embedding.length; i++) {
            ctx._source.embedding[i] = ctx._source.embedding[i] / norm;
        }
    """

    if args.dry_run:
        # Just count
        url = f"{es_url}/{args.index}/_count"
        req = request.Request(
            url,
            headers={"Authorization": f"ApiKey {api_key}"},
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                count = json.loads(resp.read())["count"]
                print(f"Index '{args.index}' has {count} documents.")
                print("Run without --dry-run to normalize them in place.")
                return 0
        except error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
            return 1

    # Real update
    url = f"{es_url}/{args.index}/_update_by_query?refresh=true&conflicts=proceed&wait_for_completion=true"
    body = json.dumps({
        "script": {"source": script.strip(), "lang": "painless"}
    }).encode("utf-8")

    print(f"Normalizing embeddings in '{args.index}'...")
    print(f"POST {url}")

    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
            print(json.dumps({
                "updated": result.get("updated"),
                "noops":   result.get("noops"),
                "total":   result.get("total"),
                "failures": result.get("failures"),
                "took_ms": result.get("took"),
            }, indent=2))
            if result.get("failures"):
                print("WARNING: some documents failed; see details above", file=sys.stderr)
                return 2
            return 0
    except error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
