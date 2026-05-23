#!/usr/bin/env python3
"""
Wipe the index and the static-image directory clean.

Use this before re-ingesting with new tiler config. The index mapping has
changed (added the `scale` field), and existing tiles use the old
single-scale tile_id format. Cleanest path is to delete and rebuild.

What it does:
    1. DELETE /wheres-waldo-tiles  (full index drop)
    2. DELETE everything in backend/static/  (the per-image jpegs)

The backend will recreate the index with the current mapping on next startup.

Usage:
    python3 scripts/wipe_index.py
    python3 scripts/wipe_index.py --keep-files   # drop index, keep jpegs

Safety: requires --confirm to actually run.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from urllib import request, error


def env_or_die(key: str) -> str:
    val = os.environ.get(key)
    if not val:
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
    ap.add_argument("--static-dir", default="backend/static")
    ap.add_argument("--keep-files", action="store_true",
                    help="Keep the static jpeg files (only drop the index)")
    ap.add_argument("--confirm", action="store_true",
                    help="Required. Without this, prints what would happen.")
    args = ap.parse_args()

    es_url = env_or_die("ES_URL").rstrip("/")
    api_key = env_or_die("ES_API_KEY")

    print(f"Index:     {args.index}")
    print(f"ES URL:    {es_url}")
    print(f"Static:    {args.static_dir}  (will{'' if not args.keep_files else ' NOT'} be wiped)")
    print()

    if not args.confirm:
        print("DRY RUN. Re-run with --confirm to actually wipe.")
        return 0

    # 1. Drop the index
    url = f"{es_url}/{args.index}"
    print(f"DELETE {url}")
    req = request.Request(
        url, method="DELETE",
        headers={"Authorization": f"ApiKey {api_key}"},
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            print(f"  ✓ {body}")
    except error.HTTPError as e:
        if e.code == 404:
            print("  (index didn't exist — nothing to drop)")
        else:
            print(f"  HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
            return 1

    # 2. Wipe static directory
    if not args.keep_files:
        static_path = Path(args.static_dir).resolve()
        if static_path.exists():
            count = 0
            for p in static_path.iterdir():
                if p.is_file():
                    p.unlink()
                    count += 1
            print(f"  ✓ Removed {count} files from {static_path}")
        else:
            print(f"  (static dir {static_path} doesn't exist)")

    print()
    print("Done. Restart the backend to recreate the index, then run batch_ingest.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
