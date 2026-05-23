#!/usr/bin/env python3
"""
Batch-ingest a folder of images into the running Find Waldo backend.

Usage:
    python3 scripts/batch_ingest.py /path/to/folder --backend http://localhost:8200

Reads every .jpg/.jpeg/.png/.webp/.gif in the folder (non-recursive),
POSTs each one to /api/ingest. Skips files whose label is already in the
index, so the script is safe to rerun after failures or partial runs.

Retries on transient connection drops (uvicorn keep-alive timeouts during
slow Jina-paced ingest) up to 3 times with exponential backoff.

The label sent to the backend is the filename (with extension stripped and
underscores/dashes turned into spaces).
"""
import argparse
import sys
import time
import json
import mimetypes
from pathlib import Path
from urllib import request, error

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
BOUNDARY = b"----findwaldo-batch-boundary-7f3a"


def label_from_filename(filepath: Path) -> str:
    return filepath.stem.replace("_", " ").replace("-", " ")


def encode_multipart(filepath: Path, label: str) -> tuple[bytes, str]:
    """Hand-built multipart/form-data with both file and label in one body."""
    mime, _ = mimetypes.guess_type(str(filepath))
    mime = mime or "application/octet-stream"

    parts: list[bytes] = []
    # file field
    parts.append(b"--" + BOUNDARY)
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filepath.name}"'.encode()
    )
    parts.append(f"Content-Type: {mime}".encode())
    parts.append(b"")
    parts.append(filepath.read_bytes())
    # label field
    parts.append(b"--" + BOUNDARY)
    parts.append(b'Content-Disposition: form-data; name="label"')
    parts.append(b"")
    parts.append(label.encode("utf-8"))
    # closing boundary
    parts.append(b"--" + BOUNDARY + b"--")
    parts.append(b"")
    body = b"\r\n".join(parts)
    return body, f"multipart/form-data; boundary={BOUNDARY.decode()}"


def list_existing(backend: str) -> list[dict]:
    """Return the full list of indexed images for skip-detection.
    Each entry has at least: image_id, label, width, height."""
    try:
        with request.urlopen(f"{backend.rstrip('/')}/api/images", timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (error.URLError, error.HTTPError, TimeoutError) as e:
        print(f"warning: could not list existing images ({e}); assuming none",
              file=sys.stderr)
        return []


def _normalize_for_match(s: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse spaces. Used to compare
    a candidate filename against existing labels with tolerance for
    different casing, separators, and noise tokens."""
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def should_skip(filepath: Path, existing: list[dict]) -> dict | None:
    """Decide if this file is already ingested.

    Match priority (most specific to most permissive):
      1. Exact label match against the filename-derived label
      2. Normalized substring match (handles 'beach' vs 'Wheres-Waldo-Beach-...')
      3. None — proceed with ingest

    Returns the matched existing record if a duplicate is found, else None.
    """
    label = label_from_filename(filepath)
    label_norm = _normalize_for_match(label)
    stem_norm = _normalize_for_match(filepath.stem)

    for img in existing:
        existing_label = img.get("label") or ""
        existing_norm = _normalize_for_match(existing_label)

        # Exact match
        if existing_label == label:
            return img
        # Normalized substring match (either way) — catches the 'beach' vs
        # 'wheres-waldo-beach-super-high-resolution-scaled' case.
        if existing_norm and (
            existing_norm in stem_norm or stem_norm in existing_norm
            or existing_norm in label_norm or label_norm in existing_norm
        ):
            return img
    return None


def ingest_one(backend: str, filepath: Path, max_retries: int = 3) -> dict | None:
    label = label_from_filename(filepath)
    body, content_type = encode_multipart(filepath, label)
    url = f"{backend.rstrip('/')}/api/ingest"

    for attempt in range(max_retries):
        # Before retrying, check if a previous attempt actually succeeded on
        # the server side (timeout could have happened on the client receive
        # while server-side work was finishing). If so, the image is now in
        # the index and we should skip rather than re-ingest a duplicate.
        if attempt > 0:
            existing = list_existing(backend)
            match = should_skip(filepath, existing)
            if match:
                print(f"  found {match.get('image_id')} indexed during retry — skipping",
                      file=sys.stderr)
                return {
                    "image_id": match.get("image_id"),
                    "width":   match.get("width", 0),
                    "height":  match.get("height", 0),
                    "tiles_indexed": match.get("tile_count", 0),
                    "elapsed_ms":   0,
                    "inference_ms": 0,
                    "indexing_ms":  0,
                }

        req = request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": content_type, "Accept": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read().decode())
        except error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            print(f"  HTTP {e.code}: {msg}", file=sys.stderr)
            return None  # don't retry application errors
        except (error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  transient error: {type(e).__name__}: {e} "
                      f"— retrying in {wait}s ({attempt + 1}/{max_retries})",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  transient error: {type(e).__name__}: {e} — gave up",
                      file=sys.stderr)
                return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Folder containing images")
    ap.add_argument("--backend", default="http://localhost:8000",
                    help="Backend URL (default: http://localhost:8000)")
    ap.add_argument("--no-skip", action="store_true",
                    help="Re-ingest even images whose label is already present")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 1

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_EXT)
    if not images:
        print(f"no images found in {folder}", file=sys.stderr)
        return 1

    existing = [] if args.no_skip else list_existing(args.backend)
    to_ingest: list[Path] = []
    skipped: list[tuple[Path, dict]] = []
    for img in images:
        match = should_skip(img, existing) if existing else None
        if match:
            skipped.append((img, match))
        else:
            to_ingest.append(img)

    print(f"Backend:  {args.backend}")
    if skipped:
        print(f"Skipping {len(skipped)} already-ingested:")
        for f, m in skipped:
            print(f"  • {f.name}  →  matched {m.get('image_id')} ({m.get('label') or '—'})")
    print(f"Ingesting {len(to_ingest)} of {len(images)} images")
    print()

    if not to_ingest:
        print("Nothing to do.")
        return 0

    print(f"{'#':>3}  {'file':<42} {'tiles':>6} {'embed':>9} {'index':>8} {'total':>9}  id")
    print("─" * 105)

    total_tiles = 0
    successes = 0
    failures: list[str] = []

    t_start = time.perf_counter()
    for i, img in enumerate(to_ingest, 1):
        print(f"{i:>3}  {img.name[:42]:<42}", end=" ", flush=True)
        result = ingest_one(args.backend, img)
        if result is None:
            print("  FAILED")
            failures.append(img.name)
            continue
        successes += 1
        total_tiles += result["tiles_indexed"]
        print(f"{result['tiles_indexed']:>6} "
              f"{result['inference_ms']:>7}ms "
              f"{result['indexing_ms']:>6}ms "
              f"{result['elapsed_ms']:>7}ms  "
              f"{result['image_id']}")

    wall = int((time.perf_counter() - t_start))
    print("─" * 105)
    print(f"Done.  {successes}/{len(to_ingest)} ok, {len(failures)} failed")
    print(f"Total tiles indexed: {total_tiles}")
    print(f"Wall clock: {wall}s")

    if failures:
        print("\nFailed files — rerun this script to retry just those:")
        for f in failures:
            print(f"  - {f}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
