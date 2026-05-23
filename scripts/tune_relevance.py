#!/usr/bin/env python3
"""
tune_relevance.py — Evaluation harness for the Find Waldo backend.

Original Phase 1D purpose
-------------------------
This script runs a fixed catalogue of test queries against a live backend and
records per-query metrics (top score, normalized scores, hit count, latency
breakdown, threshold behaviour, etc.) so we can decide:

    - Are the current MERGE_VOTE_BONUS / MERGE_RANK_BONUS values producing
      sensible cross-variant agreement?
    - Is SCORE_THRESHOLD_RATIO trimming weak hits without starving useful
      results?
    - Is QUERY_EXPANSION_ENABLED actually helping, or just doubling latency?
    - Do we need Phase 2 (reference vectors), or is the existing pipeline
      good enough on the test queries?

Phase 3 extensions
------------------
The script is now a full evaluation harness. Per-query test cases may carry
additional fields:

    image_id           explicit image identifier; if present, takes
                       precedence over `image_label` substring matching.
    ground_truth_bbox  the actual location of the target in the image,
                       expressed as {x, y, w, h} in original-image pixels.
                       When provided, the harness computes IoU between
                       each returned hit's bbox and the ground truth, and
                       records the best IoU in the top-k. This drives the
                       `success_top1/3/5` metrics: a hit "counts" only if
                       its bbox overlaps the ground truth at IoU ≥ 0.5.

Two grading models coexist in this script and answer different questions:

  1. Coarse pass/fail (existing, kept for backwards compat):
        expected_present=True  → pass iff num_hits >= 1
        expected_present=False → pass iff num_hits == 0
        expected_present=None  → not graded
     Answers: "Did the threshold filter behave correctly?"

  2. Top-k success (new in Phase 3):
        Requires ground_truth_bbox.
        success_top1 = 1 iff any of the top-1 hits has IoU(bbox, gt) >= 0.5
        success_top3 = 1 iff any of the top-3 hits has IoU(bbox, gt) >= 0.5
        success_top5 = 1 iff any of the top-5 hits has IoU(bbox, gt) >= 0.5
     Answers: "Did the system actually rank the target near the top?"

For `expected_present=False` queries the harness also records
`false_positive = 1 if num_hits > 0 else 0`. A non-zero false-positive rate
on absent queries is the cleanest signal that thresholding is too loose.

How to use it
-------------
1. Edit the CONFIG block near the top of this file to record the tuning
   values you intend to test.
2. Set those same values as environment variables on the backend, then
   restart the backend so they take effect:

       export QUERY_EXPANSION_ENABLED=true
       export QUERY_EXPANSION_MAX_VARIANTS=4
       export MERGE_VOTE_BONUS=0.05
       export MERGE_RANK_BONUS=0.10
       export SCORE_THRESHOLD_RATIO=0.80
       docker compose up --build -d backend

3. Run the script. It writes a JSON report to stdout, a human-readable
   summary to stderr, and (optionally) a CSV report to a file:

       python3 scripts/tune_relevance.py \
           --label baseline \
           --csv runs/baseline.csv \
           > runs/baseline.json

4. Change settings, restart the backend, rerun with a new label/CSV path.
   Diff the JSONs or load both CSVs into a spreadsheet to compare.

Why the script doesn't auto-set env vars
----------------------------------------
The script CANNOT change the backend's settings while it runs. The CONFIG
block here is a notepad: it documents what the operator intends the backend
to be running with. The script prints those values at the top of every run
so a mismatched run (script says one thing, backend running with another) is
visible at a glance. If the backend exposes a /api/settings endpoint in a
future phase, this script will read from it; for now, operator discipline.

What this script does NOT do
----------------------------
- It does NOT modify backend code, frontend code, or the running cluster.
- It does NOT depend on any frontend package or React state.
- It does NOT require any pip dependencies beyond Python 3.10 stdlib.
- It does NOT decide tuning parameter values for you. It surfaces metrics;
  the operator inspects them and adjusts.
- It does NOT validate bbox correctness or tile-level relevance. The
  pass/fail rule is a coarse "did anything confident survive" check.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# IoU threshold for top-k success metrics.
#
# A returned hit is considered a "correct" hit for a given ground-truth bbox
# when IoU(returned_bbox, ground_truth) >= IOU_HIT_THRESHOLD. 0.5 is the
# standard object-detection convention; tighten to 0.7 for stricter
# localisation evaluation, loosen to 0.3 if your tile size is large relative
# to the target (a 224x224 tile rarely has IoU > 0.5 with a small target
# like Waldo unless the tile is centred on him).
# ---------------------------------------------------------------------------
IOU_HIT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# CONFIG  — edit before each tuning run, then set matching env vars on the
# backend and restart it. The script prints these values at the top of every
# run; a mismatch between this block and the live backend will produce
# misleading output (the script will record what the BACKEND actually does,
# not what this block says).
# ---------------------------------------------------------------------------

# Backend URL. Override on the command line with --backend if needed.
BACKEND_URL = "http://localhost:8200"

# Per-query topK passed to /api/search. Higher = more candidates surfaced
# before threshold filtering. Keep this fixed across tuning runs so the
# raw response counts are comparable.
SEARCH_K = 10

# Whether to scope each query to a specific image. When True, the script
# uses the per-query image_id specified in TEST_QUERIES. When False, no
# image_id is sent and search ranges across the entire corpus.
SCOPE_TO_IMAGE = True

# These knobs are NOT set by the script — they are NOTES about what the
# backend should be running with. Set the matching env vars on the backend
# and restart before running the script.
EXPECTED_BACKEND_SETTINGS = {
    "QUERY_EXPANSION_ENABLED":      "true",
    "QUERY_EXPANSION_MAX_VARIANTS": "4",
    "MERGE_VOTE_BONUS":             "0.05",
    "MERGE_RANK_BONUS":             "0.10",
    "SCORE_THRESHOLD_RATIO":        "0.80",
    "RERANKER_ENABLED":             "true",
}

# ---------------------------------------------------------------------------
# Test query catalogue.
#
# Each entry is a dict with at least the `query` field. Optional fields:
#
#   category            : str  — grouping label for aggregates ("waldo",
#                                "generic_object", "ambiguous", "lookalike",
#                                "distractor", or anything you want).
#   image_label         : str  — substring (case-insensitive) matched against
#                                /api/images labels to pick the image_id.
#                                None = run unscoped against the whole corpus.
#   expected_present    : bool | None — True if the target should be
#                                findable, False if it should NOT be (the
#                                system should return zero confident hits),
#                                None if we don't know / aren't grading.
#   target_description  : str  — what we're looking for, in plain English.
#                                For human readability in CSV/JSON output.
#   notes               : str  — freeform comment about why this query is
#                                in the catalogue.
#
# The catalogue is INTENTIONALLY agnostic about which corpus images are
# present. Edit the entries to match your indexed images and ground truth.
# Where ground truth is uncertain, leave expected_present=None and the
# script will still record metrics without grading the query.
# ---------------------------------------------------------------------------

# Each entry may also include:
#
#   image_id            : str  — explicit image identifier. If present,
#                                takes precedence over image_label. Useful
#                                when a test case targets a specific
#                                indexed image regardless of its label.
#   ground_truth_bbox   : dict — {x, y, w, h} in original-image pixels.
#                                Required for top-k success metrics. When
#                                absent, those metrics are reported as None
#                                and the query is graded only on coarse
#                                presence/absence (if expected_present set).

TEST_QUERIES: list[dict] = [
    # --- Waldo-the-character probes -------------------------------------
    {
        "category":           "waldo",
        "query":              "find Waldo",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "Waldo himself (red/white striped shirt and beanie)",
        "notes":              "Direct character query against an image where Waldo's "
                              "presence is the operator's responsibility to confirm.",
    },
    {
        "category":           "waldo",
        "query":              "a man with a red and white striped shirt",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "Waldo himself, described visually",
        "notes":              "Tests whether descriptive phrasing finds the same target "
                              "as the named-character phrasing.",
    },
    {
        "category":           "waldo",
        "query":              "a man with a red and white striped hat",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "Waldo himself, focused on the hat",
        "notes":              "Different visual cue (hat vs shirt) — measures whether "
                              "the model agrees on the same tile across phrasings.",
    },

    # --- Queries we explicitly believe should NOT surface a confident hit ---
    # These are deliberately constructed: things very unlikely to be in any
    # Where's Waldo scene. We expect the threshold filter to leave us with
    # zero or near-zero hits.
    {
        "category":           "absent",
        "query":              "a pink dragon",
        "image_label":        None,
        "expected_present":   False,
        "target_description": "A pink dragon (definitely not in the corpus)",
        "notes":              "Absent-target probe — fails if the system surfaces "
                              "confident hits for something fantastical.",
    },
    {
        "category":           "absent",
        "query":              "a black labrador puppy",
        "image_label":        None,
        "expected_present":   False,
        "target_description": "A black labrador puppy",
        "notes":              "Absent-target probe with a plausible-sounding query.",
    },
    {
        "category":           "absent",
        "query":              "a 1970s Cadillac convertible",
        "image_label":        None,
        "expected_present":   False,
        "target_description": "A specific car model unlikely to be in the corpus",
        "notes":              "Absent-target probe — checks whether specificity helps "
                              "the model say 'not here' instead of returning a vaguely "
                              "car-shaped tile.",
    },

    # --- Generic objects whose presence depends on the image ---
    {
        "category":           "generic_object",
        "query":              "a striped beach umbrella",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "Any beach umbrella with stripes",
        "notes":              "Beach scenes typically contain umbrellas; presence in "
                              "this corpus is the operator's call.",
    },
    {
        "category":           "generic_object",
        "query":              "a sailboat with a red sail",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "A sailboat, red sail",
        "notes":              "Tests color+object combination retrieval.",
    },
    {
        "category":           "generic_object",
        "query":              "a giant lollipop",
        "image_label":        "Candy Factory",
        "expected_present":   None,
        "target_description": "A large lollipop",
        "notes":              "Probes the candy-themed image without depending on "
                              "knowing exactly what's in it.",
    },

    # --- Ambiguous / vague queries ---
    # These should produce many weak matches. We expect heavy thresholding.
    # No expected_present grading — these are degenerate-input probes, not
    # presence/absence tests.
    {
        "category":           "ambiguous",
        "query":              "red",
        "image_label":        None,
        "expected_present":   None,
        "target_description": "Anything red — query is too vague to be useful",
        "notes":              "Degenerate single-token query; expect heavy thresholding.",
    },
    {
        "category":           "ambiguous",
        "query":              "man",
        "image_label":        None,
        "expected_present":   None,
        "target_description": "Any man",
        "notes":              "Common-noun query; useful for checking that the system "
                              "doesn't lock onto Waldo for every 'man' query.",
    },
    {
        "category":           "ambiguous",
        "query":              "thing",
        "image_label":        None,
        "expected_present":   None,
        "target_description": "Any thing — semantically empty",
        "notes":              "Pathological query; useful as a noise-floor reference.",
    },

    # --- Lookalike / distractor queries ---
    # Queries crafted to resemble Waldo's visual signature without being him.
    # The interesting question is whether these score AS HIGH as actual
    # Waldo queries — if they do, the model can't distinguish, which is
    # exactly the signal that motivates Phase 2 (reference vectors).
    {
        "category":           "lookalike",
        "query":              "red and white striped umbrella",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "Anything red/white striped that ISN'T Waldo",
        "notes":              "Compare top_score to the 'find Waldo' top_score on the "
                              "same image. Close scores → distractor confusion.",
    },
    {
        "category":           "lookalike",
        "query":              "red and white striped beach chair",
        "image_label":        "Beach",
        "expected_present":   None,
        "target_description": "Striped beach chair (not Waldo)",
        "notes":              "Same probe as above with a different distractor object.",
    },
    {
        "category":           "lookalike",
        "query":              "red and white candy stripes",
        "image_label":        "Candy Factory",
        "expected_present":   None,
        "target_description": "Candy with red and white striping",
        "notes":              "Stripe pattern in a non-Waldo context.",
    },
]


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def http_get_json(url: str, timeout: float = 30.0) -> Any:
    """GET and parse JSON. Returns parsed body, or raises on non-2xx."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON and parse JSON response. Raises on non-2xx."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Per-query record. Contains everything we measured plus the operator-
    supplied metadata that travels with each query.
    """
    # --- query identity / operator-supplied metadata ---
    category:           str
    query:              str
    image_label:        Optional[str]   # the substring as configured
    image_id:           Optional[str]   # what we resolved it to (None if unscoped or unmatched)
    image_id_source:    str             # where image_id came from: "override" / "explicit" / "label" / "none"
    image_label_resolved: Optional[str]  # the actual image label we ran against
    expected_present:   Optional[bool]
    target_description: Optional[str]
    notes:              Optional[str]
    ground_truth_bbox:  Optional[dict]  # {x, y, w, h} when supplied; None otherwise
    # --- coarse pass/fail (Phase 1D, kept for backwards compat) ---
    passed:             Optional[bool]  # None when expected_present is None
    # --- raw measurements ---
    num_hits:           int
    top_score:          float
    normalized_scores:  list[float]
    tile_ids:           list[str]
    bboxes:             list[dict]
    # Convenience views surfaced for quick inspection in the JSON / CSV.
    # These are derived from `tile_ids` / `bboxes` rather than fetched
    # separately — they are subsets, not parallel lookups.
    top_1_hit:          Optional[str]
    top_3_hits:         list[str]
    top_5_hits:         list[str]
    threshold_triggered: bool
    knn_returned_max:   int
    inference_ms:       int
    search_ms:          int
    rerank_ms:          int
    fusion_ms:          int
    expansion_ms:       int
    elapsed_ms:         int
    reranked:           bool
    fusion_strategy:    Optional[str]
    expanded_queries:   list[str]
    corrected_query:    Optional[str]
    # --- Phase 3 metrics ---
    # IoU per top-k hit, in the same order as `bboxes`. None entries when
    # ground truth is unavailable. The list length matches min(num_hits, 5)
    # — we don't compute IoU on hits beyond top-5 since the success metrics
    # are top-1/3/5 only.
    iou_top_k:          list[Optional[float]]
    # Best IoU across the top-k that we computed. None when no ground truth.
    best_iou:           Optional[float]
    # 1/0 if the corresponding top-k slice contains a hit with IoU >=
    # IOU_HIT_THRESHOLD; None when no ground truth.
    success_top1:       Optional[int]
    success_top3:       Optional[int]
    success_top5:       Optional[int]
    # 1 iff this is an absent-target query that returned at least one hit.
    # None for queries where expected_present is not False.
    false_positive:     Optional[int]
    # ---
    error:              Optional[str] = None


def fetch_image_index(backend: str) -> list[dict]:
    """Pull the corpus from /api/images so we can resolve label substrings to image_ids."""
    images = http_get_json(f"{backend.rstrip('/')}/api/images")
    if not isinstance(images, list):
        raise RuntimeError(f"Unexpected /api/images shape: {type(images)}")
    return images


def resolve_image_id(images: list[dict], substring: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (image_id, image_label) for the first image whose label contains
    `substring` (case-insensitive). Returns (None, None) if substring is None
    or no match was found."""
    if substring is None:
        return None, None
    needle = substring.lower()
    for img in images:
        label = (img.get("label") or "")
        if needle in label.lower():
            return img.get("image_id"), label
    return None, None


def grade(expected_present: Optional[bool], num_hits: int) -> Optional[bool]:
    """Pass/fail rule — see module docstring.

    Returns None when expected_present is None (no grading).
    Otherwise:
        expected_present=True  → pass iff num_hits >= 1
        expected_present=False → pass iff num_hits == 0
    """
    if expected_present is None:
        return None
    if expected_present:
        return num_hits >= 1
    return num_hits == 0


# ---------------------------------------------------------------------------
# IoU + top-k success metrics (Phase 3)
# ---------------------------------------------------------------------------

def _iou_xywh(a: dict, b: dict) -> float:
    """Intersection-over-union for two bboxes in {x, y, w, h} form.

    Both inputs use the same coordinate space (original-image pixels).
    Returns 0.0 when either box has non-positive area or when there is no
    overlap. Symmetric in its arguments. No clamping to image dimensions —
    callers are responsible for passing well-formed boxes.
    """
    ax, ay = float(a.get("x", 0)), float(a.get("y", 0))
    aw, ah = float(a.get("w", 0)), float(a.get("h", 0))
    bx, by = float(b.get("x", 0)), float(b.get("y", 0))
    bw, bh = float(b.get("w", 0)), float(b.get("h", 0))
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    # Intersection rectangle in xyxy form
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _topk_success_metrics(
    bboxes: list[dict],
    ground_truth_bbox: Optional[dict],
    iou_threshold: float = IOU_HIT_THRESHOLD,
) -> dict:
    """Compute IoU per top-5 hit, the best IoU across that slice, and
    success_top1/3/5 indicators.

    Returns a dict with keys:
        iou_top_k    : list[Optional[float]]  one entry per hit in top-5
        best_iou     : Optional[float]
        success_top1 : Optional[int]   1 / 0 / None
        success_top3 : Optional[int]   1 / 0 / None
        success_top5 : Optional[int]   1 / 0 / None

    All four metrics are None when `ground_truth_bbox` is None — we do not
    silently report 0 in that case because "didn't have ground truth" is
    different from "failed to find target". The CSV / JSON keep the
    distinction explicit.
    """
    if ground_truth_bbox is None:
        return {
            "iou_top_k":    [],
            "best_iou":     None,
            "success_top1": None,
            "success_top3": None,
            "success_top5": None,
        }

    top5_bboxes = bboxes[:5]
    ious: list[Optional[float]] = [
        _iou_xywh(bb, ground_truth_bbox) if isinstance(bb, dict) else None
        for bb in top5_bboxes
    ]

    def any_hit(slice_ious: list[Optional[float]]) -> int:
        return 1 if any((iou or 0.0) >= iou_threshold for iou in slice_ious) else 0

    real_ious = [iou for iou in ious if iou is not None]
    return {
        "iou_top_k":    ious,
        "best_iou":     max(real_ious) if real_ious else 0.0,
        "success_top1": any_hit(ious[:1]),
        "success_top3": any_hit(ious[:3]),
        "success_top5": any_hit(ious[:5]),
    }


def run_one_query(
    backend: str,
    spec: dict,
    images: list[dict],
    k: int,
    scope_to_image: bool,
    image_id_override: Optional[str] = None,
) -> QueryResult:
    """Run a single search, build the QueryResult record from the response.

    Image selection precedence (most-specific wins):
        1. `image_id_override` argument (CLI --image-id flag)
        2. spec["image_id"]                 (explicit per-query)
        3. spec["image_label"] substring match against /api/images
        4. unscoped (no image_id sent to backend)

    `scope_to_image=False` (CLI --no-scope) forces (4) and skips 1-3.
    """
    category           = spec.get("category", "uncategorized")
    query              = spec["query"]
    image_label_substr = spec.get("image_label")
    image_id_explicit  = spec.get("image_id")
    expected_present   = spec.get("expected_present")
    target_description = spec.get("target_description")
    notes              = spec.get("notes")
    ground_truth_bbox  = spec.get("ground_truth_bbox")

    image_id: Optional[str] = None
    image_label_resolved: Optional[str] = None
    image_id_source: str = "none"

    if scope_to_image:
        if image_id_override is not None:
            image_id = image_id_override
            image_id_source = "override"
            # Try to resolve a friendly label for logging — best effort.
            for img in images:
                if img.get("image_id") == image_id_override:
                    image_label_resolved = img.get("label")
                    break
        elif image_id_explicit is not None:
            image_id = image_id_explicit
            image_id_source = "explicit"
            for img in images:
                if img.get("image_id") == image_id_explicit:
                    image_label_resolved = img.get("label")
                    break
        elif image_label_substr is not None:
            image_id, image_label_resolved = resolve_image_id(images, image_label_substr)
            if image_id is None:
                print(
                    f"  WARN: no image with label containing {image_label_substr!r}; "
                    f"running unscoped",
                    file=sys.stderr,
                )
            else:
                image_id_source = "label"

    payload: dict = {"query": query, "k": k}
    if image_id is not None:
        payload["image_id"] = image_id

    try:
        resp = http_post_json(f"{backend.rstrip('/')}/api/search", payload)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        return QueryResult(
            category=category, query=query, image_label=image_label_substr,
            image_id=image_id, image_id_source=image_id_source,
            image_label_resolved=image_label_resolved,
            expected_present=expected_present,
            target_description=target_description, notes=notes,
            ground_truth_bbox=ground_truth_bbox,
            passed=None, num_hits=0, top_score=0.0,
            normalized_scores=[], tile_ids=[], bboxes=[],
            top_1_hit=None, top_3_hits=[], top_5_hits=[],
            threshold_triggered=False, knn_returned_max=k,
            inference_ms=0, search_ms=0, rerank_ms=0,
            fusion_ms=0, expansion_ms=0, elapsed_ms=0,
            reranked=False, fusion_strategy=None, expanded_queries=[],
            corrected_query=None,
            iou_top_k=[], best_iou=None,
            success_top1=None, success_top3=None, success_top5=None,
            false_positive=None,
            error=f"{type(e).__name__}: {e}",
        )

    hits = resp.get("hits", []) or []
    top_score = float(hits[0]["score"]) if hits else 0.0
    normalized = [float(h.get("normalized_score", 1.0)) for h in hits]
    tile_ids = [h.get("tile_id") for h in hits]
    bboxes = [h.get("bbox") for h in hits]
    threshold_triggered = len(hits) < k

    # Top-k convenience views. We always populate these from whatever the
    # backend returned; truncation happens naturally if the backend
    # returned fewer than k hits.
    top_1_hit  = tile_ids[0] if tile_ids else None
    top_3_hits = [t for t in tile_ids[:3] if t is not None]
    top_5_hits = [t for t in tile_ids[:5] if t is not None]

    # Phase 3 metrics: IoU-driven top-k success when we have ground truth.
    success = _topk_success_metrics(bboxes, ground_truth_bbox)

    # False-positive metric: only meaningful for explicitly-absent queries.
    if expected_present is False:
        false_positive: Optional[int] = 1 if len(hits) > 0 else 0
    else:
        false_positive = None

    return QueryResult(
        category=category,
        query=query,
        image_label=image_label_substr,
        image_id=image_id,
        image_id_source=image_id_source,
        image_label_resolved=image_label_resolved,
        expected_present=expected_present,
        target_description=target_description,
        notes=notes,
        ground_truth_bbox=ground_truth_bbox,
        passed=grade(expected_present, len(hits)),
        num_hits=len(hits),
        top_score=top_score,
        normalized_scores=normalized,
        tile_ids=tile_ids,
        bboxes=bboxes,
        top_1_hit=top_1_hit,
        top_3_hits=top_3_hits,
        top_5_hits=top_5_hits,
        threshold_triggered=threshold_triggered,
        knn_returned_max=k,
        inference_ms=int(resp.get("inference_ms", 0)),
        search_ms=int(resp.get("search_ms", 0)),
        rerank_ms=int(resp.get("rerank_ms", 0)),
        fusion_ms=int(resp.get("fusion_ms", 0)),
        expansion_ms=int(resp.get("expansion_ms", 0)),
        elapsed_ms=int(resp.get("elapsed_ms", 0)),
        reranked=bool(resp.get("reranked", False)),
        fusion_strategy=resp.get("fusion_strategy"),
        expanded_queries=resp.get("expanded_queries", []) or [],
        corrected_query=resp.get("corrected_query"),
        iou_top_k=success["iou_top_k"],
        best_iou=success["best_iou"],
        success_top1=success["success_top1"],
        success_top3=success["success_top3"],
        success_top5=success["success_top5"],
        false_positive=false_positive,
        error=None,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_float(x: float, w: int = 5) -> str:
    return f"{x:0.3f}" if x is not None else " " * w


def _passed_marker(passed: Optional[bool]) -> str:
    if passed is None:
        return "  -  "
    return " PASS" if passed else " FAIL"


def print_human_summary(results: list[QueryResult], label: Optional[str]) -> None:
    """Compact stderr summary for eyeballing. JSON goes to stdout."""
    header = (
        f"{'cat':<16} {'query':<46} {'img':<18} "
        f"{'exp':>4} {'pass':>5} "
        f"{'#':>3} {'top':>6} {'norm_min':>8} {'lat_ms':>7} {'thr':>4} "
        f"{'s1':>3} {'s3':>3} {'s5':>3} {'iou':>5}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    by_cat: dict[str, list[QueryResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    def _success_cell(v: Optional[int]) -> str:
        if v is None:
            return "  -"
        return f"  {v}"

    for category, items in by_cat.items():
        for r in items:
            norm_min = min(r.normalized_scores) if r.normalized_scores else 0.0
            exp_str = (
                "T" if r.expected_present is True
                else "F" if r.expected_present is False
                else "-"
            )
            iou_cell = (
                f"{r.best_iou:0.2f}" if isinstance(r.best_iou, float) else "  -  "
            )
            row = (
                f"{r.category:<16} {r.query[:44]:<46} "
                f"{(r.image_label_resolved or '-')[:16]:<18} "
                f"{exp_str:>4} {_passed_marker(r.passed):>5} "
                f"{r.num_hits:>3} "
                f"{_fmt_float(r.top_score):>6} "
                f"{_fmt_float(norm_min):>8} "
                f"{r.elapsed_ms:>7d} "
                f"{'yes' if r.threshold_triggered else 'no':>4} "
                f"{_success_cell(r.success_top1):>3} "
                f"{_success_cell(r.success_top3):>3} "
                f"{_success_cell(r.success_top5):>3} "
                f"{iou_cell:>5}"
            )
            if r.error:
                row += f"   ERROR: {r.error}"
            print(row, file=sys.stderr)
        print("", file=sys.stderr)

    # Aggregates per category. We surface two distinct success notions side
    # by side: the coarse Phase-1D pass_rate (presence/absence at any rank)
    # and the new top-k IoU-based success rates (target localised at top-1/
    # top-3/top-5). They answer different questions; both can be useful
    # depending on what you're tuning.
    print("Aggregates by category:", file=sys.stderr)
    print(
        f"  {'category':<16} {'n':>3} {'mean_top':>9} {'mean_hits':>10} "
        f"{'mean_lat_ms':>12} {'pct_thresholded':>16} {'pass_rate':>10} "
        f"{'top1':>7} {'top3':>7} {'top5':>7} {'fp':>7}",
        file=sys.stderr,
    )
    for category, items in by_cat.items():
        ok = [r for r in items if r.error is None]
        if not ok:
            continue
        mean_top = statistics.mean(r.top_score for r in ok)
        mean_hits = statistics.mean(r.num_hits for r in ok)
        mean_lat = statistics.mean(r.elapsed_ms for r in ok)
        pct_thr = sum(1 for r in ok if r.threshold_triggered) / len(ok) * 100
        graded = [r for r in ok if r.passed is not None]
        pass_rate_str = (
            f"{sum(1 for r in graded if r.passed) / len(graded) * 100:>9.1f}%"
            if graded else "       n/a"
        )

        # Top-k success rates are computed only over queries that have
        # ground truth; n/a otherwise.
        def _success_rate(field_name: str) -> str:
            graded = [r for r in ok if getattr(r, field_name) is not None]
            if not graded:
                return "    n/a"
            rate = sum(1 for r in graded if getattr(r, field_name)) / len(graded) * 100
            return f"{rate:>6.1f}%"

        # False-positive rate is computed only over absent queries.
        fp_graded = [r for r in ok if r.false_positive is not None]
        fp_rate_str = (
            f"{sum(1 for r in fp_graded if r.false_positive) / len(fp_graded) * 100:>6.1f}%"
            if fp_graded else "    n/a"
        )

        print(
            f"  {category:<16} {len(ok):>3} {mean_top:>9.3f} "
            f"{mean_hits:>10.1f} {mean_lat:>12.1f} {pct_thr:>15.1f}% "
            f"{pass_rate_str:>10} "
            f"{_success_rate('success_top1'):>7} "
            f"{_success_rate('success_top3'):>7} "
            f"{_success_rate('success_top5'):>7} "
            f"{fp_rate_str:>7}",
            file=sys.stderr,
        )

    # Overall figures across the full run.
    ok_all = [r for r in results if r.error is None]
    graded_pass = [r for r in ok_all if r.passed is not None]
    if graded_pass:
        overall_pass = sum(1 for r in graded_pass if r.passed) / len(graded_pass) * 100
        print(
            f"\nOverall coarse pass rate: {len(graded_pass)} graded queries, "
            f"{overall_pass:.1f}%",
            file=sys.stderr,
        )

    # Overall top-k success rates over queries with ground truth
    for fname, label_str in (
        ("success_top1", "top-1"),
        ("success_top3", "top-3"),
        ("success_top5", "top-5"),
    ):
        graded = [r for r in ok_all if getattr(r, fname) is not None]
        if graded:
            rate = sum(1 for r in graded if getattr(r, fname)) / len(graded) * 100
            print(
                f"Overall {label_str} success: {len(graded)} graded queries, {rate:.1f}%",
                file=sys.stderr,
            )

    # Overall false-positive rate over absent queries
    fp_graded_all = [r for r in ok_all if r.false_positive is not None]
    if fp_graded_all:
        fp_rate = sum(1 for r in fp_graded_all if r.false_positive) / len(fp_graded_all) * 100
        print(
            f"Overall false positive rate: {len(fp_graded_all)} absent queries, {fp_rate:.1f}%",
            file=sys.stderr,
        )

    # Average latency across all successful queries (errors excluded)
    if ok_all:
        avg_latency = statistics.mean(r.elapsed_ms for r in ok_all)
        print(f"Average latency: {avg_latency:.0f}ms over {len(ok_all)} queries",
              file=sys.stderr)

    if label:
        print(f"\nRun label: {label}", file=sys.stderr)


def write_csv(path: str, results: list[QueryResult],
              label: Optional[str], k: int, scope_to_image: bool) -> None:
    """Write a flat-table CSV. List-shaped fields are JSON-encoded so the
    column count stays stable across rows.

    The Phase 3 spec calls out a specific minimum set of columns (category,
    query, image_id, expected_present, success_top1/3/5, false_positive,
    num_hits, top_score, normalized_min/max, latency_ms, best_iou). Those
    appear first — in the spec's order — so spreadsheet readers see them
    immediately. The rest of the run metadata, latency breakdown, and
    JSON-encoded blobs follow for deeper analysis.
    """
    fieldnames = [
        # ---- Phase 3 minimum spec, in spec order ----
        "category",
        "query",
        "image_id",
        "expected_present",
        "success_top1",
        "success_top3",
        "success_top5",
        "false_positive",
        "num_hits",
        "top_score",
        "normalized_min",
        "normalized_max",
        "latency_ms",
        "best_iou",
        # ---- Run-level context (repeated per row for spreadsheet filtering) ----
        "label",
        "k",
        "scope_to_image",
        # ---- Additional query identity ----
        "image_label",
        "image_id_source",
        "image_label_resolved",
        "target_description",
        "notes",
        "ground_truth_bbox_json",
        # ---- Coarse Phase-1D pass/fail (kept for backwards compat) ----
        "passed",
        "error",
        # ---- Top-k convenience views ----
        "top_1_hit",
        "top_3_hits_json",
        "top_5_hits_json",
        "iou_top_k_json",
        # ---- Other measurements ----
        "threshold_triggered",
        "knn_returned_max",
        "reranked",
        "fusion_strategy",
        # ---- Latency breakdown (ms) ----
        "inference_ms",
        "search_ms",
        "rerank_ms",
        "fusion_ms",
        "expansion_ms",
        # ---- Larger blobs (JSON-encoded) ----
        "corrected_query",
        "expanded_queries_json",
        "tile_ids_json",
        "normalized_scores_json",
        "bboxes_json",
    ]

    def _opt_int(v: Optional[int]) -> str:
        return "" if v is None else str(v)

    def _opt_bool(v: Optional[bool]) -> str:
        if v is None:
            return ""
        return "true" if v else "false"

    def _opt_float(v: Optional[float], precision: int = 6) -> str:
        return "" if v is None else f"{v:.{precision}f}"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            norm_min = min(r.normalized_scores) if r.normalized_scores else None
            norm_max = max(r.normalized_scores) if r.normalized_scores else None
            writer.writerow({
                # ---- Phase 3 minimum spec ----
                "category":               r.category,
                "query":                  r.query,
                "image_id":               r.image_id or "",
                "expected_present":       _opt_bool(r.expected_present),
                "success_top1":           _opt_int(r.success_top1),
                "success_top3":           _opt_int(r.success_top3),
                "success_top5":           _opt_int(r.success_top5),
                "false_positive":         _opt_int(r.false_positive),
                "num_hits":               r.num_hits,
                "top_score":              f"{r.top_score:.6f}",
                "normalized_min":         _opt_float(norm_min),
                "normalized_max":         _opt_float(norm_max),
                "latency_ms":             r.elapsed_ms,
                "best_iou":               _opt_float(r.best_iou),
                # ---- Run-level ----
                "label":                  label or "",
                "k":                      k,
                "scope_to_image":         scope_to_image,
                # ---- Identity ----
                "image_label":            r.image_label or "",
                "image_id_source":        r.image_id_source,
                "image_label_resolved":   r.image_label_resolved or "",
                "target_description":     r.target_description or "",
                "notes":                  r.notes or "",
                "ground_truth_bbox_json": (
                    json.dumps(r.ground_truth_bbox) if r.ground_truth_bbox else ""
                ),
                # ---- Coarse pass/fail ----
                "passed":                 _opt_bool(r.passed),
                "error":                  r.error or "",
                # ---- Top-k convenience ----
                "top_1_hit":              r.top_1_hit or "",
                "top_3_hits_json":        json.dumps(r.top_3_hits),
                "top_5_hits_json":        json.dumps(r.top_5_hits),
                "iou_top_k_json":         json.dumps(
                    [None if v is None else round(v, 6) for v in r.iou_top_k]
                ),
                # ---- Other ----
                "threshold_triggered":    "true" if r.threshold_triggered else "false",
                "knn_returned_max":       r.knn_returned_max,
                "reranked":               "true" if r.reranked else "false",
                "fusion_strategy":        r.fusion_strategy or "",
                # ---- Latency ----
                "inference_ms":           r.inference_ms,
                "search_ms":              r.search_ms,
                "rerank_ms":              r.rerank_ms,
                "fusion_ms":              r.fusion_ms,
                "expansion_ms":           r.expansion_ms,
                # ---- Blobs ----
                "corrected_query":        r.corrected_query or "",
                "expanded_queries_json":  json.dumps(r.expanded_queries),
                "tile_ids_json":          json.dumps(r.tile_ids),
                "normalized_scores_json": json.dumps([round(x, 6) for x in r.normalized_scores]),
                "bboxes_json":            json.dumps(r.bboxes),
            })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone relevance tuning harness for the Find Waldo backend",
    )
    parser.add_argument("--backend", default=BACKEND_URL,
                        help=f"Backend URL (default: {BACKEND_URL})")
    parser.add_argument("--k", type=int, default=SEARCH_K,
                        help=f"Per-query k (default: {SEARCH_K})")
    parser.add_argument("--no-scope", action="store_true",
                        help="Do not scope queries to a specific image; "
                             "search the whole corpus")
    parser.add_argument("--label", default=None,
                        help="Run label, recorded in JSON/CSV output")
    parser.add_argument("--category", default=None,
                        help="If set, run only queries in this category "
                             "(e.g. 'waldo', 'absent', 'generic_object', "
                             "'ambiguous', 'lookalike')")
    parser.add_argument("--csv", default=None,
                        help="Optional: write a flat-table CSV report to this path. "
                             "JSON is always written to stdout regardless.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of queries run (after category "
                             "filtering). Useful for quick smoke-tests of the "
                             "harness against a live backend.")
    parser.add_argument("--image-id", default=None,
                        help="Override the image scope for every query: run all "
                             "queries against this image_id, regardless of any "
                             "per-query image_id or image_label. Has no effect "
                             "when --no-scope is set.")
    args = parser.parse_args()

    scope_to_image = SCOPE_TO_IMAGE and not args.no_scope

    # Banner
    print("# tune_relevance.py", file=sys.stderr)
    print(f"# backend       : {args.backend}", file=sys.stderr)
    print(f"# k             : {args.k}", file=sys.stderr)
    print(f"# scope_to_image: {scope_to_image}", file=sys.stderr)
    print(f"# label         : {args.label or '(unset)'}", file=sys.stderr)
    print(f"# csv           : {args.csv or '(none)'}", file=sys.stderr)
    print(f"# limit         : {args.limit if args.limit is not None else '(unset)'}",
          file=sys.stderr)
    print(f"# image-id      : {args.image_id or '(unset)'}", file=sys.stderr)
    print("# expected backend env (set these and restart the backend before running):",
          file=sys.stderr)
    for k_, v_ in EXPECTED_BACKEND_SETTINGS.items():
        print(f"#   {k_}={v_}", file=sys.stderr)
    print("", file=sys.stderr)

    # Resolve corpus
    try:
        images = fetch_image_index(args.backend)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"FATAL: could not reach backend at {args.backend}: {e}",
              file=sys.stderr)
        return 2
    print(f"# corpus: {len(images)} indexed images", file=sys.stderr)
    for img in images:
        print(f"#   - {img.get('image_id')}  {img.get('label')}", file=sys.stderr)
    print("", file=sys.stderr)

    # Filter queries
    queries_to_run = TEST_QUERIES
    if args.category:
        queries_to_run = [q for q in TEST_QUERIES if q.get("category") == args.category]
        if not queries_to_run:
            print(f"FATAL: no queries in category {args.category!r}",
                  file=sys.stderr)
            return 2

    # Apply --limit AFTER category filtering so the operator sees the
    # subset of the requested category, not an arbitrary head of the
    # full catalogue.
    if args.limit is not None:
        if args.limit < 0:
            print(f"FATAL: --limit must be >= 0, got {args.limit}",
                  file=sys.stderr)
            return 2
        queries_to_run = queries_to_run[:args.limit]

    # Validate --image-id, if provided, against the corpus. Warn but don't
    # fail — there's a legitimate reason to run a query against an
    # image_id the script doesn't know about (e.g., debugging an
    # ingestion-vs-corpus skew).
    if args.image_id is not None and args.image_id not in {
        img.get("image_id") for img in images
    }:
        print(
            f"  WARN: --image-id {args.image_id!r} is not in /api/images; "
            f"backend will likely return zero hits for every query",
            file=sys.stderr,
        )

    # Run queries
    results: list[QueryResult] = []
    t_run = time.perf_counter()
    for idx, spec in enumerate(queries_to_run, start=1):
        category = spec.get("category", "?")
        query = spec.get("query", "?")
        image_substr = spec.get("image_label")
        print(
            f"[{idx}/{len(queries_to_run)}] {category} :: {query!r} -> "
            f"{image_substr or 'unscoped'}",
            file=sys.stderr,
        )
        r = run_one_query(
            backend=args.backend,
            spec=spec,
            images=images,
            k=args.k,
            scope_to_image=scope_to_image,
            image_id_override=args.image_id,
        )
        results.append(r)
    total_run_s = time.perf_counter() - t_run

    # Human summary on stderr
    print("", file=sys.stderr)
    print_human_summary(results, args.label)
    print(f"\nTotal wall clock: {total_run_s:.1f}s", file=sys.stderr)

    # Optional CSV
    if args.csv:
        try:
            write_csv(args.csv, results, args.label, args.k, scope_to_image)
            print(f"CSV written to {args.csv}", file=sys.stderr)
        except OSError as e:
            print(f"WARN: failed to write CSV to {args.csv}: {e}",
                  file=sys.stderr)

    # JSON to stdout (always)
    report = {
        "label":                     args.label,
        "backend":                   args.backend,
        "k":                         args.k,
        "scope_to_image":            scope_to_image,
        "image_id_override":         args.image_id,
        "limit":                     args.limit,
        "category_filter":           args.category,
        "iou_hit_threshold":         IOU_HIT_THRESHOLD,
        "expected_backend_settings": EXPECTED_BACKEND_SETTINGS,
        "corpus": [
            {"image_id": img.get("image_id"), "label": img.get("label"),
             "tile_count": img.get("tile_count")}
            for img in images
        ],
        "results":                   [asdict(r) for r in results],
        "total_wall_clock_s":        round(total_run_s, 3),
    }
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
