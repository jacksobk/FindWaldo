"""
Result list merging by tile_id with vote count and best-rank bookkeeping.

Phase 1A fans out a single user query into N variant queries (rule-based
expansions: original, character-specific descriptions, etc.) and runs kNN
once per variant. Each variant produces its own ranked candidate list of
tiles. We then merge those lists into one ranked list before handing the
candidates off to the reranker.

Algorithm (score-merge with vote count and rank tie-break):

    1. For each tile that appears in any variant's results, compute:
        - max_score:    the highest kNN cosine score the tile got across
                        any variant. This is the strongest single piece of
                        evidence we have that this tile matches.
        - votes:        the number of variants whose top-N included this
                        tile. Cross-variant agreement is a separate signal
                        from raw similarity strength: a tile that lights
                        up for "find Waldo" AND "man with red striped
                        shirt" is more likely to be a real match than one
                        that lights up only for one phrasing.
        - best_rank:    the lowest (best) rank position the tile achieved
                        in any variant. A tile that was rank-1 somewhere
                        is preferred over a tile that was rank-30 in ten
                        variants.

    2. Combine into a composite score:

           composite = max_score
                     + vote_bonus * (votes - 1)
                     + rank_bonus * (1 / best_rank)

       The (votes - 1) term means a tile that appeared in only one variant
       gets no vote bonus (its composite equals its max_score). Each
       additional variant that surfaced the tile adds a small additive
       boost. The (1 / best_rank) term rewards tiles that were ranked
       highly in some variant.

       Default weights are conservative: ranking is still mostly driven by
       raw similarity. Vote count and rank position are secondary signals
       used to break ties and to lift cross-variant winners over single-
       variant outliers.

    3. Sort by composite descending. Truncate to top_n.

What this is NOT:

    - It is NOT vector averaging. We never combine embeddings. The merge
      operates entirely on per-variant ranked result lists.
    - It is NOT RRF. Reciprocal Rank Fusion uses 1/(k+rank) summed across
      lists; this merge uses a different combination (max score + vote
      bonus + rank bonus). Both are valid list-merging strategies; this
      one weights raw similarity strength more directly, which matches
      Phase 1A's spec.
    - It is NOT a probability. The composite score is an ordinal ranking
      key, not a calibrated confidence.

Tile identity:

    Tiles are identified by `tile_id`. The same tile across variants is the
    same row in the merged output. Other fields (bbox, image_url, etc.) are
    taken from the variant where the tile achieved its best rank — that's
    the appearance we trust most.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("findwaldo.fusion")


# Default weights for the composite score. Conservative: max_score is in
# [0, 1] for cosine, so a vote_bonus of 0.05 means an extra vote adds at
# most 5% of the max possible similarity. A rank_bonus of 0.10 with rank=1
# adds 0.10; with rank=30 adds 0.0033 — diminishing fast.
DEFAULT_VOTE_BONUS: float = 0.05
DEFAULT_RANK_BONUS: float = 0.10


def merge_by_score(
    ranked_lists: list[list[dict]],
    *,
    vote_bonus: float = DEFAULT_VOTE_BONUS,
    rank_bonus: float = DEFAULT_RANK_BONUS,
    top_n: Optional[int] = None,
) -> list[dict]:
    """Merge multiple ranked lists of hits by tile_id.

    Each input list is expected to be in rank order (best first). Each hit
    must contain at least a `tile_id` (identity key) and a `score` (the
    per-variant kNN similarity). Other fields (`image_id`, `image_url`,
    `bbox`, etc.) are passed through to the merged output, taken from the
    variant where the tile achieved its best rank.

    Args:
        ranked_lists: One list of hits per source query. Empty lists are
                      allowed and skipped. The outer list itself must be
                      non-empty; the caller should not invoke this if no
                      variants ran.
        vote_bonus:   Additive boost per extra variant beyond the first.
                      A tile that appears in 1 variant gets no bonus; in 3
                      variants it gets `2 * vote_bonus`. Default 0.05.
        rank_bonus:   Coefficient for the (1/best_rank) reward. Larger
                      values weight rank-1 placements more heavily.
                      Default 0.10.
        top_n:        If set, truncate the merged output to this many
                      hits. None means return everything.

    Returns:
        A single ranked list of hits, sorted by composite score descending.
        Each hit includes:

          - all original fields from the variant where the tile achieved
            its best rank (image_id, image_url, bbox, tile_id, ...)
          - "max_score":       float, highest kNN score across variants
          - "votes":           int, number of variants this tile appeared in
          - "best_rank":       int, lowest (best) rank achieved in any variant
          - "composite_score": float, the final ranking key
          - "score":           float, overwritten with composite_score so
                               downstream consumers that sort/display on
                               `score` continue to work unchanged

    Raises:
        ValueError: if `ranked_lists` is empty (caller bug; we don't return
                    an empty list silently because that almost always means
                    upstream had no variants, which is a different error
                    state than "kNN found no matches").
    """
    if not ranked_lists:
        raise ValueError("merge_by_score requires at least one ranked list")
    if vote_bonus < 0 or rank_bonus < 0:
        raise ValueError("vote_bonus and rank_bonus must be >= 0")

    # Per-tile accumulator. We track the full hit dict from the variant
    # where the tile achieved its best rank — that's the appearance we
    # trust most for downstream rendering.
    merged: dict[str, dict] = {}

    for variant_idx, hits in enumerate(ranked_lists):
        for rank, hit in enumerate(hits, start=1):
            tile_id = hit.get("tile_id")
            if not tile_id:
                # Hits with no identity can't be deduplicated. Skip with
                # a warning rather than crashing the whole search.
                log.warning(
                    "merge_by_score: hit missing tile_id in variant %d rank %d; skipping",
                    variant_idx, rank,
                )
                continue

            score = float(hit.get("score", 0.0))
            existing = merged.get(tile_id)

            if existing is None:
                # First time we see this tile. Snapshot its fields and
                # initialize bookkeeping.
                snapshot = dict(hit)
                snapshot["max_score"] = score
                snapshot["votes"] = 1
                snapshot["best_rank"] = rank
                merged[tile_id] = snapshot
            else:
                existing["votes"] += 1
                # Track the strongest single-variant signal.
                if score > existing["max_score"]:
                    existing["max_score"] = score
                # If this variant ranked the tile higher, replace the
                # snapshot with this variant's hit dict so we surface the
                # bbox / fields from the strongest appearance. Tie on rank
                # → keep the earlier snapshot (stable behaviour).
                if rank < existing["best_rank"]:
                    # Preserve bookkeeping fields when overwriting.
                    new_snap = dict(hit)
                    new_snap["max_score"] = existing["max_score"]
                    new_snap["votes"] = existing["votes"]
                    new_snap["best_rank"] = rank
                    merged[tile_id] = new_snap

    # Compute composite score for each tile and write final ranking fields.
    # We do this in a second pass so all bookkeeping is finalized before
    # we compute the composite.
    for snap in merged.values():
        max_score = snap["max_score"]
        votes = snap["votes"]
        best_rank = snap["best_rank"]
        composite = (
            max_score
            + vote_bonus * (votes - 1)
            + rank_bonus * (1.0 / best_rank)
        )
        snap["composite_score"] = composite
        # Overwrite `score` so downstream code that sorts/displays by
        # `score` continues to work unchanged.
        snap["score"] = composite

    # Sort: composite score desc, then votes desc as a deterministic
    # tie-breaker, then best_rank asc.
    out = sorted(
        merged.values(),
        key=lambda h: (h["composite_score"], h["votes"], -h["best_rank"]),
        reverse=True,
    )

    if top_n is not None:
        out = out[:top_n]

    return out


def explain_merge(
    ranked_lists: list[list[dict]],
    merged: list[dict],
    *,
    top_n: int = 5,
) -> dict:
    """Build a small dict describing what merge did, for telemetry.

    Returns sample diagnostic information without holding refs to the full
    candidate lists (those can be large and chatty in logs).
    """
    return {
        "num_variants": len(ranked_lists),
        "candidates_per_variant": [len(lst) for lst in ranked_lists],
        "merged_unique_tiles": len(merged),
        "top_summary": [
            {
                "tile_id": h.get("tile_id"),
                "max_score": round(h.get("max_score", 0.0), 4),
                "votes": h.get("votes", 0),
                "best_rank": h.get("best_rank", 0),
                "composite": round(h.get("composite_score", 0.0), 4),
            }
            for h in merged[:top_n]
        ],
    }
