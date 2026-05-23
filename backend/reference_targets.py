"""
Reference target system.

Some queries refer to specific recurring visual subjects whose appearance
is stable across images. For these, a small set of reference crops gives
the search pipeline a much sharper signal than text alone — the model
encodes the actual visual identity (Waldo's red/white stripe pattern,
specific facial features, characteristic clothing) instead of relying
on a description that may match many distractors.

This module is INTENTIONALLY agnostic about which targets exist. It is
NOT a Waldo-finder. Waldo is the first registered target, but the system
must work identically for Odlaw, Wenda, Wizard Whitebeard, Woof, or any
generic recurring subject we want to add later (a specific brand of
striped umbrella, a custom mascot, etc.). All target identities are data,
not code.

Contract
--------
A target has:

    target_id           stable string identifier ("waldo", "odlaw", ...)
    display_name        human-readable label for logs and (eventual) UI
    aliases             list of strings the user might type to invoke it.
                        Matched case-insensitively against the corrected
                        query (with imperative prefixes like "find" already
                        stripped). The first match wins.
    reference_crop_paths list of file paths on disk; each crop is a tight
                        bounding box around the target. More crops = more
                        robust prototype, but each is one EIS embedding
                        call at startup.
    prototype_vector    The mean of the reference crop embeddings (then
                        L2-normalized). This is the single vector we
                        kNN-search with to find the target in tile space.
                        None when no crops were available — the target is
                        registered but inactive, and the orchestrator
                        falls through to the generic Phase 1 pipeline.

What this module does
---------------------
- At startup: reads `manifest.json` describing each target, loads every
  referenced crop, base64-encodes them, embeds them via the existing EIS
  endpoint, computes a per-target prototype vector by averaging then
  L2-normalizing the reference embeddings.
- At search time: `match_query(text)` returns the matched target (or
  None) for a given query string.
- At search time: `prototype_for(target_id)` returns the prototype
  vector (or None if the target has no crops yet).

What this module does NOT do
----------------------------
- It does NOT run kNN itself. The orchestrator handles search calls;
  this module provides only the prototype vectors to search WITH.
- It does NOT classify queries with an LLM. Matching is exact alias
  string-comparison, deliberately.
- It does NOT modify the Elasticsearch index. Reference crops are
  embedded at startup only; the resulting vectors live in process
  memory. There is no persistent storage of prototype vectors.
- It does NOT silently degrade. If the manifest is malformed, the
  store fails loudly at startup so misconfiguration is obvious.

Operational notes
-----------------
- Reference crops live in `reference_targets/<target_id>/*.{jpg,png,...}`.
  The folder is read at startup and again only when the application
  restarts. Hot-reload is out of scope.
- A target with no crops is still registered (so its aliases are still
  recognized) but its `prototype_vector` is None. The orchestrator
  treats this as "matched but inactive" — alias detection logs the
  match, but only the Phase 1 text pipeline runs.
- Cost at startup: one EIS image-embedding call per crop, plus a
  trivial mean-and-normalize. With 5 targets × 5 crops each = 25 image
  embeddings, this adds roughly 1-2 seconds of one-time startup cost.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("findwaldo.reference_targets")


# Imperative prefixes that should be stripped from a query before alias
# matching. Mirrors the list in query_expansion.py — kept here as a
# private duplicate so this module has no dependency on query_expansion.
# Keeping these in sync is a small operational cost; the alternative
# (importing) creates a tighter coupling than the value justifies.
_IMPERATIVE_PREFIXES: tuple[str, ...] = (
    "show me where", "show me a", "show me an", "show me the", "show me",
    "where is the", "where is a", "where is an", "where is",
    "where's the", "where's a", "where's an", "where's",
    "look for the", "look for a", "look for an", "look for",
    "find me the", "find me a", "find me an", "find me",
    "find the", "find a", "find an", "find",
    "show", "highlight", "circle",
)


@dataclass(frozen=True)
class ReferenceTarget:
    """One registered target.

    Frozen so the orchestrator can pass it around without worrying about
    mutation. The store hands out these instances; downstream code reads
    them and never modifies.
    """
    target_id:            str
    display_name:         str
    aliases:              tuple[str, ...]
    reference_crop_paths: tuple[str, ...]
    # The prototype is a list[float] when computed, None when crops were
    # missing or embedding failed. Distinguishing "registered but inactive"
    # from "not registered" lets the orchestrator log the alias match
    # while still falling through to Phase 1 for the actual search.
    prototype_vector:     Optional[list[float]] = None


class ReferenceTargetStore:
    """In-memory registry of reference targets.

    Construct with a manifest path and a base directory; call
    `await build(es_client, inference_id)` once at startup to embed crops
    and compute prototypes. After build() the store is read-only.

    Built-in safety: if no targets are registered, the store is empty
    and `match_query` always returns None — the orchestrator falls
    through to the existing pipeline with zero added cost.
    """

    def __init__(self, manifest_path: str, base_dir: str):
        self._manifest_path = manifest_path
        self._base_dir = base_dir
        self._targets: list[ReferenceTarget] = []
        # Compiled alias index: lowercased exact alias → target_id. Built
        # in `build()` after we know which targets actually have aliases.
        self._alias_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Construction / startup
    # ------------------------------------------------------------------

    async def build(self, es_client, inference_id: str) -> None:
        """Read the manifest, load and embed crops, compute prototypes.

        Idempotent: calling build() twice replaces the in-memory state with
        a fresh load. No partial state is left if the call fails partway
        through — failures raise without touching `self._targets`.

        Args:
            es_client:    an ElasticClient instance (we only call its
                          `embed_images` method; passing the whole client
                          rather than a callable keeps the interface
                          stable across future ElasticClient changes).
            inference_id: the EIS endpoint id to use for crop embedding.
                          Must be the same endpoint used to embed tiles
                          at ingest time, so the resulting prototype
                          vectors live in the same space as the index.
        """
        if not os.path.exists(self._manifest_path):
            log.info(
                "Reference target manifest not found at %s; "
                "no targets will be registered (this is fine for new deployments).",
                self._manifest_path,
            )
            self._targets = []
            self._alias_index = {}
            return

        manifest = self._load_manifest(self._manifest_path)
        targets = self._build_targets(manifest, self._base_dir)

        if not targets:
            log.info("Manifest at %s declared no targets; store is empty.",
                     self._manifest_path)
            self._targets = []
            self._alias_index = {}
            return

        log.info(
            "Reference target store: loading %d targets, %d total crops",
            len(targets),
            sum(len(t.reference_crop_paths) for t in targets),
        )

        # Embed crops per target. Targets whose crops can't be loaded or
        # whose folder is empty get a None prototype but are still
        # registered (so aliases are still recognized for logging /
        # debugging — we just won't run prototype search for them).
        built: list[ReferenceTarget] = []
        for t in targets:
            prototype = await self._compute_prototype(es_client, inference_id, t)
            built.append(ReferenceTarget(
                target_id=t.target_id,
                display_name=t.display_name,
                aliases=t.aliases,
                reference_crop_paths=t.reference_crop_paths,
                prototype_vector=prototype,
            ))

        # Build the alias index now that we know which targets exist.
        # Aliases are stored normalized (lowercased + whitespace-collapsed).
        # First-write wins on conflicts: if two targets register the same
        # alias, log a warning and keep the first.
        index: dict[str, str] = {}
        for t in built:
            for alias in t.aliases:
                key = _normalize_alias(alias)
                if not key:
                    continue
                if key in index:
                    log.warning(
                        "Duplicate alias %r registered to %r (already on %r); skipping",
                        alias, t.target_id, index[key],
                    )
                    continue
                index[key] = t.target_id

        self._targets = built
        self._alias_index = index

        active = [t for t in built if t.prototype_vector is not None]
        inactive = [t for t in built if t.prototype_vector is None]
        log.info(
            "Reference target store ready: %d active (%s), %d inactive (%s)",
            len(active),  ", ".join(t.target_id for t in active)   or "-",
            len(inactive), ", ".join(t.target_id for t in inactive) or "-",
        )

    @staticmethod
    def _load_manifest(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {path}: expected JSON object, got {type(data).__name__}")
        if "targets" not in data or not isinstance(data["targets"], list):
            raise ValueError(f"Manifest {path}: expected top-level 'targets': [...] array")
        return data

    @staticmethod
    def _build_targets(manifest: dict, base_dir: str) -> list[ReferenceTarget]:
        """Parse the manifest into ReferenceTarget shells (no prototypes yet).

        Each entry is validated for required fields. Crop paths are
        resolved relative to `base_dir`. Missing crops are dropped with a
        warning; a target with zero existing crops still gets through
        (it'll have prototype_vector=None and be inactive).
        """
        out: list[ReferenceTarget] = []
        seen_ids: set[str] = set()
        for entry in manifest["targets"]:
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest target entry is not an object: {entry!r}")
            target_id = entry.get("target_id")
            if not target_id or not isinstance(target_id, str):
                raise ValueError(f"Target entry missing/invalid 'target_id': {entry!r}")
            if target_id in seen_ids:
                raise ValueError(f"Duplicate target_id in manifest: {target_id!r}")
            seen_ids.add(target_id)

            display_name = entry.get("display_name") or target_id
            aliases = tuple(entry.get("aliases", []))
            crop_files = entry.get("reference_crops", [])
            if not isinstance(crop_files, list):
                raise ValueError(
                    f"Target {target_id!r}: 'reference_crops' must be a list"
                )

            resolved_crops: list[str] = []
            for cf in crop_files:
                if not isinstance(cf, str):
                    log.warning("Target %s: skipping non-string crop entry %r",
                                target_id, cf)
                    continue
                # Crops in the manifest are relative to base_dir unless the
                # path is already absolute. Allows the same manifest to work
                # inside Docker containers and on the host.
                full_path = cf if os.path.isabs(cf) else os.path.join(base_dir, cf)
                if not os.path.exists(full_path):
                    log.warning("Target %s: crop file not found: %s",
                                target_id, full_path)
                    continue
                resolved_crops.append(full_path)

            out.append(ReferenceTarget(
                target_id=target_id,
                display_name=display_name,
                aliases=aliases,
                reference_crop_paths=tuple(resolved_crops),
                prototype_vector=None,  # filled in later in build()
            ))
        return out

    @staticmethod
    async def _compute_prototype(
        es_client,
        inference_id: str,
        target: ReferenceTarget,
    ) -> Optional[list[float]]:
        """Embed each crop, average the vectors, L2-normalize.

        Returns None when no crops are available or when embedding fails.
        Failure is logged but does not raise — a single broken target
        should not prevent other targets from loading or the application
        from starting.

        Embedding path: tries the cluster's EIS endpoint first; falls back
        to direct Jina REST API if EIS rejects the image input. Some EIS
        endpoints provisioned with task_type=text_embedding accept text
        input but reject image input even when the underlying model is
        multimodal — in that environment we still want the prototype to
        build, since the resulting embedding lives in the same vector
        space regardless of which path produced it.
        """
        if not target.reference_crop_paths:
            log.info("Target %s has no reference crops; will be inactive",
                     target.target_id)
            return None

        # Load each crop, base64-encode it. The EIS image embedding path
        # accepts the same b64 strings the ingest pipeline uses for tiles.
        b64_crops: list[str] = []
        for path in target.reference_crop_paths:
            try:
                with open(path, "rb") as f:
                    b64_crops.append(base64.b64encode(f.read()).decode("ascii"))
            except OSError as e:
                log.warning("Target %s: failed to read crop %s: %s",
                            target.target_id, path, e)

        if not b64_crops:
            log.warning("Target %s: no crops could be loaded; inactive",
                        target.target_id)
            return None

        # Try EIS first. If the endpoint accepts image input, this is the
        # preferred path — same model, same connector, same vector space
        # as everything else.
        embeddings: Optional[list[list[float]]] = None
        try:
            _, embeddings = await es_client.embed_images(
                inference_id=inference_id,
                tiles_b64=b64_crops,
                batch_size=8,
            )
        except Exception as e:
            # The most common failure mode: EIS endpoint provisioned as
            # task_type=text_embedding rejects image input with a generic
            # 400. We can't probe the endpoint config from here without
            # adding ES client surface, so we just try the fallback path.
            log.warning(
                "Target %s: EIS embedding failed (%s); attempting Jina API fallback",
                target.target_id, type(e).__name__,
            )
            embeddings = None

        # Fallback: call Jina's REST API directly. Same model (jina-clip-v2),
        # same dimensions, same vector space. Requires JINA_API_KEY to be
        # set in the environment — if not, we surface the original failure
        # instead of pretending to recover.
        if not embeddings:
            jina_api_key = os.environ.get("JINA_API_KEY", "")
            if not jina_api_key:
                log.error(
                    "Target %s: EIS rejected image input and no JINA_API_KEY "
                    "is set for fallback. Either provision an EIS endpoint "
                    "that accepts image input, or set JINA_API_KEY in the "
                    "backend environment so the prototype can be built via "
                    "the Jina REST API directly.",
                    target.target_id,
                )
                return None
            try:
                embeddings = await _embed_via_jina_rest(b64_crops, jina_api_key)
                log.info(
                    "Target %s: prototype built via Jina REST fallback "
                    "(%d crops embedded)",
                    target.target_id, len(embeddings),
                )
            except Exception as e:
                log.error(
                    "Target %s: Jina REST fallback also failed (%s); marking inactive",
                    target.target_id, e,
                )
                return None

        if not embeddings:
            log.warning("Target %s: no embeddings returned; inactive",
                        target.target_id)
            return None

        # Mean across crops, then L2-normalize. The mean is the simplest
        # form of "prototype" — it captures the centroid of the target's
        # appearance in vector space. More sophisticated methods (e.g.
        # learning a per-target weighting) are out of scope here.
        dim = len(embeddings[0])
        accum = [0.0] * dim
        for v in embeddings:
            if len(v) != dim:
                log.warning(
                    "Target %s: embedding dim mismatch (expected %d, got %d); "
                    "skipping this crop",
                    target.target_id, dim, len(v),
                )
                continue
            for i in range(dim):
                accum[i] += v[i]
        n = len(embeddings)
        mean = [x / n for x in accum]
        norm = math.sqrt(sum(x * x for x in mean))
        if norm < 1e-12:
            log.warning("Target %s: prototype norm is ~zero; inactive",
                        target.target_id)
            return None
        return [x / norm for x in mean]

    # ------------------------------------------------------------------
    # Query-time interface
    # ------------------------------------------------------------------

    def match_query(self, text: str) -> Optional[ReferenceTarget]:
        """Return the target whose alias the query matches, or None.

        Matching is exact (after normalization): the cleaned query must
        match an alias verbatim. Substring matching would create false
        positives ("a man with red and white stripes" should not match
        the alias "red"), and we aren't doing fuzzy matching this phase.

        Imperative prefixes are stripped before matching, so "find Waldo"
        and "Waldo" both match the alias "waldo".
        """
        if not self._alias_index or not text:
            return None
        cleaned = _strip_imperative(text)
        key = _normalize_alias(cleaned)
        if not key:
            return None
        target_id = self._alias_index.get(key)
        if target_id is None:
            return None
        for t in self._targets:
            if t.target_id == target_id:
                return t
        return None  # defensive; unreachable if build() ran cleanly

    def prototype_for(self, target_id: str) -> Optional[list[float]]:
        """Return the prototype vector for a target, or None if the target
        is not registered or has no crops."""
        for t in self._targets:
            if t.target_id == target_id:
                return t.prototype_vector
        return None

    def all_targets(self) -> list[ReferenceTarget]:
        """Snapshot the registered targets. Useful for diagnostics endpoints."""
        return list(self._targets)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


async def _embed_via_jina_rest(
    b64_images: list[str],
    api_key: str,
) -> list[list[float]]:
    """Embed a list of base64-encoded images via Jina's REST API.

    Used as a fallback when the cluster's EIS inference endpoint rejects
    image input (typical when the endpoint was provisioned as
    task_type=text_embedding rather than as a true multimodal endpoint).
    The resulting vectors live in the same space as EIS-produced ones
    because the underlying model (jina-clip-v2) is identical.

    Uses aiohttp (already a backend dependency for the reranker path).
    Sequential batches of 8 to mirror the EIS code path — small enough
    to avoid any per-request size pressure on Jina, large enough that
    startup cost stays in the single-digit seconds even for many crops.
    """
    import aiohttp  # local import: only needed on the fallback path

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    out: list[list[float]] = []
    batch_size = 8

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for start in range(0, len(b64_images), batch_size):
            batch = b64_images[start:start + batch_size]
            payload = {
                "model": "jina-clip-v2",
                "input": [{"image": b} for b in batch],
            }
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"Jina REST API returned {resp.status}: {body[:300]}"
                    )
                data = await resp.json()
            # Jina returns {data: [{embedding: [...]}, ...]} — same shape
            # for both text and image inputs.
            for item in data.get("data", []):
                emb = item.get("embedding")
                if emb:
                    out.append(emb)

    return out


def _normalize_alias(s: str) -> str:
    """Lowercase + collapse whitespace. Returns "" for empty / whitespace-only."""
    return _WHITESPACE_RE.sub(" ", (s or "").strip().lower())


def _strip_imperative(query: str) -> str:
    """Remove a leading imperative or question prefix from the query.

    Mirrors query_expansion._strip_imperative but kept independent so this
    module has no cross-dependency on the expansion pipeline.
    """
    lower = (query or "").lower()
    for prefix in _IMPERATIVE_PREFIXES:
        if lower.startswith(prefix + " "):
            return query[len(prefix) + 1:].strip()
        if lower == prefix:
            return ""
    return (query or "").strip()
