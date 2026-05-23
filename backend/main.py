"""
Find Waldo — Backend API
========================
FastAPI service that orchestrates multimodal visual search using
Elasticsearch + Jina CLIP v2 via the Open Inference API.

Responsibilities:
  - Accept image uploads
  - Tile images into overlapping patches
  - Generate embeddings for each tile via the EIS inference endpoint
  - Bulk index tiles into Elasticsearch
  - Convert text queries → embeddings → kNN search → bounding boxes
  - Optional: hybrid search combining vector kNN with BM25 over OCR text

The frontend never speaks to Elasticsearch directly. All credentials live
on the server. This is the single trust boundary.
"""

import os
import uuid
import asyncio
import base64
import logging
import time
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from elastic_client import ElasticClient
from tiling import tile_image, ImageTiler
from reranker import JinaReranker
from query_correction import correct_query
from query_expansion import expand_query, ExpansionResult
from fusion import merge_by_score
from reference_targets import ReferenceTargetStore
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
log = logging.getLogger("findwaldo")


# ---------------------------------------------------------------------
# Lifespan: connect to Elastic, ensure inference endpoint and index exist
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Find Waldo backend")
    # ElasticClient handles only Elasticsearch + EIS inference. The reranker
    # path lives in JinaReranker (constructed below) and is the only piece
    # that still talks to Jina's HTTP API directly.
    client = ElasticClient(
        url=settings.es_url,
        api_key=settings.es_api_key,
        verify_certs=settings.es_verify_certs,
    )
    await client.bootstrap(
        inference_id=settings.inference_id,
        model_id=settings.eis_model_id,
        index_name=settings.index_name,
        embedding_dims=settings.embedding_dims,
    )
    app.state.es = client
    app.state.tiler = ImageTiler(
        tile_sizes=settings.tile_sizes,
        overlap=settings.tile_overlap,
        max_dim=settings.max_image_dim,
    )
    # Reranker is optional. When enabled, search pulls more candidates from
    # kNN and the reranker refines them before returning to the client.
    app.state.reranker = None
    if settings.reranker_enabled and settings.jina_api_key:
        app.state.reranker = JinaReranker(
            api_key=settings.jina_api_key,
            static_dir=settings.static_dir,
            url=settings.reranker_url,
            model=settings.reranker_model,
        )
        log.info("Reranker enabled: %s", settings.reranker_model)
    elif settings.reranker_enabled:
        log.warning("RERANKER_ENABLED=true but JINA_API_KEY is not set; reranker disabled")

    log.info(
        "Bootstrapped: index=%s inference=%s tile_sizes=%s overlap=%.0f%% reranker=%s",
        settings.index_name, settings.inference_id,
        list(settings.tile_sizes), settings.tile_overlap * 100,
        "on" if app.state.reranker else "off",
    )

    # Reference target store (Phase 2). When enabled, this loads the
    # manifest at startup, embeds each target's reference crops via the
    # same EIS endpoint used for tile ingest, and computes one prototype
    # vector per target. The store is read-only after build(); search-time
    # access is a dictionary lookup. When disabled, an empty store is
    # installed so all alias lookups return None and the orchestrator
    # falls through to the Phase 1 pipeline unchanged.
    app.state.reference_targets = ReferenceTargetStore(
        manifest_path=settings.reference_targets_manifest,
        base_dir=settings.reference_targets_dir,
    )
    if settings.reference_targets_enabled:
        try:
            await app.state.reference_targets.build(
                es_client=client,
                inference_id=settings.inference_id,
            )
        except Exception as e:
            # Failing the entire app start because reference targets
            # couldn't load is too aggressive — the rest of the search
            # pipeline still works without them. Log the error loudly so
            # the operator notices, then carry on with an empty store.
            log.error(
                "Failed to build reference target store: %s. "
                "Search will fall through to the Phase 1 pipeline for all queries.",
                e,
                exc_info=True,
            )
    else:
        log.info(
            "Reference target system disabled (REFERENCE_TARGETS_ENABLED=false). "
            "All queries use the Phase 1 pipeline."
        )

    yield
    await client.close()
    if app.state.reranker:
        await app.state.reranker.close()


app = FastAPI(
    title="Find Waldo — Multimodal Visual Search",
    description="Elasticsearch + Jina CLIP v2 reference implementation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for the local React dev server. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class IngestResponse(BaseModel):
    image_id: str
    width: int
    height: int
    tiles_indexed: int
    elapsed_ms: int
    inference_ms: int
    indexing_ms: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    image_id: Optional[str] = None
    k: int = Field(default=10, ge=1, le=50)
    num_candidates: int = Field(default=200, ge=10, le=10_000)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hybrid: bool = False


class SearchHit(BaseModel):
    tile_id: str
    image_id: str
    image_url: str
    bbox: dict   # {x, y, w, h} in original image pixels
    score: float
    # Phase 1C: score divided by the top hit's score after thresholding.
    # The top-ranked hit always has 1.0; subsequent hits express their
    # similarity strength as a fraction of the strongest match (e.g. 0.92
    # = 92% of the top score). Useful for UI cues like opacity, bar fill,
    # or a relative-strength badge. Defaults to 1.0 so a deserialized
    # response from an older backend doesn't surface a 0.0 here.
    normalized_score: float = 1.0
    rank: int


class SearchResponse(BaseModel):
    query: str
    corrected_query: Optional[str] = None  # set when input had typos
    # Each correction is [original, corrected] pair. We use list-of-list rather
    # than list-of-tuple because Pydantic v2 serializes tuples inconsistently.
    corrections: list[list[str]] = []
    # Phase 1A: expansion + merge telemetry. Optional fields — absent in
    # responses where expansion is disabled, so existing clients don't break.
    # `expanded_queries[0]` is always the original (corrected) user query when
    # expansion ran; subsequent items are the variants the orchestrator fanned
    # out to. `fusion_strategy` is "score_merge" when N>1 variants ran,
    # None otherwise. The field name is kept generic ("fusion_strategy")
    # rather than "merge_strategy" so the schema is stable across future
    # ranking-method experiments.
    expanded_queries: list[str] = []
    fusion_strategy: Optional[str] = None
    expansion_ms: int = 0
    fusion_ms: int = 0
    # Phase 2: which registered target the query matched, if any. None when
    # no target was matched (the typical case for generic queries) or when
    # the reference target system is disabled.
    matched_target: Optional[str] = None
    hits: list[SearchHit]
    total_candidates: int
    inference_ms: int
    search_ms: int
    rerank_ms: int = 0
    reranked: bool = False
    elapsed_ms: int


class ImageSummary(BaseModel):
    image_id: str
    width: int
    height: int
    tile_count: int
    image_url: str
    label: Optional[str] = None
    uploaded_at: Optional[str] = None


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/api/health")
async def health():
    es: ElasticClient = app.state.es
    return await es.health()


# ---------------------------------------------------------------------
# Ingest: upload an image, tile it, embed every tile, bulk-index
# ---------------------------------------------------------------------

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    label: Optional[str] = Form(None),
):
    """
    The full ingest pipeline in one request.

    1. Read the upload, normalize size
    2. Persist the original under /static/{image_id}.jpg for the frontend
    3. Tile with overlap
    4. Send tiles to Jina CLIP v2 (via the Elastic inference endpoint)
    5. Bulk-index tiles with their bounding boxes and embeddings
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds size limit")

    image_id = uuid.uuid4().hex[:12]
    tiler: ImageTiler = app.state.tiler
    es: ElasticClient = app.state.es

    # 1. Decode + normalize. tile_image returns (full_jpeg_bytes, width, height, tiles)
    full_bytes, width, height, tiles = tiler.tile(raw)

    # 2. Persist original for the frontend's <img>
    static_path = os.path.join(settings.static_dir, f"{image_id}.jpg")
    os.makedirs(settings.static_dir, exist_ok=True)
    with open(static_path, "wb") as f:
        f.write(full_bytes)
    image_url = f"/static/{image_id}.jpg"

    # 3. Embed all tiles in batches via the inference endpoint
    inference_ms, embeddings = await es.embed_images(
        inference_id=settings.inference_id,
        tiles_b64=[t.b64 for t in tiles],
        batch_size=settings.embedding_batch_size,
    )

    # 4. Bulk-index. Each tile carries its `scale` so search can prefer
    #    finer scales when the user query is about small objects.
    docs = [
        {
            "image_id":   image_id,
            "tile_id":    f"{image_id}-s{t.scale}-{t.row}-{t.col}",
            "row":        t.row,
            "col":        t.col,
            "scale":      t.scale,
            "bbox": {
                "x": t.x, "y": t.y, "w": t.w, "h": t.h,
            },
            "image_url":  image_url,
            "image_w":    width,
            "image_h":    height,
            "label":      label,
            "embedding":  emb,
        }
        for t, emb in zip(tiles, embeddings)
    ]
    indexing_ms, indexed = await es.bulk_index(settings.index_name, docs)

    elapsed_ms = inference_ms + indexing_ms
    log.info(
        "Ingested image %s: %dx%d, %d tiles, embed=%dms index=%dms",
        image_id, width, height, indexed, inference_ms, indexing_ms,
    )

    return IngestResponse(
        image_id=image_id,
        width=width,
        height=height,
        tiles_indexed=indexed,
        elapsed_ms=elapsed_ms,
        inference_ms=inference_ms,
        indexing_ms=indexing_ms,
    )


# ---------------------------------------------------------------------
# Search: text → embedding → kNN → bounding boxes
# ---------------------------------------------------------------------

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Search orchestration:

        1. Typo correction on the input query.
        2. (Phase 2) Check whether the corrected query matches a registered
           reference target. If it does and the target has a prototype
           vector, the prototype path is taken in addition to the Phase 1
           text pipeline.
        3. Phase 1 text pipeline:
             - (optional) expand the corrected query into N variants
             - run kNN once per variant in parallel via the EIS endpoint
             - merge per-variant ranked lists by tile_id
        4. (Phase 2) If a target was matched, run kNN with its prototype
           vector in parallel with the text pipeline, then merge the text
           result list and the prototype result list together.
        5. Hand the merged candidates to the reranker.
        6. Apply relative score thresholding and normalize scores.

    Branches that DO NOT trigger the prototype path:
        - Reference target system disabled (settings flag off)
        - Query does not match any registered target alias
        - Matched target has no reference crops (prototype is None)

    The reranker always receives the user's *original* (corrected) query —
    not an expansion variant, not a target identifier — because the
    reranker's job is to align with the user's intent, not with our
    internal recall-broadening steps.
    """
    es: ElasticClient = app.state.es
    reranker: Optional[JinaReranker] = app.state.reranker
    ref_store: ReferenceTargetStore = app.state.reference_targets

    # 1. Typo correction. CLIP is robust to misspellings but obvious typos
    # ("stripped" → "striped") still hurt; correcting them client-side
    # improves recall without any extra inference cost.
    corrected, corrections = correct_query(req.query)
    if corrections:
        log.info("Query corrections: %s → %s (%s)",
                 req.query, corrected, corrections)
    effective_query = corrected

    # 2. (Phase 2) Reference target match. The store handles its own
    # imperative-prefix stripping; we pass the corrected query verbatim.
    matched_target = ref_store.match_query(effective_query)
    matched_target_id: Optional[str] = matched_target.target_id if matched_target else None
    prototype_vector = (
        matched_target.prototype_vector if matched_target else None
    )
    if matched_target:
        log.info(
            "Query matched reference target: %s (%s); prototype %s",
            matched_target.target_id,
            matched_target.display_name,
            "available" if prototype_vector is not None else "missing (running Phase 1 only)",
        )

    # 3. Decide kNN sizing. Used by both the text pipeline and the
    # prototype path — they pull the same number of candidates so the
    # downstream merge weights both lists equally.
    if reranker:
        knn_k = max(req.k, settings.reranker_candidates)
        knn_num_candidates = max(req.num_candidates, settings.reranker_candidates * 4)
    else:
        knn_k = req.k
        knn_num_candidates = req.num_candidates

    # ----- Phase 1 text pipeline (always runs) -------------------------
    t_expand = time.perf_counter()
    if settings.query_expansion_enabled:
        try:
            expansion: ExpansionResult = expand_query(
                effective_query,
                strategy=settings.query_expansion_strategy,
                max_variants=settings.query_expansion_max_variants,
            )
        except NotImplementedError as e:
            raise HTTPException(status_code=500, detail=str(e))
        variants = expansion.variants
        expansion_strategy = expansion.strategy
    else:
        variants = [effective_query]
        expansion_strategy = None
    expansion_ms = int((time.perf_counter() - t_expand) * 1000)

    async def _run_text_variant(qtext: str) -> tuple[int, int, list[dict], int]:
        """Single text-driven kNN call for one expansion variant."""
        return await es.knn_search(
            index=settings.index_name,
            inference_id=settings.inference_id,
            query_text=qtext,
            k=knn_k,
            num_candidates=knn_num_candidates,
            image_id=req.image_id,
            min_score=req.min_score,
            hybrid=req.hybrid,
        )

    async def _run_prototype() -> Optional[tuple[int, int, list[dict], int]]:
        """kNN with the matched target's prototype vector. Returns None if
        no prototype was available so the caller can short-circuit cleanly."""
        if prototype_vector is None:
            return None
        return await es.knn_search_with_vector(
            index=settings.index_name,
            query_vector=prototype_vector,
            k=knn_k,
            num_candidates=knn_num_candidates,
            image_id=req.image_id,
            min_score=req.min_score,
        )

    # 4. Fan out: text variants in parallel, plus the prototype call if
    # applicable. asyncio.gather surfaces the first exception so a single
    # failed leg fails the whole search rather than silently returning a
    # partial / degraded result.
    text_tasks = [_run_text_variant(v) for v in variants]
    proto_task = _run_prototype()
    text_results, proto_result = await asyncio.gather(
        asyncio.gather(*text_tasks),
        proto_task,
    )

    # Per-variant timing maxima represent wall-clock cost. Summing would
    # over-report because the variants ran concurrently with each other
    # AND with the prototype call.
    inference_ms_text = max((r[0] for r in text_results), default=0)
    search_ms_text    = max((r[1] for r in text_results), default=0)
    per_variant_hits  = [r[2] for r in text_results]
    total_text        = max((r[3] for r in text_results), default=0)

    # 5a. Merge text-pipeline variants (Phase 1 step). Keeps the same
    # behaviour as before: single-variant skips the merge and uses the
    # only list as-is.
    fusion_ms = 0
    fusion_strategy: Optional[str] = None
    if len(per_variant_hits) > 1:
        candidates_before_merge = sum(len(lst) for lst in per_variant_hits)
        t_merge = time.perf_counter()
        text_hits = merge_by_score(
            per_variant_hits,
            vote_bonus=settings.merge_vote_bonus,
            rank_bonus=settings.merge_rank_bonus,
            top_n=knn_k,
        )
        fusion_ms += int((time.perf_counter() - t_merge) * 1000)
        fusion_strategy = "score_merge"
        top_breakdown = [
            {
                "tile_id":   h.get("tile_id"),
                "max_score": round(h.get("max_score", 0.0), 4),
                "votes":     h.get("votes", 0),
                "best_rank": h.get("best_rank", 0),
                "composite": round(h.get("composite_score", 0.0), 4),
            }
            for h in text_hits[:5]
        ]
        log.info(
            "Search expansion: query=%r variants=%d (%s) "
            "candidates_before_merge=%d candidates_after_merge=%d top=%s",
            effective_query, len(variants), variants,
            candidates_before_merge, len(text_hits), top_breakdown,
        )
    else:
        text_hits = per_variant_hits[0] if per_variant_hits else []

    # 5b. (Phase 2) Merge text-pipeline hits with prototype-pipeline hits
    # if a prototype search ran. We re-use `merge_by_score` because the
    # signal is the same shape: two ranked lists of hit dicts keyed by
    # tile_id. Tiles that show up in BOTH lists get a vote bonus, which
    # is exactly the behaviour we want — cross-method agreement on a tile
    # is a stronger positive signal than either method alone.
    inference_ms = inference_ms_text
    search_ms = search_ms_text
    total = total_text
    if proto_result is not None:
        proto_inference_ms, proto_search_ms, proto_hits, proto_total = proto_result
        # Wall-clock max across the two parallel paths.
        inference_ms = max(inference_ms, proto_inference_ms)
        search_ms = max(search_ms, proto_search_ms)
        total = max(total, proto_total)

        if text_hits or proto_hits:
            t_merge2 = time.perf_counter()
            merged_hits = merge_by_score(
                [text_hits, proto_hits],
                vote_bonus=settings.merge_vote_bonus,
                rank_bonus=settings.merge_rank_bonus,
                top_n=knn_k,
            )
            fusion_ms += int((time.perf_counter() - t_merge2) * 1000)
            # Tag the strategy so the response makes the path obvious.
            fusion_strategy = "score_merge+prototype"
            log.info(
                "Reference-target merge: target=%s text_hits=%d proto_hits=%d "
                "merged=%d top=%s",
                matched_target_id, len(text_hits), len(proto_hits),
                len(merged_hits),
                [
                    {
                        "tile_id":   h.get("tile_id"),
                        "max_score": round(h.get("max_score", 0.0), 4),
                        "votes":     h.get("votes", 0),
                        "best_rank": h.get("best_rank", 0),
                        "composite": round(h.get("composite_score", 0.0), 4),
                    }
                    for h in merged_hits[:5]
                ],
            )
            raw_hits = merged_hits
        else:
            raw_hits = []
    else:
        raw_hits = text_hits

    # 6. Optional reranker refinement. The reranker sees the user's query,
    # not the expansion variants — alignment with intent matters at this
    # stage, recall has already been broadened upstream.
    rerank_ms = 0
    reranked_flag = False
    if reranker and raw_hits:
        rerank_ms, raw_hits = await reranker.rerank(
            query=effective_query,
            candidates=raw_hits,
            top_k=req.k,
        )
        reranked_flag = True
    else:
        # Truncate to req.k if no reranker
        raw_hits = raw_hits[: req.k]

    # 7. Relative score thresholding (Phase 1C / 1D).
    #
    # Order of operations (Phase 1D fix):
    #   a) Identify top_score from the final ranked list (post-reranker if
    #      reranker ran, post-merge / single-variant kNN otherwise).
    #   b) Compute normalized_score = score / top_score for every hit.
    #      The rank-1 hit always reads 1.0 by construction.
    #   c) Filter to keep only hits with normalized_score >= threshold_ratio.
    #   d) Do NOT recompute normalization after filtering. The normalized
    #      values are an absolute property of each hit relative to the
    #      strongest match the system found, not a property of the
    #      surviving subset.
    #
    # Why this order matters: `normalized_score` should be stable across
    # threshold changes. If a future filter could drop the top hit (e.g.,
    # an absolute score floor), normalizing against the surviving top
    # would silently rebase. Computing it once against the original top
    # keeps the field semantically clean.
    #
    # The filter is not padded back up to req.k. If only 2 strong matches
    # remain, we return 2. Setting threshold_ratio to 0.0 disables the
    # filter; every hit passes (each carrying its normalized_score).
    threshold_ratio = settings.score_threshold_ratio
    hits_before_threshold = len(raw_hits)

    # (a) + (b): identify top_score and normalize all hits in one pass.
    if raw_hits:
        top_score = raw_hits[0]["score"]
        for h in raw_hits:
            if top_score > 0:
                h["normalized_score"] = h["score"] / top_score
            else:
                # Degenerate: top_score <= 0. Don't divide; surface 1.0
                # for the top hit and 0.0 for the rest so the UI doesn't
                # render NaN/Inf badges.
                h["normalized_score"] = 1.0 if h is raw_hits[0] else 0.0

    # (c): filter on normalized_score. With top_score > 0, the rank-1 hit
    # has normalized_score = 1.0 and always passes any threshold in [0, 1].
    # With top_score <= 0, every hit was assigned 1.0 or 0.0; we leave the
    # list untouched in that degenerate case to avoid returning an empty
    # response purely because of a pathological score distribution.
    if raw_hits and threshold_ratio > 0 and raw_hits[0]["score"] > 0:
        raw_hits = [h for h in raw_hits if h["normalized_score"] >= threshold_ratio]

    # (d) is implicit: we do not touch normalized_score after the filter.

    # (e) Phase 3-tuning: hard top-score floor.
    #
    # If the top hit's raw score is below `min_top_score_to_return`, the
    # system has nothing credible to say about this query. Returning zero
    # hits is more honest than padding the response with low-confidence
    # tiles. This is the single highest-leverage filter for absent-target
    # queries ("a pink dragon", "a 1970s Cadillac") whose top scores tend
    # to land well below those of legitimate matches.
    #
    # Operates on the rank-1 hit's score, not normalized_score, because
    # the cutoff is calibrated against the absolute score distribution
    # observed in the Phase 3 baseline.
    if (
        raw_hits
        and settings.min_top_score_to_return > 0
        and raw_hits[0]["score"] < settings.min_top_score_to_return
    ):
        log.info(
            "Top-score floor (%.2f): dropping all %d hits, top_score=%.4f",
            settings.min_top_score_to_return, len(raw_hits), raw_hits[0]["score"],
        )
        raw_hits = []

    # (f) Phase 3-tuning: per-hit absolute score floor.
    #
    # Layers underneath the relative threshold and the top-score floor. A
    # noise gate that catches individual weak hits the relative threshold
    # missed because the entire result set was tightly clustered. Default
    # 0.0 makes this a no-op for backwards compatibility.
    if raw_hits and settings.min_absolute_score > 0:
        before = len(raw_hits)
        raw_hits = [h for h in raw_hits if h["score"] >= settings.min_absolute_score]
        if before != len(raw_hits):
            log.info(
                "Absolute score floor (%.2f): %d → %d hits",
                settings.min_absolute_score, before, len(raw_hits),
            )

    if hits_before_threshold != len(raw_hits):
        log.info(
            "Score threshold (ratio=%.2f): %d → %d hits (top_score=%.4f)",
            threshold_ratio,
            hits_before_threshold,
            len(raw_hits),
            raw_hits[0]["score"] if raw_hits else 0.0,
        )

    hits = [
        SearchHit(
            tile_id=h["tile_id"],
            image_id=h["image_id"],
            image_url=h["image_url"],
            bbox=h["bbox"],
            score=h["score"],
            normalized_score=h.get("normalized_score", 1.0),
            rank=i + 1,
        )
        for i, h in enumerate(raw_hits)
    ]

    return SearchResponse(
        query=req.query,
        corrected_query=effective_query if corrections else None,
        corrections=[[orig, fixed] for orig, fixed in corrections],
        expanded_queries=variants if expansion_strategy else [],
        fusion_strategy=fusion_strategy,
        expansion_ms=expansion_ms,
        fusion_ms=fusion_ms,
        matched_target=matched_target_id,
        hits=hits,
        total_candidates=total,
        inference_ms=inference_ms,
        search_ms=search_ms,
        rerank_ms=rerank_ms,
        reranked=reranked_flag,
        elapsed_ms=inference_ms + search_ms + rerank_ms + expansion_ms + fusion_ms,
    )


# ---------------------------------------------------------------------
# Image management
# ---------------------------------------------------------------------

@app.get("/api/images", response_model=list[ImageSummary])
async def list_images():
    es: ElasticClient = app.state.es
    rows = await es.list_images(settings.index_name)
    return [ImageSummary(**r) for r in rows]


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    es: ElasticClient = app.state.es
    deleted = await es.delete_image(settings.index_name, image_id)
    static_path = os.path.join(settings.static_dir, f"{image_id}.jpg")
    if os.path.exists(static_path):
        os.remove(static_path)
    return {"image_id": image_id, "tiles_deleted": deleted}


# ---------------------------------------------------------------------
# Phase 2 debug endpoint  — TEMPORARY, diagnostic only
# ---------------------------------------------------------------------
#
# /api/debug/compare exposes the three intermediate result lists that
# Phase 2's orchestrator merges together:
#
#   1. text_results       — what the Phase 1 text pipeline produces
#                           (expansion + per-variant kNN + variant merge)
#   2. prototype_results  — what kNN against the prototype vector alone
#                           returns
#   3. merged_results     — what merge_by_score([text, prototype]) returns,
#                           which is what production hands to the reranker
#
# This endpoint INTENTIONALLY differs from /api/search in two ways:
#   - reranker is NOT applied (it would reorder by query alignment and
#     hide the merge structure under reranker scores)
#   - score threshold is NOT applied (we want the full distribution so
#     the operator can see where the score cliff is)
#
# Everything else — retrieval, merge logic, expansion behaviour — uses
# the same code paths as production. This is observation, not a parallel
# implementation.
#
# Requires REFERENCE_TARGETS_ENABLED=true at startup so the store has
# been built and prototype vectors are in memory. Without that, the
# endpoint cannot do its job and returns 400 with a clear remediation
# message.
#
# Remove this section when Phase 2 tuning is complete.
# ---------------------------------------------------------------------

class DebugCompareRequest(BaseModel):
    target_id: str = Field(..., min_length=1, max_length=64)
    image_id:  Optional[str] = None
    # Mirrors /api/search defaults so apples-to-apples comparison is easy.
    k:              int = Field(default=10,  ge=1, le=50)
    num_candidates: int = Field(default=200, ge=10, le=10_000)


class DebugCompareHit(BaseModel):
    tile_id:          str
    image_id:         str
    image_url:        str
    bbox:             dict
    score:            float
    # Per-list normalization: each list is normalized against ITS OWN top
    # score, so the rank-1 hit in each of the three lists reads 1.0. This
    # makes the three lists directly comparable rank-by-rank without
    # coupling them to a single global top.
    normalized_score: float
    rank:             int
    # "text", "prototype", or "merged" — indicates which list this hit
    # came from. The same tile_id may appear in multiple lists with
    # different scores; that's the whole point of the comparison.
    source:           str
    # Merge bookkeeping. Populated only on entries from the merged list,
    # since these are the only fields that exist for merged candidates.
    # `votes` is the number of source lists that contributed this tile,
    # `best_rank` is the lowest rank it achieved across them, and
    # `composite_score` is the value that drove its position in the
    # merged ranking. None for text-only and prototype-only entries.
    votes:           Optional[int]   = None
    best_rank:       Optional[int]   = None
    composite_score: Optional[float] = None


class DebugCompareResponse(BaseModel):
    target_id:           str
    image_id:            Optional[str]
    # True when the matched target has a prototype vector loaded in
    # memory. False when the target is registered (alias matching works)
    # but no crops were available at startup. The endpoint short-circuits
    # with a 400 when the target isn't registered at all — there's nothing
    # useful it can produce in that case.
    prototype_available: bool
    # The aliases registered for this target. Surfaced so the operator
    # can see WHICH alias choices would route a real query through the
    # production prototype path.
    aliases:             list[str]
    # The synthetic query string used to drive the text pipeline. We use
    # the target's display name (lowercased) so the text leg is asking
    # "can text retrieval find this target by name?" — that's the
    # comparison that's actually informative.
    query_used_for_text: str
    # When expansion is enabled (production setting), this lists every
    # variant the text pipeline ran. Empty list when expansion is off.
    text_expansion_variants: list[str]
    # Did the reranker run? Always false here — documented for clarity
    # so debug consumers can't accidentally read these scores as
    # reranker scores.
    reranker_applied:    bool
    # The three result lists. Each is independently sorted by score and
    # capped at `k`.
    text_results:        list[DebugCompareHit]
    prototype_results:   list[DebugCompareHit]
    merged_results:      list[DebugCompareHit]
    # Aggregate counts for at-a-glance comparison.
    summary:             dict


@app.post("/api/debug/compare", response_model=DebugCompareResponse)
async def debug_compare(req: DebugCompareRequest):
    """Run text-only, prototype-only, and merged retrieval side-by-side
    against the same target/image so the operator can see exactly what
    each leg contributes to the production ranking.

    No reranker, no thresholding — raw retrieval and raw merge only.
    See module-level comments above for rationale.
    """
    es: ElasticClient = app.state.es
    ref_store: ReferenceTargetStore = app.state.reference_targets

    # Reference-target system must have been built at startup. Without
    # it the store is empty and there's nothing to compare against.
    if not settings.reference_targets_enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                "Reference target system is disabled. Set "
                "REFERENCE_TARGETS_ENABLED=true in backend/.env, ensure "
                "the manifest exists, and restart the backend."
            ),
        )

    # Locate the target. We use prototype_for() and a separate aliases
    # walk because the public store API doesn't expose a get-by-id; the
    # alternative would be to add one, but that's API surface I don't
    # want to grow for a temporary endpoint.
    target = next(
        (t for t in ref_store.all_targets() if t.target_id == req.target_id),
        None,
    )
    if target is None:
        registered_ids = [t.target_id for t in ref_store.all_targets()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Target {req.target_id!r} is not registered. "
                f"Registered targets: {registered_ids}"
            ),
        )

    prototype_vector = target.prototype_vector
    prototype_available = prototype_vector is not None

    # Build the text query the same way a user would type it. Using the
    # display_name lowercased gives a representative natural-language
    # query without smuggling in alias-specific phrasing that might
    # accidentally trigger production target matching during downstream
    # tests. (This endpoint doesn't go through ref_store.match_query;
    # the comparison is structural, not behavioural.)
    query_used_for_text = target.display_name.lower()

    # ----- Text leg --------------------------------------------------
    # Mirror the Phase 1 text path: expansion (if enabled) + per-variant
    # kNN + variant merge. We re-use the exact same orchestration logic
    # the production handler uses, just without the surrounding reranker
    # and threshold steps.
    if settings.query_expansion_enabled:
        try:
            expansion: ExpansionResult = expand_query(
                query_used_for_text,
                strategy=settings.query_expansion_strategy,
                max_variants=settings.query_expansion_max_variants,
            )
        except NotImplementedError as e:
            raise HTTPException(status_code=500, detail=str(e))
        variants = expansion.variants
    else:
        variants = [query_used_for_text]

    async def _text_variant(qtext: str):
        return await es.knn_search(
            index=settings.index_name,
            inference_id=settings.inference_id,
            query_text=qtext,
            k=req.k,
            num_candidates=req.num_candidates,
            image_id=req.image_id,
            min_score=0.0,
            hybrid=False,
        )

    async def _prototype_call():
        if not prototype_available:
            return None
        return await es.knn_search_with_vector(
            index=settings.index_name,
            query_vector=prototype_vector,
            k=req.k,
            num_candidates=req.num_candidates,
            image_id=req.image_id,
            min_score=0.0,
        )

    text_results, proto_result = await asyncio.gather(
        asyncio.gather(*[_text_variant(v) for v in variants]),
        _prototype_call(),
    )
    per_variant_hits = [r[2] for r in text_results]

    if len(per_variant_hits) > 1:
        text_hits = merge_by_score(
            per_variant_hits,
            vote_bonus=settings.merge_vote_bonus,
            rank_bonus=settings.merge_rank_bonus,
            top_n=req.k,
        )
    else:
        text_hits = per_variant_hits[0] if per_variant_hits else []

    # ----- Prototype leg --------------------------------------------
    proto_hits = proto_result[2] if proto_result is not None else []

    # ----- Cross-method merge ---------------------------------------
    # Same merge function as production. When the prototype list is
    # empty (target is registered but inactive) the merged list is
    # identical to the text list.
    if text_hits or proto_hits:
        merged_hits = merge_by_score(
            [text_hits, proto_hits],
            vote_bonus=settings.merge_vote_bonus,
            rank_bonus=settings.merge_rank_bonus,
            top_n=req.k,
        )
    else:
        merged_hits = []

    # ----- Format the three lists for response ----------------------
    # Per-list normalization: each list is normalized against its OWN
    # top score so rank-1 always reads 1.0. This is intentionally
    # different from the production single-top normalization — here we
    # WANT the lists to be directly comparable on equal footing.
    def _format(raw_list: list[dict], source: str) -> list[DebugCompareHit]:
        if not raw_list:
            return []
        top = raw_list[0].get("score", 0.0)
        out: list[DebugCompareHit] = []
        for i, h in enumerate(raw_list):
            score = float(h.get("score", 0.0))
            if top > 0:
                normalized = score / top
            else:
                normalized = 1.0 if i == 0 else 0.0
            out.append(DebugCompareHit(
                tile_id=h["tile_id"],
                image_id=h["image_id"],
                image_url=h["image_url"],
                bbox=h["bbox"],
                score=score,
                normalized_score=normalized,
                rank=i + 1,
                source=source,
                # Merge bookkeeping is only meaningful for the merged list.
                votes=h.get("votes")           if source == "merged" else None,
                best_rank=h.get("best_rank")   if source == "merged" else None,
                composite_score=(
                    round(h["composite_score"], 6)
                    if source == "merged" and "composite_score" in h
                    else None
                ),
            ))
        return out

    text_formatted   = _format(text_hits, "text")
    proto_formatted  = _format(proto_hits, "prototype")
    merged_formatted = _format(merged_hits, "merged")

    # Overlap = tiles present in BOTH the text list and the prototype
    # list (independent of merge). Useful for seeing how much agreement
    # the two methods have on a per-query basis.
    text_ids  = {h.tile_id for h in text_formatted}
    proto_ids = {h.tile_id for h in proto_formatted}
    overlap_count = len(text_ids & proto_ids)

    # Also log a one-line summary so the operator can grep backend logs
    # alongside the response payload during debugging sessions.
    log.info(
        "debug_compare: target=%s image_id=%s prototype_available=%s "
        "text=%d prototype=%d merged=%d overlap=%d",
        req.target_id, req.image_id, prototype_available,
        len(text_formatted), len(proto_formatted),
        len(merged_formatted), overlap_count,
    )

    return DebugCompareResponse(
        target_id=req.target_id,
        image_id=req.image_id,
        prototype_available=prototype_available,
        aliases=list(target.aliases),
        query_used_for_text=query_used_for_text,
        text_expansion_variants=variants if len(variants) > 1 else [],
        reranker_applied=False,
        text_results=text_formatted,
        prototype_results=proto_formatted,
        merged_results=merged_formatted,
        summary={
            "text_count":      len(text_formatted),
            "prototype_count": len(proto_formatted),
            "merged_count":    len(merged_formatted),
            "overlap_count":   overlap_count,
        },
    )


# Serve original images so the frontend can draw bounding boxes over them
os.makedirs(settings.static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
