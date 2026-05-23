"""
Centralized configuration for the Find Waldo backend.

All environment-driven settings are declared here so that no other module
reaches into os.environ. Frozen dataclass for immutability after import.

Inference policy:
    Image ingest and search-time text embedding both go through the same
    Elastic Inference Service (EIS) endpoint:

        PUT _inference/text_embedding/{INFERENCE_ID}
        { "service": "elastic",
          "service_settings": { "model_id": "{EIS_MODEL_ID}" } }

    The endpoint is provisioned operationally (Kibana / IaC), not by the
    application. The application only reads from it. There are no external
    API keys for the embedding path; EIS hosts the model.

    A single endpoint serves both modalities because CLIP-class models are
    dual encoders that share a vector space across text and image. The same
    vectors get produced regardless of which side originated them, so cosine
    similarity between a query vector and a tile vector is well-defined
    end-to-end.

Reranker policy (separate from the embedding path):
    The reranker (jina-reranker-m0) is currently called via the Jina HTTP
    API directly from `reranker.py`. That path still exists pending its own
    migration to EIS in a later phase. The Jina-side settings below
    (jina_api_key / reranker_url / reranker_model) belong to the reranker
    only — they are NOT used by the image-search embedding path.

A note on scoring:
    Elasticsearch kNN _score values are NOT probabilities and should never
    be displayed as percentages. For cosine similarity in Elasticsearch,
    `_score = (1 + cosine) / 2`, which lands in [0, 1] but is a *similarity*
    score, not a confidence. A score of 0.92 does not mean "92% likely to
    be correct"; it means "this vector is closer to the query than a vector
    scoring 0.85". Treat scores as ordinal, not as calibrated probabilities.
"""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # --- Elasticsearch -------------------------------------------------
    es_url:          str  = os.environ.get("ES_URL", "http://localhost:9200")
    es_api_key:      str  = os.environ.get("ES_API_KEY", "")
    es_verify_certs: bool = os.environ.get("ES_VERIFY_CERTS", "true").lower() == "true"

    # --- Inference (Elastic Inference Service) -------------------------
    # The id of the cluster's EIS endpoint. Used by query_vector_builder
    # at search time to embed the query text inside the kNN call, and by
    # the ingest path to embed image tiles. The endpoint must be backed
    # by a multimodal CLIP-class model so query vectors and image vectors
    # share a space.
    inference_id: str = os.environ.get("INFERENCE_ID", "eis-jina-clip-v2")
    # The EIS-side model id. Informational here — the cluster determines
    # the actual model from the endpoint configuration. Surfaced for log
    # readability and for the bootstrap probe's mismatch warning.
    eis_model_id: str = os.environ.get("EIS_MODEL_ID", "jina-clip-v2")

    # --- Index ---------------------------------------------------------
    index_name:     str = os.environ.get("INDEX_NAME", "wheres-waldo-tiles")
    embedding_dims: int = int(os.environ.get("EMBEDDING_DIMS", "1024"))

    # --- Tiling --------------------------------------------------------
    # Multi-scale tiling. CLIP-class models can't infer scale from a query —
    # "a striped shirt" could refer to a 30px Waldo or a 700px giant. Indexing
    # tiles at multiple sizes lets kNN pick the right scale automatically.
    # Comma-separated list of tile sizes in pixels.
    #   224 — fine scale, catches Waldo-sized objects
    #   384 — medium scale, catches character/prop sized objects
    # Adding 768 catches scene-level context but multiplies per-image tile
    # count significantly; opt in via TILE_SIZES=224,384,768.
    tile_sizes:           tuple = field(default_factory=lambda: tuple(
        int(x.strip()) for x in os.environ.get("TILE_SIZES", "224,384").split(",")
        if x.strip()
    ))
    # Backwards-compat single tile_size (used as fallback when TILE_SIZES is
    # not set). Read from env directly — supports older deployments.
    tile_size:            int   = int(os.environ.get("TILE_SIZE", "384"))
    # 20% overlap is right for multi-scale tiling: the multiple scales
    # already cover the "object straddles a tile boundary" case from
    # different angles, so within each scale we don't need much overlap.
    # Higher overlap (33%, 50%) only makes sense for single-scale tiling
    # where each tile must catch every possible object position alone.
    tile_overlap:         float = float(os.environ.get("TILE_OVERLAP", "0.20"))
    max_image_dim:        int   = int(os.environ.get("MAX_IMAGE_DIM", "3072"))
    max_upload_bytes:     int   = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    embedding_batch_size: int   = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))

    # --- Reranker (Jina-direct, scope of a future phase) ---------------
    # The reranker still calls Jina's HTTP API directly. Its migration to
    # an EIS-mediated path is a separate piece of work; until then, these
    # settings remain so reranker.py continues to function.
    reranker_enabled:    bool  = os.environ.get("RERANKER_ENABLED", "true").lower() == "true"
    reranker_model:      str   = os.environ.get("RERANKER_MODEL", "jina-reranker-m0")
    reranker_url:        str   = os.environ.get("RERANKER_URL", "https://api.jina.ai/v1/rerank")
    reranker_candidates: int   = int(os.environ.get("RERANKER_CANDIDATES", "30"))
    # Used only by the reranker. NOT used by the embedding path (which goes
    # through EIS and requires no external key).
    jina_api_key:        str   = os.environ.get("JINA_API_KEY", "")

    # --- Query prompt ensemble (legacy flag, unused) -----------------
    # Originally introduced for vector-averaging across prompt variants;
    # never wired in. Kept for backward compatibility with any environment
    # files that may set it. The new query expansion path (Phase 1A) is
    # controlled by `query_expansion_enabled` below and does NOT average
    # vectors — it merges ranked result lists by tile_id with vote count
    # and best-rank bookkeeping.
    query_ensemble_enabled: bool  = os.environ.get("QUERY_ENSEMBLE", "false").lower() == "true"

    # --- Query expansion + score merge (Phase 1A) --------------------
    # When enabled, the search orchestrator expands the user's query into
    # multiple variants, runs kNN once per variant in parallel, and merges
    # the ranked candidate lists by tile_id before passing the merged list
    # to the reranker.
    #
    # The merge is score-based, not vector-based: per tile we track the
    # max kNN score across variants, the number of variants the tile
    # appeared in (votes), and the best rank achieved. The composite
    # ranking score is:
    #     composite = max_score
    #               + merge_vote_bonus * (votes - 1)
    #               + merge_rank_bonus * (1 / best_rank)
    #
    # Off by default so A/B comparisons against the single-query path are
    # easy. Flip on by setting QUERY_EXPANSION_ENABLED=true.
    query_expansion_enabled: bool = os.environ.get("QUERY_EXPANSION_ENABLED", "false").lower() == "true"
    # Strategy: "rule" (deterministic templates, zero-cost) or "llm"
    # (not yet implemented — will raise if selected).
    query_expansion_strategy: str = os.environ.get("QUERY_EXPANSION_STRATEGY", "rule")
    # Hard cap on variants per search, including the original query. Each
    # variant is one extra kNN call against the cluster, so this directly
    # bounds the per-search inference cost. 4 is a good balance: original
    # plus character expansion plus 2-3 prompt templates.
    query_expansion_max_variants: int = int(os.environ.get("QUERY_EXPANSION_MAX_VARIANTS", "4"))
    # Composite-score weights. Conservative defaults: ranking is mostly
    # driven by raw similarity (max_score in [0, 1] for cosine). Vote bonus
    # of 0.05 means each extra cross-variant vote adds 5% of the max
    # similarity. Rank bonus of 0.10 with rank=1 adds 0.10 to the composite.
    merge_vote_bonus: float = float(os.environ.get("MERGE_VOTE_BONUS", "0.05"))
    merge_rank_bonus: float = float(os.environ.get("MERGE_RANK_BONUS", "0.10"))

    # --- Relative score thresholding (Phase 1C) ----------------------
    # After the final ranked list is produced (post-reranker if reranker
    # is on, post-merge otherwise), drop any hit whose score is less than
    # `score_threshold_ratio * top_score`. This trims weak matches that the
    # system itself ranks far below its strongest hit, instead of padding
    # the response to a fixed size with low-quality results.
    #
    # 0.80 is conservative: at the default, the surviving hits are within
    # 20% of the top score's value. Tighten to 0.90 for stricter precision
    # (more aggressive trimming), loosen to 0.50 for higher recall, or set
    # to 0.0 to disable thresholding entirely (every hit returned, padded
    # only by the reranker / merge top_n).
    #
    # The threshold operates on whichever score is final — reranker score
    # if reranker ran, otherwise merge composite (expansion mode) or raw
    # kNN cosine (single-query mode). It is the user-visible score, so
    # "within 20% of the top score" is a uniform user-facing semantic.
    score_threshold_ratio: float = float(os.environ.get("SCORE_THRESHOLD_RATIO", "0.80"))

    # Absolute score floor applied AFTER relative thresholding. Any hit
    # whose final score is below this value is dropped, regardless of
    # how it ranks relative to the top hit in the same query.
    #
    # The relative threshold (above) only filters within a query; if every
    # hit is bad, every hit still passes. The absolute floor catches that
    # case. From the Phase 3 baseline, legitimate-target top scores all
    # land >= 0.83, while absent-target top scores cap at 0.74. A floor
    # at 0.55 is a conservative noise gate; a floor at 0.75 is closer to
    # "only return high-confidence matches".
    #
    # Default 0.0 = disabled (preserves byte-for-byte previous behaviour).
    min_absolute_score: float = float(os.environ.get("MIN_ABSOLUTE_SCORE", "0.0"))

    # Hard floor on the top-1 hit's score. If the strongest hit a query
    # can produce is below this value, return zero hits — the system is
    # signalling that nothing in the corpus is a credible match for the
    # query, and it's better to say nothing than to surface noise.
    #
    # Tuned from Phase 3 baseline where absent queries (pink dragon,
    # 1970s Cadillac, black labrador) topped out at 0.587–0.744 while
    # legitimate Waldo queries scored 0.832–0.929. A floor at 0.75 sits
    # cleanly between those two distributions.
    #
    # Default 0.0 = disabled (preserves byte-for-byte previous behaviour).
    min_top_score_to_return: float = float(os.environ.get("MIN_TOP_SCORE_TO_RETURN", "0.0"))

    # --- Reference target system (Phase 2) ---------------------------
    # When enabled, the search orchestrator checks whether the user query
    # matches a registered target alias. If yes, it runs the existing
    # Phase 1 text pipeline AND a kNN search using the target's prototype
    # vector, then merges the two result lists before reranking.
    #
    # When the query does NOT match any alias, behaviour is byte-for-byte
    # identical to Phase 1. Generic queries like "red umbrella" or
    # "woman in blue" never touch the prototype path.
    #
    # Off by default during initial rollout. Flip on by setting
    # REFERENCE_TARGETS_ENABLED=true, then provide a manifest at
    # `reference_targets/manifest.json` listing the targets and their
    # crop file paths.
    reference_targets_enabled: bool = (
        os.environ.get("REFERENCE_TARGETS_ENABLED", "false").lower() == "true"
    )
    # Manifest path describing the registered targets. JSON file with the
    # shape:
    #   { "targets": [
    #       { "target_id":      "waldo",
    #         "display_name":   "Waldo",
    #         "aliases":        ["waldo", "find waldo", "striped guy"],
    #         "reference_crops": ["waldo/crop1.jpg", "waldo/crop2.jpg"]
    #       },
    #       ...
    #     ]
    #   }
    # Crop paths are relative to `reference_targets_dir` unless absolute.
    reference_targets_manifest: str = os.environ.get(
        "REFERENCE_TARGETS_MANIFEST", "./reference_targets/manifest.json",
    )
    reference_targets_dir: str = os.environ.get(
        "REFERENCE_TARGETS_DIR", "./reference_targets",
    )

    # --- Storage -------------------------------------------------------
    static_dir: str = os.environ.get("STATIC_DIR", "./static")

    # --- CORS ----------------------------------------------------------
    allowed_origins: tuple = field(default_factory=lambda: tuple(
        os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    ))


settings = Settings()
