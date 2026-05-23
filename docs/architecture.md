# Architecture

This document describes the request lifecycle and the reasoning behind each
stage of the pipeline. For the inference-layer war story (why image
embedding takes a different path than text), see
[`engineering-notes.md`](engineering-notes.md). For how the relevance
thresholds were chosen, see [`relevance-tuning.md`](relevance-tuning.md).

## Design goal

Given a single busy image and a natural-language query, return bounding
boxes over the regions of the image that match the query — using
multimodal embeddings rather than a trained object detector, so the system
generalizes to arbitrary query text and arbitrary subjects without
per-class training.

## The trust boundary

The browser talks only to the FastAPI backend. The backend is the only
component holding cluster credentials or a model-provider key. The browser
never receives a query vector, an API key, or a direct cluster URL. This
keeps the security surface to one service and one set of secrets.

## Ingest lifecycle

```
image upload
   │
   ▼
multi-scale overlapping tiler (Pillow)
   │   produces tiles at multiple sizes (e.g. 224, 384) with overlap so a
   │   subject straddling a tile boundary still lands whole in some tile
   ▼
embed each tile  ── EIS first, Jina REST fallback on image-input rejection
   │   tiles are base64 JPEG; output is 1024-d CLIP vectors, L2-normalized
   ▼
bulk index into Elasticsearch
       dense_vector(1024, cosine, HNSW) + bbox{x,y,w,h} + tile/image ids
```

Multi-scale tiling matters because the right tile size depends on how large
the subject is in the frame. A single fixed tile size either chops large
subjects or buries small ones in clutter. Overlap (default 33%) ensures a
subject near a tile seam still appears intact in at least one tile.

Tiles are stored as **vectors only** — the pixels are not kept in
Elasticsearch. The original full-size image is kept on disk (or object
storage) so the rerank stage can re-crop candidate tiles on demand.

## Search lifecycle

```
user query  ("find Waldo")
   │
   ▼
query expansion  ── 1 query → N CLIP-friendly visual phrasings
   │   "find Waldo" → ["find Waldo",
   │                    "a man in a red and white striped shirt",
   │                    "a person with glasses in a busy scene", …]
   ▼
kNN per variant  ── text embedded INSIDE the cluster via
   │               query_vector_builder.text_embedding (one round-trip
   │               each: embed + ANN search, no client-side vector)
   ▼
list fusion  ── merge the N ranked lists into one
   │   composite = max_score
   │             + vote_bonus * (votes - 1)     # cross-variant agreement
   │             + rank_bonus * (1 / best_rank) # rewards strong placements
   ▼
cross-encoder rerank  ── re-crop top-N tiles from the original image,
   │                     score each jointly with the query via
   │                     jina-reranker-m0, re-sort
   ▼
confidence thresholds  ── relative ratio, per-hit absolute floor,
   │                       top-1 hard floor (return [] if nothing credible)
   ▼
bounding boxes → browser
```

### Why embed the query inside the cluster

Elasticsearch's `query_vector_builder.text_embedding` lets the kNN query
carry the *text* of the query plus a model reference; the cluster embeds
the text and runs ANN search in a single request. The alternative —
embedding client-side and sending a raw vector — adds a network hop and
puts a model dependency in the application. Keeping it in-cluster is one
fewer moving part and one fewer place for the embedding to drift out of
sync with the index.

### Why expand the query

A bare query like `find Waldo` produces a thin embedding that under-uses
the model's representational capacity — many tiles match it weakly and the
true target rarely dominates. Expanding into several visual phrasings
exercises different facets of the vision encoder; the true target tends to
score well across multiple phrasings, which the fusion step rewards. The
original query is always variant 0, so expansion only ever *adds* recall,
never removes a match the literal phrasing would have found.

### Why a custom fusion instead of RRF

Reciprocal Rank Fusion (`Σ 1/(k+rank)`) discards similarity magnitude
entirely — it ranks purely on position. For this problem the raw cosine
score carries real signal (a 0.94 match is meaningfully stronger than a
0.82), so the fusion keeps `max_score` as the primary term and uses vote
count and best-rank as *secondary* boosts. This is a deliberate departure
from RRF, documented inline in [`backend/fusion.py`](../backend/fusion.py).
It is explicitly *not* vector averaging — averaging the per-variant
embeddings would collapse the multi-faceted signal the expansion exists to
create.

### Why rerank

The kNN stage is a **bi-encoder**: query and image are embedded
independently and compared by vector distance. It's fast and gives good
recall, but it can't model fine interactions between the query and a
specific tile. The rerank stage is a **cross-encoder**: it feeds the query
text and the candidate tile's pixels through one model that attends across
both, producing a much sharper relevance score. Running it only on the
top-N candidates keeps the expensive model off the full corpus. On reranker
failure (network error, rate limit exhausted), the pipeline falls back to
the kNN/fusion order rather than erroring — degraded but still useful.

### Why threshold

Without a confidence floor, every query returns `k` results — including
queries whose subject isn't in the image at all. That's actively
misleading. The layered threshold (see
[`relevance-tuning.md`](relevance-tuning.md)) returns zero hits when the
strongest match is too weak to be credible, which is the honest behavior
for an absent target.

## The reference-target subsystem

An optional subsystem ([`backend/reference_targets.py`](../backend/reference_targets.py))
lets you register known recurring subjects (e.g. Waldo) with a handful of
reference crop images. At startup it embeds those crops, averages them into
an L2-normalized **prototype vector**, and — for queries that name a
registered subject — runs an additional prototype-kNN leg that's fused with
the text results.

It's data-driven (a manifest of subjects, aliases, and crop paths — no
subject identity is hardcoded) and degrades cleanly: a subject with zero
crops on disk is registered but inactive. In practice the prototype helped
recall but hurt precision on near-identical decoys (Waldo vs. Wenda) with
the small crop set used here, so it's **disabled by default** for clean
demos. The subsystem and that finding are a good illustration of measuring
a feature honestly and shipping it off-by-default rather than pretending it
works. See [`relevance-tuning.md`](relevance-tuning.md) for the measured
comparison.

## Going to production

Deliberately left at demo defaults:

- **One shard, zero replicas.** Set replicas per your availability SLA.
- **No auth on the FastAPI service.** Front it with OIDC / API gateway / mTLS.
- **No rate limiting** on `/api/ingest` or `/api/search`. Add it.
- **Static images on local disk.** Move to object storage (S3 / GCS).
- **HNSW defaults.** For large corpora, tune `m` and `ef_construction`.
- **The image-embedding fallback** to Jina REST should become a properly
  provisioned multimodal EIS endpoint (see
  [`engineering-notes.md`](engineering-notes.md)) — the workaround is fine
  for a demo but bypasses the cluster's inference governance.

The core pattern — tile → embed → index → expand → kNN → fuse → rerank →
threshold — is sound as written.
