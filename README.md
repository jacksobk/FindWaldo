# Multimodal Visual Search on Elasticsearch

**Natural-language search for *where* something is inside a single image — built on Elasticsearch dense-vector kNN, Jina CLIP v2 embeddings, and a vision-language cross-encoder reranker.**

Type `find Waldo` (or `a man in a red striped shirt`, or `줄무늬 셔츠를 입은 남자`) and the system returns bounding boxes over the matching regions of a busy image. No object-detection model, no per-class training — just multimodal embeddings, approximate-nearest-neighbor retrieval, and a reranking stage, orchestrated end to end.

The "Where's Waldo" framing is a deliberately adversarial test bed: hundreds of small, visually similar figures per scene, where the target is near-indistinguishable from decoys (Waldo vs. Wenda, both in red-and-white stripes). The same pipeline works unchanged on retail shelves, aerial imagery, and document layouts — any domain where CLIP-class models have training coverage.

---

## Why this is interesting (the ML/retrieval engineering)

The hard part of this project isn't the web app — it's the retrieval pipeline and the inference plumbing behind it. Five things worth a hiring manager's attention:

### 1. Two-stage retrieve-then-rerank, both stages multimodal

The system embeds image *tiles* and text *queries* into a shared 1024-dimensional CLIP space, retrieves candidates with HNSW kNN, then re-scores the top candidates with `jina-reranker-m0` — a vision-language cross-encoder that *jointly* attends to the query text and each candidate tile's pixels, rather than comparing pre-computed vectors.

This is the standard high-precision IR pattern (cheap broad recall, expensive precise rerank) applied to vision. The bi-encoder kNN stage encodes query and image *independently*; the cross-encoder rerank stage encodes them *together*, which is strictly more expressive and is where most of the top-1 precision comes from. See [`backend/reranker.py`](backend/reranker.py) — including the on-the-fly tile re-cropping (tiles are stored as vectors, not pixels, so candidates are re-cropped from the original image at rerank time) and graceful fallback to kNN order on reranker failure.

### 2. Multi-query expansion + custom list fusion

A bare query like `find Waldo` embeds into a thin, under-specified vector. The pipeline expands it into several CLIP-friendly visual phrasings (`a man in a red and white striped shirt`, `a person with glasses in a busy scene`, …), runs kNN per variant, and **fuses the ranked lists** with a custom scheme that combines max-similarity, cross-variant vote count, and best-rank — deliberately *not* RRF and *not* vector averaging, with the reasoning documented inline. Tiles that surface across multiple phrasings are rewarded as higher-confidence matches.

See [`backend/query_expansion.py`](backend/query_expansion.py) and [`backend/fusion.py`](backend/fusion.py). Both are dependency-free, fully deterministic (for reproducible benchmarks), and carry extensive rationale comments explaining *why* each design choice was made over the alternatives.

### 3. Relevance thresholding tuned from a real benchmark, not vibes

Returning ten weak matches for a query whose target isn't even in the image is worse than returning nothing. The system applies a layered confidence filter — relative score ratio, per-hit absolute floor, and a top-1 hard floor — whose values were derived from an actual measured score distribution across query categories (present / absent / lookalike / ambiguous), not guessed.

The methodology, the score bands that justified each threshold, and the false-positive-rate trade-offs are written up in [`docs/relevance-tuning.md`](docs/relevance-tuning.md). The reusable benchmark harness that produced those numbers — IoU-based top-k success metrics, false-positive tracking, per-category aggregates, CSV/JSON output — is in [`scripts/tune_relevance.py`](scripts/tune_relevance.py).

### 4. Inference-layer engineering: graceful degradation across two providers

The cluster's Elasticsearch Inference Service (EIS) endpoint for `jina-clip-v2` was provisioned as `task_type=text_embedding` — it embeds query *text* but rejects *image* input with a generic `400`. Rather than block on a Kibana reconfiguration, the image-embedding path transparently falls back to Jina's REST API on that specific error, producing vectors in the *same* embedding space (same model), so cosine similarity between an EIS-embedded query and a Jina-REST-embedded tile remains well-defined.

This is documented honestly as an operational workaround, not hidden — see [`docs/engineering-notes.md`](docs/engineering-notes.md) for the diagnostic process (probing input shapes against the connector, isolating the failure to `task_type`, validating the fallback produced compatible vectors). The detective work is arguably the most representative sample of real-world inference engineering in the repo.

### 5. Single trust boundary, embedding inside the cluster

Search embeds the query text *inside* the kNN call via Elasticsearch's `query_vector_builder.text_embedding` — the cluster runs the embedding and the ANN search in a single round-trip. The browser never holds a query vector; the backend never ships a model. The FastAPI service is the only trust boundary.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                          Browser (React)                          │
│   Search bar │ Image viewer + bbox overlay │ Guided demo mode     │
└───────────────────────────────────────────────────────────────────┘
                               │ HTTPS  (only talks to the backend)
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Python, async)               │
│                                                                   │
│   /api/ingest   /api/search   /api/images   /api/health           │
│                                                                   │
│   INGEST:  tile (multi-scale, overlapping)                        │
│            → embed each tile (image)                               │
│            → bulk-index vectors + bboxes                           │
│                                                                   │
│   SEARCH:  expand query into N visual phrasings                   │
│            → kNN per variant (text embedded inside the cluster)   │
│            → fuse ranked lists (votes + max-score + best-rank)    │
│            → cross-encoder rerank top-N candidates                │
│            → confidence thresholds (ratio / abs floor / top floor)│
│            → bounding boxes back to the browser                   │
└───────────────────────────────────────────────────────────────────┘
        │ text embed + kNN (EIS)        │ image embed + rerank (Jina REST)
        ▼                               ▼
┌────────────────────────────────┐  ┌──────────────────────────────────┐
│  Elasticsearch 9.x + EIS       │  │  Jina AI REST API                │
│  _inference (jina-clip-v2)     │  │  jina-clip-v2  (image embed)     │
│  index: dense_vector(1024,     │  │  jina-reranker-m0 (rerank)       │
│         cosine, HNSW)          │  │                                  │
│  bbox{x,y,w,h}, tile/image ids │  │  Used for the image-embedding    │
│  query_vector_builder kNN      │  │  fallback + the rerank stage.    │
└────────────────────────────────┘  └──────────────────────────────────┘
```

For the full request lifecycle and the rationale behind each stage, see [`docs/architecture.md`](docs/architecture.md).

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Vector store + ANN | **Elasticsearch 9.x** `dense_vector`, HNSW, cosine | kNN with `query_vector_builder` — embedding runs in-cluster |
| Embeddings | **Jina CLIP v2** (1024-d, multimodal, 89 languages) | via Elastic Inference Service (text) + Jina REST (image fallback) |
| Reranking | **jina-reranker-m0** (vision-language cross-encoder) | re-scores top-N candidates against the query jointly |
| Backend | **FastAPI** (async Python) | tiling, orchestration, fusion, thresholding |
| Image processing | **Pillow** | multi-scale overlapping tiler, on-the-fly rerank crops |
| Frontend | **React + TypeScript + Vite + Tailwind** | bbox overlay, ranked results, guided demo mode |
| Eval | stdlib-only **benchmark harness** | IoU top-k success, false-positive rate, CSV/JSON |

---

## Quick start

> Requires an Elasticsearch 9.x cluster with the Elastic Inference Service, a Jina AI API key, Python 3.11+, and Node 20+ (or Docker).

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>

cp backend/.env.example backend/.env
# edit backend/.env — set ES_URL, ES_API_KEY, JINA_API_KEY
```

Provision the inference endpoint once, in Kibana → Dev Console (see [`docs/elasticsearch-setup.md`](docs/elasticsearch-setup.md) for the exact commands and the text-vs-image caveat).

**With Docker (recommended):**

```bash
docker compose up -d
# backend  → http://localhost:8200
# frontend → http://localhost:8280
```

**Without Docker:**

```bash
# Backend
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate shell)
cd frontend && npm install && npm run dev   # → http://localhost:5173
```

Then open the frontend, ingest a busy image, and search. Try `find Waldo`, `a sailboat with a red sail`, `بحار` (Arabic: *sailor*), or an absent-target query like `a pink dragon` to see the confidence threshold return zero hits.

---

## Repository layout

```
.
├── backend/                     FastAPI service (the interesting half)
│   ├── main.py                  Routes + search orchestration
│   ├── elastic_client.py        ES + EIS integration; image-embed fallback
│   ├── reranker.py              jina-reranker-m0 cross-encoder stage
│   ├── query_expansion.py       Multi-query visual phrasing expansion
│   ├── fusion.py                Ranked-list merge (votes + score + rank)
│   ├── reference_targets.py     Prototype-vector subsystem for known subjects
│   ├── tiling.py                Multi-scale overlapping image tiler
│   ├── config.py                Env-driven settings (all thresholds here)
│   └── .env.example
│
├── frontend/                    Vite + React + TypeScript + Tailwind
│   └── src/ …                   Image canvas, bbox overlay, demo mode
│
├── scripts/
│   ├── tune_relevance.py        Benchmark harness (IoU top-k, FP rate, CSV)
│   └── test_phase3_metrics.py   Harness self-tests (IoU math, CLI plumbing)
│
└── docs/
    ├── architecture.md          Request lifecycle, design rationale
    ├── relevance-tuning.md      Threshold methodology + measured score bands
    ├── engineering-notes.md     The EIS text-only discovery + fallback
    └── elasticsearch-setup.md   Kibana Dev Console provisioning
```

---

## What this is and isn't

**It is** a reference implementation and portfolio project demonstrating multimodal retrieval, a two-stage retrieve-rerank pipeline, inference-layer integration, and an evidence-driven approach to relevance tuning.

**It isn't** a turnkey product. The "Going to production" section of [`docs/architecture.md`](docs/architecture.md) is explicit about what's left at demo defaults (single shard, no auth on the API, local-disk static files, HNSW defaults).

**On domain fit:** the pipeline works well where CLIP has training coverage — product photography, natural scenes, document layouts, illustrations. It works *poorly* on specialist domains like medical imaging without a domain-adapted embedding model. That limitation is real and is discussed honestly in [`docs/engineering-notes.md`](docs/engineering-notes.md); the architecture (tile → embed → kNN → rerank) is unchanged regardless of which embedding model is plugged in.

---

## License

MIT — see [LICENSE](LICENSE). Reference implementation; use it however you like.
