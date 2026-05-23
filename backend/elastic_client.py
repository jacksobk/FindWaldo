"""
Elasticsearch client wrapper for the Find Waldo demo.

This is the only module in the application that knows about Elasticsearch.
Everything else talks to this class.

Inference contract — split paths (text via EIS, images via Jina REST):
    The application uses an EIS-managed inference endpoint for TEXT:

        PUT _inference/text_embedding/{inference_id}
        {
          "service": "jinaai",
          "service_settings": { "model_id": "{model_id}",
                                 "api_key": "..." }
        }

    For IMAGES, this code calls Jina's REST API directly (api.jina.ai)
    using JINA_API_KEY. This is a workaround: the cluster's JinaAI EIS
    connector is registered with task_type=text_embedding and rejects
    image input with a generic 400, even though the underlying model
    (jina-clip-v2) is multimodal. The Jina REST API accepts the same
    base64 image payload and returns vectors in the same space as the
    EIS text path, so cosine similarity between an EIS-embedded query
    vector and a Jina-REST-embedded tile vector is well-defined.

    Both paths use the same model (jina-clip-v2). The split is purely
    operational — if the cluster gets a true multimodal endpoint, the
    fallback can go away and embed_images() can route everything
    through EIS again. Until then, this code transparently retries
    image input via Jina REST when EIS rejects it.

A note on scoring:
    Elasticsearch kNN _score values returned in `hits[].score` are NOT
    probabilities. They are similarity scores in [0, 1] (for cosine):
    higher = more similar to the query, but a score of 0.92 does not mean
    "92% confident". Use scores ordinally for ranking; do not show them as
    percentages in any UI surface.
"""
import asyncio
import logging
import math
import os
import time
from typing import Optional

from elasticsearch import AsyncElasticsearch, NotFoundError, ApiError
from elasticsearch.helpers import async_bulk

log = logging.getLogger("findwaldo.elastic")


def _l2_normalize(vec: list[float]) -> list[float]:
    """
    L2-normalize a vector to unit length.

    Defensive: even with cosine similarity at the index level (which
    normalizes at query time and accepts any vector), normalizing on
    ingest gives consistent magnitudes if the index similarity is ever
    switched to dot_product for performance. Cheap to do, costs nothing
    if the index is already cosine.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


class ElasticClient:
    def __init__(self, url: str, api_key: str, verify_certs: bool = True):
        self.client = AsyncElasticsearch(
            url,
            api_key=api_key if api_key else None,
            verify_certs=verify_certs,
            request_timeout=120,
            retry_on_timeout=True,
            max_retries=2,
        )

    async def close(self):
        await self.client.close()

    # ------------------------------------------------------------------
    # Bootstrap: idempotent setup of the inference endpoint and index
    # ------------------------------------------------------------------

    async def bootstrap(
        self,
        inference_id: str,
        model_id: str,
        index_name: str,
        embedding_dims: int,
    ) -> None:
        """Verify connectivity, inspect the inference endpoint, ensure index exists."""
        if not await self.client.ping():
            raise RuntimeError("Elasticsearch is not reachable. Check ES_URL / ES_API_KEY.")

        endpoint_cfg = await self._ensure_inference_endpoint(inference_id, model_id)

        # We use cosine similarity unconditionally. Cosine normalizes at
        # query time and accepts any vector — robust to small numerical
        # drift in embeddings. Equivalent to dot_product on unit vectors,
        # marginally slower at scale, completely irrelevant at our size.
        ss = endpoint_cfg.get("service_settings", {}) if endpoint_cfg else {}
        similarity = "cosine"
        reported_dims = ss.get("dimensions", embedding_dims)

        if reported_dims != embedding_dims:
            log.warning(
                "Configured EMBEDDING_DIMS=%d differs from endpoint-reported dimensions=%d. "
                "Using endpoint value for the index mapping.",
                embedding_dims, reported_dims,
            )

        log.info(
            "Inference endpoint '%s' configured for model_id=%s dims=%d similarity=%s",
            inference_id,
            ss.get("model_id", "(unknown)"),
            reported_dims,
            similarity,
        )

        await self._ensure_index(index_name, reported_dims, similarity)

    async def _ensure_inference_endpoint(self, inference_id: str, model_id: str) -> dict:
        """
        Verify the inference endpoint exists and return its configuration.

        Does NOT create the endpoint — provisioning is operational work that
        belongs in Kibana or IaC, not in application bootstrap. Failing fast
        with a clear remediation message is preferable to silently creating
        an endpoint with assumptions that may not match the deployment.
        """
        try:
            resp = await self.client.inference.get(inference_id=inference_id)
            body = resp.body if hasattr(resp, "body") else resp
            endpoints = body.get("endpoints", [body]) if isinstance(body, dict) else [body]
            ep = endpoints[0] if endpoints else {}
            log.info(
                "Found inference endpoint '%s' (service=%s, model_id=%s)",
                inference_id,
                ep.get("service", "?"),
                ep.get("service_settings", {}).get("model_id", "?"),
            )
            return ep
        except NotFoundError:
            pass

        log.error(
            "Inference endpoint '%s' does not exist on the cluster.",
            inference_id,
        )
        raise RuntimeError(
            f"Inference endpoint '{inference_id}' was not found.\n\n"
            f"This application expects the endpoint to already exist, hosted by\n"
            f"Elastic Inference Service. Provision it in Kibana Dev Console, then\n"
            f"restart the backend:\n\n"
            f"  PUT _inference/text_embedding/{inference_id}\n"
            f'  {{ "service": "elastic",\n'
            f'    "service_settings": {{ "model_id": "{model_id}" }} }}\n'
        )

    async def _ensure_index(self, index_name: str, dims: int, similarity: str) -> None:
        if await self.client.indices.exists(index=index_name):
            log.info("Index '%s' already exists", index_name)
            return
        log.info("Creating index '%s' (dims=%d, similarity=%s)", index_name, dims, similarity)
        await self.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "image_id":  {"type": "keyword"},
                    "tile_id":   {"type": "keyword"},
                    "row":       {"type": "integer"},
                    "col":       {"type": "integer"},
                    "scale":     {"type": "integer"},  # tile size in px (224, 384, 768)
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "w": {"type": "integer"},
                            "h": {"type": "integer"},
                        },
                    },
                    "image_url": {"type": "keyword"},
                    "image_w":   {"type": "integer"},
                    "image_h":   {"type": "integer"},
                    "label":     {"type": "text"},   # optional caption / OCR for hybrid search
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": similarity,
                    },
                }
            },
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            },
        )

    async def health(self) -> dict:
        try:
            cluster = await self.client.cluster.health()
            return {
                "status": "ok",
                "cluster_status": cluster.get("status"),
                "cluster_name":   cluster.get("cluster_name"),
                "active_nodes":   cluster.get("number_of_nodes"),
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # ------------------------------------------------------------------
    # Inference: embed images and text via the same EIS endpoint
    # ------------------------------------------------------------------

    async def embed_images(
        self,
        inference_id: str,
        tiles_b64: list[str],
        batch_size: int = 16,
    ) -> tuple[int, list[list[float]]]:
        """
        Embed image tiles via the cluster's EIS inference endpoint.

        The same endpoint serves both modalities. We send the tile base64
        strings as `input` and let EIS dispatch them to the multimodal
        model. Returned vectors live in the same space as the search-time
        text embeddings produced by `query_vector_builder.text_embedding`.

        Batching is preserved as an interface concern — EIS accepts arrays
        of inputs, and chunking lets us bound payload size per HTTP call.
        Sequential rather than concurrent because ingest is one-time setup,
        not serving traffic, and we'd rather be polite to the cluster.

        Returns (total_ms, embeddings_in_input_order).
        """
        t0 = time.perf_counter()
        all_embeddings: list[list[float]] = []

        batches = [tiles_b64[i:i + batch_size]
                   for i in range(0, len(tiles_b64), batch_size)]
        log.info(
            "Embedding %d tiles in %d sequential batches of %d via inference '%s'",
            len(tiles_b64), len(batches), batch_size, inference_id,
        )

        for batch in batches:
            embeddings = await self._embed_image_batch(inference_id, batch)
            all_embeddings.extend(embeddings)

        total_ms = int((time.perf_counter() - t0) * 1000)
        return total_ms, all_embeddings

    async def _embed_image_batch(
        self,
        inference_id: str,
        batch_b64: list[str],
    ) -> list[list[float]]:
        """
        Single batched call to embed image inputs.

        Tries EIS first. If EIS rejects the input with a 400 (the
        expected failure mode when the endpoint is text_embedding only),
        falls back to calling Jina's REST API directly. Same model, same
        vector space — see module docstring for why this split exists.

        Retries on cluster-level 429 (rate limit) before falling back,
        because rate-limit pressure is transient and the right response
        is to wait, not to hop to a different provider.
        """
        max_retries = 3
        wait_table = [10, 30, 60]
        for attempt in range(max_retries + 1):
            try:
                resp = await self.client.inference.inference(
                    inference_id=inference_id,
                    input=batch_b64,
                )
            except ApiError as e:
                msg = str(e)
                is_rate_limit = (
                    getattr(e, "status_code", None) == 429
                    or "status [429]" in msg
                )
                if is_rate_limit and attempt < max_retries:
                    wait = wait_table[attempt]
                    log.warning(
                        "EIS inference 429 (attempt %d/%d); backing off %ds",
                        attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                # 400 "input validation" — the canonical signature of an
                # EIS endpoint provisioned text-only refusing image input.
                # Don't burn retries on this; it's deterministic. Fall
                # through to Jina REST.
                is_input_validation = (
                    getattr(e, "status_code", None) == 400
                    and "input validation" in msg.lower()
                )
                if is_input_validation:
                    log.warning(
                        "EIS rejected image input (status 400, input validation). "
                        "Falling back to Jina REST API for this batch of %d images.",
                        len(batch_b64),
                    )
                    return await _embed_via_jina_rest(batch_b64)
                raise

            blocks = resp.get("text_embedding") or resp.get("embedding") or []
            if len(blocks) != len(batch_b64):
                log.warning(
                    "EIS returned %d embeddings for batch of %d inputs",
                    len(blocks), len(batch_b64),
                )
            # Normalize on ingest as a defensive measure (see _l2_normalize).
            return [_l2_normalize(b["embedding"]) for b in blocks]

        # Unreachable, defensive
        raise RuntimeError("EIS inference exhausted retries without returning")

    async def embed_text(self, inference_id: str, text: str) -> tuple[int, list[float]]:
        """Embed a single text string. Used for client-side debugging only —
        production search embeds the query inside the kNN call via
        query_vector_builder, eliminating one network hop."""
        t0 = time.perf_counter()
        resp = await self.client.inference.inference(
            inference_id=inference_id,
            input=[text],
        )
        blocks = resp.get("text_embedding") or resp.get("embedding") or []
        if not blocks:
            raise RuntimeError(f"EIS inference returned no embedding: {resp}")
        return int((time.perf_counter() - t0) * 1000), blocks[0]["embedding"]

    # ------------------------------------------------------------------
    # Bulk indexing
    # ------------------------------------------------------------------

    async def bulk_index(self, index: str, docs: list[dict]) -> tuple[int, int]:
        t0 = time.perf_counter()
        actions = ({"_index": index, "_source": d} for d in docs)
        success, errors = await async_bulk(
            self.client, actions, chunk_size=200, max_retries=2,
            raise_on_error=False, request_timeout=60,
        )
        if errors:
            log.warning("Bulk indexing had %d errors (showing first): %s",
                        len(errors), errors[0] if errors else None)
        await self.client.indices.refresh(index=index)
        return int((time.perf_counter() - t0) * 1000), success

    # ------------------------------------------------------------------
    # Search: kNN with optional hybrid BM25 (RRF)
    # ------------------------------------------------------------------

    async def knn_search(
        self,
        index: str,
        inference_id: str,
        query_text: str,
        k: int,
        num_candidates: int,
        image_id: Optional[str] = None,
        min_score: float = 0.0,
        hybrid: bool = False,
    ) -> tuple[int, int, list[dict], int]:
        """
        Run kNN search. The query text is embedded server-side via
        query_vector_builder.text_embedding, which references the EIS
        inference endpoint by id — one round trip, the cluster handles
        the embedding.

        Hybrid mode fuses kNN with a BM25 retriever over `label` using RRF.

        Returned `score` values are kNN similarity scores, NOT probabilities.
        Treat them as ordinal rankings; do not display as percentages.
        """
        knn_body = {
            "field": "embedding",
            "k": k,
            "num_candidates": num_candidates,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": inference_id,
                    "model_text": query_text,
                }
            },
        }
        if image_id:
            knn_body["filter"] = {"term": {"image_id": image_id}}

        return await self._run_knn(
            index=index,
            knn_body=knn_body,
            k=k,
            num_candidates=num_candidates,
            query_text_for_hybrid=query_text,
            min_score=min_score,
            hybrid=hybrid,
        )

    async def knn_search_with_vector(
        self,
        index: str,
        query_vector: list[float],
        k: int,
        num_candidates: int,
        image_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> tuple[int, int, list[dict], int]:
        """
        Run kNN search with a pre-computed query vector instead of a text
        prompt. Used by the reference-target prototype path: the prototype
        is the mean of crop embeddings, computed once at startup and held
        in memory; we feed it directly to kNN.

        The vector MUST live in the same embedding space as the index
        (same EIS-managed CLIP model). Hybrid mode is not offered here
        because there is no text to BM25 against.

        Returns the same shape as `knn_search`:
          (inference_ms, search_ms, hits, total)

        `inference_ms` will always be 0 here — the embedding work was done
        at startup, not during this request — but the field is preserved
        in the return tuple so the orchestrator can sum latencies
        uniformly across both code paths.
        """
        knn_body = {
            "field": "embedding",
            "k": k,
            "num_candidates": num_candidates,
            "query_vector": query_vector,
        }
        if image_id:
            knn_body["filter"] = {"term": {"image_id": image_id}}

        return await self._run_knn(
            index=index,
            knn_body=knn_body,
            k=k,
            num_candidates=num_candidates,
            query_text_for_hybrid=None,
            min_score=min_score,
            hybrid=False,
        )

    async def _run_knn(
        self,
        index: str,
        knn_body: dict,
        k: int,
        num_candidates: int,
        query_text_for_hybrid: Optional[str],
        min_score: float,
        hybrid: bool,
    ) -> tuple[int, int, list[dict], int]:
        """
        Execute a kNN search request and return parsed hits.

        Shared backend for `knn_search` (text-driven) and
        `knn_search_with_vector` (prototype-driven). The two callers
        differ only in how `knn_body` is constructed; everything from
        request execution through response parsing is identical.
        """
        t0 = time.perf_counter()

        fields = [
            "image_id", "tile_id", "image_url", "row", "col",
            "image_w", "image_h",
            # Object fields are returned as flattened leaves; request each one explicitly.
            "bbox.x", "bbox.y", "bbox.w", "bbox.h",
        ]

        if hybrid:
            if query_text_for_hybrid is None:
                raise ValueError(
                    "Hybrid kNN requires query_text_for_hybrid (BM25 leg has no text)"
                )
            body = {
                "retriever": {
                    "rrf": {
                        "retrievers": [
                            {"knn": knn_body},
                            {
                                "standard": {
                                    "query": {
                                        "match": {"label": {"query": query_text_for_hybrid, "operator": "or"}}
                                    }
                                }
                            },
                        ],
                        "rank_window_size": max(num_candidates, 50),
                        "rank_constant": 60,
                    }
                },
                "size": k,
                "fields": fields,
                "_source": False,
            }
        else:
            body = {
                "knn": knn_body,
                "size": k,
                "fields": fields,
                "_source": False,
            }
            if min_score > 0:
                body["min_score"] = min_score

        # Defensive retry on 429 from the cluster's inference layer. EIS may
        # rate-limit at the cluster level; if so, back off and retry rather
        # than surfacing a 500 to the user.
        t_search = time.perf_counter()
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = await self.client.search(index=index, body=body)
                break
            except ApiError as e:
                msg = str(e)
                is_rate_limit = (
                    getattr(e, "status_code", None) == 429
                    or "status [429]" in msg
                )
                if is_rate_limit and attempt < max_retries:
                    wait = 5 * (2 ** attempt)
                    log.warning(
                        "Search hit inference rate limit; retrying in %ds (%d/%d)",
                        wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
        search_ms = int((time.perf_counter() - t_search) * 1000)

        hits_raw = resp["hits"]["hits"]
        total = (
            resp["hits"]["total"]["value"]
            if isinstance(resp["hits"]["total"], dict)
            else len(hits_raw)
        )

        hits = []
        for h in hits_raw:
            f = h.get("fields", {})
            # Reconstruct the bbox object from flattened leaf fields. The
            # Elasticsearch `fields` API returns object subfields as e.g.
            # `bbox.x: [123]` rather than `bbox: {x: 123, ...}`.
            bbox = {
                "x": _first(f, "bbox.x") or 0,
                "y": _first(f, "bbox.y") or 0,
                "w": _first(f, "bbox.w") or 0,
                "h": _first(f, "bbox.h") or 0,
            }
            hits.append({
                "tile_id":   _first(f, "tile_id"),
                "image_id":  _first(f, "image_id"),
                "image_url": _first(f, "image_url"),
                "bbox":      bbox,
                # NOTE: kNN _score is a similarity score in [0, 1] for cosine,
                # NOT a probability. Do not present as a percentage in the UI.
                "score":     float(h["_score"]),
            })

        total_ms     = int((time.perf_counter() - t0) * 1000)
        inference_ms = max(0, total_ms - search_ms)
        return inference_ms, search_ms, hits, total

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    async def list_images(self, index: str) -> list[dict]:
        try:
            resp = await self.client.search(
                index=index,
                size=0,
                aggs={
                    "by_image": {
                        "terms": {"field": "image_id", "size": 1000},
                        "aggs": {
                            "first": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": ["image_id", "image_url", "image_w", "image_h", "label"],
                                }
                            }
                        },
                    }
                },
            )
        except NotFoundError:
            return []

        out = []
        for bucket in resp["aggregations"]["by_image"]["buckets"]:
            src = bucket["first"]["hits"]["hits"][0]["_source"]
            out.append({
                "image_id":    src["image_id"],
                "image_url":   src["image_url"],
                "width":       src["image_w"],
                "height":      src["image_h"],
                "label":       src.get("label"),
                "tile_count":  bucket["doc_count"],
                "uploaded_at": None,
            })
        return out

    async def delete_image(self, index: str, image_id: str) -> int:
        resp = await self.client.delete_by_query(
            index=index,
            query={"term": {"image_id": image_id}},
            refresh=True,
        )
        return resp.get("deleted", 0)


def _first(fields: dict, key: str):
    """The fields-format response returns lists; flatten our scalar lookups."""
    v = fields.get(key)
    if isinstance(v, list):
        return v[0] if v else None
    return v


async def _embed_via_jina_rest(
    b64_images: list[str],
) -> list[list[float]]:
    """
    Embed a list of base64-encoded images via Jina's REST API directly.

    Used as a fallback when the cluster's EIS inference endpoint rejects
    image input. Same model (jina-clip-v2), same vector space as the
    EIS text-embedding path, so the resulting tile vectors and
    EIS-produced query vectors compare correctly under cosine similarity.

    Reads JINA_API_KEY from the environment at call time (not import
    time) so a deployment that doesn't need the fallback doesn't have to
    set the var. If the fallback IS taken and the var is missing, we
    raise loudly rather than producing degenerate vectors silently.

    Uses aiohttp — already a backend dependency for the reranker. Output
    is L2-normalised to match the EIS path; downstream code should not
    have to know which leg produced which vector.
    """
    import aiohttp  # local import: only needed on the fallback path

    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "JINA_API_KEY is not set, but EIS rejected image input and the "
            "Jina REST fallback is the only remaining image-embedding path. "
            "Either set JINA_API_KEY in the backend environment, or provision "
            "an EIS endpoint that accepts image input."
        )

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "jina-clip-v2",
        "input": [{"image": b} for b in b64_images],
    }

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Jina REST API returned {resp.status}: {body[:300]}"
                )
            data = await resp.json()

    out: list[list[float]] = []
    for item in data.get("data", []):
        emb = item.get("embedding")
        if emb:
            out.append(_l2_normalize(emb))

    if len(out) != len(b64_images):
        log.warning(
            "Jina REST returned %d embeddings for batch of %d images",
            len(out), len(b64_images),
        )

    return out
