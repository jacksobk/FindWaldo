"""
Multimodal reranker via Jina's jina-reranker-m0.

The kNN stage gives us top-N candidates fast. The reranker then scores each
candidate against the query in a more refined way using a vision-language
model that *jointly* attends to the query text and the tile image — unlike
the embedding model which encodes them separately and compares vectors.

In standard search benchmarks, this two-stage pattern (retrieve broad with
embeddings, refine with reranker) lifts top-1 precision by 20-40 percentage
points over kNN alone. For visual search the lift is similar.

API: POST https://api.jina.ai/v1/rerank
Request:
    {
      "model": "jina-reranker-m0",
      "query": "red striped shirt",
      "documents": [
        {"image": "<base64 jpeg>"},
        {"image": "<base64 jpeg>"}, ...
      ],
      "top_n": 5
    }
Response:
    {
      "results": [
        {"index": 7, "relevance_score": 0.92},
        {"index": 2, "relevance_score": 0.85}, ...
      ],
      "usage": {"total_tokens": 1234}
    }

Indices in the response refer to positions in the input `documents` list.
"""
import asyncio
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp
from PIL import Image

log = logging.getLogger("findwaldo.reranker")


class JinaReranker:
    """
    Reranks a list of candidate tiles against a query using jina-reranker-m0.

    Tiles aren't stored as image bytes in Elasticsearch (only embeddings),
    so we re-crop them on the fly from the original full-size image stored
    on disk. For 30 candidates this takes ~50ms total — much less than the
    reranker's network roundtrip.

    Usage:
        reranker = JinaReranker(api_key=..., static_dir="./static")
        ranked = await reranker.rerank(query, candidate_tiles, top_k=5)
        # candidate_tiles is the list of dicts returned by knn_search;
        # `ranked` is the same list re-sorted, with each item gaining a
        # `rerank_score` field.
    """

    def __init__(
        self,
        api_key: str,
        static_dir: str,
        url: str = "https://api.jina.ai/v1/rerank",
        model: str = "jina-reranker-m0",
        timeout_s: float = 30.0,
    ):
        if not api_key:
            raise ValueError("Jina API key is required for reranking")
        self.api_key = api_key
        self.static_dir = Path(static_dir)
        self.url = url
        self.model = model
        self.timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _crop_tile_b64(self, image_url: str, bbox: dict) -> Optional[str]:
        """Re-crop a tile from the on-disk full image. Returns base64 JPEG
        or None if the source file is missing.

        The image_url returned by /api/images is e.g. `/static/abcdef.jpg`.
        Strip the leading `/static/` and read from self.static_dir."""
        if not image_url:
            return None
        # URL is `/static/<id>.jpg` — extract the filename
        filename = os.path.basename(image_url)
        path = (self.static_dir / filename).resolve()
        if not path.exists():
            log.warning("Cannot find original image at %s for reranking (cwd=%s, static_dir=%s)",
                        path, os.getcwd(), self.static_dir)
            return None

        try:
            img = Image.open(path).convert("RGB")
            x, y = int(bbox.get("x", 0)), int(bbox.get("y", 0))
            w, h = int(bbox.get("w", 0)), int(bbox.get("h", 0))
            if w <= 0 or h <= 0:
                return None
            patch = img.crop((x, y, x + w, y + h))
            # Reranker has 56x56 minimum; downscale very large patches a bit
            # to control payload size. Max edge 512 keeps quality high while
            # cutting bytes.
            if max(patch.size) > 512:
                patch.thumbnail((512, 512), Image.LANCZOS)
            buf = io.BytesIO()
            patch.save(buf, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            log.warning("Failed to crop tile %s: %s", image_url, e)
            return None

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> tuple[int, list[dict]]:
        """
        Args:
            query:       user query text
            candidates:  list of hit dicts (must contain `bbox` and `image_url`)
            top_k:       how many results to return after reranking

        Returns: (elapsed_ms, reranked_hits)

            elapsed_ms includes cropping, network, and parsing.
            reranked_hits is the input list re-sorted in relevance order, with
            each item augmented with `rerank_score` and `original_score`.
        """
        if not candidates:
            return 0, []

        t0 = time.perf_counter()

        # Crop all candidates concurrently in a thread pool. PIL is CPU-bound
        # but cropping is fast (each crop is ~5ms on a 2560x1644 image).
        loop = asyncio.get_event_loop()
        crops = await asyncio.gather(*[
            loop.run_in_executor(
                None, self._crop_tile_b64, c["image_url"], c["bbox"]
            )
            for c in candidates
        ])

        # Build the documents array; track original positions so a None crop
        # doesn't break the response-index alignment.
        documents = []
        kept_indices: list[int] = []
        for i, b64 in enumerate(crops):
            if b64:
                documents.append({"image": b64})
                kept_indices.append(i)

        if not documents:
            log.warning("No candidates could be cropped; returning kNN order")
            return int((time.perf_counter() - t0) * 1000), candidates[:top_k]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }

        session = await self._http()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Retry on Jina rate-limit (429). The reranker shares the Jina free-
        # tier token budget with the embedder, so the same windowing applies.
        max_retries = 3
        wait_table = [10, 30, 60]
        body = None
        for attempt in range(max_retries + 1):
            try:
                async with session.post(self.url, headers=headers, json=payload) as resp:
                    body = await resp.json()
                    if resp.status == 429 and attempt < max_retries:
                        wait = wait_table[attempt]
                        log.warning(
                            "Reranker 429: %s — backing off %ds (retry %d/%d)",
                            body.get("detail", "rate limited"), wait,
                            attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        log.error("Reranker API %d: %s", resp.status, body)
                        # Fall back to kNN order — the original ranking is still good
                        return (
                            int((time.perf_counter() - t0) * 1000),
                            candidates[:top_k],
                        )
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("Reranker network error, falling back to kNN order: %s", e)
                return int((time.perf_counter() - t0) * 1000), candidates[:top_k]

        if body is None:
            return int((time.perf_counter() - t0) * 1000), candidates[:top_k]

        results = body.get("results", [])

        # Build the reranked list by mapping reranker-document-indices back to
        # candidate-list-indices via kept_indices.
        reranked: list[dict] = []
        for r in results:
            doc_idx = r.get("index")
            if doc_idx is None or doc_idx >= len(kept_indices):
                continue
            cand_idx = kept_indices[doc_idx]
            hit = dict(candidates[cand_idx])
            hit["rerank_score"] = float(r.get("relevance_score", 0.0))
            hit["original_score"] = hit.get("score", 0.0)
            # Replace the displayed score with the reranker score so the UI
            # naturally shows the refined number.
            hit["score"] = hit["rerank_score"]
            reranked.append(hit)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "Reranked %d candidates → top %d in %dms (tokens=%s)",
            len(candidates), len(reranked), elapsed_ms,
            body.get("usage", {}).get("total_tokens", "?"),
        )
        # Defense in depth: even if the API ignored top_n, enforce client-side.
        return elapsed_ms, reranked[:top_k]
