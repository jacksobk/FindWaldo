#!/usr/bin/env python3
"""
EIS Contract Probe — Phase 2 Validation Checkpoint
===================================================

Validates three contracts against the live Elasticsearch 9.4 cluster
*before* any application code is written that depends on them:

    1. Text embedding via _inference (input: string)
    2. Image embedding via _inference (input: data URI / base64)
    3. query_vector_builder.text_embedding inside a kNN search

For each contract, the probe:
    - sends one minimal request
    - prints the raw response
    - reports the discovered keys, types, and dimensions
    - flags any deviation from the assumptions in elastic_client.py

Nothing is asserted. The script reports what the cluster does, then
prints a contract summary. The operator (or you) decides whether the
shapes match what elastic_client.py expects, and we adapt the *client*
if they don't — never the application code that consumes it.

Usage
-----
    cd backend
    cp .env.example .env       # fill in ES_URL and ES_API_KEY
    python ../scripts/probe_eis_contract.py

Exit code is 0 on successful probe (regardless of shape findings),
1 on connectivity / inference failure.
"""
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make backend modules importable regardless of CWD
HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

# Load .env if present (no python-dotenv dep — parse manually)
env_file = BACKEND / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from elasticsearch import AsyncElasticsearch, NotFoundError  # noqa: E402


# ---------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------

C_BOLD  = "\033[1m"
C_DIM   = "\033[2m"
C_TEAL  = "\033[36m"
C_RED   = "\033[31m"
C_GREEN = "\033[32m"
C_YEL   = "\033[33m"
C_OFF   = "\033[0m"

def header(label: str) -> None:
    print()
    print(f"{C_BOLD}{C_TEAL}{'─' * 70}{C_OFF}")
    print(f"{C_BOLD}{C_TEAL}{label}{C_OFF}")
    print(f"{C_BOLD}{C_TEAL}{'─' * 70}{C_OFF}")

def step(label: str) -> None:
    print(f"\n  {C_BOLD}{label}{C_OFF}")

def kv(key: str, val: Any, color: str = "") -> None:
    print(f"    {C_DIM}{key:<26}{C_OFF}{color}{val}{C_OFF}")

def ok(msg: str) -> None:
    print(f"    {C_GREEN}✓ {msg}{C_OFF}")

def warn(msg: str) -> None:
    print(f"    {C_YEL}! {msg}{C_OFF}")

def fail(msg: str) -> None:
    print(f"    {C_RED}✗ {msg}{C_OFF}")

def show_response(resp: dict, max_chars: int = 400) -> None:
    rendered = json.dumps(resp, indent=2, default=str)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + f"\n    ... [{len(rendered) - max_chars} more chars truncated]"
    for line in rendered.splitlines():
        print(f"    {C_DIM}{line}{C_OFF}")


# ---------------------------------------------------------------------
# Tiny PNG generator — no Pillow dependency
# ---------------------------------------------------------------------

def tiny_jpeg_b64() -> str:
    """
    Produce a minimal but valid 8x8 grayscale JPEG, base64 encoded.
    Uses Pillow if available, falls back to a hand-coded PNG if not.
    The probe just needs *valid image bytes* — content doesn't matter.
    """
    try:
        from PIL import Image
        img = Image.new("RGB", (8, 8), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        # 1x1 transparent PNG as last-resort fallback
        png_1x1 = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc"
            b"\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return base64.b64encode(png_1x1).decode("ascii")


# ---------------------------------------------------------------------
# Contract findings — accumulated, printed at the end
# ---------------------------------------------------------------------

class Findings:
    def __init__(self) -> None:
        self.text_response_key: str | None = None
        self.image_response_key: str | None = None
        self.embedding_dims: int | None = None
        self.image_input_shape: str | None = None
        self.knn_works_with_model_text: bool | None = None
        self.knn_works_with_query_text: bool | None = None
        self.knn_response_status: str | None = None
        self.notes: list[str] = []


# ---------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------

async def probe_connectivity(client: AsyncElasticsearch) -> dict:
    header("0. CONNECTIVITY")
    info = await client.info()
    kv("cluster_name",  info.get("cluster_name"))
    kv("version",       info.get("version", {}).get("number"))
    kv("build_flavor",  info.get("version", {}).get("build_flavor"))
    return info


async def probe_inference_endpoint(client: AsyncElasticsearch, inference_id: str) -> dict | None:
    header("1. INFERENCE ENDPOINT EXISTS")
    try:
        resp = await client.inference.get(inference_id=inference_id)
        body = resp.body if hasattr(resp, "body") else resp
        endpoints = body.get("endpoints", [body]) if isinstance(body, dict) else [body]
        if endpoints:
            ep = endpoints[0]
            kv("inference_id",        ep.get("inference_id", "(unknown)"))
            kv("task_type",           ep.get("task_type", "(unknown)"))
            kv("service",             ep.get("service", "(unknown)"))
            ss = ep.get("service_settings", {})
            kv("service_settings",    json.dumps(ss))
            ok(f"endpoint '{inference_id}' is registered")
        return body
    except NotFoundError:
        fail(f"endpoint '{inference_id}' not found")
        warn("create it via Kibana Dev Console with:")
        print(f"      {C_DIM}PUT _inference/text_embedding/{inference_id}{C_OFF}")
        print(f"      {C_DIM}{{ \"service\": \"elastic\", \"service_settings\": {{ \"model_id\": \"jina-clip-v2\" }} }}{C_OFF}")
        return None


async def probe_text_embedding(client: AsyncElasticsearch, inference_id: str, findings: Findings) -> None:
    header("2. CONTRACT: TEXT EMBEDDING")
    step(f"POST _inference/{inference_id}  with input=[\"...string...\"]")
    sample = "a man in a red and white striped shirt"
    try:
        resp = await client.inference.inference(inference_id=inference_id, input=[sample])
        body = resp.body if hasattr(resp, "body") else resp
    except Exception as e:
        fail(f"inference call raised: {type(e).__name__}: {e}")
        findings.notes.append("text_embedding: inference call failed")
        return

    print()
    show_response(body)
    print()

    # Discover the response key
    candidates = ["text_embedding", "embedding", "inference_results", "predicted_value"]
    found_key = None
    for k in candidates:
        if k in body and body[k]:
            found_key = k
            break

    if found_key is None:
        fail(f"none of the expected keys present: {candidates}")
        kv("actual top-level keys", list(body.keys()))
        findings.notes.append("text_embedding: unknown response shape")
        return

    findings.text_response_key = found_key
    blocks = body[found_key]
    kv("response key",        found_key, C_GREEN)
    kv("blocks length",       len(blocks))

    # Extract the embedding vector regardless of nesting
    first = blocks[0] if isinstance(blocks, list) else blocks
    vec = None
    if isinstance(first, dict):
        for vk in ("embedding", "predicted_value", "values"):
            if vk in first:
                vec = first[vk]
                kv("vector key (nested)",  vk, C_GREEN)
                break
    elif isinstance(first, list):
        vec = first
        kv("vector layout",       "top-level list of floats")

    if vec is None or not isinstance(vec, list):
        fail("could not locate the float vector in the response")
        return

    findings.embedding_dims = len(vec)
    kv("embedding dimension", len(vec), C_GREEN)
    kv("first 3 values",      [round(float(x), 4) for x in vec[:3]])
    ok("text embedding contract validated")


async def probe_image_embedding(client: AsyncElasticsearch, inference_id: str, findings: Findings) -> None:
    header("3. CONTRACT: IMAGE EMBEDDING (BASE64)")
    img_b64 = tiny_jpeg_b64()

    # Try multiple input shapes — some EIS deployments accept data URIs,
    # others want raw base64, others require an object form.
    shapes_to_try = [
        ("data URI",                 [f"data:image/jpeg;base64,{img_b64}"]),
        ("raw base64 string",        [img_b64]),
        ("object {image: <b64>}",    [{"image": img_b64}]),
        ("object {image: <dataURI>}",[{"image": f"data:image/jpeg;base64,{img_b64}"}]),
    ]

    for label, payload in shapes_to_try:
        step(f"trying input shape: {label}")
        try:
            resp = await client.inference.inference(inference_id=inference_id, input=payload)
            body = resp.body if hasattr(resp, "body") else resp
        except Exception as e:
            err = f"{type(e).__name__}"
            msg = str(e).splitlines()[0][:200]
            warn(f"{err}: {msg}")
            continue

        # Did we get a vector back?
        vec = None
        used_key = None
        for k in ("text_embedding", "embedding", "inference_results", "predicted_value"):
            if k in body and body[k]:
                used_key = k
                blocks = body[k]
                first = blocks[0] if isinstance(blocks, list) else blocks
                if isinstance(first, dict):
                    vec = first.get("embedding") or first.get("predicted_value")
                elif isinstance(first, list):
                    vec = first
                break

        if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec[:3]):
            findings.image_input_shape = label
            findings.image_response_key = used_key
            kv("response key",        used_key, C_GREEN)
            kv("vector dimension",    len(vec), C_GREEN)
            kv("first 3 values",      [round(float(x), 4) for x in vec[:3]])
            ok(f"image input shape '{label}' WORKS")

            if findings.embedding_dims and len(vec) != findings.embedding_dims:
                warn(f"image dim {len(vec)} ≠ text dim {findings.embedding_dims} — model may not be unified")
            else:
                ok("image and text vectors live in the same dimensional space")
            return
        else:
            warn(f"shape '{label}' returned no usable vector")

    fail("no input shape produced an image embedding")
    findings.notes.append(
        "image_embedding: cluster did not accept any of "
        f"{[s[0] for s in shapes_to_try]} — the EIS endpoint may be text-only "
        "or expect an undocumented shape. elastic_client.py.embed_images "
        "must be adapted before image ingest will work."
    )


async def probe_query_vector_builder(
    client: AsyncElasticsearch,
    inference_id: str,
    findings: Findings,
) -> None:
    header("4. CONTRACT: query_vector_builder.text_embedding")

    # Build a throwaway index just for this probe so we don't depend on
    # any application data being present.
    probe_index = "eis-contract-probe-tmp"
    dims = findings.embedding_dims or 1024

    try:
        if await client.indices.exists(index=probe_index):
            await client.indices.delete(index=probe_index)

        await client.indices.create(
            index=probe_index,
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    }
                }
            },
        )

        # Index one stub document so kNN has something to search against.
        # The vector content is irrelevant — we only care whether the search
        # request *parses and dispatches* with each query_vector_builder shape.
        stub_vec = [0.0] * dims
        stub_vec[0] = 1.0
        await client.index(
            index=probe_index, id="probe-1",
            document={"embedding": stub_vec}, refresh=True,
        )

        for variant_name, body in [
            ("model_text (legacy)", {
                "knn": {
                    "field": "embedding", "k": 1, "num_candidates": 10,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": inference_id,
                            "model_text": "test query",
                        }
                    },
                },
                "size": 1, "_source": False,
            }),
            ("query_text (newer)", {
                "knn": {
                    "field": "embedding", "k": 1, "num_candidates": 10,
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": inference_id,
                            "query_text": "test query",
                        }
                    },
                },
                "size": 1, "_source": False,
            }),
        ]:
            step(f"trying variant: {variant_name}")
            try:
                resp = await client.search(index=probe_index, body=body)
                body_resp = resp.body if hasattr(resp, "body") else resp
                hits = body_resp.get("hits", {}).get("hits", [])
                ok(f"variant '{variant_name}' accepted — returned {len(hits)} hit(s)")
                if variant_name == "model_text (legacy)":
                    findings.knn_works_with_model_text = True
                else:
                    findings.knn_works_with_query_text = True
            except Exception as e:
                msg = str(e).splitlines()[0][:240]
                warn(f"{type(e).__name__}: {msg}")
                if variant_name == "model_text (legacy)":
                    findings.knn_works_with_model_text = False
                else:
                    findings.knn_works_with_query_text = False

    finally:
        try:
            await client.indices.delete(index=probe_index)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(findings: Findings, expected_dims_env: int) -> None:
    header("CONTRACT SUMMARY — what elastic_client.py must match")

    print()
    print(f"  {C_BOLD}TEXT EMBEDDING{C_OFF}")
    if findings.text_response_key:
        kv("response key", findings.text_response_key, C_GREEN)
        kv("dimension", findings.embedding_dims)
        if expected_dims_env != findings.embedding_dims:
            warn(f"EMBEDDING_DIMS={expected_dims_env} in env, live={findings.embedding_dims} — update env to silence")
        else:
            ok("EMBEDDING_DIMS matches live endpoint")
    else:
        fail("text embedding contract NOT validated")

    print()
    print(f"  {C_BOLD}IMAGE EMBEDDING{C_OFF}")
    if findings.image_input_shape:
        kv("accepted input shape", findings.image_input_shape, C_GREEN)
        kv("response key", findings.image_response_key)
        if findings.image_input_shape != "data URI":
            warn(
                "elastic_client.py currently sends 'data URI' (data:image/jpeg;base64,...). "
                f"Live endpoint accepts '{findings.image_input_shape}'. "
                "embed_images() must be adapted."
            )
        else:
            ok("elastic_client.py.embed_images data-URI shape is correct")
    else:
        fail("image embedding contract NOT validated — image ingest will fail")

    print()
    print(f"  {C_BOLD}query_vector_builder{C_OFF}")
    if findings.knn_works_with_model_text and not findings.knn_works_with_query_text:
        ok("use model_text (current code is correct)")
    elif findings.knn_works_with_query_text and not findings.knn_works_with_model_text:
        warn("9.4 requires query_text — elastic_client.py.knn_search must be updated")
    elif findings.knn_works_with_model_text and findings.knn_works_with_query_text:
        ok("both model_text and query_text accepted — current code (model_text) is fine")
    else:
        fail("neither variant accepted — kNN search contract not validated")

    if findings.notes:
        print()
        print(f"  {C_BOLD}{C_YEL}NOTES{C_OFF}")
        for n in findings.notes:
            print(f"    - {n}")

    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

async def main() -> int:
    es_url = os.environ.get("ES_URL")
    es_key = os.environ.get("ES_API_KEY")
    inference_id = os.environ.get("INFERENCE_ID", "jinaai-embeddings")
    expected_dims = int(os.environ.get("EMBEDDING_DIMS", "1024"))

    if not es_url or not es_key:
        fail("ES_URL and ES_API_KEY must be set (export them or fill backend/.env)")
        return 1

    print()
    print(f"  {C_BOLD}EIS Contract Probe{C_OFF}")
    kv("ES_URL",       es_url)
    kv("INFERENCE_ID", inference_id)
    kv("EMBEDDING_DIMS (env)", expected_dims)

    client = AsyncElasticsearch(
        es_url,
        api_key=es_key,
        verify_certs=os.environ.get("ES_VERIFY_CERTS", "true").lower() == "true",
        request_timeout=60,
    )

    findings = Findings()
    try:
        await probe_connectivity(client)
        ep = await probe_inference_endpoint(client, inference_id)
        if ep is None:
            return 1
        await probe_text_embedding(client, inference_id, findings)
        await probe_image_embedding(client, inference_id, findings)
        await probe_query_vector_builder(client, inference_id, findings)
        print_summary(findings, expected_dims)
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
