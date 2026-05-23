# EIS Contract Probe — Kibana Dev Console Companion

This is the manual equivalent of `scripts/probe_eis_contract.py`. Run these
in Kibana Dev Console against your 9.4 cluster. **Do not advance to Phase 2
application code until each contract resolves.**

For automation, prefer the Python script — it discovers shapes by trying
multiple variants, where Dev Console is one-shot.

---

## 0. Sanity check — endpoint exists

```
GET _inference/jinaai-embeddings
```

**Expect:** a 200 with `service: "elastic"` and `service_settings.model_id`
present. If 404, create it first:

```
PUT _inference/text_embedding/jinaai-embeddings
{
  "service": "elastic",
  "service_settings": {
    "model_id": "jina-clip-v2"
  }
}
```

---

## 1. Contract: Text Embedding

```
POST _inference/jinaai-embeddings
{
  "input": ["a man in a red and white striped shirt"]
}
```

**What to record from the response:**

| Question | Where to look |
|---|---|
| Top-level key wrapping the embedding? | One of `text_embedding`, `embedding`, `inference_results` |
| Is each block an object with `embedding`? Or a raw float list? | First element of that array |
| Vector dimension? | `length(response.<key>[0].embedding)` |

**The probe script handles all three keys; the application code in
`elastic_client.py.embed_images` and `embed_text` already falls through
to either `text_embedding` or `embedding`. If the actual key is something
else (e.g. `inference_results`), update the fallback list in those two
methods only.**

---

## 2. Contract: Image Embedding (Base64)

The probe script tries four input shapes against the live endpoint.
Run them yourself in Dev Console **in order**, stopping at the first
that returns a vector. Use any small JPEG and base64-encode it; for
quick testing a 1×1 pixel works.

A 1×1 pixel red JPEG (paste this verbatim — already base64-encoded):

```
/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD/2Q==
```

### Try variant A — data URI

```
POST _inference/jinaai-embeddings
{
  "input": ["data:image/jpeg;base64,/9j/4AAQ...redacted...AwD3+iiigD/2Q=="]
}
```

### Try variant B — raw base64 string (no data: prefix)

```
POST _inference/jinaai-embeddings
{
  "input": ["/9j/4AAQ...redacted...AwD3+iiigD/2Q=="]
}
```

### Try variant C — object form

```
POST _inference/jinaai-embeddings
{
  "input": [{ "image": "/9j/4AAQ...redacted...AwD3+iiigD/2Q==" }]
}
```

### Try variant D — object with data URI

```
POST _inference/jinaai-embeddings
{
  "input": [{ "image": "data:image/jpeg;base64,/9j/4AAQ...AwD3+iiigD/2Q==" }]
}
```

**Stop at the first variant that returns a numeric vector.** Record:

| Finding | Action |
|---|---|
| Variant A works | No change required — `elastic_client.py.embed_images` already sends data URIs |
| Variant B works | Change one line in `embed_images`: drop the `data:image/jpeg;base64,` prefix |
| Variant C or D works | Change `embed_images` to wrap each input in `{"image": ...}` |
| None work | Image ingest cannot proceed against this EIS endpoint. Open a support ticket — the EIS deployment may be text-only |

---

## 3. Contract: query_vector_builder.text_embedding

This contract has historically used the field name `model_text`. There is
some indication newer Elasticsearch versions also accept `query_text`. The
probe checks both.

### Setup — create a throwaway index

```
DELETE eis-contract-probe-tmp

PUT eis-contract-probe-tmp
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

(Replace `1024` with the dimension you observed in step 1 if different.)

### Index one stub document

```
POST eis-contract-probe-tmp/_doc/probe-1?refresh=true
{
  "embedding": [1.0, 0.0, 0.0, ... ]   // padded to your dim
}
```

### Try variant 1 — `model_text` (current code)

```
POST eis-contract-probe-tmp/_search
{
  "knn": {
    "field": "embedding",
    "k": 1,
    "num_candidates": 10,
    "query_vector_builder": {
      "text_embedding": {
        "model_id": "jinaai-embeddings",
        "model_text": "test query"
      }
    }
  },
  "size": 1,
  "_source": false
}
```

### Try variant 2 — `query_text`

```
POST eis-contract-probe-tmp/_search
{
  "knn": {
    "field": "embedding",
    "k": 1,
    "num_candidates": 10,
    "query_vector_builder": {
      "text_embedding": {
        "model_id": "jinaai-embeddings",
        "query_text": "test query"
      }
    }
  },
  "size": 1,
  "_source": false
}
```

**Decision matrix:**

| Variant 1 (`model_text`) | Variant 2 (`query_text`) | Action |
|---|---|---|
| OK | OK | No change required |
| OK | error | No change required |
| error | OK | Update `elastic_client.py.knn_search`: rename `model_text` → `query_text` |
| error | error | Open support ticket. Application code blocked. |

### Cleanup

```
DELETE eis-contract-probe-tmp
```

---

## When to advance to Phase 2 application code

Only after:

1. Text contract: confirmed the response key (`text_embedding` / `embedding` / other).
2. Image contract: confirmed at least one input shape returns a vector.
3. kNN contract: confirmed at least one of `model_text` / `query_text` works.

If any of the three fails to validate, the action is to **adjust
`elastic_client.py` only** — never to spread workaround logic into
ingest, search routes, or the frontend.
