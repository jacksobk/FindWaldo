# Elasticsearch Setup

One-time provisioning for the inference endpoint and index. Everything here
runs in **Kibana → Dev Console** against an Elasticsearch 9.x cluster with
the Elastic Inference Service available.

## 1. Provision the inference endpoint

The application uses a JinaAI inference endpoint for `jina-clip-v2`:

```
PUT _inference/text_embedding/jinaai-multimodal
{
  "service": "jinaai",
  "service_settings": {
    "model_id": "jina-clip-v2",
    "api_key": "<your-jina-api-key>",
    "dimensions": 1024,
    "similarity": "cosine"
  }
}
```

Verify it embeds text:

```
POST _inference/jinaai-multimodal
{ "input": "a man in a red and white striped shirt" }
```

You should get back a 1024-element float array.

### ⚠️ The text-vs-image caveat

A `text_embedding` task-type endpoint embeds query **text** but may reject
**image** input with a generic `400 input validation error`, even though
`jina-clip-v2` is multimodal. This repo handles that by falling back to the
Jina REST API for image embedding (set `JINA_API_KEY` in `backend/.env`).
The full diagnosis is in [`engineering-notes.md`](engineering-notes.md).

For a clean production setup where image embedding goes through the cluster,
provision an endpoint that accepts image input and remove the fallback. The
application reads `INFERENCE_ID` from the environment, so you can point it at
whichever endpoint you provision without code changes.

## 2. The index

The backend creates the index automatically on first startup if it doesn't
exist. The mapping it creates:

```json
{
  "mappings": {
    "properties": {
      "image_id":  { "type": "keyword" },
      "tile_id":   { "type": "keyword" },
      "row":       { "type": "integer" },
      "col":       { "type": "integer" },
      "scale":     { "type": "integer" },
      "bbox": {
        "type": "object",
        "properties": {
          "x": { "type": "integer" },
          "y": { "type": "integer" },
          "w": { "type": "integer" },
          "h": { "type": "integer" }
        }
      },
      "image_url": { "type": "keyword" },
      "image_w":   { "type": "integer" },
      "image_h":   { "type": "integer" },
      "label":     { "type": "text" },
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      }
    }
  },
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "refresh_interval": "1s"
  }
}
```

`label` is optional — populate it during ingest with OCR text or a caption
to enable hybrid (BM25 + vector) search.

## 3. The search query

The kNN query embeds the user's text *inside* the cluster via
`query_vector_builder`, so the application never holds a query vector:

```
POST <index>/_search
{
  "knn": {
    "field": "embedding",
    "k": 10,
    "num_candidates": 200,
    "query_vector_builder": {
      "text_embedding": {
        "model_id": "jinaai-multimodal",
        "model_text": "a man in a red striped shirt"
      }
    },
    "filter": { "term": { "image_id": "abc123" } }
  },
  "fields": ["image_id", "tile_id", "image_url", "bbox"],
  "_source": false
}
```

The `query_vector_builder` block is the key construct: Elasticsearch embeds
`model_text` with the named endpoint and runs HNSW kNN against the stored
tile vectors in a single round-trip.

## 4. Required API key privileges

The cluster API key in `backend/.env` (`ES_API_KEY`) needs:

- `manage_inference`, `read_inference` — endpoint verification
- `read`, `write`, `create_index`, `manage` on the target index

## Index mapping notes for scale

- **Shards / replicas** — the auto-created index uses 1 shard, 0 replicas
  (demo defaults). Set replicas per your availability SLA.
- **HNSW** — defaults are fine to tens of millions of vectors per node. For
  larger corpora, tune `m` and `ef_construction` in the `dense_vector`
  mapping.
- **`refresh_interval`** — `1s` suits a demo. Heavy bulk ingest should widen
  this and refresh explicitly after the batch.
