# Engineering Notes

Honest write-ups of the non-obvious problems in this project and how they
were diagnosed. These are the parts that don't fit in a clean architecture
diagram but are the most representative of real inference-engineering work.

---

## 1. The inference endpoint that embedded text but rejected images

### Symptom

The `/api/ingest` path failed with a generic Elasticsearch error whenever it
tried to embed image tiles:

```
elasticsearch.BadRequestError: BadRequestError(400, 'status_exception',
  'Received an input validation error response for request from
   inference entity id [jinaai-multimodal] status [400]')
```

Text queries worked perfectly. Image embedding always 400'd. The error
message was unhelpful — "input validation error" with no field-level detail,
because the Elasticsearch JinaAI connector wraps the upstream provider error
and strips the body.

### Diagnosis

The model behind the endpoint (`jina-clip-v2`) is genuinely multimodal, so
"the model can't do images" was wrong. The question was *where* the image
input was being rejected — at the connector, or upstream at Jina.

Step 1 — **probe Jina directly.** Bypassing Elasticsearch entirely and
calling Jina's REST API with the exact same base64 crops succeeded, returning
1024-d vectors. So the crops were valid and the model accepted them. The
problem was the Elasticsearch path specifically.

Step 2 — **probe the connector with multiple input shapes.** Sending the
same crop to the EIS endpoint in seven different request shapes (raw base64
string, data URI, `{image: …}` object, with/without `input_type` hints,
etc.) — every image-shaped request failed; a plain *text* control request
succeeded with a 1024-d vector.

Step 3 — **inspect the endpoint config:**

```
GET _inference/jinaai-multimodal
→ "task_type": "text_embedding"
```

There it was. The endpoint had been provisioned as `task_type=text_embedding`.
Despite the multimodal model and the "multimodal" name, the endpoint accepts
*text input only*. The existing indexed tiles had been embedded under a
different configuration that no longer existed; the live endpoint could
embed query text (which is why search worked) but would never accept an
image again.

### Fix

Two options:

1. **Reprovision** a true multimodal EIS endpoint in Kibana. The clean fix,
   but operational — a Dev Console change outside the application.
2. **Fall back in code:** when the EIS endpoint rejects image input with the
   specific `400 input validation` signature, embed the images via Jina's
   REST API directly. Same model (`jina-clip-v2`), same 1024-d output, same
   vector space — so cosine similarity between an EIS-embedded *query* vector
   and a Jina-REST-embedded *tile* vector is well-defined and correct.

This repo implements (2) as a transparent fallback in
[`backend/elastic_client.py`](../backend/elastic_client.py) (`_embed_via_jina_rest`),
applied on exactly that error signature and nowhere else. It's the right
call for getting unblocked without a cluster change, and it's documented
honestly rather than hidden: the production recommendation in
[`architecture.md`](architecture.md) is to provision the proper multimodal
endpoint and retire the fallback.

### Why this is worth writing down

The instinct to *not* trust the surface error ("the model can't do images"),
to isolate the failure with direct probes, and to find a correct workaround
that preserves vector-space compatibility is the actual job. The fallback is
a few dozen lines; the diagnosis is the engineering.

---

## 2. Relevance tuning needs an absolute floor, not just a relative one

The first thresholding attempt used only a *relative* score ratio — keep
hits within X% of the top hit. It barely changed anything, because the
within-query score spread was always tight: even for a query whose target
wasn't in the image, the top ten tiles clustered within ~20% of each other.
The relative filter had nothing to remove.

The signal that distinguishes "target present" from "target absent" lives in
the **absolute top score across queries**, not in the spread within a single
query. Present-target queries topped out at 0.83–0.95; absent-target queries
("a pink dragon", "a 1970s Cadillac") capped at ~0.59–0.76. That gap is the
thing to threshold on.

The fix was a layered filter: relative ratio (keeps the within-query shape),
a per-hit absolute floor (a defensive noise gate), and a **top-1 hard floor**
that returns zero hits when the single best match is below a value sitting in
the empirical gap between the present and absent score bands. Full
methodology and the measured numbers are in
[`relevance-tuning.md`](relevance-tuning.md).

The lesson that generalizes: *threshold on the axis where the signal actually
lives.* It took a benchmark across query categories to see that the
discriminating axis was absolute-top-score, not within-query-ratio.

---

## 3. The prototype subsystem worked, and still got shipped off-by-default

The reference-target subsystem builds an averaged "prototype" vector from a
few reference crops of a known subject and runs an extra kNN leg for queries
that name that subject. Architecturally it works: it finds the subject and
contributes recall the text query misses.

But measured honestly against near-identical decoys, it *hurt* precision. In
the Where's Waldo set, Waldo and Wenda wear the same red-and-white stripes; a
prototype built from five small Waldo crops scored 0.94–0.95 on *both*
characters and on generic striped figures — a band too tight to discriminate
identity. On the one query where the prototype merged into the results
(`find Waldo`), it dragged the top score *below* what the plain text path
achieved on the same image, and its top-ranked tile was visually confirmed to
not contain Waldo.

So it's **disabled by default**. The code stays (it's a clean, data-driven
subsystem and works fine for visually distinctive subjects), but the demo
runs without it because measured precision matters more than an
architecturally cool feature. Re-enabling is a one-line env change once the
crop set improves (more crops, larger context, or contrastive prototyping —
Waldo-prototype minus Wenda-prototype).

The general point: build the feature, *measure* it, and have the discipline
to ship it off when the measurement says so.

---

## 4. Domain fit: where CLIP works and where it doesn't

CLIP-class embeddings reflect their training distribution — web-scraped
image/caption pairs. That makes the pipeline strong on:

- product photography (retail shelf search, e-commerce),
- natural scenes (crowd photos, aerial imagery),
- document and UI layouts,
- illustrations and cartoons (hence Waldo working at all).

And weak on specialist domains the training data barely covers:

- **medical imaging** (tested on heart-anatomy diagrams — results were
  mediocre; queries like "left ventricle" require relative-spatial reasoning
  CLIP doesn't do, and the technical vocabulary is anchored to common-language
  meanings in the text encoder),
- radiology, pathology, CAD drawings, sheet music.

This was tested rather than assumed — an unlabeled heart diagram was ingested
and queried, and it under-performed exactly as the training-distribution
argument predicts. The fix for those domains is a domain-adapted embedding
model, which is a *model* project, not an *architecture* project: the
tile → embed → kNN → rerank pipeline is unchanged; only the embedding model
behind `_inference` swaps out.

Stating this limitation plainly is more useful than a demo that overclaims
and gets caught — and it points cleanly at where the real strengths are
(retail / inventory / document search), all of which the existing pipeline
handles with no changes.
