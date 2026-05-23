# Relevance Tuning

How the confidence thresholds in this system were chosen — from a measured
benchmark across query categories, not guessed. The benchmark harness that
produced these numbers is [`scripts/tune_relevance.py`](../scripts/tune_relevance.py).

## The benchmark harness

`tune_relevance.py` runs a catalogue of test queries against the live backend
and records, per query: top score, normalized-score range, hit count after
thresholding, latency breakdown, and — when a ground-truth bounding box is
supplied — IoU-based top-k success metrics. It grades two ways:

- **Coarse pass/fail** — did a present-target query return anything, and did
  an absent-target query correctly return nothing.
- **Top-k success** — for queries with a ground-truth bbox, did a returned
  tile overlap the truth at IoU ≥ 0.5 within top-1 / top-3 / top-5.

It also tracks **false-positive rate** on absent-target queries (the cleanest
signal that thresholding is too loose), emits JSON to stdout and optional CSV,
and prints per-category and overall aggregates to stderr. Stdlib only, fully
read-only against the backend, deterministic given a fixed catalogue.

Query categories: `waldo` (present target), `generic_object`, `ambiguous`
(single words like "red", "man"), `lookalike` (red/white stripes that aren't
the target), and `absent` (subjects not in the image at all).

## The measured score distribution

Running the catalogue with thresholds disabled revealed the score bands by
category (top-1 cosine/rerank score per query):

| Category | top-score range | mean |
|---|---|---|
| `waldo` (present) | 0.83 – 0.95 | 0.89 |
| `lookalike` | 0.90 – 0.93 | 0.91 |
| `generic_object` | 0.65 – 0.95 | 0.85 |
| `ambiguous` | 0.67 – 0.84 | 0.75 |
| `absent` | **0.59 – 0.76** | 0.65 |

The key observation: **absent-target top scores (≤ 0.76) sit below
present-target top scores (≥ 0.83), with a clear gap in between.** That gap
is where an absolute threshold separates signal from noise.

The within-query *normalized* spread, by contrast, was uniformly tight
(0.81–0.97 across every category) — which is why a purely *relative*
threshold (keep hits within X% of the top) did almost nothing: there was
nothing in the tail to trim. The discriminating axis is the absolute top
score, not the within-query ratio.

## The layered filter

Three filters applied in order, all env-configurable, all defaulting to
no-op so the change is byte-for-byte inert when unset:

1. **`SCORE_THRESHOLD_RATIO`** (relative) — drop hits whose normalized score
   is below this fraction of the top hit. Shapes the within-query result
   list. Kept at `0.80` (and later tightened toward `0.92` to make
   present-target queries return ~3–5 tightly-grouped hits instead of a full
   ten).

2. **`MIN_ABSOLUTE_SCORE`** (per-hit floor) — drop any individual hit below
   this absolute score. A defensive noise gate set at `0.55`, below the
   lowest legitimate top score, so it never clips a real match — it only
   catches pathologically weak individual tiles.

3. **`MIN_TOP_SCORE_TO_RETURN`** (top-1 hard floor) — if the single best hit
   is below this, return **zero** hits. This is the primary absent-target
   rejection mechanism. Set into the empirical gap between the absent band
   (≤ 0.76) and the present band (≥ 0.83).

## Trade-offs

- **Tighten `MIN_TOP_SCORE_TO_RETURN`** → fewer false positives on absent
  queries, but risk of false negatives if a legitimate query happens to score
  low. The safe value lives strictly between the two score bands; pushing it
  above the present-target floor (0.83) starts rejecting real matches.

- **Tighten `SCORE_THRESHOLD_RATIO`** → cleaner, shorter result lists, but
  risk of trimming borderline-useful tiles. Because the within-query spread is
  tight, this knob mostly controls *how many* of the top cluster you keep.

- **A collateral effect:** a `generic_object` query whose best legitimate
  match scored ~0.65 ("giant lollipop") gets rejected by a top-floor set high
  enough to also reject absent queries (whose best reached ~0.76). There's no
  single floor that admits the 0.65 match *and* rejects the 0.76 false
  positive — they overlap. For a demo this is the right trade (better to say
  "no confident match" than to surface a wrong tile); raising that match's
  score is a data problem — better crops or rephrasing — not a threshold
  problem.

## Result

Applying the thresholds took the absent-target false-positive rate from
**100% → 0%** on the benchmark catalogue (every "pink dragon" / "Cadillac" /
"labrador" query correctly returns nothing), while present-target queries
continued to return their matches. Disabling the prototype subsystem (see
[`engineering-notes.md`](engineering-notes.md)) and trimming the reranker
candidate pool also cut average query latency by roughly a third.

The point of the exercise: the thresholds aren't magic numbers, they're read
off a measured distribution, and the harness makes re-tuning for a new image
corpus a single command — which matters because the bands shift by domain
(retail photos and anatomy diagrams do *not* share Where's Waldo's score
distribution).
