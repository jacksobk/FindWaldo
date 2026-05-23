"""
Multi-query expansion for visual search.

Why expand:
    A user query like "find Waldo" embeds into a thin, abstract vector. The
    image-side embeddings are dense — they encode color, pose, clothing,
    composition. The thin query under-uses the model's representational
    capacity: many tiles match weakly, and the right tile rarely dominates.

    Expanding "find Waldo" into 3-5 visual phrasings — "a man in a red and
    white striped shirt", "a person with glasses and a hat in a crowd",
    "a character with striped clothing in a busy scene" — exercises
    different facets of the model's vision encoder. Each variant lights up
    a slightly different region of the embedding space; tiles that genuinely
    match the user's intent tend to score well across multiple variants,
    which is the signal the merge step rewards downstream.

What this module does:
    Given a user query, return a list of expansion variants. The first item
    is always the original query verbatim — that guarantees we never *remove*
    matches the user's literal phrasing would have found, only *add* coverage
    from synonymous phrasings.

What this module is NOT:
    - It does not embed anything. Embedding stays in the search orchestrator.
    - It does not average or combine vectors. Averaging collapses the
      multi-faceted signal expansion is meant to preserve.
    - It does not run kNN. The orchestrator fans out to the search backend
      with each variant separately.

Strategies:
    Two strategies are wired in. The orchestrator picks one based on config:

      "rule"  — deterministic, zero-cost templates. Strips imperative verbs
                ("find X" -> "X"), wraps the cleaned noun phrase in a small
                set of CLIP-friendly prompt forms ("a photo of {n}", "{n}
                in a busy scene", etc.), and adds a domain-specific visual
                description when the query mentions a known character.
                Always works, no external dependencies, no latency budget.

      "llm"   — calls an LLM to generate richer variants. Stub for a future
                phase. Currently raises NotImplementedError so a misconfig
                fails loudly rather than silently degrading to "rule".

    Both strategies return at most `max_variants` items, with the original
    query as item 0. Duplicates after normalization are removed.

Determinism:
    The "rule" strategy is fully deterministic given the same input — same
    query in, same variants out. Important for reproducible benchmarks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("findwaldo.query_expansion")


# Imperative / question prefixes we strip before applying noun-phrase
# templates. "find Waldo" → "Waldo"; "show me a striped umbrella" →
# "a striped umbrella". Order matters — longer phrases first so we don't
# leave dangling words. Matched case-insensitively at the start of the
# query only.
IMPERATIVE_PREFIXES: tuple[str, ...] = (
    "show me where",
    "show me a",
    "show me an",
    "show me the",
    "show me",
    "where is the",
    "where is a",
    "where is an",
    "where is",
    "where's the",
    "where's a",
    "where's an",
    "where's",
    "look for the",
    "look for a",
    "look for an",
    "look for",
    "find me the",
    "find me a",
    "find me an",
    "find me",
    "find the",
    "find a",
    "find an",
    "find",
)

# Prompt templates applied to the cleaned noun phrase. CLIP literature shows
# small prompt ensembles ("a photo of {x}", "an illustration of {x}", etc.)
# add ~10% accuracy over a bare label. We adapt that here for the cartoon /
# busy-scene domain by adding scene-context variants ("{x} in a busy scene",
# "{x} in a crowd") that match the way Where's Waldo images are composed.
NOUN_TEMPLATES: tuple[str, ...] = (
    "a photo of {n}",
    "an illustration of {n}",
    "{n} in a busy scene",
    "{n} in a crowd",
)

# Domain-specific descriptive expansions for known characters. Each entry
# maps a trigger phrase to a fuller visual description. When the user query
# contains a trigger, we append the description as one of the variants.
# Triggers are matched case-insensitively as whole words.
CHARACTER_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("waldo",
     "a man wearing a red and white horizontally striped shirt, a red and "
     "white striped beanie hat, round black glasses, and blue jeans"),
    ("wenda",
     "a woman wearing a red and white horizontally striped shirt, a red and "
     "white striped beanie hat, and round black glasses"),
    ("wizard whitebeard",
     "an old wizard with a long flowing white beard, wearing red robes and "
     "a tall pointed hat"),
    ("whitebeard",
     "an old wizard with a long flowing white beard, wearing red robes"),
    ("odlaw",
     "a man wearing a yellow and black horizontally striped shirt, "
     "yellow and black striped hat, dark sunglasses, and a black mustache"),
    ("woof",
     "a small dog with a red and white striped tail visible"),
)


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExpansionResult:
    """Output of `expand_query`.

    Attributes:
        variants: List of query strings to issue to the search backend.
                  Index 0 is always the user's original query, verbatim.
        strategy: Which strategy produced these variants. Surfaced for
                  telemetry / UI badge.
    """
    variants: list[str]
    strategy: str


def _normalize_for_dedup(s: str) -> str:
    """Lowercase + collapse whitespace for de-duplication only.

    The actual variants we return preserve the original casing the templates
    produce. We just don't want two variants that differ only in trailing
    whitespace to both be sent to the backend.
    """
    return _WHITESPACE_RE.sub(" ", s.strip().lower())


def _matches_trigger(query: str, trigger: str) -> bool:
    """Whole-word, case-insensitive match for a trigger phrase in the query.

    "wizard" should match a query of "find the wizard" but not "wizardry".
    A simple substring check would be wrong; using \\b in a regex compiled
    per call is fine — these patterns are short and triggers are few.
    """
    pattern = r"\b" + re.escape(trigger.lower()) + r"\b"
    return re.search(pattern, query.lower()) is not None


def _strip_imperative(query: str) -> str:
    """Remove an imperative or question prefix from the query if present.

    "find Waldo" -> "Waldo"
    "show me a striped umbrella" -> "a striped umbrella"
    "where is the wizard" -> "the wizard"
    "Waldo" (no prefix) -> "Waldo"

    The result is always a noun phrase suitable for wrapping in templates
    like "a photo of {n}". Matching is case-insensitive and only at the
    start of the query.
    """
    lower = query.lower()
    for prefix in IMPERATIVE_PREFIXES:
        # Match the prefix followed by whitespace OR end-of-string. Without
        # the boundary check, "find" would match "finder".
        if lower.startswith(prefix + " "):
            return query[len(prefix) + 1:].strip()
        if lower == prefix:
            return ""
    return query.strip()


def _expand_rule_based(query: str, max_variants: int) -> list[str]:
    """Generate variants using prompt templates + character expansions.

    Order is significant for the dedup pass:
        1. Original query verbatim (always)
        2. Character-specific descriptive expansion (most targeted)
        3. Prompt templates wrapped around the noun phrase

    Earlier (more targeted) variants win when there's textual overlap.
    """
    out: list[str] = [query]
    noun_phrase = _strip_imperative(query)

    # Character-specific expansion (at most one applies per query — we take
    # the first match, since multiple character mentions in one query is
    # rare and combining their descriptions would be incoherent).
    for trigger, description in CHARACTER_DESCRIPTIONS:
        if _matches_trigger(query, trigger):
            out.append(description)
            break

    # Prompt templates wrap around the noun phrase, not the original query.
    # Skip when the noun phrase ended up empty (the user typed only an
    # imperative like "find") — wrapping nothing produces nonsense like
    # "a photo of ".
    if noun_phrase:
        for tmpl in NOUN_TEMPLATES:
            out.append(tmpl.format(n=noun_phrase))

    # De-duplicate while preserving order, then cap.
    seen: set[str] = set()
    unique: list[str] = []
    for v in out:
        key = _normalize_for_dedup(v)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(v)
        if len(unique) >= max_variants:
            break
    return unique


def expand_query(
    query: str,
    *,
    strategy: str = "rule",
    max_variants: int = 4,
) -> ExpansionResult:
    """Expand a user query into multiple variants for downstream search.

    Args:
        query: The user's input query. Stripped but otherwise untouched.
        strategy: "rule" (default, deterministic templates) or "llm"
                  (currently unimplemented — raises). Unknown strategy
                  values fall back to "rule" with a warning.
        max_variants: Hard upper bound on the variants returned, including
                      the original query. Must be >= 1.

    Returns:
        An ExpansionResult with at least one variant (the original query,
        unless the input is empty in which case a single empty-string
        variant is returned). Never raises for empty input — that's the
        orchestrator's job to reject upstream.
    """
    if max_variants < 1:
        raise ValueError("max_variants must be >= 1")

    cleaned = query.strip() if query else ""
    if not cleaned:
        return ExpansionResult(variants=[""], strategy=strategy)

    if strategy == "rule":
        variants = _expand_rule_based(cleaned, max_variants)
    elif strategy == "llm":
        # Hook for a later phase. Failing loudly is better than silently
        # downgrading because a stale config could mask a real outage.
        raise NotImplementedError(
            "LLM-based query expansion is not implemented yet. "
            "Set QUERY_EXPANSION_STRATEGY=rule or leave unset."
        )
    else:
        log.warning(
            "Unknown query expansion strategy %r; falling back to 'rule'",
            strategy,
        )
        variants = _expand_rule_based(cleaned, max_variants)
        strategy = "rule"

    return ExpansionResult(variants=variants, strategy=strategy)
