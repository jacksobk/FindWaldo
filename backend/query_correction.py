"""
Lightweight query correction for visual search.

Why: CLIP embedding models are reasonably robust to misspellings, but obvious
typos ("stripped" → "striped", "Waldoe" → "Waldo") still degrade match quality.
A full-blown spell-checker is overkill for a demo; what we want is to fix
the small set of words people typically misspell when searching this kind of
scene, while leaving everything else alone.

Approach: small curated vocabulary of "correct" terms for the Where's Waldo
domain plus general visual descriptors. For each input word, if it's not
already in the vocab and there's a single high-confidence close match in the
vocab, swap it. Conservative — when in doubt, leave the word as-is rather
than risk a wrong correction.

This runs in microseconds (Python stdlib only, no model calls), so we apply
it on every search before embedding.
"""
import difflib
import re

# Domain vocabulary. Keep this short and high-signal — every word here is a
# correction target. Adding too many obscure words causes false positives.
# Lower-cased; matched case-insensitively.
DOMAIN_VOCAB: frozenset[str] = frozenset({
    # Where's Waldo characters and themes
    "waldo", "wenda", "wizard", "whitebeard", "odlaw", "woof",
    # Scene types
    "beach", "candy", "factory", "ocean", "skiing", "space", "station",
    "underground", "toys", "giants", "film", "set", "food", "court",
    # Common visual descriptors
    "striped", "checkered", "polkadot", "spotted", "plaid",
    "umbrella", "umbrellas", "sailboat", "sailboats", "boat", "boats",
    "ship", "ships", "horse", "horses", "dragon", "dragons", "ladder",
    "ladders", "balloon", "balloons", "kite", "kites", "tower", "towers",
    "windmill", "windmills", "castle", "tent", "tents", "stripes",
    # Colors
    "red", "white", "blue", "yellow", "green", "orange", "purple", "pink",
    "black", "brown", "gold", "silver",
    # People descriptors
    "man", "woman", "child", "person", "people", "wizard", "robe", "robes",
    "shirt", "shirts", "hat", "hats", "glasses", "scarf", "cape", "boots",
    "lifeguard", "lifeguards", "swimmer", "swimmers", "soldier", "soldiers",
    # Actions
    "running", "swimming", "falling", "carrying", "holding", "riding",
    "wearing", "playing", "throwing", "catching", "reading", "watching",
    # Objects
    "ball", "book", "books", "flag", "flags", "sword", "swords",
    "candy", "ice", "cream", "pizza", "burger",
})


def _correct_word(word: str, cutoff: float = 0.82) -> str:
    """
    Correct a single word against the domain vocab. Conservative: only swaps
    when there's exactly one strong match.

    cutoff is the difflib similarity ratio — 0.82 means "very close" (about
    one-character edits in short words). Higher = more conservative.
    """
    lower = word.lower()
    # Already in vocab — no correction needed.
    if lower in DOMAIN_VOCAB:
        return word
    # Don't correct very short words (high false-positive rate) or long
    # phrases (likely proper nouns or things we don't know about).
    if len(lower) < 4 or len(lower) > 20:
        return word
    # Don't correct anything with non-alphabetic characters (numbers, etc.)
    if not lower.isalpha():
        return word

    matches = difflib.get_close_matches(lower, DOMAIN_VOCAB, n=2, cutoff=cutoff)
    if len(matches) != 1:
        # Either no match or ambiguous — leave alone.
        return word

    # Preserve original case pattern as best we can. If original was all-caps,
    # uppercase the correction; if titlecase, titlecase; otherwise lowercase.
    correction = matches[0]
    if word.isupper():
        return correction.upper()
    if word.istitle():
        return correction.title()
    return correction


def correct_query(query: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Apply word-level correction to a query string.

    Returns: (corrected_query, list_of_corrections)
        corrected_query: the new query string with substitutions applied
        list_of_corrections: list of (original, corrected) pairs, useful for
                             telling the user what got autocorrected. Empty
                             if no corrections were made.

    Punctuation, spaces, and word order are preserved. Only individual words
    that have unambiguous near-matches in the domain vocab get swapped.
    """
    if not query.strip():
        return query, []

    corrections: list[tuple[str, str]] = []
    out_parts: list[str] = []

    # Split preserving whitespace so we can rejoin without losing the original
    # spacing. The split pattern captures both word characters and the
    # separators between them.
    tokens = re.findall(r"\w+|\W+", query)
    for tok in tokens:
        if tok.isalnum() or (tok and tok[0].isalpha()):
            corrected = _correct_word(tok)
            if corrected != tok:
                corrections.append((tok, corrected))
            out_parts.append(corrected)
        else:
            out_parts.append(tok)

    return "".join(out_parts), corrections
