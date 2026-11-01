"""Locate verbatim quotes from the ranker in the transcript.

The ranker sees second-resolution segment timestamps and its start/end/payoff numbers can be
far off (seen: ~1000 s) or land mid-sentence. Quotes are reliable: we find them in the text and
take the times from there.
"""

import re
from difflib import SequenceMatcher

MIN_RATIO = 0.75  # fuzzy match threshold (whisper base vs small transcripts differ slightly)


def norm(text: str) -> list[str]:
    return [t for t in re.sub(r"[^\w%'\s]", " ", text.lower().replace("’", "'")).split() if t]


def segment_words(segments: list[dict]) -> list[dict]:
    """Segment-level transcript -> word list with times spread evenly over each segment."""
    out = []
    for s in segments:
        toks = s["text"].split()
        if not toks:
            continue
        dur = max(s["e"] - s["s"], 0.01)
        for i, t in enumerate(toks):
            out.append({"w": t, "s": s["s"] + dur * i / len(toks), "e": s["s"] + dur * (i + 1) / len(toks)})
    return out


def find_quote(words: list[dict], quote: str, near: float | None = None,
               lo: float | None = None, hi: float | None = None) -> tuple[int, int] | None:
    """(first index, last index) of the best match of `quote` in `words`, or None.

    Exact token match first; otherwise the best fuzzy window (ratio >= MIN_RATIO). Ties go to
    the match nearest `near`. lo/hi restrict the search to a time range.
    """
    q = norm(quote)
    if not q:
        return None
    toks = [norm(w["w"]) for w in words]
    flat, owner = [], []
    for i, t in enumerate(toks):
        for x in t:
            flat.append(x)
            owner.append(i)
    n = len(q)
    best, best_key = None, None
    for k in range(0, max(len(flat) - n + 1, 0)):
        i0, i1 = owner[k], owner[k + n - 1]
        if lo is not None and words[i0]["s"] < lo or hi is not None and words[i1]["e"] > hi:
            continue
        window = flat[k:k + n]
        if window == q:
            ratio = 1.0
        else:
            if window[0] != q[0] and window[-1] != q[-1]:
                continue  # cheap prefilter: at least one end must agree
            ratio = SequenceMatcher(None, " ".join(window), " ".join(q)).ratio()
            if ratio < MIN_RATIO:
                continue
        dist = abs(words[i0]["s"] - near) if near is not None else 0.0
        key = (-ratio, dist)
        if best_key is None or key < best_key:
            best, best_key = (i0, i1), key
    return best
