"""Visual-variety filter for rank: reject a window that is one long talking-head shot with no b-roll plan.

182604-c1 v2r3 (reviewer round 3, 7/10): 29 s of one talking head in one composition, so visual richness blocked
a PASS. Fix at selection time, not in QA: count source shot changes (scene detection) and drop the candidate
when its longest shot exceeds MAX_SHOT_SEC and the transcript offers no number cards (broll.plan_inserts).
"""
from __future__ import annotations

from pathlib import Path

MAX_SHOT_SEC = 12.0   # longest allowed single shot without a b-roll or graphics plan (Codee #1776)


def longest_shot(cut_times: list[float], dur: float) -> float:
    """Longest stretch between camera cuts (seconds from window start) in a window of `dur` seconds."""
    edges = [0.0] + sorted(t for t in cut_times if 0.0 < t < dur) + [dur]
    return max(b - a for a, b in zip(edges, edges[1:]))


def broll_cards(words: list[dict], start: float, end: float) -> int:
    """Number of number cards broll.plan_inserts would place for the window (words use s/e/w, source time)."""
    from . import broll
    ow = [{"w": w["w"], "s": w["s"] - start, "e": w["e"] - start} for w in words if start <= w["s"] < end]
    return len(broll.plan_inserts(broll.number_phrases(ow), end - start))


def variety_problem(video: Path | None, words: list[dict], start: float, end: float) -> str | None:
    """None if the window is fine, otherwise the reason it is dropped."""
    from .render import scene_cuts
    if video is None:
        return None
    dur = end - start
    longest = longest_shot([t for t, _ in scene_cuts(video, start, dur)], dur)
    if longest <= MAX_SHOT_SEC:
        return None
    if broll_cards(words, start, end):
        return None
    return f"one shot of {longest:.0f} s (limit {MAX_SHOT_SEC:.0f} s) and no b-roll plan"
