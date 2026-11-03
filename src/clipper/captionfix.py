"""Caption word times corrected from the finished audio (reviewer #350/#367).

The transcript's forced alignment can be 0.3-0.6 s off after fillers ("um"): alliance c1 FEELS captioned at
1.76 s, spoken at 1.30 s. The cut audio is transcribed again (whisper on the exact segments) and words whose
start disagrees by more than RETIME_MIN are moved to the heard start.
"""
from __future__ import annotations

from .syncmeter import pair

RETIME_MIN = 0.20   # s: smaller differences are the forced aligner's normal noise, left alone
MAX_MOVED_DUR = 0.45  # s: a moved word's length is capped so its midpoint stays inside its segment
RETIME_MAX = 1.00   # s: a larger difference is a mismatched word, not a timing error


def output_times(words: list[dict], segments: list[tuple[float, float, float]]) -> dict[int, tuple[float, int]]:
    """word index -> (output time, segment index) for words build_ass would caption (same inclusion rule).
    A word that falls into two segments is left out (retiming it would move both)."""
    found: dict[int, list[tuple[float, int]]] = {}
    for k, (seg_s, seg_e, off) in enumerate(segments):
        for i, w in enumerate(words):
            if w["s"] >= seg_s - 0.02 and (w["s"] + w["e"]) / 2 <= seg_e:
                found.setdefault(i, []).append((w["s"] - seg_s + off, k))
    return {i: v[0] for i, v in found.items() if len(v) == 1}


def retime_words(words: list[dict], segments: list[tuple[float, float, float]],
                 heard: list[tuple[str, float]], bias: float = 0.0) -> tuple[list[dict], list[str]]:
    """words: caption words (source time); heard: whisper (text, output start) of the finished audio.
    bias: build_ass later shifts every word earlier by sync_shift, so a moved word is placed at heard + bias.
    Returns the words with grossly misplaced starts moved to the heard start, and a log line per moved word."""
    times = output_times(words, segments)
    idx = sorted(times)
    cap = [(words[i]["w"], times[i][0]) for i in idx]
    out = [dict(w) for w in words]
    moved = []
    for ci, hj in pair(cap, heard):
        t_cap, t_heard = cap[ci][1], heard[hj][1]
        diff = t_cap - t_heard
        if RETIME_MIN < abs(diff) <= RETIME_MAX:
            i = idx[ci]
            w = out[i]
            seg_s, seg_e = segments[times[i][1]][:2]
            new_s = w["s"] - (diff - bias)
            # build_ass captions a word only if it starts inside its segment and its middle is inside too. A word
            # moved out of its segment (whisper heard it in the neighbouring segment's output span) would
            # vanish from the captions and leave a hole under speech: keep its aligned time instead.
            if not seg_s - 0.02 <= new_s <= seg_e:
                continue
            dur = min(w["e"] - w["s"], MAX_MOVED_DUR)  # aligned durations after fillers are inflated (incident: 1.1 s)
            w["s"] = new_s
            w["e"] = min(new_s + dur, seg_e)
            moved.append(f"{w['w']}@{t_cap:.2f}->{t_heard:.2f}")
    # A moved word keeps its (shifted) duration, but the heard word often lasts longer: hold it until the next
    # word starts when the gap is small, or the caption leaves a hole under speech (QA caption gap).
    touched = {idx[ci] for ci, _ in pair(cap, heard) if RETIME_MIN < abs(cap[ci][1] - heard[_][1]) <= RETIME_MAX}
    order = sorted(range(len(out)), key=lambda i: out[i]["s"])
    for a, b in zip(order, order[1:]):
        if (a in touched or b in touched) and 0 < out[b]["s"] - out[a]["e"] <= 0.6:
            # never past the end of a's own segment: its middle would leave the segment and build_ass drops it
            # (a word replayed in two segments, e.g. the cold open, is held to the earliest of their ends)
            ends = [e_ for s_, e_, _o in segments
                    if out[a]["s"] >= s_ - 0.02 and (out[a]["s"] + out[a]["e"]) / 2 <= e_]
            out[a]["e"] = min([out[b]["s"]] + ends)
    return out, moved


def heard_in_output(wav, wav_t0: float, segments: list[tuple[float, float, float]]) -> list[tuple[str, float]]:
    """Whisper words (text, output start) of the audio the segments produce, read from the segment WAV."""
    import numpy as np
    from faster_whisper import WhisperModel
    from .transcribe import load_audio
    audio = np.concatenate([load_audio(wav, s - wav_t0, e - s) for s, e, _ in segments])
    segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(
        audio, language="en", word_timestamps=True)
    return [(w.word, float(w.start)) for s in segs for w in (s.words or [])]
