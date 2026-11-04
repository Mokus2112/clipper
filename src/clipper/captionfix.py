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


def output_occurrences(words: list[dict], segments: list[tuple[float, float, float]]) -> list[tuple[int, int, float]]:
    """(word index, segment index, output time) for every caption occurrence build_ass would draw (same inclusion
    rule), sorted by output time. A word replayed in two segments (cold open + main) has one occurrence per segment."""
    occ = [(i, k, w["s"] - seg_s + off)
           for k, (seg_s, seg_e, off) in enumerate(segments)
           for i, w in enumerate(words)
           if w["s"] >= seg_s - 0.02 and (w["s"] + w["e"]) / 2 <= seg_e]
    return sorted(occ, key=lambda o: o[2])


def retime_words(words: list[dict], segments: list[tuple[float, float, float]],
                 heard: list[tuple[str, float]], bias: float = 0.0) -> tuple[list[dict], list[str]]:
    """words: caption words (source time); heard: whisper (text, output start) of the finished audio.
    bias: build_ass later shifts every word earlier by sync_shift, so a moved word is placed at heard + bias.
    Returns the words with grossly misplaced starts moved to the heard start, and a log line per moved word.
    Each occurrence is timed on its own: a word replayed in two segments gets a per-segment override
    w["ov"] = {segment index: (s, e)} that build_ass uses for that segment (its base s/e stay the source time)."""
    occ = output_occurrences(words, segments)
    cap = [(words[i]["w"], t) for i, _k, t in occ]
    cur = {(i, k): [words[i]["s"], words[i]["e"]] for i, k, _t in occ}  # (word, segment) -> new [s, e]
    moved, touched = [], set()
    for ci, hj in pair(cap, heard):
        i, k, t_cap = occ[ci]
        diff = t_cap - heard[hj][1]
        if RETIME_MIN < abs(diff) <= RETIME_MAX:
            seg_s, seg_e = segments[k][:2]
            w = words[i]
            new_s = w["s"] - (diff - bias)
            # build_ass captions a word only if it starts inside its segment and its middle is inside too. A word
            # moved out of its segment (whisper heard it in the neighbouring segment's output span) would
            # vanish from the captions and leave a hole under speech: keep its aligned time instead.
            if not seg_s - 0.02 <= new_s <= seg_e:
                continue
            dur = min(w["e"] - w["s"], MAX_MOVED_DUR)  # aligned durations after fillers are inflated (incident: 1.1 s)
            cur[(i, k)] = [new_s, min(new_s + dur, seg_e)]
            touched.add((i, k))
            moved.append(f"{w['w']}@{t_cap:.2f}->{heard[hj][1]:.2f}")
    # A moved word keeps its (shifted) duration, but the heard word often lasts longer: hold it until the next
    # word of the same segment starts when the gap is small, or the caption leaves a hole under speech.
    for k, (seg_s, seg_e, _o) in enumerate(segments):
        order = sorted((key for key in cur if key[1] == k), key=lambda key: cur[key][0])
        for a, b in zip(order, order[1:]):
            if (a in touched or b in touched) and 0 < cur[b][0] - cur[a][1] <= 0.6:
                # never past the end of the segment: the middle would leave it and build_ass drops the word
                cur[a][1] = min(cur[b][0], seg_e)
    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        ks = [k for (j, k) in cur if j == i]
        if len(ks) == 1:
            (k,) = ks
            if (i, k) in touched or cur[(i, k)][1] != w["e"]:
                w["s"], w["e"] = cur[(i, k)]
        elif ks and any((i, k) in touched or cur[(i, k)][1] != w["e"] for k in ks):
            w["ov"] = {k: tuple(cur[(i, k)]) for k in ks}
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
