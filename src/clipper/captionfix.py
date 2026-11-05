"""Caption word times corrected from the finished audio (reviewer #350/#367).

The transcript's forced alignment can be 0.3-0.6 s off after fillers ("um"): alliance c1 FEELS captioned at
1.76 s, spoken at 1.30 s. The cut audio is transcribed again (whisper on the exact segments) and words whose
start disagrees by more than RETIME_MIN are moved to the heard start.
"""
from __future__ import annotations

import numpy as np

from .syncmeter import HOP, _norm, energy_onset, pair

RETIME_MIN = 0.20   # s: smaller differences are the forced aligner's normal noise, left alone
MAX_MOVED_DUR = 0.45  # s: a moved word's length is capped so its midpoint stays inside its segment
RETIME_MAX = 1.00   # s: a larger difference is a mismatched word, not a timing error
SNAP_WINDOW = 0.30  # s: a whisper start is snapped to the nearest audio-energy onset only within this window
MIN_WORD = 0.08     # s: a snapped word never starts closer than this to the previous word of its segment
HOLE_MAX = 0.40     # s: a gap this short with speech energy in it is closed by holding the previous caption
HOLE_SPEECH_MAX = 0.80  # s: a longer gap is closed too when speech energy fills it (>85%): a held-out word tail
SPEECH_DB = -40.0   # dB: envelope level that counts as speech energy


def output_occurrences(words: list[dict], segments: list[tuple[float, float, float]]) -> list[tuple[int, int, float]]:
    """(word index, segment index, output time) for every caption occurrence build_ass would draw (same inclusion
    rule), sorted by output time. A word replayed in two segments (cold open + main) has one occurrence per segment."""
    occ = [(i, k, w["s"] - seg_s + off)
           for k, (seg_s, seg_e, off) in enumerate(segments)
           for i, w in enumerate(words)
           if w["s"] >= seg_s - 0.02 and (w["s"] + w["e"]) / 2 <= seg_e]
    return sorted(occ, key=lambda o: o[2])


def pair_with_leftovers(cap: list[tuple[str, float]], heard: list[tuple[str, float]]) -> list[tuple[int, int]]:
    """pair() plus a second pass: a caption word the sequence match skipped (typically a one-letter word like "a"
    next to a changed neighbour) is linked to the nearest unused heard word with the same text within RETIME_MAX."""
    pairs = list(pair(cap, heard))
    used_c, used_h = {c for c, _ in pairs}, {h for _, h in pairs}
    for ci, (w, t) in enumerate(cap):
        if ci in used_c:
            continue
        cands = [(abs(t - ht), hj) for hj, (hw, ht) in enumerate(heard)
                 if hj not in used_h and _norm(hw) == _norm(w) and abs(t - ht) <= RETIME_MAX]
        if cands:
            hj = min(cands)[1]
            pairs.append((ci, hj))
            used_c.add(ci)
            used_h.add(hj)
    return sorted(pairs)


def _speech_between(env: np.ndarray, t0: float, t1: float, ratio: float = 0.5) -> bool:
    """True when more than `ratio` of the output span [t0, t1] carries speech energy (env: 10 ms output envelope)."""
    i0, i1 = max(0, int(t0 / HOP)), min(len(env), int(t1 / HOP))
    return i1 > i0 and float((env[i0:i1] >= SPEECH_DB).mean()) > ratio


def retime_words(words: list[dict], segments: list[tuple[float, float, float]],
                 heard: list[tuple[str, float]], bias: float = 0.0,
                 env: np.ndarray | None = None) -> tuple[list[dict], list[str]]:
    """words: caption words (source time); heard: whisper (text, output start) of the finished audio.
    bias: build_ass later shifts every word earlier by sync_shift, so a moved word is placed at heard + bias.
    Returns the words with grossly misplaced starts moved to the heard start, and a log line per moved word.
    env: optional 10 ms dB envelope of the output audio. With it a heard start is snapped to the audio-energy onset
    within SNAP_WINDOW (whisper starts are only good to ~+-0.5 s), word order is kept with MIN_WORD spacing, and a
    short gap with speech energy is closed (HOLE_MAX). Without a nearby onset the whisper time stays (log: "~").
    Each occurrence is timed on its own: a word replayed in two segments gets a per-segment override
    w["ov"] = {segment index: (s, e)} that build_ass uses for that segment (its base s/e stay the source time)."""
    occ = output_occurrences(words, segments)
    cap = [(words[i]["w"], t) for i, _k, t in occ]
    cur = {(i, k): [words[i]["s"], words[i]["e"]] for i, k, _t in occ}  # (word, segment) -> new [s, e]
    moved, touched = [], set()
    for ci, hj in pair_with_leftovers(cap, heard):
        i, k, t_cap = occ[ci]
        t_heard, mark = heard[hj][1], ""
        if env is not None:
            on = energy_onset(env, t_heard, SNAP_WINDOW, SNAP_WINDOW)
            if on is not None and abs(on - t_heard) <= SNAP_WINDOW:
                t_heard = on
            else:
                mark = "~"  # no onset in reach: whisper time kept
        diff = t_cap - t_heard
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
            moved.append(f"{w['w']}@{t_cap:.2f}->{t_heard:.2f}{mark}")
    # A moved word keeps its (shifted) duration, but the heard word often lasts longer: hold it until the next
    # word of the same segment starts when the gap is small, or the caption leaves a hole under speech.
    for k, (seg_s, seg_e, off) in enumerate(segments):
        if env is None:
            order = sorted((key for key in cur if key[1] == k), key=lambda key: cur[key][0])
        else:  # spoken order is the aligned order: a snap never swaps words
            order = sorted((key for key in cur if key[1] == k), key=lambda key: words[key[0]]["s"])
            for a, b in zip(order, order[1:]):
                if b in touched and cur[b][0] < cur[a][0] + MIN_WORD:
                    new_s = min(cur[a][0] + MIN_WORD, seg_e)
                    cur[b] = [new_s, max(cur[b][1], new_s)]
        for a, b in zip(order, order[1:]):
            gap = cur[b][0] - cur[a][1]
            closing = (a in touched or b in touched) and 0 < gap <= 0.6
            if env is not None and 0 < gap <= HOLE_SPEECH_MAX and _speech_between(
                    env, cur[a][1] - seg_s + off, cur[b][0] - seg_s + off, 0.5 if gap <= HOLE_MAX else 0.85):
                closing = True
            if closing:
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


def output_audio(wav, wav_t0: float, segments: list[tuple[float, float, float]]) -> np.ndarray:
    """The audio the segments produce, concatenated from the segment WAV."""
    from .transcribe import load_audio
    return np.concatenate([load_audio(wav, s - wav_t0, e - s) for s, e, _ in segments])


def heard_in_output(wav, wav_t0: float, segments: list[tuple[float, float, float]],
                    audio: np.ndarray | None = None) -> list[tuple[str, float]]:
    """Whisper words (text, output start) of the audio the segments produce, read from the segment WAV."""
    from faster_whisper import WhisperModel
    if audio is None:
        audio = output_audio(wav, wav_t0, segments)
    segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(
        audio, language="en", word_timestamps=True)
    return [(w.word, float(w.start)) for s in segs for w in (s.words or [])]
