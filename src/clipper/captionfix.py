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


def occ_count(occ, i):
    return [o for o in occ if o[0] == i]


def _speech_between(env: np.ndarray, t0: float, t1: float, ratio: float = 0.5) -> bool:
    """True when more than `ratio` of the output span [t0, t1] carries speech energy (env: 10 ms output envelope)."""
    i0, i1 = max(0, int(t0 / HOP)), min(len(env), int(t1 / HOP))
    return i1 > i0 and float((env[i0:i1] >= SPEECH_DB).mean()) > ratio


def _speech_run_end(env: np.ndarray, t0: float, t1: float, dip: float = 0.06) -> float:
    """End (output s) of the continuous speech run that starts at t0, looking no further than t1. Dips below the
    speech level shorter than `dip` count as inside the run. t0 itself must carry speech, else t0 is returned."""
    i, n, end, quiet = int(t0 / HOP), min(len(env), int(t1 / HOP)), t0, 0
    while i < n:
        if env[i] >= SPEECH_DB:
            end, quiet = (i + 1) * HOP, 0
        else:
            quiet += 1
            if quiet * HOP > dip:
                break
        i += 1
    return end


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
    moved, touched, crossed, dropped = [], set(), set(), set()
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
            # moved out of its segment would vanish and leave a hole under speech. If it landed in the NEIGHBOUR
            # segment (the aligner put it past a cut, the speaker said it before it: "a" at a join), it moves there
            # (a replayed word gets ov[k] = None: not captioned in its old segment); otherwise it keeps its aligned time.
            dur = min(w["e"] - w["s"], MAX_MOVED_DUR)  # aligned durations after fillers are inflated (incident: 1.1 s)
            if seg_s - 0.02 <= new_s <= seg_e:
                cur[(i, k)] = [new_s, min(new_s + dur, seg_e)]
                touched.add((i, k))
            else:
                k2 = next((j for j in (k - 1, k + 1) if 0 <= j < len(segments)
                           and segments[j][0] - 0.02 <= new_s <= segments[j][1]), None)
                if k2 is None or (i, k2) in cur:
                    continue
                cur.pop((i, k))
                dropped.add((i, k))
                cur[(i, k2)] = [new_s, min(new_s + dur, segments[k2][1])]
                touched.add((i, k2))
                crossed.add((i, k2))
            moved.append(f"{w['w']}@{t_cap:.2f}->{t_heard:.2f}{mark}")
    # A moved word keeps its (shifted) duration, but the heard word often lasts longer: hold it until the next
    # word of the same segment starts when the gap is small, or the caption leaves a hole under speech.
    for k, (seg_s, seg_e, off) in enumerate(segments):
        if env is None:
            order = sorted((key for key in cur if key[1] == k), key=lambda key: cur[key][0])
        else:  # spoken order is the aligned order: a snap never swaps words
            order = sorted((key for key in cur if key[1] == k), key=lambda key: cur[key][0] if key in crossed else words[key[0]]["s"])
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
            if not closing and env is not None and 0 < gap <= HOLE_SPEECH_MAX:
                # a word tail held past its caption: speech runs on from the gap start. Extend over the run only
                # (silence after it stays uncaptioned); a run reaching the next word closes the gap fully.
                t0, t1 = cur[a][1] - seg_s + off, cur[b][0] - seg_s + off
                run = _speech_run_end(env, t0, t1)
                if run - t0 >= 0.08:
                    if t1 - run <= 0.12:
                        closing = True
                    else:
                        cur[a][1] = min(run + 0.03 + seg_s - off, cur[b][0], seg_e)
            if closing:
                # never past the end of the segment: the middle would leave it and build_ass drops the word
                cur[a][1] = min(cur[b][0], seg_e)
    if env is not None:
        _close_joins(cur, segments, env)
    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        ks = [k for (j, k) in cur if j == i]
        gone = {k for (j, k) in dropped if j == i}
        if len(ks) == 1 and not (gone and len(occ_count(occ, i)) > 1):
            (k,) = ks
            if (i, k) in touched or cur[(i, k)] != [w["s"], w["e"]]:
                w["s"], w["e"] = cur[(i, k)]
        elif ks and (gone or any((i, k) in touched or cur[(i, k)] != [w["s"], w["e"]] for k in ks)):
            w["ov"] = {**{k: tuple(cur[(i, k)]) for k in ks}, **{k: None for k in gone}}
    return out, moved


def _speech_run_start(env: np.ndarray, t0: float, t1: float, dip: float = 0.06) -> float:
    """Start (output s) of the continuous speech run that ends at t1, looking back no further than t0."""
    i, lo, start, quiet = int(t1 / HOP) - 1, int(t0 / HOP), t1, 0
    while i >= lo and i < len(env):
        if env[i] >= SPEECH_DB:
            start, quiet = i * HOP, 0
        else:
            quiet += 1
            if quiet * HOP > dip:
                break
        i -= 1
    return start


def _close_joins(cur: dict, segments: list[tuple[float, float, float]], env: np.ndarray) -> None:
    """Close a short caption hole that straddles a segment join (the per-segment pass never sees it): the last caption
    of segment k-1 is held to the speech run end (at most the segment end), the first caption of segment k starts at
    the speech start (at least the segment start). Silence in between stays uncaptioned."""
    for k in range(1, len(segments)):
        ps, pe, po = segments[k - 1]
        ns, ne, no = segments[k]
        prev = [key for key in cur if key[1] == k - 1]
        nxt = [key for key in cur if key[1] == k]
        if not prev or not nxt:
            continue
        a = max(prev, key=lambda key: cur[key][1])
        b = min(nxt, key=lambda key: cur[key][0])
        ea, sb = cur[a][1] - ps + po, cur[b][0] - ns + no
        if not 0 < sb - ea <= HOLE_SPEECH_MAX:
            continue
        join = no
        if _speech_between(env, ea, sb, 0.5 if sb - ea <= HOLE_MAX else 0.85):
            cur[a][1] = pe
            cur[b][0] = ns
            continue
        run = _speech_run_end(env, ea, min(join, sb))
        if run - ea >= 0.08:
            cur[a][1] = min(run + 0.03 + ps - po, pe)
        st = _speech_run_start(env, max(join, ea), sb)
        if sb - st >= 0.08:
            cur[b][0] = max(st - 0.03 + ns - no, ns)


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
