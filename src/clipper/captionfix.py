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
ON_ONSET = 0.06    # s: a caption this close to an audio-energy onset is already on its word
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
                 env: np.ndarray | None = None, audio: np.ndarray | None = None) -> tuple[list[dict], list[str]]:
    """words: caption words (source time); heard: whisper (text, output start) of the finished audio.
    bias: build_ass later shifts every word earlier by sync_shift, so a moved word is placed at heard + bias.
    Returns the words with grossly misplaced starts moved to the heard start, and a log line per moved word.
    env: optional 10 ms dB envelope of the output audio. With it a heard start is snapped to the audio-energy onset
    within SNAP_WINDOW (whisper starts are only good to ~+-0.5 s), word order is kept with MIN_WORD spacing, and a
    short gap with speech energy is closed (HOLE_MAX). Without a nearby onset the whisper time stays (log: "~").
    Each occurrence is timed on its own: a word replayed in two segments gets a per-segment override
    w["ov"] = {segment index: (s, e)} that build_ass uses for that segment (its base s/e stay the source time).
    audio: optional output audio (16 kHz). With it the first caption of every segment is moved to the real speech onset
    when the cut leaves 120-300 ms of silence in front of it (see _onset_pass)."""
    occ = output_occurrences(words, segments)
    cap = [(words[i]["w"], t) for i, _k, t in occ]
    cur = {(i, k): [words[i]["s"], words[i]["e"]] for i, k, _t in occ}  # (word, segment) -> new [s, e]
    moved, touched, crossed, dropped = [], set(), set(), set()
    prev_heard, prev_skipped = -9.0, False
    for ci, hj in sorted(pair_with_leftovers(cap, heard)):
        i, k, t_cap = occ[ci]
        t_heard, mark = heard[hj][1], ""
        if words[i].get("fill"):
            continue  # inserted from the heard audio by repair_words: already at its heard time
        # Whisper on the finished audio sometimes gives two neighbouring words the same start ("And" and "I" both
        # 9.2 s). When the first of them was left alone, the collapsed start says nothing about the second: it keeps
        # its aligned time (if the first did move, the second is placed after it by the MIN_WORD spacing below).
        collided = heard[hj][1] < prev_heard + MIN_WORD
        prev_heard = max(prev_heard, heard[hj][1])
        if collided and prev_skipped:
            continue
        prev_skipped = True
        if env is not None and energy_onset(env, t_cap, ON_ONSET, ON_ONSET) is not None:
            # The caption already sits on a speech onset (chip-roy c1: "And," at 8.94, burst at 8.93): a heard start
            # that is a long way off is whisper noise, and snapping it to another onset would move a correct word.
            continue
        prev_skipped = False
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
            nxt = next((occ[c2] for c2 in range(ci + 1, len(occ)) if occ[c2][1] == k), None)
            if nxt is not None and new_s > words[nxt[0]]["s"] - MIN_WORD and new_s > w["s"]:
                continue  # moving later must not jump over the next word (chip-roy c1: "would" past "say,")
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
    _spread_collapsed(cur, occ)
    if env is not None:
        _close_joins(cur, segments, env)
    if audio is not None:
        moved += _onset_pass(cur, words, segments, audio, bias)
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


COLLAPSE_MAX = 0.06   # s: two neighbouring caption words starting this close are a collapsed pair, not two words
COLLAPSE_GAP = 0.16   # s: the earlier word of a collapsed pair is put this far before the later one


def _spread_collapsed(cur: dict, occ) -> None:
    """The forced aligner sometimes stacks a short word on its neighbour ("I" 0.04 s before "would"). Nobody speaks
    two words in 40 ms, so the earlier one is moved back to COLLAPSE_GAP before the later, but never closer than
    MIN_WORD+0.02 to the word before it."""
    by_seg: dict[int, list[int]] = {}
    for i, k, _t in occ:
        if (i, k) in cur:  # a word retimed into another segment has no entry here
            by_seg.setdefault(k, []).append(i)
    for k, idx in by_seg.items():
        for n in range(1, len(idx)):
            a, b = (idx[n - 1], k), (idx[n], k)
            if not (0 <= cur[b][0] - cur[a][0] < COLLAPSE_MAX):
                continue
            floor = cur[(idx[n - 2], k)][0] + MIN_WORD + 0.02 if n >= 2 else -1e9
            new_s = max(cur[b][0] - COLLAPSE_GAP, floor)
            if new_s < cur[a][0]:
                cur[a][0] = new_s


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


ONSET_SEARCH = 0.30   # s: the onset is looked for at most this far after the lit time; none found -> start unchanged
ONSET_REL_DB = 22.0   # dB below the clip's speech level (p90 of the 20 ms curve) that counts as speech
ONSET_FLOOR_DB = -50.0
ONSET_PUSH_MAX = 0.07   # the next word may be pushed this far later (QA tolerates 80 ms)
ONSET_MIN_WORD = 0.04   # a short function word ("to") may start only ~90 ms before the next word


def db_curve20(a: np.ndarray) -> np.ndarray:
    """The same curve qa._db_curve reads: 20 ms RMS window, 10 ms hop, dB (index i describes [i*10 ms, +20 ms])."""
    n = max(len(a) - 320, 0)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(0, n, 160)
    sq = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
    ms = (sq[np.minimum(idx + 320, len(a))] - sq[idx]) / 320.0
    return 10 * np.log10(ms + 1e-18)


def onset_threshold(db: np.ndarray) -> float:
    """Relative speech threshold: 22 dB under the clip's speech level (p90), never above QA's -35 dB, never below -50."""
    if not len(db):
        return -35.0
    return float(min(-35.0, max(ONSET_FLOOR_DB, np.percentile(db, 90) - ONSET_REL_DB)))


def _onset_pass(cur: dict, words: list[dict], segments: list[tuple[float, float, float]], audio: np.ndarray,
                bias: float) -> list[str]:
    """The first caption of a segment sits at the aligner start, which after a cut can be 120-150 ms before the
    speech (QA: 'lights up during silence'). Measure the onset on the finished audio with QA's own curve and a
    threshold relative to the clip's speech level; search at most ONSET_SEARCH forward. No onset -> unchanged, logged."""
    db = db_curve20(audio)
    if not len(db):
        return []
    thr = onset_threshold(db)
    log = []
    for k, (ss, se, so) in enumerate(segments):
        keys = [key for key in cur if key[1] == k]
        if not keys:
            continue
        key = min(keys, key=lambda kk: cur[kk][0])
        s, e = cur[key]
        lit = s - ss + so - bias   # output time the highlight starts (build_ass shifts by sync_shift)
        if lit < so - bias - 1e-6 or lit > so + (se - ss):
            continue
        i0 = max(int(round((lit - 0.03) / HOP)), int(np.ceil(so / HOP)), 0)   # frames of the previous segment do not count
        i1 = min(int(round((lit + 0.04) / HOP)) + 1, len(db))   # speech must already be there, not 100 ms ahead
        i1 = max(i1, i0 + 4)   # the highlight leads the join by the bias: look at the first frames after the cut
        if i1 > len(db) or db[i0:i1].max() >= thr:
            continue   # speech already under the highlight
        j1 = min(int(round((lit + ONSET_SEARCH) / HOP)) + 1, len(db))
        hit = next((j for j in range(i1, j1) if db[j] >= thr), None)
        name = words[key[0]]["w"]
        if hit is None:
            log.append(f"onset-none:{name}@{lit:.2f}")
            continue
        ons = hit * HOP
        new_s = ons - so + ss + bias
        others = [kk for kk in keys if kk != key]
        if others:
            nk = min(others, key=lambda kk: cur[kk][0])
            need = new_s + ONSET_MIN_WORD + 0.02   # a short word ("to") sits right in front of the next one
            if need > cur[nk][0]:
                if need - cur[nk][0] > ONSET_PUSH_MAX or need > cur[nk][1] + ONSET_PUSH_MAX:
                    log.append(f"onset-blocked:{name}@{lit:.2f}")
                    continue
                cur[nk] = [need, max(cur[nk][1], need + MIN_WORD)]
        cur[key] = [new_s, min(max(e, new_s + ONSET_MIN_WORD), se)]
        log.append(f"onset:{name}@{lit:.2f}>{ons:.2f}")
    return log


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


FILL_MAX_WORDS = 6   # a run of heard words missing from the transcript is filled in only up to this long
FILL_MIN_SPAN = 0.30  # s: ... and only when it spans at least this much speech


def repair_words(words: list[dict], segments: list[tuple[float, float, float]],
                 heard: list[tuple[str, float]], env: np.ndarray | None) -> tuple[list[dict], list[str]]:
    """Fixes of the transcript itself, before retime_words (which only moves words):
    - "going to" spoken as "gonna" (whisper on the finished audio heard "gonna"): one caption word "gonna", so the
      caption matches the audio;
    - a run of words the finished audio has but the transcript lacks (whisper on the full audio and the aligner dropped
      "If I could say," in chip-roy c1, 14.1-14.6 s, caption gap under speech): inserted, spread over the speech.
    Returns (words, log lines). Never touches words that pair with a heard word."""
    occ = output_occurrences(words, segments)
    cap = [(words[i]["w"], t) for i, _k, t in occ]
    pairs = pair_with_leftovers(cap, heard)
    used_h = {h for _c, h in pairs}
    log: list[str] = []
    drop: set[int] = set()
    words = [dict(w) for w in words]
    # going to -> gonna
    for hj, (hw, ht) in enumerate(heard):
        if hj in used_h or _norm(hw) != "gonna":
            continue
        for ci in range(len(cap) - 1):
            i, _k, t = occ[ci]
            j = occ[ci + 1][0]
            if (_norm(cap[ci][0]) == "going" and _norm(cap[ci + 1][0]) == "to" and abs(t - ht) <= 0.5
                    and j == i + 1 and i not in drop):
                g = words[i]["w"]
                words[i]["w"] = ("Gonna" if g[:1].isupper() else "gonna") + g[len("going"):].replace("to", "")
                words[i]["e"] = max(words[i]["e"], words[j]["e"])
                drop.add(j)
                log.append(f"going to->gonna@{t:.2f}")
                break
    words = [w for i, w in enumerate(words) if i not in drop]
    # a transcript word that repeats its neighbour and has no heard counterpart is a transcript duplicate ("And," +
    # "and" in chip-roy c1 8.8-9.6 s: the audio has one "and", the caption lit I 0.2 s late)
    occ = output_occurrences(words, segments)
    cap = [(words[i]["w"], t) for i, _k, t in occ]
    paired_c = {c for c, _h in pair_with_leftovers(cap, heard)}
    drop = set()
    for ci in range(1, len(cap)):
        i, _k, t = occ[ci]
        pi, _pk, pt = occ[ci - 1]
        if (i == pi + 1 and i not in drop and pi not in drop and (ci in paired_c) != (ci - 1 in paired_c)
                and _norm(cap[ci][0]) == _norm(cap[ci - 1][0]) and len(occ_count(occ, i)) == 1
                and len(occ_count(occ, pi)) == 1 and 0 <= t - pt <= 0.45 and not words[pi]["w"].rstrip().endswith((".", "!", "?"))):
            gone = pi if ci in paired_c else i
            drop.add(gone)
            log.append(f"dup {words[gone]['w']}@{(pt if gone == pi else t):.2f}")
    words = [w for i, w in enumerate(words) if i not in drop]
    if env is None:
        return words, log
    # runs of unpaired heard words in a caption gap
    occ = output_occurrences(words, segments)
    cap = [(words[i]["w"], t) for i, _k, t in occ]
    used_h = {h for _c, h in pair_with_leftovers(cap, heard)}
    cap_t = sorted(t for _w, t in cap)
    runs, cur_run = [], []
    for hj in range(len(heard)):
        if hj not in used_h and (not cur_run or hj == cur_run[-1] + 1):
            cur_run.append(hj)
        else:
            if cur_run:
                runs.append(cur_run)
            cur_run = [hj] if hj not in used_h else []
    if cur_run:
        runs.append(cur_run)
    inserts = []
    for run in runs:
        t0 = heard[run[0]][1]
        nxt = next((t for t in cap_t if t > t0 + 0.02), None)
        prv = max((t for t in cap_t if t <= t0 + 0.02), default=None)
        t1 = min(heard[run[-1]][1] + 0.30, nxt - 0.02) if nxt is not None else heard[run[-1]][1] + 0.30
        if len(run) > FILL_MAX_WORDS or t1 - t0 < FILL_MIN_SPAN or any(t0 - 0.12 <= t <= t1 - 0.05 for t in cap_t):
            continue
        if prv is not None and t0 - prv > 1.5 or not _speech_between(env, t0, t1, 0.6):
            continue
        k = next((k for k, (s, e, off) in enumerate(segments) if off - 0.02 <= t0 <= off + (e - s)), None)
        if k is None:
            continue
        step = (t1 - t0) / len(run)
        for n, hj in enumerate(run):
            ts = t0 + n * step
            ss = segments[k][0] + ts - segments[k][2]
            inserts.append((prv, {"w": heard[hj][0].strip(), "s": ss, "e": ss + step * 0.9, "fill": True}))
        log.append(f"filled {' '.join(heard[h][0].strip() for h in run)}@{t0:.2f}")
    # positions are taken against the original word list, then inserted back to front so earlier indices stay valid
    placed = []
    for n, (prv, nw) in enumerate(inserts):
        pos = 0
        if prv is not None:
            pos = 1 + min(occ, key=lambda o: abs(o[2] - prv))[0]
        placed.append((pos, n, nw))
    for pos, _n, nw in sorted(placed, key=lambda x: (x[0], x[1]), reverse=True):
        words.insert(pos, nw)
    return words, log
