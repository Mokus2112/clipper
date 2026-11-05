"""Automatic QA at the end of every render: defects the reviewer found once must not reach him again.

Each check returns a list of problems (empty = ok). See data/campaigns/LESSONS.md for the history.
"""

import re
from difflib import SequenceMatcher
import subprocess
from pathlib import Path

import numpy as np

from .common import ROOT, path_of, tool

# Strictest safe zone over YouTube Shorts, TikTok and Instagram Reels (1080x1920).
TOP, BOTTOM, RIGHT = 220, 1500, 920   # faces
TEXT_BOTTOM, TEXT_RIGHT = 1250, 860    # captions and hook (card H): above the Shorts/Reels UI, left of the buttons
SPEAKER_FACE_W = 0.10      # faces wider than 10% of the frame are speakers
END_SILENCE_DB = -30.0     # last 50 ms of a segment must be at most this loud (no cut mid-word)
MIN_SHOT_FRAMES = 3        # a shot shorter than this is a flash
MIN_CAPTION = 0.25         # s a caption chunk must stay on screen
FACE_WARN_PX = 60          # a face overshooting the safe zone by at most this is a WARN, not a FAIL (Codee #1310)
WARN = "WARN "             # problem lines with this prefix are reported in qa.txt but never fail a clip


def as_warn(problems: list[str]) -> list[str]:
    return [WARN + p for p in problems]


def is_warn(problem: str) -> bool:
    return problem.startswith(WARN)


def _duration(video: Path) -> float:
    return float(subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout)


def text_boxes(cdir: Path, dur: float, fonts: Path, split: bool = False) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Bounding box of all on-screen text (captions + hook) every 0.1 s, rendered alone on mid grey.

    Grey, not black: the black outline/shadow and the dark hook box count as drawn pixels too."""
    fdir = Path(fonts).resolve()
    if not fdir.is_dir() or not any(fdir.glob("*.[ot]tf")):
        raise RuntimeError(f"QA font dir missing or empty: {fdir}")  # never measure with a fallback font
    raw = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"color=0x808080:s=1080x1920:r=10:d={dur:.2f}",
         "-vf", f"subtitles=captions.ass:fontsdir={Path(fonts).resolve()},format=gray", "-f", "rawvideo", "-"],
        cwd=cdir, capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 1920, 1080)
    out = []
    for i, f in enumerate(frames):
        ys, xs = np.nonzero(np.abs(f.astype(np.int16) - 128) > 20)
        if not len(xs):
            continue
        if not split:
            out.append((i / 10, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
            continue
        # one box per text block (hook, caption): rows separated by more than 80 px are different blocks
        rows = np.unique(ys)
        starts = [0] + [k for k in range(1, len(rows)) if rows[k] - rows[k - 1] > 80]
        for a, b in zip(starts, starts[1:] + [len(rows)]):
            m = (ys >= rows[a]) & (ys <= rows[b - 1])
            out.append((i / 10, (int(xs[m].min()), int(rows[a]), int(xs[m].max()), int(rows[b - 1]))))
    return out


def face_boxes(video: Path, dur: float) -> list[tuple[float, tuple[int, int, int, int]]]:
    import cv2
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                    (540, 960), 0.8, 0.3, 50)
    cap = cv2.VideoCapture(str(video))
    out, t = [], 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            _, faces = det.detect(cv2.resize(fr, (540, 960)))
            raw = [tuple(float(v) * 2 for v in f[:4]) + (float(f[14]),) for f in (faces if faces is not None else [])]
            for x, y, w, h, _sc in drop_false_faces(raw):
                if w >= 1080 * SPEAKER_FACE_W:
                    out.append((t, (int(x), int(y), int(x + w), int(y + h))))
        t += 0.5
    cap.release()
    return out


def check_safe_zone(cdir: Path, fonts: Path) -> list[str]:
    video = cdir / "short.mp4"
    dur = _duration(video)
    bad = []
    for t, (x0, y0, x1, y1) in text_boxes(cdir, dur, fonts):
        if y0 < TOP or y1 > TEXT_BOTTOM or x1 > TEXT_RIGHT:
            bad.append(f"safe zone: text at {t:.1f}s box x{x0}-{x1} y{y0}-{y1}")
    for t, (x0, y0, x1, y1) in face_boxes(video, dur):
        if y0 < TOP or y1 > BOTTOM or x1 > RIGHT:
            over = max(TOP - y0, y1 - BOTTOM, x1 - RIGHT)
            line = f"safe zone: face at {t:.1f}s box x{x0}-{x1} y{y0}-{y1}"
            bad.append(WARN + line if over <= FACE_WARN_PX else line)
    return bad


def check_safe_zone_text_only(cdir: Path, fonts: Path) -> list[str]:
    """Safe-zone check for montage sources: in-game characters are not faces, only our own text counts."""
    return [p for p in check_safe_zone(cdir, fonts) if not p.removeprefix(WARN).startswith("safe zone: face")]


def check_segment_ends(audio_fn, source_ends: list[tuple[float, float]]) -> list[str]:
    """Each segment end must land in silence in the source audio (last 30 ms before the cut).

    source_ends: (source end time, output end time). The output itself is faded, so it is
    measured on the source.
    """
    bad = []
    for src_t, out_t in source_ends:
        a = audio_fn(src_t - 0.06, 0.06)
        db = 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9) if len(a) else -99
        if db > -35.0:
            bad.append(f"audio: segment ending at {out_t:.2f}s (source {src_t:.2f}) is cut during sound ({db:.0f} dB)")
    return bad


def check_flashes(video: Path) -> list[str]:
    """Camera cuts closer than MIN_SHOT_FRAMES apart: a 1-2 frame shot is a flash."""
    res = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-i", str(video), "-an",
         "-vf", "scale=320:-2,scdet=threshold=100,metadata=print:file=-", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    cuts, n = [], None
    for line in res.stdout.splitlines():
        if line.startswith("frame:"):
            n = int(line.split()[0].split(":")[1])
        elif "scd.score=" in line and n is not None and float(line.split("=", 1)[1]) >= 4.0:
            cuts.append(n)
    return [f"flash: shots of {b - a} frame(s) at frames {a}-{b}"
            for a, b in zip([0] + cuts, cuts) if 0 < b - a < MIN_SHOT_FRAMES]


def check_caption_durations(ass: Path) -> list[str]:
    """Every caption chunk (consecutive lines with the same text) stays >= MIN_CAPTION on screen."""
    def secs(t: str) -> float:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    chunks: list[list[float]] = []
    last_text = None
    for line in ass.read_text().splitlines():
        if not line.startswith("Dialogue: 0,"):
            continue
        p = line.split(",", 9)
        text = re.sub(r"\{[^}]*\}", "", p[9])
        if text == last_text and chunks and abs(chunks[-1][1] - secs(p[1])) < 0.02:
            chunks[-1][1] = secs(p[2])
        else:
            chunks.append([secs(p[1]), secs(p[2])])
        last_text = text
    return [f"caption: chunk at {a:.2f}s only {b - a:.2f}s on screen" for a, b in chunks if b - a < MIN_CAPTION - 0.005]


def check_true_peak(video: Path) -> list[str]:
    res = subprocess.run([tool("ffmpeg"), "-hide_banner", "-nostats", "-i", str(video), "-vn",
                          "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", res.stderr)
    if not m or m[-1] == "-inf":
        return []
    tp = float(m[-1])
    return [f"audio: true peak {tp:.1f} dBFS (must be < -1.0)"] if tp >= -1.0 else []


def check_segment_starts(audio_fn, source_starts: list[tuple[float, float]]) -> list[str]:
    bad = []
    for src_t, out_t in source_starts:
        a = audio_fn(src_t, 0.03)
        db = 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9) if len(a) else -99
        if db > END_SILENCE_DB:
            bad.append(f"audio: segment starting at {out_t:.2f}s (source {src_t:.2f}) starts during sound ({db:.0f} dB)")
    return bad


FILLERS = {"like", "so", "and", "um", "uh", "yeah", "the", "a", "i", "it's", "you", "know", "just", "oh"}


LEGACY_HIGHLIGHT = "&H0000E0FF"   # brand yellow, the highlight colour before captions.highlight was read from config
_hl_cache: dict[str, re.Pattern] = {}


def highlight_re(color: str | None = None) -> re.Pattern:
    """Pattern for the active-word tag `\\c<highlight>}word` of a caption line.

    `color` is an ASS colour (&HAABBGGRR, any case, optional leading 00 alpha, optional trailing &);
    None reads captions.highlight from config.yaml and falls back to the legacy yellow when it is
    missing or not a colour. The sentence colour never matches: only the highlight tag does.
    """
    if color is None:
        try:
            from .common import load_config
            color = load_config().get("captions", {}).get("highlight")
        except Exception:  # noqa: BLE001 - QA must run without a readable config
            color = None
    hexes = re.sub(r"(?i)^&?h|&$", "", (color or "").strip())
    if not re.fullmatch(r"(?i)[0-9a-f]{6}|[0-9a-f]{8}", hexes):
        hexes = LEGACY_HIGHLIGHT[2:]
    rgb, alpha = hexes[-6:], hexes[:-6] if len(hexes) == 8 else "00"
    key = (alpha + rgb).upper()
    if key not in _hl_cache:
        a = "(?:00)?" if alpha == "00" else re.escape(alpha)
        _hl_cache[key] = re.compile(r"\\c&H" + a + re.escape(rgb) + r"&?\}([^{]+)", re.I)
    return _hl_cache[key]


def _highlights(ass: Path) -> list[tuple[float, float, str]]:
    """(start, end, highlighted word) of every caption line."""
    def secs(t: str) -> float:
        h, m, s_ = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s_)
    out = []
    for line in ass.read_text().splitlines():
        if line.startswith("Dialogue: 0,"):
            p = line.split(",", 9)
            m = highlight_re().search(p[9])
            out.append((secs(p[1]), secs(p[2]), m.group(1).strip() if m else ""))
    return out


def _db_curve(a: np.ndarray) -> np.ndarray:
    return np.array([20 * np.log10(np.sqrt(np.mean(a[i:i + 320] ** 2)) + 1e-9) for i in range(0, max(len(a) - 320, 0), 160)])


def check_caption_sync(video: Path, ass: Path) -> list[str]:
    """Highlights vs audio: a word after a pause lights up within 150 ms of its onset, and no new
    word lights up while the audio is silent."""
    from .transcribe import load_audio
    a = load_audio(video, 0.0, _duration(video))
    db = _db_curve(a)
    speech = db >= -35.0
    onsets = [i * 0.01 for i in range(6, len(speech)) if speech[i] and not speech[i - 6:i].any()]
    hl = _highlights(ass)
    bad = []
    starts = sorted({round(h[0], 2) for h in hl})
    # Mutual nearest pairing: a pair counts only if the onset's nearest highlight t also has that
    # onset as its nearest onset. Only words after >= 150 ms of silence are checked; unpaired
    # onsets (laughs, breaths, other speakers, second onsets inside one word) are ignored.
    for t in starts:
        if not onsets:
            continue
        o = min(onsets, key=lambda o: abs(o - t))
        if abs(o - t) > 0.5:
            continue
        j = int(round(o / 0.01))
        if j < 15 or speech[j - 15:j].any():
            continue
        if min(starts, key=lambda s: abs(s - o)) != t:
            continue
        ti = int(round(t / 0.01))
        if ti >= 15 and speech[ti - 12:ti - 2].any():
            continue  # the highlight sits inside continuous speech, so the onset is another sound
        if abs(t - o) > 0.15:
            bad.append(f"caption sync: word lights up at {t:.2f}s, audio onset {o:.2f}s ({(t - o) * 1000:+.0f} ms)")
    seen = set()
    for s0, _e, w in hl:
        key = round(s0, 2)
        if key in seen:
            continue
        seen.add(key)
        i0, i1 = int(max(s0 - 0.05, 0) / 0.01), int((s0 + 0.15) / 0.01)
        if i1 <= len(db) and i0 < i1 and db[i0:i1].max() < -35.0:
            bad.append(f"caption sync: '{w}' lights up at {s0:.2f}s during silence")
    return bad


SYNC_FAIL_SHARE = 0.15        # FAIL only if more than 15% of the caption words are off ...
SYNC_FAIL_MEDIAN_MS = 120.0   # ... or the median offset (a systematic shift) exceeds this


def check_caption_sync_words(video: Path, ass: Path) -> list[str]:
    """Word-by-word sync from the shared meter (clipper.syncmeter): whisper on the finished audio is the
    reference, confirmed by the audio-energy onset. Words where the two disagree are 'uncertain' and never fail."""
    from .syncmeter import measure, summary, unmatched_alarming, FAIL_MS, WHISPER_ONLY_FAIL_MS
    rows = measure(video, ass)
    bad = [r for r in rows if r.verdict in ("late", "early")]
    out = []
    if bad:
        sure = [r for r in rows if not r.uncertain and not r.unmatched]
        med = abs(float(np.median([r.d_whisper for r in sure]))) if sure else 0.0
        hard = len(bad) / len(rows) > SYNC_FAIL_SHARE or med > SYNC_FAIL_MEDIAN_MS
        shown = ", ".join(f"{r.word}@{r.cap:.2f} {r.d_whisper:+.0f}ms" for r in bad[:6])
        line = (f"caption sync (whisper+energy): {len(bad)} words off ({FAIL_MS:.0f} ms with an energy onset, "
                f"{WHISPER_ONLY_FAIL_MS:.0f} ms whisper-only): {shown} [{summary(rows)}]")
        out.append(line if hard else WARN + line)
    # unmatched = caption words whisper did not find in the audio: sync unknown, counted apart from late/early
    un = [r for r in rows if r.unmatched]
    if un and unmatched_alarming(rows):
        shown = ", ".join(f"{r.word}@{r.cap:.2f}" for r in un[:8])
        out.append(f"caption sync unmatched: {len(un)} of {len(rows)} caption words not found in the audio, sync unknown: {shown}")
    return out


def check_dead_air(video: Path, max_gap: float = 0.9) -> list[str]:
    """Silence over max_gap seconds between speech (not at the start or end of the clip)."""
    from .transcribe import load_audio
    dur = _duration(video)
    db = _db_curve(load_audio(video, 0.0, dur))
    sp = np.flatnonzero(db >= -35.0)
    if not len(sp):
        return []
    out = []
    for a, b in zip(sp[:-1], sp[1:]):
        if (b - a) * 0.01 > max_gap:
            out.append(f"dead air: {(b - a) * 0.01:.2f}s of silence at {a * 0.01 + 0.02:.2f}s inside the speech")
    return out


def check_caption_overlap(ass: Path) -> list[str]:
    hl = sorted(_highlights(ass))
    return [f"caption: two lines on screen at {b[0]:.2f}s" for a, b in zip(hl, hl[1:]) if b[0] < a[1] - 0.005]


def check_hook_length(cdir: Path) -> list[str]:
    """Hard gate: the hook in clip.json must fit rank.hook_ok (HOOK_MAX_CHARS chars, HOOK_MAX_WORDS words); the renderer never shrinks below 80 px."""
    import json
    from .rank import hook_ok, HOOK_MAX_CHARS, HOOK_MAX_WORDS
    try:
        hook = json.loads((cdir / "clip.json").read_text()).get("hook", "")
    except (OSError, ValueError):
        return []
    if hook and not hook_ok(hook):
        return [f"hook too long: {hook!r} ({len(hook)} chars, max {HOOK_MAX_CHARS}, {HOOK_MAX_WORDS} words): shorten or drop"]
    return []


SENT_END = (".", "?", "!", "\u2026")
EDGE_GAP = 0.3  # s of silence after the last spoken word (or sentence-final punctuation) that makes a clean end


def edge_problems(words: list[tuple[str, float, float]], dur: float, head_db: float,
                  gap: float = EDGE_GAP) -> list[str]:
    """words: (text, start, end) heard in the final audio. head_db: level of the first 30 ms."""
    if not words:
        return []
    bad = []
    last = words[-1]
    tail = dur - last[2]
    txt = last[0].rstrip("\"')]")
    if (not txt.endswith(SENT_END) or txt.endswith("..")) and tail < gap:  # "..." = trailing off, not an end
        bad.append(f"edge: clip ends on '{last[0]}' with only {max(tail, 0) * 1000:.0f} ms after it and no sentence end")
    if words[0][1] < 0.08 and head_db > END_SILENCE_DB:
        bad.append(f"edge: clip starts inside a word ('{words[0][0]}' at {words[0][1]:.2f}s, {head_db:.0f} dB at 0 s)")
    return bad


def check_edge_words(video: Path) -> list[str]:
    """Hard end gate on the rendered audio: word-level re-transcription of the final clip."""
    from faster_whisper import WhisperModel
    from .transcribe import load_audio
    dur = _duration(video)
    audio = load_audio(video, 0.0, dur)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, language="en", word_timestamps=True)
    words = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]
    head = audio[:480]
    head_db = 20 * np.log10(np.sqrt(np.mean(head ** 2)) + 1e-9) if len(head) else -99
    return edge_problems(words, dur, head_db)


CAPTION_WORDS_FAIL_N = 3
CAPTION_WORDS_FAIL_SHARE = 0.10


def check_caption_words(video: Path, ass: Path) -> list[str]:
    """Every captioned word (4+ letters, not a filler) must be heard: re-transcribe the output."""
    from faster_whisper import WhisperModel
    from .transcribe import load_audio
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower().replace("*", ""))  # noqa: E731
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(load_audio(video, 0.0, _duration(video)), language="en", word_timestamps=True)
    heard: list[tuple[str, float]] = []
    for s in segs:
        for w in (s.words or []):
            if heard and re.match(r"\s*[,.]\d", w.word) and heard[-1][0][-1:].isdigit():
                heard[-1] = (heard[-1][0] + norm(w.word), heard[-1][1])  # whisper splits '1,200' into '1' + ',200'
            else:
                heard.append((norm(w.word), w.start))
    missing, checked = [], 0
    for s0, _e, w in _highlights(ass):
        if "-" in w:  # whisper hears 'ex-wife' as 'ex', 'wife': check the longest part
            w = max(w.split("-"), key=len)
        n = norm(w)
        if len(n) < 4 or n in FILLERS or "*" in w:
            continue
        checked += 1
        alts = {n}
        if re.fullmatch(r"(twen|thir|for|fif|six|seven|eigh|nine)ties", n):  # whisper writes 'nineties' as '90s'
            alts.add({"twen": "20", "thir": "30", "for": "40", "fif": "50", "six": "60", "seven": "70",
                      "eigh": "80", "nine": "90"}[n[:-4]] + "s")
        # Spelling variants of names (name_fixes: McArthur captioned, whisper writes MacArthur).
        close = lambda h: len(n) >= 6 and SequenceMatcher(None, h, n).ratio() >= 0.85  # noqa: E731
        if not any((h in alts or h == n or h.startswith(n[:4]) or close(h)) and abs(t - s0) <= 1.5 for h, t in heard):
            missing.append(f"{w}@{s0:.2f}")
    missing = sorted(set(missing))
    if not missing:
        return []
    line = f"caption words not heard in the audio: {', '.join(missing[:8])}"
    # one unheard word is whisper noise (Codee #1310): FAIL from 3 words or more than 10% of the checked words
    hard = len(missing) >= CAPTION_WORDS_FAIL_N or len(missing) > CAPTION_WORDS_FAIL_SHARE * max(checked, 1)
    return [line if hard else WARN + line]


def _repeats(seq: list[tuple[str, float]]) -> list[tuple[int, int]]:
    """Index spans (i, n) where a token or short phrase (n=1..3 tokens) is immediately repeated."""
    out = []
    for n in (1, 2, 3):
        for i in range(len(seq) - 2 * n + 1):
            a = [t for t, _ in seq[i:i + n]]
            b = [t for t, _ in seq[i + n:i + 2 * n]]
            if all(len(x) >= 1 for x in a) and all(x == y or (len(x) > 3 and x[:3] == y[:3]) for x, y in zip(a, b)):
                out.append((i, n))
    return out


def check_repeated_words(video: Path, ass: Path) -> list[str]:
    """Stutters/repeats ('I I', 'he didn't he didn't') must be captioned exactly as heard:
    a caption repeat needs an audible repeat at both onsets, an audible repeat needs a caption repeat."""
    from faster_whisper import WhisperModel
    from .transcribe import load_audio
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower().replace("*", ""))  # noqa: E731
    cap, prev = [], None  # one word may span two Dialogue lines (position change): merge those
    for line in ass.read_text().splitlines():
        if not line.startswith("Dialogue: 0,"):
            continue
        p = line.split(",", 9)
        m = highlight_re().search(p[9])
        key = re.sub(r"\{\\pos[^}]*\}", "", p[9])  # same text and same highlighted word = same word
        h, mi, sec = p[1].split(":")
        s0 = int(h) * 3600 + int(mi) * 60 + float(sec)
        h, mi, sec = p[2].split(":")
        e0 = int(h) * 3600 + int(mi) * 60 + float(sec)
        t = norm(m.group(1)) if m else ""
        if t and not (prev and prev[0] == key and s0 - prev[1] < 0.03):
            cap.append((t, s0))
        prev = (key, e0)
    audio = load_audio(video, 0.0, _duration(video))
    heard: list[tuple[str, float]] = []
    for size in ("small", "base"):
        segs, _ = WhisperModel(size, device="cpu", compute_type="int8").transcribe(
            audio, language="en", word_timestamps=True)
        h = [(norm(w.word), w.start) for s in segs for w in (s.words or [])]
        if size == "small":
            heard = [x for x in h if x[0]]
        else:
            base = [x for x in h if x[0]]
    problems = []
    cap_rep = _repeats(cap)
    for src, name in ((heard, "small"), (base, "base")):
        for i, n in _repeats(src):
            t0 = src[i][1]
            near = [(j, m) for j, m in cap_rep if abs(cap[j][1] - t0) <= 0.8]
            if not near:
                problems.append(f"repeat heard but not captioned @{t0:.2f}: "
                                f"{' '.join(t for t, _ in src[i:i + 2 * n])} ({name})")
    for j, m in cap_rep:
        t0 = cap[j][1]
        ok = any(abs(src[i][1] - t0) <= 0.8 for src in (heard, base) for i, _n in _repeats(src))
        if not ok:
            problems.append(f"repeat captioned but not heard @{t0:.2f}: {' '.join(t for t, _ in cap[j:j + 2 * m])}")
    return as_warn(sorted(set(problems)))  # Codee #1310: a repeat mismatch is advisory


def caption_gaps(db: np.ndarray, covered: np.ndarray, thr_db: float = -35.0, join: float = 0.15,
                 min_gap: float = 0.6) -> list[tuple[float, float]]:
    """(start, end) of speech with no caption on screen, longer than min_gap seconds (10 ms grid).

    Uncaptioned speech runs separated by at most `join` s of silence count as one gap: a gap of
    short syllables (each under the 0.7 s limit of check_uncaptioned_speech) is still a visible hole.
    """
    runs, i, n = [], 0, len(db)
    while i < n:
        if db[i] > thr_db and not covered[i]:
            j = i
            while j < n and db[j] > thr_db and not covered[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) / 100 <= join and not covered[merged[-1][1]:a].any():
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a / 100, b / 100) for a, b in merged if (b - a) / 100 > min_gap]


def check_caption_gaps(video: Path, ass: Path) -> list[str]:
    """Speech for more than 0.6 s with no caption line on screen (measured on the finished audio)."""
    from .transcribe import load_audio
    db = _db_curve(load_audio(video, 0.0, _duration(video)))
    covered = np.zeros(len(db), bool)
    for s0, e0, _w in _highlights(ass):
        covered[int(s0 * 100):int(e0 * 100) + 1] = True
    gaps = caption_gaps(db, covered)
    return [f"caption gap under speech: {', '.join(f'{a:.2f}-{b:.2f}' for a, b in gaps[:6])}"] if gaps else []


def check_uncaptioned_speech(video: Path, ass: Path, thr_db: float = -35.0, min_len: float = 0.7) -> list[str]:
    """Speech (RMS above thr_db for min_len) with no caption line on screen."""
    from .transcribe import load_audio
    a = load_audio(video, 0.0, _duration(video))
    db = _db_curve(a)  # 10 ms hop
    covered = np.zeros(len(db), bool)
    for s0, e0, _w in _highlights(ass):
        covered[int(s0 * 100):int(e0 * 100) + 1] = True
    out, i = [], 0
    while i < len(db):
        if db[i] > thr_db and not covered[i]:
            j = i
            while j < len(db) and db[j] > thr_db and not covered[j]:
                j += 1
            # a short but loud run (>= 400 ms, peak above -20 dB) is a spoken syllable, not a caption gap
            if (j - i) / 100 >= min_len or ((j - i) >= 40 and db[i:j].max() > -20):
                out.append(f"{i / 100:.2f}-{j / 100:.2f}")
            i = j
        else:
            i += 1
    from faster_whisper import WhisperModel
    segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(a, language="en", word_timestamps=True)
    spans = []  # whisper words with no caption on screen, merged when adjacent
    for sg in segs:
        for w in (sg.words or []):
            i0, i1 = int(w.start * 100), int(w.end * 100) + 1
            if i1 - i0 >= 3 and not covered[i0:i1].any():
                if spans and w.start - spans[-1][1] < 0.15:
                    spans[-1][1] = w.end
                else:
                    spans.append([w.start, w.end])
    out += [f"{a0:.2f}-{b0:.2f}(whisper)" for a0, b0 in spans if b0 - a0 >= 0.7]
    return [f"speech without caption at {', '.join(out[:6])}"] if out else []


def burned_text_conflicts(cdir: Path, fonts: Path, min_chars: int = 4, dy_at=None, captions_only: bool = False) -> list[tuple[float, tuple[int, int, int, int], str]]:
    """Source text/graphics burned into the picture under our captions (lower-thirds, insets).

    Needs nocap.mp4 (render output without subtitles, 540x960, 2 fps). For every sampled frame
    the box of our caption text is OCRed in the caption-free frame; readable text there is a conflict.
    Returns (t, box in 1080x1920 coordinates, text).
    """
    import cv2
    nocap = cdir / "nocap.mp4"
    if not nocap.exists():
        return []
    boxes = text_boxes(cdir, _duration(cdir / "short.mp4"), fonts, split=True)
    by_t = {}
    for t, b in boxes:
        by_t.setdefault(round(t * 2) / 2, []).append(b)
    cap = cv2.VideoCapture(str(nocap))
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / 2
        i += 1
        bs = sorted(by_t.get(t, []), key=lambda b: b[1])
        for k, (bx0, by0, bx1, by1) in enumerate(bs):
            is_cap = k == len(bs) - 1  # the lowest block is the caption, the rest is the hook
            if captions_only and not is_cap:
                continue
            dy = dy_at(t) if (dy_at and is_cap) else 0
            x0, y0, x1, y1 = bx0 // 2, (by0 + dy) // 2, bx1 // 2, (by1 + dy) // 2
            ex = 40  # look beyond the box: a label only partly under it is still unreadable
            ry0, rx0 = max(0, y0 - ex), max(0, x0 - 20)
            roi = fr[ry0:y1 + ex, rx0:x1 + 20]
            if roi.size == 0:
                continue
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if cv2.Canny(g, 100, 200).mean() / 255 < 0.02:
                continue
            g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6", "tsv"],
                                 input=cv2.imencode(".png", g)[1].tobytes(), capture_output=True)
            words = []
            for row in res.stdout.decode(errors="ignore").splitlines()[1:]:
                c = row.split("\t")
                if len(c) == 12 and float(c[10]) >= 60 and re.search(r"[A-Za-z0-9]{2,}", c[11]):
                    wx0, wy0 = rx0 + int(c[6]) / 3, ry0 + int(c[7]) / 3
                    wx1, wy1 = wx0 + int(c[8]) / 3, wy0 + int(c[9]) / 3
                    if wx0 < x1 + 10 and wx1 > x0 - 10 and wy0 < y1 + 15 and wy1 > y0 - 15:  # on or hugging our text box
                        words.append(c[11])
            txt = " ".join(words)
            if len(re.sub(r"\W", "", txt)) >= min_chars:
                out.append((t, (x0 * 2, y0 * 2, x1 * 2, y1 * 2), txt))
    cap.release()
    return out


def busy_caption_times(cdir: Path, fonts: Path, dy: int = 0) -> set[float]:
    """Sample times (0.5 s grid) where the caption-free picture is busy around our caption box.
    OCR on a sparse 2 fps grid misses lower-thirds that animate; this marks the whole span they occupy."""
    import cv2
    nocap = cdir / "nocap.mp4"
    if not nocap.exists():
        return set()
    by_t: dict[float, list] = {}
    for t, b in text_boxes(cdir, _duration(cdir / "short.mp4"), fonts, split=True):
        by_t.setdefault(round(t * 2) / 2, []).append(b)
    cap = cv2.VideoCapture(str(nocap))
    out, i = set(), 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / 2
        i += 1
        bs = sorted(by_t.get(t, []), key=lambda b: b[1])
        if not bs:
            continue
        x0, y0, x1, y1 = [v // 2 for v in bs[-1]]
        roi = fr[max(0, y0 + dy // 2 - 40):y1 + dy // 2 + 40, max(0, x0 - 20):x1 + 20]
        if roi.size and cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 100, 200).mean() / 255 >= 0.02:
            out.add(t)
    cap.release()
    return out


def plan_caption_shift(cdir: Path, fonts: Path) -> list[list[float]]:
    """[t0, t1, dy] spans (output seconds) where captions must move to clear burned-in source text."""
    hits = burned_text_conflicts(cdir, fonts, captions_only=True)
    if not hits:
        return []
    spans: list[list[float]] = []
    for t, _, _ in hits:
        if spans and t - 0.5 <= spans[-1][1]:
            spans[-1][1] = t + 0.5
        else:
            spans.append([t - 0.5, t + 0.5])
    busy = busy_caption_times(cdir, fonts)
    for s in spans:  # a lower-third lasts longer than the OCR sample that caught it: cover the whole run
        while s[0] - 0.5 >= 0 and (round((s[0] + 0.5) * 2) / 2 - 0.5) in busy:
            s[0] -= 0.5
        while (round((s[1] - 0.5) * 2) / 2 + 0.5) in busy:
            s[1] += 0.5
    boxes = text_boxes(cdir, _duration(cdir / "short.mp4"), fonts, split=True)
    out = []
    faces = face_boxes(cdir / "short.mp4", _duration(cdir / "short.mp4"))
    for a, b in spans:
        inside = [(t, bx) for t, bx in boxes if a <= t <= b and bx[3] > 1100]
        if not inside:  # conflict seen by OCR but no caption box in the lower band: nothing to shift
            continue
        top, bot = min(bx[1] for _, bx in inside), max(bx[3] for _, bx in inside)
        fallback = None
        for dy in (-260, -400, -520, 240, 360, 120, 60):
            if top + dy < TOP or bot + dy > TEXT_BOTTOM:
                continue
            left = burned_text_conflicts(cdir, fonts, dy_at=lambda t, a=a, b=b, dy=dy: dy if a <= t <= b else 0,
                                           captions_only=True)
            if any(a <= t <= b for t, _, _ in left):
                continue
            # The moved caption must not land on the speaker's face / beard either.
            on_face = any(a - 0.25 <= ft <= b + 0.25 and bx[0] < fx1 and bx[2] > fx0
                          and bx[3] + dy > fy0 and bx[1] + dy < fy1 * 1.0 + (fy1 - fy0) * FACE_BEARD_MARGIN
                          for _, bx in inside for ft, (fx0, fy0, fx1, fy1) in faces)
            if on_face:
                fallback = fallback if fallback is not None else dy
                continue
            out.append([max(0.0, a), b, dy])
            break
        else:
            if fallback is not None:
                out.append([max(0.0, a), b, fallback])
    return out


def top_text_hits(cdir: Path) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Source text burned into the top strip (y < 130, platform UI zone). (t, box in 1080x1920 coordinates).

    Words are found by OCR; a cut-off fragment at the left edge by edge density in the corner.
    """
    import cv2
    nocap = cdir / "nocap.mp4"
    if not nocap.exists():
        return []
    cap = cv2.VideoCapture(str(nocap))
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / 2
        i += 1
        band = cv2.cvtColor(fr[:65], cv2.COLOR_BGR2GRAY)
        x0 = y0 = x1 = y1 = None
        if cv2.Canny(band, 100, 200).mean() / 255 >= 0.004:
            g = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6", "tsv"],
                                 input=cv2.imencode(".png", g)[1].tobytes(), capture_output=True)
            for row in res.stdout.decode(errors="ignore").splitlines()[1:]:
                c = row.split("\t")
                if len(c) == 12 and float(c[10]) >= 50 and re.search(r"[A-Za-z0-9]{2,}", c[11]):
                    a, b = int(c[6]) // 3, int(c[7]) // 3
                    e, f = a + int(c[8]) // 3, b + int(c[9]) // 3
                    x0, y0 = (a, b) if x0 is None else (min(x0, a), min(y0, b))
                    x1, y1 = (e, f) if x1 is None else (max(x1, e), max(y1, f))
        corner = band[:, :40]
        if x0 is not None:
            out.append((t, (x0 * 2, y0 * 2, x1 * 2, y1 * 2)))
        if (x0 is None or x0 > 40) and cv2.Canny(corner, 100, 200).mean() / 255 >= 0.02:
            out.append((t, (0, 0, 100, 170)))
    cap.release()
    return out + edge_fragment_hits(cdir) + dense_top_hits(cdir)


def dense_top_hits(cdir: Path) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Every output frame (30 fps, not the 2 fps OCR samples): edges in the top 130 px mean burned-in source
    text (flat backgrounds have none). Catches the 1-3 frames a title lingers between OCR samples."""
    import cv2
    video = cdir / "short.mp4"
    if not video.exists():
        return []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        e = cv2.Canny(cv2.cvtColor(fr[:130], cv2.COLOR_BGR2GRAY), 60, 150)
        if e.mean() / 255 >= 0.003:
            ys, xs = np.nonzero(e)
            out.append((i / fps, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
        i += 1
    cap.release()
    return out


def edge_fragment_hits(cdir: Path) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Per output frame: bright pixels (>200 and well above the local median) in the left-edge window x0-40, y80-180
    (cut-off source letters). The source text drifts in time, so every frame is scanned."""
    import cv2
    video = cdir / "short.mp4"
    if not video.exists():
        return []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr[80:180, :40], cv2.COLOR_BGR2GRAY)
        if int(((g > 200) & (g > np.median(g) + 20)).sum()) >= 8:
            out.append((i / fps, (0, 0, 100, 170)))
        i += 1
    cap.release()
    return out


def _raw_top_spans(cdir: Path) -> list[list[float]]:
    spans: list[list[float]] = []
    for t, b in top_text_hits(cdir):
        sp = next((s_ for s_ in spans if t - 1.0 <= s_[1] and abs(b[0] - s_[2]) < 60 and abs(b[2] - s_[4]) < 120), None)
        if sp:
            sp[1] = t + 0.5
            sp[2], sp[3] = min(sp[2], b[0]), min(sp[3], b[1])
            sp[4], sp[5] = max(sp[4], b[2]), max(sp[5], b[3])
        else:
            spans.append([t - 0.5, t + 0.5, *b])
    return spans


def _is_persistent(sp: list[float], dur: float) -> bool:
    """Text up (nearly) the whole clip is the source credit / channel watermark: keep it (hiding the origin
    would look like a disguised reupload). Only temporary titles and captions get blurred."""
    return dur > 0 and (sp[1] - sp[0]) >= 0.85 * dur


def plan_top_blur(cdir: Path) -> list[list[float]]:
    """[t0, t1, x0, y0, x1, y1] spans (output seconds / pixels) to blur out burned-in source text in the top strip."""
    dur = _duration(cdir / "short.mp4")
    return [[max(0.0, a), b2, max(0, x0 - 40), max(0, y0 - 40), min(1080, x1 + 40), min(240, y1 + 40)]
            for a, b2, x0, y0, x1, y1 in _raw_top_spans(cdir) if not _is_persistent([a, b2], dur)]


def check_top_text(cdir: Path) -> list[str]:
    dur = _duration(cdir / "short.mp4")
    keep = [s_ for s_ in _raw_top_spans(cdir) if _is_persistent(s_, dur)]
    return [f"burned-in source text in the top strip at {t:.1f}s box {b}" for t, b in top_text_hits(cdir)
            if not any(s_[0] <= t <= s_[1] and b[0] < s_[4] and b[2] > s_[2] for s_ in keep)]


def check_burned_text(cdir: Path, fonts: Path) -> list[str]:
    return [f"burned-in source text under captions at {t:.1f}s box y{b[1]}-{b[3]}: '{txt[:40]}'"
            for t, b, txt in burned_text_conflicts(cdir, fonts)]


def check_logo(cdir: Path, box: Path | None, y: int, samples: int = 12) -> list[str]:
    """Mandated logo (Alliance) must be present and unblurred on every sampled frame: compare the
    rendered region with the logo box image (sharpness and pixel difference)."""
    if box is None:
        return []
    import cv2
    ref = cv2.imread(str(box))
    if ref is None:
        return [f"logo: cannot read {box}"]
    h, w = ref.shape[:2]
    x = (1080 - w) // 2
    ref = ref[: h // 2 * 2, : w // 2 * 2]  # yuv420 crops are even-sized
    h, w = ref.shape[:2]
    dur = _duration(cdir / "short.mp4")
    out = []
    for k in range(samples):
        t = min(dur - 0.05, (k + 0.5) * dur / samples)
        r = subprocess.run([tool("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(cdir / "short.mp4"), "-frames:v", "1",
                            "-vf", f"crop={w}:{h}:{x}:{y}", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (h, w):
            out.append(f"logo: frame at {t:.1f}s unreadable")
            continue
        diff = float(np.abs(img.astype(np.int16) - ref.astype(np.int16)).mean())
        if diff > 12:
            out.append(f"logo: not intact at {t:.1f}s (mean diff {diff:.1f})")
    return out


OVERRIDE_CHECKS = {"face_box", "safe_zone_text", "caption_sync_word"}   # visual/safe zone and caption timing only, never content
OVERRIDE_WHO = {"codee", "felulbiralo"}
OVERRIDE_BASIS = {"frame", "listening"}
OVERRIDE_DT = 0.2      # seconds around the overridden point
OVERRIDE_BOX_PX = 40   # box corners must match within this


def _problem_point(problem: str) -> tuple[str, float, tuple[int, int, int, int] | None, str | None] | None:
    """(check type, time, box, word) of a per-point problem line that may be overridden, else None."""
    m = re.match(r"safe zone: (face|text) at ([\d.]+)s box x(\d+)-(\d+) y(\d+)-(\d+)$", problem)
    if m:
        x0, x1, y0, y1 = (int(m.group(i)) for i in (3, 4, 5, 6))
        return ("face_box" if m.group(1) == "face" else "safe_zone_text"), float(m.group(2)), (x0, y0, x1, y1), None
    m = re.match(r"caption sync: '(.+)' lights up at ([\d.]+)s during silence$", problem)
    if m:
        return "caption_sync_word", float(m.group(2)), None, m.group(1).strip().lower()
    return None


def valid_override(o: dict) -> bool:
    """A complete entry: one concrete point, who checked, on what basis, when. Anything less is ignored."""
    if not isinstance(o, dict) or o.get("check") not in OVERRIDE_CHECKS:
        return False
    if str(o.get("by", "")).lower() not in OVERRIDE_WHO or o.get("basis") not in OVERRIDE_BASIS:
        return False
    if not str(o.get("date", "")).strip() or not str(o.get("reason", "")).strip():
        return False
    if not isinstance(o.get("t"), (int, float)):
        return False
    if o["check"] == "caption_sync_word":
        return bool(str(o.get("word", "")).strip())
    box = o.get("box")
    return isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)


def apply_overrides(cdir: Path, problems: list[str]) -> tuple[list[str], list[str]]:
    """Drop per-point problems covered by a valid entry in cdir/override.json (written by Codee or the reviewer only).
    Returns (remaining problems, 'OVERRIDDEN by <who>: <reason>' lines for qa.txt)."""
    import json
    try:
        entries = json.loads((cdir / "override.json").read_text()).get("overrides", [])
    except (OSError, ValueError, AttributeError):
        return problems, []
    entries = [o for o in entries if valid_override(o)]
    if not entries:
        return problems, []
    left, notes = [], []
    for p in problems:
        pt = _problem_point(p)
        hit = None
        if pt:
            kind, t, box, word = pt
            for o in entries:
                if o["check"] != kind or abs(o["t"] - t) > OVERRIDE_DT:
                    continue
                if kind == "caption_sync_word":
                    if str(o["word"]).strip().lower() != word:
                        continue
                elif any(abs(a - b) > OVERRIDE_BOX_PX for a, b in zip(o["box"], box)):
                    continue
                hit = o
                break
        if hit:
            notes.append(f"OVERRIDDEN by {hit['by']}: {hit['reason']} [{p}] ({hit['basis']}, {hit['date']})")
        else:
            left.append(p)
    return left, notes


def run_all(cfg: dict, cdir: Path, audio_fn, source_ends: list[tuple[float, float]],
            source_starts: list[tuple[float, float]] = (), logo: tuple | None = None,
            montage: bool = False) -> list[str]:
    """montage: music/footage sources without speech or captions: skip the speech, caption, face and fast-cut checks."""
    video = cdir / "short.mp4"
    problems = []
    fdir = Path(path_of(cfg, "fonts")).resolve()
    if not fdir.is_dir() or not any(fdir.glob("*.[ot]tf")):
        return [f"QA font dir missing or empty ({fdir}): caption checks cannot be trusted"]
    if montage:
        # edges are faded in render (0.4 s), so the raw-source edge checks do not apply
        problems += check_safe_zone_text_only(cdir, path_of(cfg, "fonts"))
        problems += check_true_peak(video)
    else:
        problems += check_hook_length(cdir)
        problems += check_safe_zone(cdir, path_of(cfg, "fonts"))
        problems += check_caption_face(cdir, path_of(cfg, "fonts"))
        problems += check_segment_ends(audio_fn, source_ends)
        problems += check_segment_starts(audio_fn, list(source_starts))
        problems += check_true_peak(video)
        problems += check_flashes(video)
        problems += check_caption_durations(cdir / "captions.ass")
        problems += check_dead_air(video)
        problems += check_caption_overlap(cdir / "captions.ass")
        problems += check_caption_sync(video, cdir / "captions.ass")
        problems += check_caption_sync_words(video, cdir / "captions.ass")
        problems += check_caption_words(video, cdir / "captions.ass")
        problems += check_edge_words(video)
        problems += check_repeated_words(video, cdir / "captions.ass")
        problems += check_uncaptioned_speech(video, cdir / "captions.ass")
        problems += check_caption_gaps(video, cdir / "captions.ass")
        problems += check_burned_text(cdir, path_of(cfg, "fonts"))
        problems += check_top_text(cdir)
    if logo:
        problems += check_logo(cdir, *logo)
    warns = [p for p in problems if is_warn(p)]
    problems, overridden = apply_overrides(cdir, [p for p in problems if not is_warn(p)])
    lines = problems + overridden
    (cdir / "qa.txt").write_text("\n".join(lines + warns) + "\n" if problems
                                 else "\n".join(["ok"] + overridden + warns) + "\n")
    log_qa(cdir, problems, warns)
    return problems


def qa_check_name(problem: str) -> str:
    """A stable check label from a problem line: drop times, pixels and counts."""
    return re.sub(r"\s+", " ", re.sub(r"[\d.]+|\(.*?\)|@\S*", "", problem)).strip()[:60]


def log_qa(cdir: Path, problems: list[str], warns: list[str] = ()) -> None:
    """Append one QA result to data/qa_log.jsonl (qa.txt gets overwritten on every render)."""
    import json
    from datetime import datetime
    try:
        job = json.loads((cdir.parents[1] / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    m = re.search(r"-v(\d+)$", cdir.name)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "clip_id": cdir.name,
           "version": int(m.group(1)) if m else 1, "source_url": job.get("input"),
           "campaign": job.get("source"), "ok": not problems,
           "failed_checks": sorted({qa_check_name(p) for p in problems}),
           "warnings": sorted({qa_check_name(p.removeprefix(WARN)) for p in warns})}
    out = ROOT / "data" / "qa_log.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


FACE_HAIR_MARGIN = 0.25
FACE_OVERLAP_MIN_PX = 20   # ignore hair-tip / chin grazes below this
FACE_BEARD_MARGIN = 0.04   # detector boxes stop near the chin; a little extra for the beard


FALSE_FACE_SCORE_GAP = 0.10   # a stacked box must score this much below the face above it to be dropped
FALSE_FACE_X_OVERLAP = 0.3    # share of the narrower box that must overlap in x


def drop_false_faces(faces: list[tuple[float, float, float, float, float]]) -> list[tuple[float, float, float, float, float]]:
    """Drop detector boxes (x, y, w, h, score) that are hand/hair artefacts under a real face: the box centre lies below
    another face's bottom edge, the boxes overlap in x, and its score is clearly lower. Two real stacked faces score alike and stay."""
    keep = []
    for i, (x, y, w, h, sc) in enumerate(faces):
        cy = y + h / 2
        false = False
        for j, (ox, oy, ow, oh, osc) in enumerate(faces):
            if i == j or cy <= oy + oh:
                continue
            xo = min(x + w, ox + ow) - max(x, ox)
            if xo >= FALSE_FACE_X_OVERLAP * min(w, ow) and sc <= osc - FALSE_FACE_SCORE_GAP:
                false = True
                break
        if not false:
            keep.append((x, y, w, h, sc))
    return keep


def caption_face_overlaps(cdir: Path, fonts: Path, step: float = 0.2) -> list[tuple[float, int]]:
    """(t, overlap px) where a caption block sits on a speaker's face (box extended down for beard/jaw).
    Checks every `step` s, the caption block and the hook alike."""
    import cv2
    video = cdir / "short.mp4"
    dur = _duration(video)
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                    (540, 960), 0.8, 0.3, 50)
    cap = cv2.VideoCapture(str(video))
    out = []
    for t, (x0, y0, x1, y1) in text_boxes(cdir, dur, fonts, split=True):
        if abs(t / step - round(t / step)) > 0.01:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        _, faces = det.detect(cv2.resize(fr, (540, 960)))
        boxes = [tuple(float(v) * 2 for v in f[:4]) + (float(f[14]),) for f in (faces if faces is not None else [])]
        for fx, fy, fw, fh, _sc in drop_false_faces(boxes):
            if fw < 1080 * SPEAKER_FACE_W:
                continue
            fb = fy + fh * (1 + FACE_BEARD_MARGIN)
            ft = fy + fh * FACE_HAIR_MARGIN   # the detector box includes hair; a hook grazing the hair is fine
            if x1 > fx and x0 < fx + fw and y1 > ft and y0 < fb:
                out.append((t, int(min(y1, fb) - max(y0, ft))))
    cap.release()
    return out


def check_caption_face(cdir: Path, fonts: Path) -> list[str]:
    return [f"caption/hook on the speaker's face at {t:.1f}s (overlap {ov}px)" for t, ov in caption_face_overlaps(cdir, fonts) if ov >= FACE_OVERLAP_MIN_PX]
