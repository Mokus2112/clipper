"""Automatic QA at the end of every render: defects the reviewer found once must not reach him again.

Each check returns a list of problems (empty = ok). See data/campaigns/LESSONS.md for the history.
"""

import re
import subprocess
from pathlib import Path

import numpy as np

from .common import ROOT, path_of, tool

# Strictest safe zone over YouTube Shorts, TikTok and Instagram Reels (1080x1920).
TOP, BOTTOM, RIGHT = 220, 1500, 920
SPEAKER_FACE_W = 0.10      # faces wider than 10% of the frame are speakers
END_SILENCE_DB = -30.0     # last 50 ms of a segment must be at most this loud (no cut mid-word)
MIN_SHOT_FRAMES = 3        # a shot shorter than this is a flash
MIN_CAPTION = 0.25         # s a caption chunk must stay on screen


def _duration(video: Path) -> float:
    return float(subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout)


def text_boxes(cdir: Path, dur: float, fonts: Path, split: bool = False) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Bounding box of all on-screen text (captions + hook) every 0.1 s, rendered alone on black."""
    raw = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"color=black:s=1080x1920:r=10:d={dur:.2f}",
         "-vf", f"subtitles=captions.ass:fontsdir={fonts},format=gray", "-f", "rawvideo", "-"],
        cwd=cdir, capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 1920, 1080)
    out = []
    for i, f in enumerate(frames):
        ys, xs = np.nonzero(f > 40)
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
            for f in faces if faces is not None else []:
                x, y, w, h = (float(v) * 2 for v in f[:4])
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
        if y0 < TOP or y1 > BOTTOM or x1 > RIGHT:
            bad.append(f"safe zone: text at {t:.1f}s box x{x0}-{x1} y{y0}-{y1}")
    for t, (x0, y0, x1, y1) in face_boxes(video, dur):
        if y0 < TOP or y1 > BOTTOM or x1 > RIGHT:
            bad.append(f"safe zone: face at {t:.1f}s box x{x0}-{x1} y{y0}-{y1}")
    return bad


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


def _highlights(ass: Path) -> list[tuple[float, float, str]]:
    """(start, end, highlighted word) of every caption line."""
    def secs(t: str) -> float:
        h, m, s_ = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s_)
    out = []
    for line in ass.read_text().splitlines():
        if line.startswith("Dialogue: 0,"):
            p = line.split(",", 9)
            m = re.search(r"\\c&H0000E0FF\}([^{]+)", p[9])
            out.append((secs(p[1]), secs(p[2]), m.group(1).strip() if m else ""))
    return out


def _db_curve(a: np.ndarray) -> np.ndarray:
    return np.array([20 * np.log10(np.sqrt(np.mean(a[i:i + 320] ** 2)) + 1e-9) for i in range(0, max(len(a) - 320, 0), 160)])


def check_caption_sync(video: Path, ass: Path) -> list[str]:
    """Highlights vs audio: a word after a pause lights up within 80 ms of its onset, and no new
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
        if abs(t - o) > 0.08:
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


def check_dead_air(video: Path, max_gap: float = 0.4) -> list[str]:
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


def check_caption_words(video: Path, ass: Path) -> list[str]:
    """Every captioned word (4+ letters, not a filler) must be heard: re-transcribe the output."""
    from faster_whisper import WhisperModel
    from .transcribe import load_audio
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower().replace("*", ""))  # noqa: E731
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(load_audio(video, 0.0, _duration(video)), language="en", word_timestamps=True)
    heard = [(norm(w.word), w.start) for s in segs for w in (s.words or [])]
    missing = []
    for s0, _e, w in _highlights(ass):
        if "-" in w:  # whisper hears 'ex-wife' as 'ex', 'wife': check the longest part
            w = max(w.split("-"), key=len)
        n = norm(w)
        if len(n) < 4 or n in FILLERS or "*" in w:
            continue
        if not any((h == n or h.startswith(n[:4])) and abs(t - s0) <= 1.5 for h, t in heard):
            missing.append(f"{w}@{s0:.2f}")
    missing = sorted(set(missing))
    return [f"caption words not heard in the audio: {', '.join(missing[:8])}"] if len(missing) >= 2 else []


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
        m = re.search(r"\\c&H0000E0FF\}([^{]+)", p[9])
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
    return sorted(set(problems))


def check_uncaptioned_speech(video: Path, ass: Path, thr_db: float = -35.0, min_len: float = 0.3) -> list[str]:
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
            # a short but loud run (>= 100 ms, peak above -20 dB) is a spoken syllable, not a caption gap
            if (j - i) / 100 >= min_len or ((j - i) >= 10 and db[i:j].max() > -20):
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
    out += [f"{a0:.2f}-{b0:.2f}(whisper)" for a0, b0 in spans if b0 - a0 >= 0.3]
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
                    if wx0 < x1 and wx1 > x0 and wy0 < y1 and wy1 > y0:  # overlaps our text box
                        words.append(c[11])
            txt = " ".join(words)
            if len(re.sub(r"\W", "", txt)) >= min_chars:
                out.append((t, (x0 * 2, y0 * 2, x1 * 2, y1 * 2), txt))
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
    boxes = text_boxes(cdir, _duration(cdir / "short.mp4"), fonts, split=True)
    out = []
    for a, b in spans:
        inside = [bx for t, bx in boxes if a <= t <= b and bx[3] > 1100]
        top, bot = min(bx[1] for bx in inside), max(bx[3] for bx in inside)
        for dy in (-260, -400, -520, 240, 360):
            if top + dy < TOP or bot + dy > BOTTOM:
                continue
            left = burned_text_conflicts(cdir, fonts, dy_at=lambda t, a=a, b=b, dy=dy: dy if a <= t <= b else 0,
                                           captions_only=True)
            if all(not (a <= t <= b) for t, _, _ in left):
                out.append([max(0.0, a), b, dy])
                break
    return out


def check_burned_text(cdir: Path, fonts: Path) -> list[str]:
    return [f"burned-in source text under captions at {t:.1f}s box y{b[1]}-{b[3]}: '{txt[:40]}'"
            for t, b, txt in burned_text_conflicts(cdir, fonts)]


def run_all(cfg: dict, cdir: Path, audio_fn, source_ends: list[tuple[float, float]],
            source_starts: list[tuple[float, float]] = ()) -> list[str]:
    video = cdir / "short.mp4"
    problems = []
    problems += check_safe_zone(cdir, path_of(cfg, "fonts"))
    problems += check_segment_ends(audio_fn, source_ends)
    problems += check_segment_starts(audio_fn, list(source_starts))
    problems += check_true_peak(video)
    problems += check_flashes(video)
    problems += check_caption_durations(cdir / "captions.ass")
    problems += check_dead_air(video)
    problems += check_caption_overlap(cdir / "captions.ass")
    problems += check_caption_sync(video, cdir / "captions.ass")
    problems += check_caption_words(video, cdir / "captions.ass")
    problems += check_repeated_words(video, cdir / "captions.ass")
    problems += check_uncaptioned_speech(video, cdir / "captions.ass")
    problems += check_burned_text(cdir, path_of(cfg, "fonts"))
    (cdir / "qa.txt").write_text("\n".join(problems) + "\n" if problems else "ok\n")
    return problems
