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


def text_boxes(cdir: Path, dur: float, fonts: Path) -> list[tuple[float, tuple[int, int, int, int]]]:
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
        if len(xs):
            out.append((i / 10, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return out


def face_boxes(video: Path, dur: float) -> list[tuple[float, tuple[int, int, int, int]]]:
    import cv2
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                    (540, 960), 0.6, 0.3, 50)
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
        a = audio_fn(src_t - 0.03, 0.03)
        db = 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9) if len(a) else -99
        if db > END_SILENCE_DB:
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
    (cdir / "qa.txt").write_text("\n".join(problems) + "\n" if problems else "ok\n")
    return problems
