"""Safe-zone check for a rendered clip (TikTok / Reels / Shorts strictest zone).

Usage: uv run python scripts/safecheck.py <clip dir>

Text: captions.ass rendered alone on black, bounding box of the text every 0.1 s.
Faces: YuNet on short.mp4 every 0.5 s (speaker faces: wider than 10% of the frame).
Limits: top 220, bottom 1500 (y > 1500 is UI), right 920 (x > 920 is the button column).
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from clipper.common import ROOT, path_of, load_config, tool

TOP, BOTTOM, RIGHT = 220, 1500, 920


def text_boxes(cdir: Path, dur: float) -> list[tuple[float, tuple[int, int, int, int]]]:
    fonts = path_of(load_config(), "fonts")
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
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                    (540, 960), 0.6, 0.3, 50)
    cap = cv2.VideoCapture(str(video))
    out, t = [], 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            small = cv2.resize(fr, (540, 960))
            _, faces = det.detect(small)
            for f in faces if faces is not None else []:
                x, y, w, h = (float(v) * 2 for v in f[:4])
                if w >= 1080 * 0.10:  # speaker faces; tiny detections are background or false positives
                    out.append((t, (int(x), int(y), int(x + w), int(y + h))))
        t += 0.5
    return out


def main(cdir: Path) -> int:
    video = cdir / "short.mp4"
    dur = float(subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout)
    bad = []
    for t, (x0, y0, x1, y1) in text_boxes(cdir, dur):
        if y0 < TOP or y1 > BOTTOM or x1 > RIGHT:
            bad.append(f"text {t:.1f}s box x{x0}-{x1} y{y0}-{y1}")
    for t, (x0, y0, x1, y1) in face_boxes(video, dur):
        if y0 < TOP or y1 > BOTTOM or x1 > RIGHT:
            bad.append(f"face {t:.1f}s box x{x0}-{x1} y{y0}-{y1}")
    print(f"{cdir.name}: {'OK' if not bad else f'{len(bad)} violations'}")
    for b in bad[:25]:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
