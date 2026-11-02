"""Segment audio with a sample-exact timeline.

`ffmpeg -ss S -i video` (input seek) can return audio that starts later than S: on some sources
everything between a keyframe and ~0.9 s after it is dropped (alliance c1: -ss 22.44 -t 3.84 gave
3.41 s, content from 22.87). The render then padded the missing part with silence, so captions ran
late and words vanished from the finished audio. Here the clip range is decoded once into a WAV
(wav seeks are sample-exact) and every segment reads from that file.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .common import tool

PRE = 4.0  # s decoded before the first segment: the start-of-seek loss falls into this margin


def wav_seconds(path: Path) -> float:
    """Length of a PCM wav written by extract_range (44-byte header assumed from ffprobe instead)."""
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def extract_range(video: Path, t0: float, t1: float, out_wav: Path, pre: float = PRE) -> float:
    """Decode the audio of [t0 - pre, t1] to out_wav; returns the source time of the wav's first sample.

    Input seek can lose audio at the start of the range (never at the end), so the real start is
    requested start + (requested length - decoded length). If that lands after t0, fall back to the
    slow but exact output seek.
    """
    start = max(0.0, t0 - pre)
    want = t1 - start
    base = [tool("ffmpeg"), "-v", "error", "-y"]
    tail = ["-vn", "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(base + ["-ss", f"{start:.6f}", "-t", f"{want:.6f}", "-i", str(video)] + tail, check=True)
    wav_start = start + max(0.0, want - wav_seconds(out_wav))
    if wav_start > t0 + 0.001:
        subprocess.run(base + ["-i", str(video), "-ss", f"{start:.6f}", "-t", f"{want:.6f}"] + tail, check=True)
        wav_start = start
    return wav_start
