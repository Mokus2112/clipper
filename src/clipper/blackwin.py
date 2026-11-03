"""Share of a source window that is black (title cards, fades): rank drops windows that are mostly black.

CoD c3 (reviewer): about 30% of the 21 s window were black title cards; cutting them out left an 18 s clip that
still had black stretches, so the window itself is the problem.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .common import tool

BLACK_MAX = 0.20  # share of the window; above it the pick is dropped
BLACK_D, BLACK_PIC_TH = 0.1, 0.90


def black_seconds(stderr: str) -> float:
    """Total black time from blackdetect's log output."""
    return sum(float(b) - float(a) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", stderr))


def black_fraction(video: Path, start: float, end: float) -> float:
    res = subprocess.run(
        [tool("ffmpeg"), "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video), "-an",
         "-vf", f"blackdetect=d={BLACK_D}:pic_th={BLACK_PIC_TH}:pix_th=0.10", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    return black_seconds(res.stderr) / max(end - start, 1e-6)
