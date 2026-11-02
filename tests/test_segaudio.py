import subprocess
from pathlib import Path

import pytest

from clipper.common import tool
from clipper.segaudio import extract_range, wav_seconds

# Copy of the style-test dark clip source (alliance c1): input seek loses 0.43 s of audio at -ss 22.44.
SRC = Path("/Users/badi/Codee/agents/clip-fejleszto/work/E/data/jobs/alliance-secure-ai-20261102-182604/source.mp4")
pytestmark = pytest.mark.skipif(not SRC.exists(), reason="fixture source missing")


def _dur(args, tmp_path):
    out = tmp_path / "x.wav"
    subprocess.run([tool("ffmpeg"), "-v", "error", "-y", *args, "-vn", "-c:a", "pcm_s16le", str(out)], check=True)
    return wav_seconds(out)


def test_input_seek_is_short_on_this_source(tmp_path):
    # documents the bug: this is the form the render used per segment
    assert _dur(["-ss", "22.44", "-t", "3.84", "-i", str(SRC)], tmp_path) < 3.6


def test_segments_from_extracted_wav_have_full_length(tmp_path):
    wav = tmp_path / "range.wav"
    w0 = extract_range(SRC, 22.44, 28.0, wav)
    assert w0 <= 22.44
    for s, d in [(22.44, 3.84), (26.9, 0.71), (22.2, 1.0)]:
        got = _dur(["-ss", f"{s - w0:.6f}", "-t", f"{d:.6f}", "-i", str(wav)], tmp_path)
        assert abs(got - d) < 0.01, (s, d, got)


def test_wav_content_matches_exact_seek(tmp_path):
    import numpy as np
    wav = tmp_path / "range.wav"
    w0 = extract_range(SRC, 22.44, 28.0, wav)
    def pcm(args):
        return np.frombuffer(subprocess.run([tool("ffmpeg"), "-v", "error", *args, "-vn", "-ac", "1", "-ar", "16000",
                                              "-f", "s16le", "-"], capture_output=True, check=True).stdout, np.int16).astype(float)
    a = pcm(["-i", str(SRC), "-ss", "22.44", "-t", "3.84"])
    b = pcm(["-ss", f"{22.44 - w0:.6f}", "-t", "3.84", "-i", str(wav)])
    n = min(len(a), len(b))
    assert n > 3.8 * 16000 and np.corrcoef(a[:n], b[:n])[0, 1] > 0.98
