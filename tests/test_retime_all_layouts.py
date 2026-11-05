"""Caption retiming and the segment WAV are layout independent; only the ffmpeg audio INPUTS are gated (shots hang)."""
import re
from pathlib import Path

SRC = (Path(__file__).parents[1] / "src/clipper/render.py").read_text()


def test_wav_extraction_and_retime_do_not_depend_on_layout():
    assert not re.search(r"extract_range\([^\n]*\) if wav_audio", SRC)
    assert not re.search(r"if wav_audio and clip\.get\(\"words\"\)", SRC)
    assert re.search(r"if clip\.get\(\"words\"\) and cfg\[\"captions\"\]\.get\(\"retime_from_audio\"", SRC)


def test_shots_still_renders_with_the_video_audio():
    assert 'wav_audio = layout != "shots"' in SRC
    assert re.search(r"if wav_audio:\n\s+for i, \(s, e, _\) in enumerate\(segments\):.*seg_wav", SRC, re.S)
