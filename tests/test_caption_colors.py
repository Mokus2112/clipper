"""Klip v2 captions: grey sentence (#C8C8C8), white active word, set from config only."""
import copy
import re
from pathlib import Path

import yaml

from clipper.render import build_ass

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
FONTS = ROOT / "fonts"
GREY, WHITE = "&H00C8C8C8", "&H00FFFFFF"  # ASS BGR


def ass_for(color, highlight):
    cfg = copy.deepcopy(CFG)
    cfg["captions"]["color"], cfg["captions"]["highlight"] = color, highlight
    words = [{"w": w, "s": 0.4 * i, "e": 0.4 * i + 0.3} for i, w in enumerate("this is the whole point".split())]
    return build_ass(cfg, {"words": words}, [(0.0, 5.0, 0.0)], "center", None, FONTS)


def test_style_uses_the_sentence_colour_and_active_word_uses_the_highlight():
    ass = ass_for(GREY, WHITE)
    assert re.search(r"Style: Cap,[^,]*,\d+," + GREY, ass)
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue: 0,")]
    assert lines and all(f"\\c{WHITE}" in l for l in lines)   # the active word is white
    assert all(f"\\c{GREY}" in l for l in lines)              # and goes back to grey afterwards


def test_each_line_lights_exactly_one_word_at_a_time():
    lines = [l for l in ass_for(GREY, WHITE).splitlines() if l.startswith("Dialogue: 0,")]
    assert all(len(re.findall(re.escape("\\c" + WHITE), l)) == 1 for l in lines)
