"""Keyword badge: one accent chip in the hook slot, after the hook, inside the safe zone (klip v2, reviewer #1638)."""
import re
from pathlib import Path

import yaml

from clipper import qa
from clipper.render import SAFE_LEFT, SAFE_RIGHT_X, build_ass, out_time

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
FONTS = ROOT / "fonts"
SEGS = [(10.0, 20.0, 0.0), (30.0, 40.0, 10.0)]  # output 0-20 s


def words(text, t=10.0, step=0.4):
    return [{"w": w, "s": t + i * step, "e": t + i * step + step * 0.8} for i, w in enumerate(text.split())]


def ass_with(badge, clip_extra=None, segs=SEGS):
    clip = {"hook": "A possible AI warning shot", "words": words("the clearest warning shot we ever get for loss of control"),
            "start": 10.0, "end": 40.0, "badge": badge, **(clip_extra or {})}
    return build_ass(CFG, clip, segs, "center", None, FONTS)


def badge_lines(ass):
    return [l for l in ass.splitlines() if l.startswith("Dialogue: ") and ("HookBox" in l or "HookText" in l)
            and not l.startswith("Dialogue: 0,")]


def secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_out_time_maps_source_to_output():
    assert out_time(SEGS, 12.0) == 2.0 and out_time(SEGS, 31.5) == 11.5 and out_time(SEGS, 25.0) is None


def test_badge_found_by_word_and_shown_after_the_hook():
    ass = ass_with({"text": "Clearest warning shot", "at": "clearest"})
    lines = [l for l in badge_lines(ass) if "CLEAREST WARNING SHOT" in l or "m " in l]
    hook_end = CFG["hook"]["seconds"]
    badge = [l for l in lines if "\\fad(120,120)" in l]
    assert len(badge) == 2  # chip + text
    start = secs(badge[0].split(",")[1])
    assert start >= hook_end + 0.2 - 1e-6
    assert secs(badge[0].split(",")[2]) - start >= 0.4


def test_badge_text_and_chip_stay_inside_the_safe_zone():
    ass = ass_with({"text": "An extremely long keyword badge text that must shrink to fit the safe width of the frame", "t0": 14.0})
    chip = next(l for l in badge_lines(ass) if "\\p1" in l and "\\fad(120,120)" in l)
    xs = [int(v) for v in re.findall(r"[ml] (\d+) \d+", chip.split("}")[-1])]
    assert min(xs) == SAFE_LEFT and max(xs) <= SAFE_RIGHT_X
    size = int(re.search(r"\\fs(\d+)", next(l for l in badge_lines(ass) if "\\fs" in l and "\\fad(120,120)" in l)).group(1))
    assert 36 <= size <= 64


def test_only_one_badge_and_none_without_a_valid_time():
    assert len([l for l in badge_lines(ass_with({"text": "x", "t0": 14.0})) if "\\fad(120,120)" in l]) == 2
    assert not [l for l in badge_lines(ass_with({"text": "x", "at": "nonexistentword"})) if "\\fad(120,120)" in l]
    assert not [l for l in badge_lines(ass_with({"text": "x", "t0": 25.0})) if "\\fad(120,120)" in l]  # in a cut-out part


def test_no_badge_key_changes_nothing():
    clip = {"hook": "A possible AI warning shot", "words": words("a b c"), "start": 10.0, "end": 40.0}
    assert build_ass(CFG, clip, SEGS, "center", None, FONTS) == build_ass(CFG, {**clip, "badge": None}, SEGS, "center", None, FONTS)


def test_badge_colours_default_to_the_hook_palette_and_can_be_overridden():
    ass = ass_with({"text": "x", "t0": 14.0})
    assert CFG["hook"]["box"] in next(l for l in badge_lines(ass) if "\\p1" in l and "\\fad(120,120)" in l)
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["badge"] = {"fill": "&H00FFFFFF", "color": "&H00000000"}
    clip = {"hook": "h", "words": words("a b c d e f g h i j k l"), "start": 10.0, "end": 40.0, "badge": {"text": "x", "t0": 14.0}}
    out = build_ass(cfg, clip, SEGS, "center", None, FONTS)
    assert "\\c&H00FFFFFF\\fad" in out and "\\c&H00000000\\fad" in out
