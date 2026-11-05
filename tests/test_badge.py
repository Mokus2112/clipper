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
    badge = [l for l in lines if "\\fad(133,133)" in l]
    assert len(badge) == 2  # chip + one text line
    start = secs(badge[0].split(",")[1])
    assert start >= hook_end + 0.2 - 1e-6
    assert secs(badge[0].split(",")[2]) - start >= 0.4


def test_badge_text_and_chip_stay_inside_the_safe_zone():
    ass = ass_with({"text": "An extremely long keyword badge text that must shrink to fit the safe width of the frame", "t0": 14.0})
    chip = next(l for l in badge_lines(ass) if "\\p1" in l and "\\fad(133,133)" in l)
    xs = [int(v) for v in re.findall(r"[ml] (\d+) \d+", chip.split("}")[-1])]
    assert min(xs) == SAFE_LEFT and max(xs) <= SAFE_RIGHT_X
    size = int(re.search(r"\\fs(\d+)", next(l for l in badge_lines(ass) if "\\fs" in l and "\\fad(133,133)" in l)).group(1))
    assert 36 <= size <= 64  # two lines at >= 44, or the 36 px last resort


def test_only_one_badge_and_none_without_a_valid_time():
    assert len([l for l in badge_lines(ass_with({"text": "x", "t0": 14.0})) if "\\fad(133,133)" in l]) == 2
    assert not [l for l in badge_lines(ass_with({"text": "x", "at": "nonexistentword"})) if "\\fad(133,133)" in l]
    assert not [l for l in badge_lines(ass_with({"text": "x", "t0": 25.0})) if "\\fad(133,133)" in l]  # in a cut-out part


def test_no_badge_key_changes_nothing():
    clip = {"hook": "A possible AI warning shot", "words": words("a b c"), "start": 10.0, "end": 40.0}
    assert build_ass(CFG, clip, SEGS, "center", None, FONTS) == build_ass(CFG, {**clip, "badge": None}, SEGS, "center", None, FONTS)


def test_badge_colours_default_to_the_hook_palette_and_can_be_overridden():
    ass = ass_with({"text": "x", "t0": 14.0})
    assert CFG["hook"]["box"] in next(l for l in badge_lines(ass) if "\\p1" in l and "\\fad(133,133)" in l)
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["badge"] = {"fill": "&H00FFFFFF", "color": "&H00000000"}
    clip = {"hook": "h", "words": words("a b c d e f g h i j k l"), "start": 10.0, "end": 40.0, "badge": {"text": "x", "t0": 14.0}}
    out = build_ass(cfg, clip, SEGS, "center", None, FONTS)
    assert "\\c&H00FFFFFF\\fad" in out and "\\c&H00000000\\fad" in out


from clipper.render import BADGE_FADE_MS, BADGE_MIN_SIZE, BADGE_SLIDE_PX, badge_layout
from clipper.textfit import drawn_width, font_file

LONG = "Clearest warning shot we ever get for a loss of control over agents"


def text_lines(ass):
    return [l for l in badge_lines(ass) if "\\fs" in l and "\\fad(133,133)" in l]


def test_fade_is_four_frames_in_and_out_and_slides_in():
    assert BADGE_FADE_MS == 133  # 4 frames at 30 fps
    ass = ass_with({"text": "Clearest warning shot", "t0": 14.0})
    for l in [l for l in badge_lines(ass) if "\\fad(133,133)" in l]:
        m = re.search(r"\\move\((-?\d+),(\d+),(-?\d+),(\d+),0,133\)", l)
        assert m and int(m.group(3)) - int(m.group(1)) == BADGE_SLIDE_PX and m.group(2) == m.group(4)


def test_text_is_never_below_44_px_it_wraps_to_two_lines_instead():
    ass = ass_with({"text": LONG, "t0": 14.0})
    tl = text_lines(ass)
    assert len(tl) == 2  # two lines
    assert all(int(re.search(r"\\fs(\d+)", l).group(1)) >= BADGE_MIN_SIZE for l in tl)
    ys = [int(re.search(r"\\move\(-?\d+,(\d+)", l).group(1)) for l in tl]
    assert ys[1] > ys[0]


def test_two_line_badge_chip_covers_both_lines_and_stays_in_the_safe_zone():
    ass = ass_with({"text": LONG, "t0": 14.0})
    chip = next(l for l in badge_lines(ass) if "\\p1" in l and "\\fad(133,133)" in l)
    pts = [(int(a), int(b)) for a, b in re.findall(r"[ml] (\d+) (\d+)", chip.split("}")[-1])]
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    tl = text_lines(ass)
    ly = [int(re.search(r"\\move\(-?\d+,(\d+)", l).group(1)) for l in tl]
    assert y0 < min(ly) and max(ly) < y1 and max(p[0] for p in pts) <= SAFE_RIGHT_X and y0 >= 290


def test_short_text_stays_one_line_at_the_full_size():
    ass = ass_with({"text": "Clearest warning shot", "t0": 14.0})
    tl = text_lines(ass)
    assert len(tl) == 1 and int(re.search(r"\\fs(\d+)", tl[0]).group(1)) == 64


def test_layout_uses_measured_widths():
    font = font_file(FONTS, CFG["hook"]["font"])
    lines, size = badge_layout(font, LONG.upper(), 64, 700)
    assert size >= BADGE_MIN_SIZE and all(drawn_width(font, ln, size, 0, 0, 0) <= 700 + 4 for ln in lines)


def test_nothing_fits_even_in_two_lines_falls_back_below_44_with_a_warning(capsys):
    font = font_file(FONTS, CFG["hook"]["font"])
    lines, size = badge_layout(font, "SUPERCALIFRAGILISTICEXPIALIDOCIOUSWORDTHATISVERYLONGINDEED", 64, 300)
    assert size < BADGE_MIN_SIZE and len(lines) == 1


def test_badge_is_made_in_shots_and_split_layouts_too():
    for layout, spans in (("shots", [(0.0, 30.0, "face", {"ft": 600, "fb": 1000, "cap": 1.2, "t0": 0.0})]), ("split", None)):
        clip = {"hook": "A hook", "words": words("the clearest warning shot we ever get for loss of control"),
                "start": 10.0, "end": 40.0, "badge": {"text": "Clearest warning shot", "t0": 14.0}}
        ass = build_ass(CFG, clip, SEGS, layout, spans, FONTS)
        chip = [l for l in ass.splitlines() if "\\p1" in l and "\\fad(133,133)" in l]
        assert chip, layout
        pts = [int(a) for a in re.findall(r"[ml] (\d+) \d+", chip[0].split("}")[-1])]
        assert min(pts) == SAFE_LEFT and max(pts) <= SAFE_RIGHT_X


def test_y_override_moves_the_chip_and_a_face_in_the_slot_is_warned(capsys):
    base = ass_with({"text": "Clearest warning shot", "t0": 14.0})
    moved = ass_with({"text": "Clearest warning shot", "t0": 14.0, "y": 700})
    yb = [int(v) for v in re.findall(r"[ml] \d+ (\d+)", next(l for l in badge_lines(base) if "\\p1" in l and "\\fad(133,133)" in l).split("}")[-1])]
    ym = [int(v) for v in re.findall(r"[ml] \d+ (\d+)", next(l for l in badge_lines(moved) if "\\p1" in l and "\\fad(133,133)" in l).split("}")[-1])]
    assert min(ym) - min(yb) == 700 - CFG["hook"]["y"]
    clip = {"hook": "h", "words": words("a b c d e f g h i j k l m n o p"), "start": 10.0, "end": 40.0,
            "badge": {"text": "Clearest warning shot", "t0": 14.0}}
    spans = [(0.0, 20.0, "face", {"ft": 330, "fb": 900, "cap": 1.2, "t0": 0.0})]  # forehead at the chip's height
    build_ass(CFG, clip, SEGS, "shots", spans, FONTS)
    assert "sits on the face" in capsys.readouterr().err
