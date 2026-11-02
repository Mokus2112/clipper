import re
from pathlib import Path

import yaml

from clipper.render import SAFE_TOP, build_ass
from clipper.textfit import drawn_width, font_file

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
FONTS = ROOT / "fonts"


def words(text, t=0.0, step=0.3):
    out = []
    for w in text.split():
        out.append({"w": w, "s": t, "e": t + step * 0.8})
        t += step
    return out


def caps(ass):
    return [l for l in ass.splitlines() if l.startswith("Dialogue: 0,")]


def secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_caption_lines_fit_measured_width():
    clip = {"words": words("the psychological argument for psychologically unbelievable plays here")}
    ass = build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "center", None, FONTS)
    font = font_file(FONTS, CFG["captions"]["font"])
    cc = CFG["captions"]
    for line in caps(ass):
        text = re.sub(r"\{[^}]*\}", "", line.split(",", 9)[9])
        assert drawn_width(font, text, cc["size"], cc["spacing"], cc["outline"], cc["shadow"]) <= cc["max_width"]


def test_fade_only_on_first_event_of_a_line_and_no_overlap():
    clip = {"words": words("one two three four five six")}
    lines = caps(build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "center", None, FONTS))
    fades = [l for l in lines if "\\fad(" in l]
    texts = [re.sub(r"\{[^}]*\}", "", l.split(",", 9)[9]) for l in lines]
    assert len(fades) == len(set(texts))  # one fade per caption line
    spans = sorted((secs(l.split(",")[1]), secs(l.split(",")[2])) for l in lines)
    assert all(b0 <= a1 + 1e-6 for (_, b0), (a1, _) in zip(spans, spans[1:]))


def test_hook_min_size_and_box_height():
    clip = {"hook": "Lacy got caught lacking here", "words": []}
    ass = build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "center", None, FONTS)
    size = int(re.search(r"\\fs(\d+)", ass).group(1))
    assert size >= CFG["hook"]["min_size"]
    ys = [int(v) for v in re.findall(r"m \d+ (\d+) l \d+ \d+ l \d+ (\d+)", ass)[0]]
    assert ys[1] - ys[0] <= 100


def hook_box(ass):
    m = re.search(r"HookBox.*?m \d+ (\d+) l \d+ \d+ l \d+ (\d+)", ass)
    return (int(m.group(1)), int(m.group(2))) if m else None


def hook_end(ass):
    m = re.search(r"Dialogue: 1,[\d:.]+,([\d:.]+),HookBox", ass)
    return secs(m.group(1))


def test_hook_one_fixed_position_and_short_on_tight_closeup():
    clip = {"hook": "Who pooped in Depp's bed?", "words": []}
    y = CFG["hook"]["y"]
    # room above the hair: full window, fixed y
    geo = {"ft": 700, "fb": 1000, "cap": 1.2, "t0": 0.0}
    ass = build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "shots", [(0.0, 5.0, "face", geo)], FONTS)
    box = hook_box(ass)
    assert box and abs((box[0] + box[1]) / 2 - y) < 1 and box[0] >= SAFE_TOP
    assert abs(hook_end(ass) - CFG["hook"]["seconds"]) < 0.02
    # head too high: same position, but at most 2.0 s
    geo = {"ft": 330, "fb": 900, "cap": 1.0, "t0": 0.0}
    ass = build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "shots", [(0.0, 5.0, "face", geo)], FONTS)
    box = hook_box(ass)
    assert abs((box[0] + box[1]) / 2 - y) < 1
    assert hook_end(ass) <= 2.0 + 0.02
    assert len([l for l in ass.splitlines() if l.startswith("Dialogue") and "HookBox" in l]) == 2  # no per-shot jumping


def test_no_fade_on_short_chunk_and_short_words_merge():
    clip = {"words": [{"w": "a", "s": 0.0, "e": 0.05}, {"w": "big", "s": 0.05, "e": 0.4},
                      {"w": "yes", "s": 0.4, "e": 0.9}]}
    lines = caps(build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "center", None, FONTS))
    for l in lines:
        d = secs(l.split(",")[2]) - secs(l.split(",")[1])
        assert d >= 4 / 30 - 0.011  # no 1-3 frame flashes
