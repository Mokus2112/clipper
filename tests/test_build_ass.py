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


def test_closeup_hook_never_on_the_head():
    clip = {"hook": "Who pooped in Depp's bed?", "words": []}
    # small face low in frame: room above the hair
    geo = {"ft": 700, "fb": 1000, "cap": 1.2, "t0": 0.0}
    box = hook_box(build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "shots", [(0.0, 5.0, "face", geo)], FONTS))
    assert box and box[0] >= SAFE_TOP and box[1] <= 700 - 0.35 * 300
    # big face high in frame: under the chin, clear of the captions
    geo = {"ft": 330, "fb": 900, "cap": 1.0, "t0": 0.0}
    box = hook_box(build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "shots", [(0.0, 5.0, "face", geo)], FONTS))
    assert box and box[0] >= 940 and box[1] <= 1410 - 90 - 30
    # face fills the frame: no hook at all
    geo = {"ft": 330, "fb": 1330, "cap": 1.0, "t0": 0.0}
    assert hook_box(build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "shots", [(0.0, 5.0, "face", geo)], FONTS)) is None
