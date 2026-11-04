"""Card H: captions and hook stay inside x65-860, y290-1250 (Badi 1011)."""
import re
from pathlib import Path

import yaml

from clipper import qa
from clipper.render import SAFE_RIGHT_X, build_ass

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
FONTS = ROOT / "fonts"
X_MAX, Y_TOP, Y_BOTTOM = 860, 290, 1250


def words(text, step=0.3):
    return [{"w": w, "s": i * step, "e": i * step + step * 0.8} for i, w in enumerate(text.split())]


def test_text_limits():
    assert SAFE_RIGHT_X == X_MAX
    assert (qa.TEXT_RIGHT, qa.TEXT_BOTTOM) == (X_MAX, Y_BOTTOM)


def test_every_caption_position_ends_above_1250():
    cc = CFG["captions"]
    half = (cc["size"] * 0.75) / 2 + cc["outline"] + cc["shadow"]  # Bebas cap height ~0.7-0.75 em
    for key in ("y_center", "y_split", "y_shots", "y_shots_duo", "y_shots_face", "y_fit"):
        assert cc[key] + half <= Y_BOTTOM, key


def test_hook_box_inside_zone_for_long_hook():
    clip = {"hook": "Lacy got caught lacking here", "words": words("a b c")}
    ass = build_ass(CFG, clip, [(0.0, 10.0, 0.0)], "center", None, FONTS)
    xs = [int(v) for v in re.search(r"HookBox.*?m (\d+) \d+ l (\d+)", ass).groups()]
    y0 = int(re.search(r"HookBox.*?m \d+ (\d+) l", ass).group(1))
    assert xs[1] <= X_MAX and y0 >= Y_TOP


def test_caption_text_right_edge():
    # captions are centred on (SAFE_LEFT + SAFE_RIGHT_X) / 2 and at most max_width wide
    from clipper.render import SAFE_LEFT
    assert (SAFE_LEFT + SAFE_RIGHT_X) / 2 + CFG["captions"]["max_width"] / 2 <= X_MAX
