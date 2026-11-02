"""Text width from the real font file (what libass will draw), so layout checks never guess.

libass scales a font so that `size` px = usWinAscent + usWinDescent font units; the advance
widths in hmtx then give the line width. Kerning is ignored (it only makes lines narrower).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _metrics(path: str) -> tuple[dict, dict, float, float]:
    from fontTools.ttLib import TTFont

    f = TTFont(path, lazy=True)
    os2 = f["OS/2"]
    cmap = f.getBestCmap()
    adv = {cp: f["hmtx"][g][0] for cp, g in cmap.items()}
    em = float(os2.usWinAscent + os2.usWinDescent)
    cap = float(getattr(os2, "sCapHeight", 0) or 0.7 * f["head"].unitsPerEm)
    return adv, {"asc": os2.usWinAscent, "desc": os2.usWinDescent}, em, cap


def font_file(fonts_dir: Path, family: str) -> Path:
    """The font file for a family name: 'Bebas Neue' -> BebasNeue-*.ttf/otf."""
    stem = family.replace(" ", "")
    hits = sorted(p for p in Path(fonts_dir).iterdir()
                  if p.suffix.lower() in (".ttf", ".otf") and p.stem.replace("-", "").startswith(stem))
    if not hits:
        raise FileNotFoundError(f"font {family!r} not in {fonts_dir}")
    # 'Montserrat ExtraBold' -> Montserrat-ExtraBold.ttf; plain family -> Regular first
    for h in hits:
        if h.stem.replace("-", "") == stem:
            return h
    return next((h for h in hits if h.stem.endswith("Regular")), hits[0])


def text_width(font: Path, text: str, size: float, spacing: float = 0.0) -> float:
    """Advance width of one line in px (no outline/shadow)."""
    adv, _, em, _ = _metrics(str(font))
    space = adv.get(32, em * 0.25)
    return sum(adv.get(ord(ch), space) for ch in text) * size / em + spacing * len(text)


def drawn_width(font: Path, text: str, size: float, spacing: float = 0.0, outline: float = 0.0,
                shadow: float = 0.0) -> float:
    """Width including the outline on both sides and the shadow offset: the pixels actually lit."""
    return text_width(font, text, size, spacing) + 2 * outline + shadow


def cap_height(font: Path, size: float) -> float:
    """Height of a capital letter in px at this size."""
    _, _, em, cap = _metrics(str(font))
    return cap * size / em


def cap_center_offset(font: Path, size: float) -> float:
    """px from the middle of the line box (where \\an4/\\an5 put y) down to the middle of the capitals.

    Negative: the capitals sit above the line box middle (the box includes the descender space).
    """
    _, m, em, cap = _metrics(str(font))
    # line box: top = baseline - asc, bottom = baseline + desc; caps: baseline - cap .. baseline
    line_mid = (m["desc"] - m["asc"]) / 2
    cap_mid = -cap / 2
    return (cap_mid - line_mid) * size / em
