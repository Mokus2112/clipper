"""Own-generated b-roll: number cards drawn in code (Pillow), overlaid on the main video.

Campaign rule (ICH): original b-roll required, no AI imagery, no stock. Everything here is drawn
from the clip's own words, licence 'own-generated'. Cards slide/fade in and out over IN_OUT frames
and sit in the middle band, so captions (burned in after the overlay) and the hook stay readable.
"""

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LICENSE = "own-generated"
HOOK_KEEP = 2.5     # hook stays on the face
TAIL_KEEP = 2.0     # closing sentence stays on the face
CARD_SEC = 1.8
MAX_INSERTS = 3
MIN_GAP = 2.0
IN_OUT = 7          # frames of motion in and out
CARD_BOX = (100, 420, 980, 920)   # x0, y0, x1, y1 inside the 1080x1920 safe zone (y <= 1250)
DARK = (18, 18, 18, 235)
ACCENT = (255, 224, 0, 255)       # brand yellow (config captions box 0x00E0FF in BGR)
WHITE = (255, 255, 255, 255)
GREY = (200, 200, 200, 255)

_NUM = re.compile(r"\d")
_UNITS = {"million", "billion", "thousand", "percent", "k", "m", "b", "stores", "dollars"}


def number_phrases(words: list[dict]) -> list[dict]:
    """Words (output timeline: w, s, e) -> numeric phrases such as '$1 million' with their time span."""
    out, i = [], 0
    while i < len(words):
        w = words[i]["w"]
        if _NUM.search(w):
            j = i
            text = w.strip(" ,.;:!?")
            # Whisper splits "$20,000." into "$20" + ",000."; glue the fragments back together.
            while j + 1 < len(words) and re.match(r"^,\d", words[j + 1]["w"]):
                text += "," + words[j + 1]["w"].lstrip(",").strip(" ,.;:!?")
                j += 1
            nxt = words[j + 1]["w"].strip(" ,.;:!?").lower() if j + 1 < len(words) else ""
            if nxt in _UNITS:
                text += " " + nxt
                j += 1
            out.append({"text": text, "s": words[i]["s"], "e": words[j]["e"]})
            i = j + 1
        else:
            i += 1
    return out


def plan_inserts(phrases: list[dict], total: float, avoid: list[tuple[float, float]] = ()) -> list[dict]:
    """Pick up to MAX_INSERTS cards. The card starts on the number, never inside the hook or the closing
    sentence, never overlaps another card, and never sits on `avoid` spans."""
    picked: list[dict] = []
    for p in sorted(phrases, key=lambda p: p["s"]):
        t0 = max(p["s"] - 0.1, HOOK_KEEP)
        t1 = t0 + CARD_SEC
        if t1 > total - TAIL_KEEP or len(picked) >= MAX_INSERTS:
            continue
        if picked and t0 < picked[-1]["t1"] + MIN_GAP:
            continue
        if any(t0 < b and t1 > a for a, b in avoid):
            continue
        picked.append({"kind": "number", "text": p["text"], "t0": round(t0, 3), "t1": round(t1, 3),
                       "license": LICENSE})
    return picked


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def card_frame(text: str, sub: str, k: int, n: int, font_file: Path, W: int = 1080, H: int = 1920) -> Image.Image:
    """Frame k of n: slides up and fades over IN_OUT frames at both ends."""
    p = min(_ease(k / IN_OUT), _ease((n - 1 - k) / IN_OUT))
    x0, y0, x1, y1 = CARD_BOX
    shift = int((1 - p) * 60)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0, y0 + shift, x1, y1 + shift), 36, fill=DARK)
    d.rectangle((x0, y0 + shift, x0 + 16, y1 + shift), fill=ACCENT)
    size = 210
    while size > 60:
        f = ImageFont.truetype(str(font_file), size)
        if d.textlength(text, font=f) <= (x1 - x0) - 120:
            break
        size -= 6
    cy = (y0 + y1) // 2 + shift
    d.text(((x0 + x1) // 2 + 8, cy - (30 if sub else 0)), text, font=f, fill=ACCENT, anchor="mm")
    if sub:
        fs = ImageFont.truetype(str(font_file), 54)
        d.text(((x0 + x1) // 2 + 8, cy + size // 2 + 20), sub, font=fs, fill=GREY, anchor="mm")
    if p < 1:
        a = layer.getchannel("A").point(lambda v: int(v * p))
        layer.putalpha(a)
    img.alpha_composite(layer)
    return img


def write_cards(inserts: list[dict], outdir: Path, fps: float, font_file: Path, W: int = 1080, H: int = 1920) -> list[dict]:
    """Write PNG sequences, one folder per insert; returns the inserts with 'dir' and 'frames'."""
    res = []
    for i, ins in enumerate(inserts):
        n = max(2 * IN_OUT + 2, round((ins["t1"] - ins["t0"]) * fps))
        d = outdir / f"broll{i}"
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            card_frame(ins["text"], ins.get("sub", ""), k, n, font_file, W, H).save(d / f"{k:04d}.png")
        res.append({**ins, "dir": str(d), "frames": n})
    return res


def overlay_graph(base: str, ins: list[dict], first_input: int, out: str, fps: float) -> str:
    """ffmpeg filter chain: overlay each PNG sequence (input index first_input+i) at its t0."""
    parts, cur = [], base
    for i, b in enumerate(ins):
        lab = f"bl{i}"
        nxt = out if i == len(ins) - 1 else f"bo{i}"
        parts.append(f"[{first_input + i}:v]format=rgba,setpts=PTS-STARTPTS+{b['t0']}/TB[{lab}]")
        parts.append(f"[{cur}][{lab}]overlay=0:0:eof_action=pass:format=auto[{nxt}]")
        cur = nxt
    return ";".join(parts)
