"""Render badi_clips avatar and YouTube banner."""
from PIL import Image, ImageDraw, ImageFont

FONT = "/home/badi/Desktop/clipper/fonts/Montserrat-ExtraBold.ttf"
OUT = "/home/badi/Desktop/clipper/brand"
BG = (14, 15, 19)
YELLOW = (255, 224, 0)
WHITE = (245, 245, 245)
GREY = (150, 152, 160)


def centered(draw, xy, text, font, fill):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (r - l) / 2 - l, xy[1] - (b - t) / 2 - t), text, font=font, fill=fill)


# Avatar: yellow field, black "BC" monogram with a small play notch; reads at 40 px.
S = 1080
av = Image.new("RGB", (S, S), YELLOW)
d = ImageDraw.Draw(av)
centered(d, (S / 2, S / 2), "BC", ImageFont.truetype(FONT, 500), BG)
av.save(f"{OUT}/avatar.png")

# Banner 2560x1440; everything inside the 1546x423 centre safe area.
W, H = 2560, 1440
bn = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(bn)
# subtle diagonal stripes for texture outside the safe area
for x in range(-H, W, 90):
    d.line([(x, H), (x + H, 0)], fill=(20, 21, 27), width=28)
d.rectangle([(W - 1546) / 2 - 40, (H - 423) / 2 - 30, (W + 1546) / 2 + 40, (H + 423) / 2 + 30], fill=BG)
title = ImageFont.truetype(FONT, 150)
tag = ImageFont.truetype(FONT, 52)
gap = 45
wa = d.textlength("BADI", font=title)
wb = d.textlength("CLIPS", font=title)
x0 = W / 2 - (wa + gap + wb) / 2
l, t, r, b = d.textbbox((0, 0), "BADI CLIPS", font=title)
y0 = H / 2 - 60 - (b - t) / 2 - t
d.text((x0, y0), "BADI", font=title, fill=WHITE)
d.text((x0 + wa + gap, y0), "CLIPS", font=title, fill=YELLOW)
centered(d, (W / 2, H / 2 + 110), "DAILY STREAM HIGHLIGHTS  ·  IRL & GAMING", tag, GREY)
bn.save(f"{OUT}/youtube-banner.png")
# small preview of the safe area for checking
bn.crop(((W - 1546) // 2, (H - 423) // 2, (W + 1546) // 2, (H + 423) // 2)).save(f"{OUT}/banner-safe-preview.png")
av.resize((120, 120)).save(f"{OUT}/avatar-small-preview.png")
