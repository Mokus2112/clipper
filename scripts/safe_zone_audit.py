#!/usr/bin/env python3
"""Safe zone audit (card H): caption/hook boxes of rendered clips vs the conservative zone x 65-860, y 290-1250.
Usage: uv run python scripts/safe_zone_audit.py <clip dir> [...]   (dirs holding short.mp4 + captions.ass). Measures only, never writes config."""
import sys
from pathlib import Path
from clipper import qa
X0,X1,Y0,Y1=65,860,290,1250
rows=[]
for c in sys.argv[1:]:
    c=Path(c); d=qa._duration(c/"short.mp4")
    bx=qa.text_boxes(c,d,Path(__file__).resolve().parent.parent / "fonts",split=True)
    # per time: blocks sorted by y; lowest = caption, others hook
    by={}
    for t,b in bx: by.setdefault(t,[]).append(b)
    cap=[sorted(v,key=lambda b:b[1])[-1] for v in by.values()]
    hook=[b for v in by.values() for b in sorted(v,key=lambda b:b[1])[:-1]]
    def st(bs,name):
        if not bs: return f"{name}: none"
        x0=min(b[0] for b in bs);x1=max(b[2] for b in bs);y0=min(b[1] for b in bs);y1=max(b[3] for b in bs)
        n=len(bs)
        out_r=sum(b[2]>X1 for b in bs);out_b=sum(b[3]>Y1 for b in bs);out_l=sum(b[0]<X0 for b in bs);out_t=sum(b[1]<Y0 for b in bs)
        return f"{name}: n={n} x{x0}-{x1} y{y0}-{y1} | frames beyond right>{X1}:{out_r} bottom>{Y1}:{out_b} left<{X0}:{out_l} top<{Y0}:{out_t}"
    print(c.name); print("  ",st(cap,"caption")); print("  ",st(hook,"hook"))
