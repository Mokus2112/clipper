"""Safe-zone check for a rendered clip. Usage: uv run python scripts/safecheck.py <clip dir>

The checks live in src/clipper/qa.py and also run automatically at the end of every render.
"""

import sys
from pathlib import Path

from clipper.common import load_config, path_of
from clipper.qa import check_safe_zone, check_flashes, check_caption_durations

cdir = Path(sys.argv[1])
bad = (check_safe_zone(cdir, path_of(load_config(), "fonts")) + check_flashes(cdir / "short.mp4")
       + check_caption_durations(cdir / "captions.ass"))
print(f"{cdir.name}: {'OK' if not bad else f'{len(bad)} problems'}")
for b in bad[:25]:
    print("  " + b)
sys.exit(1 if bad else 0)
