import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(os.environ.get("SMOKE_RENDER") != "1",
                                reason="slow ffmpeg render; run with SMOKE_RENDER=1 before render-touching commits")


@pytest.mark.parametrize("layout", ["center", "shots"])
def test_smoke_render_finishes_in_time(layout):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_render.py"), layout],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
