#!/usr/bin/env python3
"""Smoke render per layout: a ~6 s slice of a fixture clip must render in LIMIT seconds (default 60).

Run before every commit that touches render.py / the ffmpeg graph:
    uv run python scripts/smoke_render.py            # all layouts
    uv run python scripts/smoke_render.py center     # one layout
Fails (exit 1) when a render does not produce short.mp4 inside the limit, i.e. when ffmpeg hangs
(the b86d356 shots-layout deadlock sat at 100% CPU for 15+ minutes). Everything runs in a temp data dir
with absolute paths; the real data/jobs is never touched.

Fixtures: SMOKE_FIXTURES (dir holding data/jobs/<job>/ with source.mp4, job.json, clips.json).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(os.environ.get("SMOKE_FIXTURES", "/Users/badi/Codee/agents/clip-fejleszto/work/E"))
LIMIT = float(os.environ.get("SMOKE_LIMIT", "60"))
SLICE = 6.0
# layout label -> (job dir, clip id suffix). center = single speaker crop, shots = per-shot layouts (face/duo).
CASES = {
    "center": ("alliance-secure-ai-20261102-182604", "c1"),
    "shots": ("clipfarm-solved-20261102-091333", "c2"),
}


def make_job(tmp: Path, job: str, suffix: str) -> tuple[str, str]:
    src = FIX / "data" / "jobs" / job
    clips = json.loads((src / "clips.json").read_text())
    clip = next(c for c in clips if c["id"].endswith(suffix))
    t0 = clip["start"]
    shift = t0 - 1.0
    jd = tmp / "data" / "jobs" / job
    jd.mkdir(parents=True)
    shutil.copy(src / "job.json", jd / "job.json")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{shift:.3f}", "-t", f"{SLICE + 3:.3f}", "-i", str(src / "source.mp4"),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-c:a", "aac", str(jd / "source.mp4")], check=True)
    end = t0 + SLICE
    words = [{**w, "s": w["s"] - shift, "e": w["e"] - shift}
             for w in clip.get("words", []) if t0 - 0.5 <= w["s"] <= end + 0.5]
    small = {k: v for k, v in clip.items() if k in ("id", "title", "story", "caption", "hashtags", "hook", "score", "name_fixes")}
    small.update(start=1.0, end=SLICE + 1.0, words=words, payoff_start=None, payoff_end=None, quotes=None,
                 words_whisper=clip.get("words_whisper") and words)
    (jd / "clips.json").write_text(json.dumps([small]))
    return job, small["id"]


def smoke(label: str) -> tuple[bool, float, str]:
    job, suffix = CASES[label]
    with tempfile.TemporaryDirectory(prefix=f"smoke-{label}-") as t:
        tmp = Path(t)
        job, cid = make_job(tmp, job, suffix)
        code = ("import os,sys;from clipper.common import load_config;from clipper.render import render;"
                "cfg=load_config();cfg['paths']['data']=sys.argv[1];os.environ['CLIPPER_ONLY']=sys.argv[3];"
                "render(cfg,sys.argv[2])")
        start = time.time()
        p = subprocess.Popen([sys.executable, "-I", "-c", code, str(tmp / "data"), job, cid], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=LIMIT)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)  # the python process and its ffmpeg children
            return False, time.time() - start, f"TIMEOUT after {LIMIT:.0f}s (ffmpeg hang?)"
        took = time.time() - start
        ok = (tmp / "data" / "jobs" / job / "clips" / cid / "short.mp4").exists()
        return ok, took, "" if ok else out[-600:]


def main(argv: list[str]) -> int:
    labels = argv[1:] or list(CASES)
    bad = 0
    for lb in labels:
        ok, took, msg = smoke(lb)
        print(f"{lb:7s} {'OK  ' if ok else 'FAIL'} {took:5.1f}s {msg}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
