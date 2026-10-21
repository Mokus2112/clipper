"""Shared helpers: config, paths, subprocess wrappers, JSON IO."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BIN_DIRS = [ROOT / "bin", Path.home() / ".local" / "bin", Path.home() / ".bun" / "bin"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tool(name: str) -> str:
    for d in BIN_DIRS:
        p = d / name
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if not found:
        sys.exit(f"missing tool: {name}")
    return found


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir(cfg: dict, job_id: str) -> Path:
    d = path_of(cfg, "data") / "jobs" / job_id
    if not d.exists():
        sys.exit(f"unknown job: {job_id}")
    return d


def find_source(cfg: dict, name: str) -> dict:
    for s in cfg.get("sources", []):
        if s["name"] == name:
            return s
    sys.exit(f"unknown source '{name}', add it to config.yaml")


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd)[:300])
    return subprocess.run(cmd, check=True, **kw)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def source_video(jd: Path) -> Path:
    for p in sorted(jd.glob("source.*")):
        if p.suffix in (".mp4", ".mkv", ".webm", ".mov"):
            return p
    sys.exit(f"no source video in {jd}")


def probe(path: Path) -> dict:
    out = subprocess.run(
        [tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    st = info["streams"][0]
    return {"width": st["width"], "height": st["height"], "duration": float(info["format"]["duration"])}


def remove_source(jd: Path) -> int:
    """Delete the downloaded VOD (or the symlink to a local file) of a job; returns bytes freed."""
    freed = 0
    for p in list(jd.glob("source.*")) + list(jd.glob("*.live_chat.json")):
        if not p.is_symlink():
            freed += p.stat().st_size
        p.unlink()
    return freed
