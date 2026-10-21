"""Step 1: bring a VOD into a new job directory (download via yt-dlp or link a local file)."""

import subprocess
from datetime import datetime
from pathlib import Path

from .common import find_source, log, path_of, run, source_video, tool, write_json


def ytdlp(cfg: dict) -> list[str]:
    # YouTube extraction needs a JS runtime; system node is too old, bun works.
    return [tool("yt-dlp"), "--no-playlist", "--js-runtimes", f"bun:{tool('bun')}"]


def download(cfg: dict, jd: Path, url: str) -> None:
    """Native DASH download of the full VOD (~100 MB/s here); far faster than yt-dlp's ffmpeg-based section mode."""
    h = cfg["fetch"]["max_height"]
    chat = None
    if cfg["fetch"]["chat"]:
        # Live chat replay (YouTube only) downloads in parallel; it can be slow for long streams, so it is optional.
        chat = subprocess.Popen(
            ytdlp(cfg) + ["--skip-download", "--write-subs", "--sub-langs", "live_chat",
                          "-o", str(jd / "source.%(ext)s"), url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    run(ytdlp(cfg) + [
        # Prefer H.264: AV1/VP9 decode is much slower on this CPU-only machine.
        "-f", f"bv*[height<={h}][fps<=60][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}][fps<=60]+ba/b[height<={h}]/b",
        "--merge-output-format", "mp4", "-o", str(jd / "source.%(ext)s"), url,
    ])
    if chat:
        try:
            chat.wait(timeout=cfg["fetch"]["chat_timeout"])
        except subprocess.TimeoutExpired:
            chat.kill()
            for p in jd.glob("*.live_chat.json*"):
                p.unlink()
            log("live chat download timed out, continuing without chat")


def cut_section(jd: Path, section: str) -> None:
    """Keep only START-END (HH:MM:SS) of the source, stream copy (cuts snap to keyframes)."""
    start, end = section.split("-")
    full = source_video(jd)
    part = jd / "section.mp4"
    run([tool("ffmpeg"), "-v", "error", "-y", "-ss", start, "-to", end, "-i", full, "-c", "copy",
         "-avoid_negative_ts", "make_zero", part])
    full.unlink()  # removes the download, or only the link for a local file
    part.rename(jd / "source.mp4")


def fetch(cfg: dict, source_name: str, url_or_path: str, section: str | None = None) -> str:
    source = find_source(cfg, source_name)
    job_id = f"{source_name}-{datetime.now():%Y%m%d-%H%M%S}"
    jd = path_of(cfg, "data") / "jobs" / job_id
    jd.mkdir(parents=True)

    local = Path(url_or_path).expanduser()
    if local.exists():
        (jd / f"source{local.suffix}").symlink_to(local.resolve())
    else:
        download(cfg, jd, url_or_path)
    if section:
        cut_section(jd, section)

    write_json(jd / "job.json", {
        "id": job_id,
        "source": source["name"],
        "input": url_or_path,
        "section": section,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    log(f"job created: {job_id}")
    return job_id
