"""Step 1: bring a VOD into a new job directory (download via yt-dlp or link a local file)."""

from datetime import datetime
from pathlib import Path

from .common import find_source, log, path_of, run, tool, write_json


def fetch(cfg: dict, source_name: str, url_or_path: str, section: str | None = None) -> str:
    source = find_source(cfg, source_name)
    job_id = f"{source_name}-{datetime.now():%Y%m%d-%H%M%S}"
    jd = path_of(cfg, "data") / "jobs" / job_id
    jd.mkdir(parents=True)

    local = Path(url_or_path).expanduser()
    if local.exists():
        (jd / f"source{local.suffix}").symlink_to(local.resolve())
    else:
        h = cfg["fetch"]["max_height"]
        cmd = [
            tool("yt-dlp"), "--no-playlist",
            # YouTube extraction needs a JS runtime; system node is too old, bun works.
            "--js-runtimes", f"bun:{tool('bun')}",
            "-f", f"bv*[height<={h}][fps<=60]+ba/b[height<={h}]/b",
            "--merge-output-format", "mp4",
            "-o", str(jd / "source.%(ext)s"),
        ]
        if section:
            # No --force-keyframes-at-cuts: it re-encodes the whole section, which is very slow on CPU.
            cmd += ["--download-sections", f"*{section}"]
        elif cfg["fetch"]["chat"]:
            # Live chat replay is only available for YouTube streams; yt-dlp warns and continues otherwise.
            cmd += ["--write-subs", "--sub-langs", "live_chat"]
        run(cmd + [url_or_path])

    write_json(jd / "job.json", {
        "id": job_id,
        "source": source["name"],
        "input": url_or_path,
        "section": section,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    log(f"job created: {job_id}")
    return job_id
