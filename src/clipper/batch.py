"""Overnight practice batch: fetch the latest VOD for each practice source and run the full pipeline."""

import subprocess
from datetime import datetime

from .common import find_source, load_config, log, tool
from . import review_queue


def latest_vod_url(channel_url: str) -> str | None:
    """Return the URL of the most recent upload on a YouTube channel."""
    res = subprocess.run(
        [tool("yt-dlp"), "--no-playlist", "--playlist-items", "1",
         "--get-id", "--flat-playlist", channel_url + "/streams"],
        capture_output=True, text=True, timeout=60,
    )
    video_id = res.stdout.strip().splitlines()[0] if res.stdout.strip() else None
    if not video_id:
        # Fall back to /videos (VODs) if /streams returned nothing
        res = subprocess.run(
            [tool("yt-dlp"), "--no-playlist", "--playlist-items", "1",
             "--get-id", "--flat-playlist", channel_url + "/videos"],
            capture_output=True, text=True, timeout=60,
        )
        video_id = res.stdout.strip().splitlines()[0] if res.stdout.strip() else None
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else None


def run_batch(cfg: dict, source_names: list[str], notify: bool = True) -> dict:
    """Run the full pipeline for each source. Returns a summary dict."""
    results = {}
    for name in source_names:
        src = find_source(cfg, name)
        if not src:
            log(f"batch: unknown source {name}, skipping")
            results[name] = {"status": "error", "reason": "unknown source"}
            continue
        channel_url = src.get("channel_url", "")
        if not channel_url:
            log(f"batch: {name} has no channel_url, skipping")
            results[name] = {"status": "skip", "reason": "no channel_url"}
            continue
        log(f"batch: fetching latest VOD for {name}")
        url = latest_vod_url(channel_url)
        if not url:
            log(f"batch: no VOD found for {name}")
            results[name] = {"status": "skip", "reason": "no VOD found"}
            continue
        log(f"batch: running pipeline for {name} -> {url}")
        from .fetch import fetch
        from .detect import detect
        from .transcribe import transcribe_clips, transcribe_full
        from .rank import rank
        from .render import render
        from .common import job_dir, remove_source
        try:
            job = fetch(cfg, name, url, section=None)
            detect(cfg, job)
            transcribe_full(cfg, job)
            if rank(cfg, job):
                transcribe_clips(cfg, job)
                render(cfg, job)
                clips = [i for i in review_queue.load(cfg) if i.get("source") == name and i["status"] == "pending"]
                results[name] = {"status": "ok", "clips": len(clips), "url": url}
            else:
                results[name] = {"status": "no_clips", "url": url}
            if not cfg["fetch"]["keep_source"]:
                remove_source(job_dir(cfg, job))
        except Exception as exc:
            log(f"batch: pipeline failed for {name}: {exc}")
            results[name] = {"status": "error", "reason": str(exc)}
    if notify and cfg.get("telegram", {}).get("notify"):
        _send_summary(cfg, source_names, results)
    return results


def _send_summary(cfg: dict, sources: list[str], results: dict) -> None:
    from .notify import send, telegram_env, review_keyboard
    token, chat_id = telegram_env(cfg)
    lines = [f"Overnight batch ({datetime.now().strftime('%Y-%m-%d')})"]
    for name in sources:
        r = results.get(name, {})
        status = r.get("status", "?")
        if status == "ok":
            lines.append(f"  {name}: {r['clips']} klip jovahagyasra var")
        elif status == "no_clips":
            lines.append(f"  {name}: lefutott, nincs erdekes klip")
        elif status == "skip":
            lines.append(f"  {name}: athagyva ({r.get('reason','')})")
        else:
            lines.append(f"  {name}: hiba -- {r.get('reason','')}")
    pending = [i["id"] for i in review_queue.load(cfg) if i["status"] == "pending"]
    keyboard_json = review_keyboard(pending) if pending else None
    import json
    args = ["-d", f"chat_id={chat_id}", "-d", f"text={chr(10).join(lines)}"]
    if keyboard_json:
        args += ["-d", f"reply_markup={keyboard_json}"]
    send(token, args, "sendMessage")
