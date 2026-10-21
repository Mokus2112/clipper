"""Command line entry point: `uv run clipper <command>`."""

import argparse
import json

from .common import job_dir, load_config, path_of, remove_source
from . import review_queue


def main() -> None:
    p = argparse.ArgumentParser(prog="clipper", description="Streamer VOD -> YouTube Shorts pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("run", "fetch"):
        s = sub.add_parser(name, help="full pipeline" if name == "run" else "download/link a VOD into a new job")
        s.add_argument("source", help="source name from config.yaml")
        s.add_argument("input", help="VOD URL or local file path")
        s.add_argument("--section", help="only keep a range, e.g. 01:10:00-01:40:00")
        s.add_argument("--no-notify", action="store_true", help="don't send the clips to Telegram")
    for name in ("detect", "transcribe", "rank", "captions", "render"):
        s = sub.add_parser(name, help=f"run the {name} step on a job")
        s.add_argument("job")
        if name == "rank":
            s.add_argument("--min-score", type=int, help="override rank.min_score")
    s = sub.add_parser("review", help="list clips in the review queue")
    s.add_argument("--all", action="store_true")
    s.add_argument("--json", action="store_true")
    for name in ("approve", "reject"):
        sub.add_parser(name).add_argument("clip_ids", nargs="+")
    s = sub.add_parser("upload", help="upload approved clips (respects daily_limit)")
    s.add_argument("--limit", type=int)
    sub.add_parser("auth", help="run the YouTube OAuth flow and store the token")
    s = sub.add_parser("notify", help="send pending clips to Telegram for review")
    s.add_argument("clip_ids", nargs="*")
    s = sub.add_parser("cleanup", help="delete source VODs, keep transcripts and rendered clips")
    s.add_argument("jobs", nargs="*", help="job ids (default: all jobs)")

    a = p.parse_args()
    cfg = load_config()

    if a.cmd in ("run", "fetch"):
        from .fetch import fetch
        job = fetch(cfg, a.source, a.input, a.section)
        if a.cmd == "run":
            from .detect import detect
            from .transcribe import transcribe_clips, transcribe_full
            from .rank import rank
            from .render import render
            detect(cfg, job)
            transcribe_full(cfg, job)
            if rank(cfg, job):
                transcribe_clips(cfg, job)
                render(cfg, job)
                if cfg["telegram"]["notify"] and not a.no_notify:
                    from .notify import notify
                    notify(cfg)
            if not cfg["fetch"]["keep_source"]:
                remove_source(job_dir(cfg, job))
        print(job)
    elif a.cmd == "detect":
        from .detect import detect
        detect(cfg, a.job)
    elif a.cmd == "transcribe":
        from .transcribe import transcribe_full
        transcribe_full(cfg, a.job)
    elif a.cmd == "rank":
        from .rank import rank
        rank(cfg, a.job, a.min_score)
    elif a.cmd == "captions":
        from .transcribe import transcribe_clips
        transcribe_clips(cfg, a.job)
    elif a.cmd == "render":
        from .render import render
        render(cfg, a.job)
    elif a.cmd == "review":
        items = [i for i in review_queue.load(cfg) if a.all or i["status"] == "pending"]
        if a.json:
            print(json.dumps(items, indent=2, ensure_ascii=False))
        for i in [] if a.json else items:
            print(f"[{i['status']}] {i['id']}  score {i['score']}  {i['duration']}s\n  {i['title']}\n  {i['file']}\n")
    elif a.cmd in ("approve", "reject"):
        for cid in a.clip_ids:
            status = "approved" if a.cmd == "approve" else "rejected"
            review_queue.set_status(cfg, cid, status)
            print(f"{cid}: {status}")
    elif a.cmd == "upload":
        from .upload import upload
        upload(cfg, a.limit)
    elif a.cmd == "notify":
        from .notify import notify
        print(f"sent {notify(cfg, a.clip_ids or None)} clips")
    elif a.cmd == "cleanup":
        jobs = a.jobs or [d.name for d in (path_of(cfg, "data") / "jobs").iterdir() if d.is_dir()]
        freed = sum(remove_source(job_dir(cfg, j)) for j in jobs)
        print(f"freed {freed / 1e9:.2f} GB from {len(jobs)} jobs")
    elif a.cmd == "auth":
        from .upload import credentials
        credentials(cfg)
        print("token stored")


if __name__ == "__main__":
    main()
