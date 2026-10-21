"""Human review queue: every rendered clip waits here until it is approved or rejected."""

from datetime import date, datetime
from pathlib import Path

from .common import path_of, read_json, write_json


def _path(cfg: dict) -> Path:
    return path_of(cfg, "data") / "queue.json"


def load(cfg: dict) -> list[dict]:
    return read_json(_path(cfg), [])


def save(cfg: dict, items: list[dict]) -> None:
    write_json(_path(cfg), items)


def add(cfg: dict, clip: dict, source: dict, file: Path) -> None:
    items = [i for i in load(cfg) if i["id"] != clip["id"]]
    tags = source.get("tags", [])
    clip_tags = clip.get("hashtags") or [t.replace(" ", "") for t in tags]
    hashtags = " ".join(f"#{t}" for t in clip_tags[:3])
    footer = cfg["upload"].get("description_footer", "")
    # Campaign briefs often require a tag, mention or link in every post.
    required = source.get("required_text", "")
    social = " ".join(x for x in [clip.get("caption") or clip["title"], " ".join(f"#{t}" for t in clip_tags), required] if x)
    items.append({
        "id": clip["id"],
        "file": str(file),
        "title": clip["title"],
        "description": "\n\n".join(x for x in [source.get("credit", ""), required, hashtags, footer] if x),
        "tags": list(dict.fromkeys(tags + clip_tags)),
        "tiktok": social,
        "instagram": "\n\n".join(x for x in [social, source.get("credit", "")] if x),
        "score": clip["score"],
        "reason": clip["reason"],
        "duration": round(clip["end"] - clip["start"], 1),
        "status": "pending",
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    save(cfg, items)


def set_status(cfg: dict, clip_id: str, status: str, **extra) -> dict:
    items = load(cfg)
    for i in items:
        if i["id"] == clip_id:
            i["status"] = status
            i.update(extra)
            save(cfg, items)
            return i
    raise SystemExit(f"unknown clip: {clip_id}")


def uploaded_today(cfg: dict) -> int:
    today = date.today().isoformat()
    return sum(1 for i in load(cfg) if i.get("uploaded_at", "").startswith(today))
