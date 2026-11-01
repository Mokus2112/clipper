"""Human review queue: every rendered clip waits here until it is approved or rejected."""

from datetime import date, datetime
from pathlib import Path

from .common import log, path_of, read_json, write_json


def _path(cfg: dict) -> Path:
    return path_of(cfg, "data") / "queue.json"


def load(cfg: dict) -> list[dict]:
    return read_json(_path(cfg), [])


def save(cfg: dict, items: list[dict]) -> None:
    write_json(_path(cfg), items)


def _campaign_title(title: str, source: dict) -> str:
    """Apply campaign title rules: a mandatory prefix (e.g. "Ad: Scape -") and/or a name that must appear."""
    mention = source.get("title_mention", "")
    if mention and mention.lower() not in title.lower():
        title = f"{mention}: {title}"
    prefix = source.get("title_prefix", "")
    if prefix and not title.startswith(prefix):
        title = f"{prefix} {title}"
    return title[:100]  # YouTube title limit


def add(cfg: dict, clip: dict, source: dict, file: Path) -> None:
    existing = next((i for i in load(cfg) if i["id"] == clip["id"]), None)
    # A re-render must never wipe an approved or uploaded entry (its video_id/privacy would be lost
    # and the clip would show as pending again). Re-render under a new id (e.g. -v2) instead.
    if existing and existing["status"] in ("approved", "uploaded"):
        log(f"{clip['id']}: already {existing['status']}, not re-queued; use a new clip id for a new version")
        return
    items = [i for i in load(cfg) if i["id"] != clip["id"]]
    tags = source.get("tags", [])
    clip_tags = clip.get("hashtags") or [t.replace(" ", "") for t in tags]
    hashtags = " ".join(f"#{t}" for t in clip_tags[:3])
    footer = cfg["upload"].get("description_footer", "")
    # Campaign briefs often require a tag, mention or link in every post.
    required = source.get("required_text", "")
    title = _campaign_title(clip["title"], source)
    social = " ".join(x for x in [clip.get("caption") or title, " ".join(f"#{t}" for t in clip_tags), required] if x)
    items.append({
        "id": clip["id"],
        "file": str(file),
        "title": title,
        "description": "\n\n".join(x for x in [source.get("credit", ""), required, hashtags, footer] if x),
        "tags": list(dict.fromkeys(tags + clip_tags)),
        "tiktok": social,
        "instagram": "\n\n".join(x for x in [social, source.get("credit", "")] if x),
        "score": clip["score"],
        "music": clip.get("music"),
        "reason": clip["reason"],
        "duration": round(clip["end"] - clip["start"], 1),
        "source": source["name"],
        # Practice sources are for pipeline evaluation only: never offered for approval or upload.
        "status": "practice" if "practice" in source["name"] else "pending",
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    save(cfg, items)
    write_posts(items[-1], source)


PLATFORM_NAMES = {"youtube": "YouTube Shorts", "tiktok": "TikTok", "instagram": "Instagram Reels", "x": "X"}


def write_posts(item: dict, source: dict) -> Path:
    """posts.md next to the clip: copy-paste texts for every platform the campaign allows."""
    plats = source.get("platforms", ["youtube"])
    hashtags = " ".join(f"#{t}" for t in item["tags"][:5])
    required = source.get("required_text", "")
    credit = source.get("credit", "")
    lines = [f"# {item['id']}", "",
             f"Kampany: {source['name']}  |  Engedelyezett platformok: "
             + ", ".join(PLATFORM_NAMES.get(p, p) for p in plats), ""]
    if "youtube" in plats:
        lines += ["## YouTube Shorts", "", "Cim:", "", item["title"], "", "Leiras:", "", item["description"], ""]
    if "tiktok" in plats:
        lines += ["## TikTok", "", "Caption:", "", item["tiktok"], ""]
    if "instagram" in plats:
        lines += ["## Instagram Reels", "", "Caption:", "", item["instagram"], ""]
    if "x" in plats:
        x_text = " ".join(t for t in [item["title"], required, credit] if t)
        lines += ["## X", "", x_text[:280], ""]
    lines += ["## Megjegyzes", "",
              "Kampanyhoz bekuldes: Whop / Content Rewards, a poszt linkjevel"
              + (" (Reach: 30 percen belul)." if source["name"].startswith("reach") else "."), ""]
    out = Path(item["file"]).with_name("posts.md")
    out.write_text("\n".join(lines))
    return out


def set_status(cfg: dict, clip_id: str, status: str, **extra) -> dict:
    items = load(cfg)
    for i in items:
        if i["id"] == clip_id:
            if i["status"] == "practice" and status in ("approved", "uploaded"):
                raise SystemExit(f"{clip_id} is a practice clip and can never be approved or uploaded")
            i["status"] = status
            i.update(extra)
            save(cfg, items)
            return i
    raise SystemExit(f"unknown clip: {clip_id}")


def uploaded_today(cfg: dict) -> int:
    today = date.today().isoformat()
    return sum(1 for i in load(cfg) if i.get("uploaded_at", "").startswith(today))
