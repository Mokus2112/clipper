"""Step 6: upload approved clips through the YouTube Data API (OAuth installed-app flow)."""

from datetime import datetime
from pathlib import Path

from .common import log, path_of
from . import review_queue

SCOPE_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"
SCOPE_MANAGE = "https://www.googleapis.com/auth/youtube"  # videos.list/update (publish, unpublish)
SCOPES = [SCOPE_UPLOAD, SCOPE_MANAGE]


def credentials(cfg: dict, need: str = SCOPE_UPLOAD, interactive: bool = True):
    """OAuth credentials that include the `need` scope.

    The stored token keeps the scopes it was granted (an old upload-only token still uploads).
    If `need` is missing, `interactive` runs the consent flow for all SCOPES, otherwise it exits
    with a hint to run `clipper auth`.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = path_of(cfg, "secrets")
    token = secrets / "token.json"
    creds = Credentials.from_authorized_user_file(str(token)) if token.exists() else None
    if creds and need not in (creds.scopes or []):
        if not interactive:
            raise SystemExit(f"token has no {need} scope: run `uv run clipper auth` to re-consent")
        creds = None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        client = secrets / "client_secret.json"
        if not client.exists():
            raise SystemExit(f"missing {client}: create an OAuth desktop client in Google Cloud Console")
        flow = InstalledAppFlow.from_client_secrets_file(str(client), SCOPES)
        creds = flow.run_local_server(port=8765, open_browser=False,
                                      authorization_prompt_message="Open this URL in a browser on this machine: {url}")
    token.write_text(creds.to_json())
    token.chmod(0o600)
    return creds


def metadata(cfg: dict, item: dict) -> dict:
    """videos.insert body. Category is per source (IRL/podcast/news clips are not Gaming)."""
    uc = cfg["upload"]
    source = next((s for s in cfg.get("sources", []) if s["name"] == item.get("source")), {})
    lang = uc.get("language", "en")
    return {
        "snippet": {
            "title": item["title"],
            "description": item["description"],
            "tags": item["tags"],
            "categoryId": str(source.get("category_id") or uc["category_id"]),
            "defaultLanguage": lang,
            "defaultAudioLanguage": lang,
        },
        # Real footage only (Reach campaigns ban AI content). A source with AI voice/visuals must set
        # synthetic_media: true, so this is not hardcoded.
        "status": {"privacyStatus": uc["privacy"], "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": bool(source.get("synthetic_media", False))},
    }


def youtube_allowed(cfg: dict, item: dict) -> bool:
    """False if the clip's campaign lists platforms without youtube (e.g. CoD: TikTok/IG only)."""
    source = next((s for s in cfg.get("sources", []) if s["name"] == item.get("source")), {})
    return "youtube" in source.get("platforms", ["youtube"])


def upload(cfg: dict, limit: int | None = None, clip_ids: list[str] | None = None) -> list[str]:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    uc = cfg["upload"]
    budget = uc["daily_limit"] - review_queue.uploaded_today(cfg)
    if limit is not None:
        budget = min(budget, limit)
    approved = [i for i in review_queue.load(cfg) if i["status"] == "approved"]
    if clip_ids:
        not_ready = set(clip_ids) - {i["id"] for i in approved}
        if not_ready:
            raise SystemExit(f"not approved (or unknown): {', '.join(sorted(not_ready))}")
        approved = [i for i in approved if i["id"] in clip_ids]
        blocked = [i["id"] for i in approved if not youtube_allowed(cfg, i)]
        if blocked:
            raise SystemExit(f"campaign does not allow YouTube: {', '.join(blocked)}")
    approved = [i for i in approved if youtube_allowed(cfg, i)]
    if budget <= 0 or not approved:
        log(f"nothing to upload (approved: {len(approved)}, remaining daily budget: {max(budget, 0)})")
        return []

    yt = build("youtube", "v3", credentials=credentials(cfg))
    done = []
    for item in approved[:budget]:
        body = metadata(cfg, item)
        media = MediaFileUpload(item["file"], mimetype="video/mp4", chunksize=-1, resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        # Report what YouTube actually set: unverified API projects can be forced to private.
        privacy = (resp.get("status") or {}).get("privacyStatus") or f"unknown, requested {uc['privacy']}"
        review_queue.set_status(cfg, item["id"], "uploaded", video_id=resp["id"], privacy=privacy,
                                uploaded_at=datetime.now().isoformat(timespec="seconds"))
        log(f"uploaded {item['id']} -> https://youtube.com/shorts/{resp['id']} (privacy: {privacy}, requested {uc['privacy']})")
        done.append(resp["id"])
        if cfg.get("telegram", {}).get("notify"):
            from .notify import send_upload_result
            send_upload_result(cfg, item["id"], resp["id"], privacy)
    return done


def set_privacy(cfg: dict, clip_id: str, privacy: str) -> str:
    """Change an uploaded clip's privacy (publish = public, unpublish = private).

    Returns the privacy YouTube reports back after the update.
    """
    from googleapiclient.discovery import build

    item = next((i for i in review_queue.load(cfg) if i["id"] == clip_id), None)
    if not item:
        raise SystemExit(f"unknown clip: {clip_id}")
    if not item.get("video_id"):
        raise SystemExit(f"{clip_id} is not uploaded yet (status: {item['status']})")
    yt = build("youtube", "v3", credentials=credentials(cfg, need=SCOPE_MANAGE, interactive=False))
    found = yt.videos().list(part="status", id=item["video_id"]).execute().get("items", [])
    if not found:
        raise SystemExit(f"video {item['video_id']} not found on the channel")
    # Send back the full current status so update does not reset other fields (license, embeddable...).
    status = found[0]["status"]
    status["privacyStatus"] = privacy
    status.pop("publishAt", None)  # a scheduled publish time is only valid while private
    resp = yt.videos().update(part="status", body={"id": item["video_id"], "status": status}).execute()
    actual = (resp.get("status") or {}).get("privacyStatus", f"unknown, requested {privacy}")
    extra = {"privacy": actual}
    if actual == "public":
        extra["published_at"] = datetime.now().isoformat(timespec="seconds")
    review_queue.set_status(cfg, clip_id, item["status"], **extra)
    log(f"{clip_id}: privacy -> {actual} (requested {privacy}) https://youtube.com/shorts/{item['video_id']}")
    if cfg.get("telegram", {}).get("notify"):
        from .notify import send_privacy_result
        send_privacy_result(cfg, clip_id, item["video_id"], actual)
    return actual


def sync(cfg: dict) -> dict:
    """Pull the real state of every uploaded clip from YouTube into the queue.

    Badi can change privacy or delete videos in YouTube Studio; without this the dashboard
    keeps showing stale data. Deleted videos move to the archive (status rejected,
    deleted_on_youtube set) so counts match the channel.
    """
    from googleapiclient.discovery import build

    uploaded = [i for i in review_queue.load(cfg) if i["status"] == "uploaded" and i.get("video_id")]
    if not uploaded:
        return {"checked": 0, "changed": 0, "deleted": 0}
    yt = build("youtube", "v3", credentials=credentials(cfg, need=SCOPE_MANAGE, interactive=False))
    found: dict[str, str] = {}
    ids = [i["video_id"] for i in uploaded]
    for n in range(0, len(ids), 50):
        resp = yt.videos().list(part="status", id=",".join(ids[n:n + 50])).execute()
        for v in resp.get("items", []):
            found[v["id"]] = v["status"]["privacyStatus"]
    # An empty answer for many ids is more likely an auth/API problem than a mass deletion.
    if not found and len(ids) >= 3:
        raise SystemExit(f"YouTube returned none of {len(ids)} videos; not marking anything deleted")
    changed = deleted = 0
    now = datetime.now().isoformat(timespec="seconds")
    for i in uploaded:
        vid = i["video_id"]
        if vid not in found:
            review_queue.set_status(cfg, i["id"], "rejected", deleted_on_youtube=now)
            log(f"{i['id']}: video {vid} no longer on the channel, moved to archive")
            deleted += 1
        elif found[vid] != i.get("privacy"):
            extra = {"privacy": found[vid]}
            if found[vid] == "public" and not i.get("published_at"):
                extra["published_at"] = now
            review_queue.set_status(cfg, i["id"], "uploaded", **extra)
            log(f"{i['id']}: privacy {i.get('privacy')} -> {found[vid]} (changed on YouTube)")
            changed += 1
    return {"checked": len(ids), "changed": changed, "deleted": deleted}
