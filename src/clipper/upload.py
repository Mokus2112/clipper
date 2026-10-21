"""Step 6: upload approved clips through the YouTube Data API (OAuth installed-app flow)."""

from datetime import datetime
from pathlib import Path

from .common import log, path_of
from . import review_queue

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def credentials(cfg: dict):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = path_of(cfg, "secrets")
    token = secrets / "token.json"
    creds = Credentials.from_authorized_user_file(str(token), SCOPES) if token.exists() else None
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


def upload(cfg: dict, limit: int | None = None) -> list[str]:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    uc = cfg["upload"]
    budget = uc["daily_limit"] - review_queue.uploaded_today(cfg)
    if limit is not None:
        budget = min(budget, limit)
    approved = [i for i in review_queue.load(cfg) if i["status"] == "approved"]
    if budget <= 0 or not approved:
        log(f"nothing to upload (approved: {len(approved)}, remaining daily budget: {max(budget, 0)})")
        return []

    yt = build("youtube", "v3", credentials=credentials(cfg))
    done = []
    for item in approved[:budget]:
        body = {
            "snippet": {
                "title": item["title"],
                "description": item["description"],
                "tags": item["tags"],
                "categoryId": uc["category_id"],
            },
            "status": {"privacyStatus": uc["privacy"], "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(item["file"], mimetype="video/mp4", chunksize=-1, resumable=True)
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        review_queue.set_status(cfg, item["id"], "uploaded", video_id=resp["id"],
                                uploaded_at=datetime.now().isoformat(timespec="seconds"))
        log(f"uploaded {item['id']} -> https://youtube.com/shorts/{resp['id']} ({uc['privacy']})")
        done.append(resp["id"])
    return done
