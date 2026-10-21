"""Send rendered clips to Telegram for review and manual upload (API uploads stay private until the audit passes)."""

import json
import subprocess
from pathlib import Path

from .common import log, tool
from . import review_queue


def telegram_env(cfg: dict) -> tuple[str, str]:
    env = {}
    for line in Path(cfg["telegram"]["env_file"]).expanduser().read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env["TELEGRAM_BOT_TOKEN"], env["ALLOWED_CHAT_ID"]


def caption(item: dict) -> str:
    return f"Klip: {item['id']}\nPontszám: {item['score']}/10, {item['duration']} mp\nMiért: {item['reason']}"[:1024]


def post_texts(item: dict) -> str:
    return (
        f"{item['id']}\n\n"
        f"YOUTUBE CÍM:\n{item['title']}\n\nYOUTUBE LEÍRÁS:\n{item['description']}\n\n"
        f"TIKTOK:\n{item.get('tiktok', item['title'])}\n\n"
        f"INSTAGRAM:\n{item.get('instagram', item['title'])}"
    )[:4096]


def send(token: str, args: list[str], method: str) -> bool:
    res = subprocess.run([tool("curl"), "-s", *args, f"https://api.telegram.org/bot{token}/{method}"],
                         capture_output=True, text=True, timeout=600)
    ok = json.loads(res.stdout or "{}").get("ok", False)
    if not ok:
        log(f"telegram {method} failed: {res.stdout[:300]}")
    return ok


def notify(cfg: dict, clip_ids: list[str] | None = None) -> int:
    token, chat_id = telegram_env(cfg)
    items = [i for i in review_queue.load(cfg)
             if (i["id"] in clip_ids if clip_ids else i["status"] == "pending" and not i.get("notified"))]
    for item in items:
        video = ["-F", f"chat_id={chat_id}", "-F", f"caption={caption(item)}",
                 "-F", "supports_streaming=true", "-F", f"video=@{item['file']}"]
        if not send(token, video, "sendVideo"):
            continue
        send(token, ["-F", f"chat_id={chat_id}", "-F", f"text={post_texts(item)}"], "sendMessage")
        review_queue.set_status(cfg, item["id"], item["status"], notified=True)
        log(f"sent {item['id']} to telegram")
    return len(items)
