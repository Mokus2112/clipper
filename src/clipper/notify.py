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
    return (
        f"Klip: {item['id']}\nPontszám: {item['score']}/10, {item['duration']} mp\n\n"
        f"CÍM:\n{item['title']}\n\nLEÍRÁS:\n{item['description']}\n\n"
        f"Miért: {item['reason']}"
    )[:1024]


def notify(cfg: dict, clip_ids: list[str] | None = None) -> int:
    token, chat_id = telegram_env(cfg)
    items = [i for i in review_queue.load(cfg)
             if (i["id"] in clip_ids if clip_ids else i["status"] == "pending" and not i.get("notified"))]
    for item in items:
        res = subprocess.run(
            [tool("curl"), "-s", "-F", f"chat_id={chat_id}", "-F", f"caption={caption(item)}",
             "-F", "supports_streaming=true", "-F", f"video=@{item['file']}",
             f"https://api.telegram.org/bot{token}/sendVideo"],
            capture_output=True, text=True, timeout=600,
        )
        if not json.loads(res.stdout or "{}").get("ok"):
            log(f"telegram send failed for {item['id']}: {res.stdout[:300]}")
            continue
        review_queue.set_status(cfg, item["id"], item["status"], notified=True)
        log(f"sent {item['id']} to telegram")
    return len(items)
