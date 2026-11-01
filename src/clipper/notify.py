"""Send rendered clips to Telegram for review, with copy-paste post texts per platform."""

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
    music = f", zene: {item['music']:.0%}" if item.get("music") is not None else ""
    return f"Klip: {item['id']}\nPontszám: {item['score']}/10, {item['duration']} mp{music}\nMiért: {item['reason']}"[:1024]


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


def review_keyboard(pending_ids: list[str]) -> str:
    """ReplyKeyboardMarkup with ✅/❌ buttons for every pending clip."""
    rows = [[f"✅ ok {cid}", f"❌ no {cid}"] for cid in pending_ids]
    return json.dumps({"keyboard": rows, "resize_keyboard": True, "one_time_keyboard": False})


def send_upload_result(cfg: dict, clip_id: str, video_id: str, privacy: str) -> None:
    """Notify Badi after a successful upload."""
    token, chat_id = telegram_env(cfg)
    is_reach = any(
        s.get("campaign_url", "")
        for s in cfg.get("sources", [])
        if s["name"] == clip_id.split("-")[0]
    )
    text = f"Feltoltve: https://youtube.com/shorts/{video_id} ({privacy})"
    if is_reach or any(
        "reach" in s["name"] for s in cfg.get("sources", [])
        if s["name"] in clip_id
    ):
        text += "\n\nReach kampany: 30 percen belul kuldsd be a Content Rewardsba!"
    all_pending = [i["id"] for i in review_queue.load(cfg) if i["status"] == "pending"]
    keyboard_json = review_keyboard(all_pending) if all_pending else json.dumps({"remove_keyboard": True})
    send(token, ["-d", f"chat_id={chat_id}", "-d", f"text={text}",
                 "-d", f"reply_markup={keyboard_json}"], "sendMessage")


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

    # After all clips are sent, refresh the reply keyboard with all pending clip buttons.
    all_pending = [i["id"] for i in review_queue.load(cfg) if i["status"] == "pending"]
    if all_pending:
        keyboard_json = review_keyboard(all_pending)
        send(token, ["-d", f"chat_id={chat_id}", "-d", "text=Jovahagyas:",
                     "-d", f"reply_markup={keyboard_json}"], "sendMessage")
    return len(items)
