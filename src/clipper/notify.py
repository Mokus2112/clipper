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


def review_keyboard(pending_ids: list[str], private_ids: list[str] = ()) -> str:
    """ReplyKeyboardMarkup: ✅/❌ for every pending clip, 🌍 for every uploaded but not public clip."""
    rows = [[f"✅ ok {cid}", f"❌ no {cid}"] for cid in pending_ids]
    rows += [[f"🌍 publikus {cid}"] for cid in private_ids]
    return json.dumps({"keyboard": rows, "resize_keyboard": True, "one_time_keyboard": False})


def current_keyboard(cfg: dict) -> str:
    """Keyboard for the current queue state, or remove_keyboard when nothing needs a decision."""
    items = review_queue.load(cfg)
    pending = [i["id"] for i in items if i["status"] == "pending"]
    # superseded: an uploaded clip replaced by a better render; never offer it for publishing.
    private = [i["id"] for i in items if i["status"] == "uploaded" and i.get("privacy") != "public"
               and not i.get("superseded_by")]
    if not pending and not private:
        return json.dumps({"remove_keyboard": True})
    return review_keyboard(pending, private)


def _is_reach(cfg: dict, clip_id: str) -> bool:
    item = next((i for i in review_queue.load(cfg) if i["id"] == clip_id), {})
    source = item.get("source") or next(
        (s["name"] for s in cfg.get("sources", []) if clip_id.startswith(s["name"] + "-")), "")
    return source.startswith("reach")


def _send_text(cfg: dict, text: str) -> None:
    token, chat_id = telegram_env(cfg)
    send(token, ["--data-urlencode", f"chat_id={chat_id}", "--data-urlencode", f"text={text}",
                 "--data-urlencode", f"reply_markup={current_keyboard(cfg)}"], "sendMessage")


def send_upload_result(cfg: dict, clip_id: str, video_id: str, privacy: str) -> None:
    """After upload: link + the privacy YouTube set. Publishing is Badi's manual 🌍 button."""
    text = f"Feltoltve: https://youtube.com/shorts/{video_id}\nYouTube privacy: {privacy}"
    if privacy != "public":
        text += f"\nPublikusra a 🌍 publikus {clip_id} gombbal."
    elif _is_reach(cfg, clip_id):
        text += "\n\nReach kampany: 30 percen belul kuldsd be a Content Rewardsba!"
    _send_text(cfg, text)


def send_privacy_result(cfg: dict, clip_id: str, video_id: str, privacy: str) -> None:
    """After publish/unpublish: the privacy YouTube reports, plus the Reach 30-minute reminder."""
    text = f"{clip_id}\nhttps://youtube.com/shorts/{video_id}\nYouTube privacy: {privacy}"
    if privacy == "public" and _is_reach(cfg, clip_id):
        text += "\n\nReach kampany: 30 percen belul kuldsd be a Content Rewardsba!"
    _send_text(cfg, text)


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

    # After all clips are sent, refresh the reply keyboard (pending ✅/❌ + uploaded 🌍).
    if any(i["status"] == "pending" for i in review_queue.load(cfg)):
        send(token, ["-d", f"chat_id={chat_id}", "-d", "text=Jovahagyas:",
                     "--data-urlencode", f"reply_markup={current_keyboard(cfg)}"], "sendMessage")
    return len(items)
