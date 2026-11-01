"""Daily 20:00 Telegram summary: what was rendered, decided and uploaded today, what is waiting."""

from collections import Counter
from datetime import date

from . import review_queue


def build(cfg: dict) -> str:
    today = date.today().isoformat()
    items = review_queue.load(cfg)
    made = [i for i in items if i.get("created", "").startswith(today)]
    kinds = Counter(i["status"] for i in made)
    uploaded = [i for i in items if i.get("uploaded_at", "").startswith(today)]
    pending = [i for i in items if i["status"] == "pending"]
    approved = [i for i in items if i["status"] == "approved"]

    lines = [f"Clipper napi osszefoglalo ({today})"]
    if made:
        parts = ", ".join(f"{n} {s}" for s, n in kinds.most_common())
        lines.append(f"Ma renderelve: {len(made)} klip ({parts})")
    else:
        lines.append("Ma nem keszult klip.")
    for i in uploaded:
        note = i.get("privacy", "?")
        if i.get("superseded_by"):
            note += f", lecserelve: {i['superseded_by']}"
        lines.append(f"Feltoltve ({note}): {i['title']} https://youtube.com/shorts/{i.get('video_id', '?')}")
    if approved:
        lines.append(f"Jovahagyva, feltoltesre var: {len(approved)}")
    if pending:
        lines.append(f"Dontesre var ({len(pending)}):")
        lines += [f"  {i['id']}  {i['score']}/10  {i['title']}" for i in pending[:10]]
    return "\n".join(lines)


def send_summary(cfg: dict, dry: bool = False) -> str:
    text = build(cfg)
    if not dry:
        from .notify import send, telegram_env
        token, chat_id = telegram_env(cfg)
        send(token, ["--data-urlencode", f"chat_id={chat_id}", "--data-urlencode", f"text={text}"], "sendMessage")
    return text
