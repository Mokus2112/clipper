"""Step 2: find candidate highlight moments from loudness spikes, reaction sounds (YAMNet) and chat velocity."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .common import job_dir, log, probe, read_json, source_video, tool, write_json

SR = 16000


def audio_features(video: Path, threads: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One decode pass over the audio: per-second loudness (dB), YAMNet reaction score and music score."""
    from .audio_ai import Yamnet

    yam = Yamnet(threads)
    proc = subprocess.Popen(
        [tool("ffmpeg"), "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    db, reaction, music = [], [], []
    while True:
        buf = proc.stdout.read(SR * 2 * 60)
        if not buf:
            break
        a = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        for i in range(0, len(a), SR):
            sec = a[i:i + SR]
            db.append(20 * np.log10(np.sqrt(np.mean(sec ** 2)) + 1e-6))
            r, m = yam.tag(sec)
            reaction.append(r)
            music.append(m)
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg audio decode failed")
    return np.array(db), np.array(reaction), np.array(music)


def hms(t: str) -> int:
    secs = 0
    for part in t.split(":"):
        secs = secs * 60 + int(float(part))
    return secs


# Chat reactions that usually mean "something just happened"; such messages count double.
HYPE = re.compile(r"\b(LUL|LULW|KEKW|OMEGALUL|ICANT|LMAO+|LMFAO|W+|L+|POG|POGGERS|CLIP ?IT|\+2|NO WAY|WTF)\b", re.I)


def chat_messages(jd: Path) -> list[tuple[int, str]]:
    """(offset_seconds, text) from a YouTube live_chat.json (yt-dlp) or a Twitch chat JSON (TwitchDownloaderCLI)."""
    out = []
    for path in jd.glob("*.live_chat.json"):
        with open(path) as f:
            for line in f:
                try:
                    item = json.loads(line)["replayChatItemAction"]
                    t = int(item["videoOffsetTimeMsec"]) // 1000
                    for action in item.get("actions", []):
                        runs = action["addChatItemAction"]["item"]["liveChatTextMessageRenderer"]["message"]["runs"]
                        out.append((t, "".join(r.get("text", "") for r in runs)))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
    twitch = jd / "twitch_chat.json"
    if twitch.exists():
        for c in json.loads(twitch.read_text()).get("comments", []):
            out.append((int(c["content_offset_seconds"]), c.get("message", {}).get("body", "")))
    return out


def chat_per_second(jd: Path, seconds: int, offset: int = 0) -> np.ndarray:
    """Weighted chat messages per second. offset: start of a cut section (chat times refer to the full VOD)."""
    counts = np.zeros(seconds)
    for t, text in chat_messages(jd):
        t -= offset
        if 0 <= t < seconds:
            counts[t] += 2.0 if HYPE.search(text) else 1.0
    return counts


def robust_z(x: np.ndarray, window: int) -> np.ndarray:
    """Deviation from a rolling median, scaled by the global MAD, so level changes across a stream don't dominate."""
    if len(x) == 0:
        return x
    window = min(window, len(x) // 2 * 2 + 1)
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    baseline = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    dev = x - baseline
    mad = np.median(np.abs(dev - np.median(dev))) * 1.4826
    return dev / (mad if mad > 1e-9 else 1.0)


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same") if len(x) >= k else x


def detect(cfg: dict, job_id: str) -> list[dict]:
    dc = cfg["detect"]
    jd = job_dir(cfg, job_id)
    video = source_video(jd)
    duration = probe(video)["duration"]

    log("measuring loudness and tagging audio...")
    db, reaction, music = audio_features(video)
    np.savez_compressed(jd / "audio_tags.npz", db=db, reaction=reaction, music=music)
    audio_z = smooth(robust_z(db, dc["baseline_window"]))
    # Laughter/screaming/cheering probability; already 0..1, so scale it instead of z-scoring a mostly-zero signal.
    reaction_s = smooth(reaction, 3) * dc["reaction_scale"]
    section = (read_json(jd / "job.json", {}) or {}).get("section")
    chat = chat_per_second(jd, len(db), hms(section.split("-")[0]) if section else 0)
    chat_z = smooth(robust_z(chat, dc["baseline_window"]), 5) if chat.any() else np.zeros_like(audio_z)
    score = audio_z + dc["chat_weight"] * chat_z + dc["reaction_weight"] * reaction_s

    edge = dc["skip_edges"] if duration > 4 * dc["skip_edges"] else 0
    order = np.argsort(score)[::-1]
    peaks: list[int] = []
    for t in order:
        if t < edge or t > len(score) - edge:
            continue
        if all(abs(t - p) >= dc["min_gap"] for p in peaks):
            peaks.append(int(t))
        if len(peaks) >= dc["candidates"]:
            break

    candidates = [
        {
            "id": i,
            "peak": p,
            "start": max(0.0, p - dc["pre_roll"]),
            "end": min(duration, p + dc["post_roll"]),
            "audio_z": round(float(audio_z[p]), 2),
            "chat_z": round(float(chat_z[p]), 2),
            "reaction": round(float(reaction[p]), 2),
            "score": round(float(score[p]), 2),
        }
        for i, p in enumerate(sorted(peaks))
    ]
    write_json(jd / "candidates.json", candidates)
    log(f"{len(candidates)} candidate windows (chat data: {'yes' if chat.any() else 'no'})")
    return candidates
