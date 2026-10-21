"""Step 2: find candidate highlight windows from audio loudness spikes and chat velocity."""

import json
import subprocess
from pathlib import Path

import numpy as np

from .common import job_dir, log, probe, source_video, tool, write_json

SR = 16000


def loudness_per_second(video: Path) -> np.ndarray:
    """Stream mono 16 kHz PCM out of ffmpeg and return RMS loudness in dB for each second."""
    proc = subprocess.Popen(
        [tool("ffmpeg"), "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    out = []
    while True:
        buf = proc.stdout.read(SR * 2 * 60)
        if not buf:
            break
        a = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(a) // SR
        if n:
            out.append(np.sqrt(np.mean(a[: n * SR].reshape(n, SR) ** 2, axis=1)))
        if len(a) % SR:
            tail = a[n * SR:]
            out.append(np.array([np.sqrt(np.mean(tail ** 2))]))
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg audio decode failed")
    rms = np.concatenate(out) if out else np.zeros(0)
    return 20 * np.log10(rms + 1e-6)


def chat_per_second(jd: Path, seconds: int) -> np.ndarray:
    """Count YouTube live chat replay messages per second (yt-dlp live_chat.json, one JSON object per line)."""
    counts = np.zeros(seconds)
    files = list(jd.glob("*.live_chat.json"))
    if not files:
        return counts
    with open(files[0]) as f:
        for line in f:
            try:
                item = json.loads(line)["replayChatItemAction"]
                t = int(item["videoOffsetTimeMsec"]) // 1000
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
            if 0 <= t < seconds:
                counts[t] += len(item.get("actions", []))
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

    log("measuring loudness...")
    db = loudness_per_second(video)
    audio_z = smooth(robust_z(db, dc["baseline_window"]))
    chat = chat_per_second(jd, len(db))
    chat_z = smooth(robust_z(chat, dc["baseline_window"]), 5) if chat.any() else np.zeros_like(audio_z)
    score = audio_z + dc["chat_weight"] * chat_z

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
            "score": round(float(score[p]), 2),
        }
        for i, p in enumerate(sorted(peaks))
    ]
    write_json(jd / "candidates.json", candidates)
    log(f"{len(candidates)} candidate windows (chat data: {'yes' if chat.any() else 'no'})")
    return candidates
