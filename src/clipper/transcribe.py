"""Transcription: a fast full-VOD pass for story finding, and an accurate word-level pass for the chosen clips."""

import subprocess
from pathlib import Path

import numpy as np

from .common import job_dir, log, probe, read_json, source_video, tool, write_json

SR = 16000


def load_audio(video: Path, start: float, duration: float) -> np.ndarray:
    pcm = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-ss", f"{start:.2f}", "-t", f"{duration:.2f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    # Hand Whisper a float array decoded by our ffmpeg (avoids faster-whisper's PyAV decoder).
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def model(cfg: dict, name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=cfg["whisper"]["threads"])


def transcribe_full(cfg: dict, job_id: str) -> list[dict]:
    """Segment-level transcript of the whole VOD, processed in chunks to keep memory flat. Resumable."""
    wc = cfg["whisper"]
    jd = job_dir(cfg, job_id)
    video = source_video(jd)
    duration = probe(video)["duration"]
    state = read_json(jd / "transcript.json", {"done_until": 0.0, "language": wc["language"], "segments": []})
    if state["done_until"] >= duration - 1:
        return state["segments"]

    m = model(cfg, wc["full_model"])
    chunk = wc["chunk_sec"]
    t = state["done_until"]
    while t < duration:
        length = min(chunk, duration - t)
        segments, info = m.transcribe(load_audio(video, t, length), vad_filter=True, language=state["language"])
        for seg in segments:
            text = seg.text.strip()
            if text:
                state["segments"].append({"s": round(t + seg.start, 1), "e": round(t + seg.end, 1), "text": text})
        # Lock the language after the first chunk so later chunks don't flip on noise.
        state["language"] = state["language"] or info.language
        t += length
        state["done_until"] = t
        write_json(jd / "transcript.json", state)
        log(f"transcribed {t / 60:.0f}/{duration / 60:.0f} min")
    return state["segments"]


def transcribe_clips(cfg: dict, job_id: str) -> list[dict]:
    """Accurate word timestamps for each selected clip, then snap clip boundaries to word edges."""
    wc, cc = cfg["whisper"], cfg["clip"]
    jd = job_dir(cfg, job_id)
    video = source_video(jd)
    clips = read_json(jd / "clips.json")
    if clips is None:
        raise SystemExit("run rank first")
    language = read_json(jd / "transcript.json", {}).get("language")

    m = model(cfg, wc["clip_model"])
    for clip in clips:
        lo = max(0.0, min(clip["start"], clip.get("payoff_start", clip["start"])) - 3)
        hi = max(clip["end"], clip.get("payoff_end", clip["end"])) + 3
        segments, _ = m.transcribe(load_audio(video, lo, hi - lo), word_timestamps=True, vad_filter=True,
                                   language=language)
        words = [
            {"w": w.word.strip(), "s": round(lo + w.start, 2), "e": round(lo + w.end, 2)}
            for seg in segments for w in (seg.words or []) if w.word.strip()
        ]
        clip["words"] = words
        clip["start"], clip["end"] = snap(words, clip["start"], clip["end"], cc)
        log(f"{clip['id']}: {len(words)} words, {clip['end'] - clip['start']:.1f}s")
    write_json(jd / "clips.json", clips)
    return clips


def snap(words: list[dict], start: float, end: float, cc: dict) -> tuple[float, float]:
    """Move boundaries to word edges so clips never cut mid-word, then enforce length limits."""
    inside = [w for w in words if w["s"] >= start - 0.4 and w["e"] <= end + 0.4]
    if inside:
        start, end = inside[0]["s"], inside[-1]["e"]
    start = max(0.0, start - cc["pad_start"])
    end = end + cc["pad_end"]
    if end - start < cc["min_sec"]:
        end = start + cc["min_sec"]
    if end - start > cc["max_sec"]:
        end = start + cc["max_sec"]
    return round(start, 2), round(end, 2)
