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
    from .anchors import find_quote
    for clip in clips:
        ps = clip.get("payoff_start")
        lo = max(0.0, min(clip["start"], ps if ps is not None else clip["start"]) - 5)
        hi = max(clip["end"], clip.get("payoff_end") or clip["end"]) + 5
        # The prompt nudges Whisper to keep spoken fillers ("like", "you know", "it's") verbatim;
        # captions that skip words people can hear look machine-made.
        segments, _ = m.transcribe(load_audio(video, lo, hi - lo), word_timestamps=True, vad_filter=True,
                                   initial_prompt="Um, so, like, you know, I mean, it's, uh, yeah. Verbatim.",
                                   language=language)
        words = [
            {"w": w.word.strip(), "s": round(lo + w.start, 2), "e": round(lo + w.end, 2)}
            for seg in segments for w in (seg.words or []) if w.word.strip()
        ]
        clip["words"] = words
        quotes = clip.get("quotes")
        if quotes:
            # Exact cut points from the ranker's quotes in the word-level transcript.
            qs = find_quote(words, quotes.get("start_quote", ""), near=clip["start"])
            qe = find_quote(words, quotes.get("end_quote", ""), near=clip["end"],
                            lo=words[qs[0]]["s"] if qs else None) if qs else None
            if qs and qe:
                i0 = qs[0]
                # The ranker picks sentence starts on the punctuated transcript. Word-level
                # punctuation from Whisper is too unreliable to move the start automatically.
                if i0 > 0 and not words[i0 - 1]["w"].rstrip("\"'”’)").endswith((".", "?", "!")):
                    log(f"{clip['id']}: note: no sentence end before the start quote in the word transcript")
                clip["start"], clip["end"] = words[i0]["s"], words[qe[1]]["e"]
            if clip.get("payoff_start") is not None:
                qp = find_quote(words, quotes.get("payoff_quote", ""), near=clip["payoff_start"],
                                lo=clip["start"] - 0.5, hi=clip["end"] + 0.5)
                if qp:
                    i0, i1 = qp
                    # A long payoff is shortened from the front (cold opens work at 1-3.5 s),
                    # preferring to start right after a comma: "..., nobody knows who pooped in the bed."
                    while words[i1]["e"] - words[i0]["s"] > 3.5 and i0 < i1:
                        i0 += 1
                    if i0 != qp[0]:
                        # Trimmed: start after the first comma that leaves <= 3.5 s, if any.
                        for j in range(qp[0], i1):
                            if words[j]["w"].endswith(",") and words[i1]["e"] - words[j + 1]["s"] <= 3.5:
                                i0 = j + 1
                                break
                    clip["payoff_start"], clip["payoff_end"] = words[i0]["s"], words[i1]["e"]
                    # Strong profanity in the first seconds risks limited ads: no cold open then.
                    from .render import _PROFANITY
                    if any(_PROFANITY.search(w["w"]) for w in words[i0:i1 + 1]):
                        log(f"{clip['id']}: payoff has strong profanity, no cold open")
                        clip["payoff_start"] = clip["payoff_end"] = None
            log(f"{clip['id']}: quotes start={'ok' if qs else 'MISS'} end={'ok' if qe else 'MISS'}")
            log(f"{clip['id']}: {len(words)} words, {clip['end'] - clip['start']:.1f}s")
            continue
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
