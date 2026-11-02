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
    from .common import find_source
    src = find_source(cfg, read_json(jd / "job.json")["source"]) or {}
    # Names in the prompt make Whisper spell them right ("Lacy", not "Lacey").
    source_names = f"{src['streamer']}. " if src.get("streamer") else ""
    for clip in clips:
        ps = clip.get("payoff_start")
        lo = max(0.0, min(clip["start"], ps if ps is not None else clip["start"]) - 5)
        hi = max(clip["end"], clip.get("payoff_end") or clip["end"]) + 5
        # The prompt nudges Whisper to keep spoken fillers ("like", "you know", "it's") verbatim;
        # captions that skip words people can hear look machine-made.
        segments, _ = m.transcribe(load_audio(video, lo, hi - lo), word_timestamps=True, vad_filter=True,
                                   initial_prompt=f"{source_names}Um, so, like, you know, I mean, it's, uh, yeah. Verbatim.",
                                   language=language)
        words = [
            {"w": w.word.strip(), "s": round(lo + w.start, 2), "e": round(lo + w.end, 2)}
            for seg in segments for w in (seg.words or []) if w.word.strip()
        ]
        quotes0 = clip.get("quotes") or {}
        if quotes0.get("end_quote") and not find_quote(words, quotes0["end_quote"], near=clip["end"]):
            # Whisper sometimes drops the last words of a window (crosstalk, VAD, profanity):
            # re-transcribe the tail without VAD and splice the missing words in.
            t0 = max(lo, clip["end"] - 10)
            seg2, _ = m.transcribe(load_audio(video, t0, hi - t0), word_timestamps=True, vad_filter=False,
                                   condition_on_previous_text=False, language=language,
                                   initial_prompt=f"{source_names}Um, so, like, you know, I mean, it's, uh, yeah. Verbatim.")
            tail = [{"w": w.word.strip(), "s": round(t0 + w.start, 2), "e": round(t0 + w.end, 2)}
                    for sg in seg2 for w in (sg.words or []) if w.word.strip()]
            if tail:
                cut = tail[0]["s"] - 0.05
                words = [w for w in words if w["e"] <= cut] + tail
                log(f"{clip['id']}: end quote missing, re-transcribed tail from {t0:.1f}s ({len(tail)} words)")
        # Whisper's text, forced-aligned times (Whisper word times drift 0.2-0.6 s).
        from .align import align_words
        win_audio = load_audio(video, lo, hi - lo)
        aligned = align_words(words, win_audio, lo)
        if aligned:
            aligned = snap_to_onsets(aligned, win_audio, lo, ref=words)
        if aligned:
            clip["words_whisper"] = words
            words = aligned
        else:
            log(f"{clip['id']}: forced alignment failed, using Whisper word times")
        # Manual fix for words Whisper and the aligner both missed (clip["word_inserts"], survives re-runs).
        for ins in clip.get("word_inserts") or []:
            if not any(abs(w["s"] - ins["s"]) < 0.05 and w["w"].lower() == ins["w"].lower() for w in words):
                words.append(dict(ins))
        # Manual removal of a misaligned duplicate (clip["word_drops"] = [{"w":..,"s":..}], survives re-runs).
        for dr in clip.get("word_drops") or []:
            words = [w for w in words if not (abs(w["s"] - dr["s"]) < 0.05 and w["w"].lower() == dr["w"].lower())]
        words.sort(key=lambda w: w["s"])
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
                    # A long payoff (> 3.5 s) is shortened from the front, but only at a word that
                    # follows an audible pause: "To this day, nobody" has no pause before "nobody",
                    # a cut there leaves "day" audible. Without such a point the whole quote stays (<= 5 s).
                    if words[i1]["e"] - words[i0]["s"] > 3.5:
                        from .render import silence_runs
                        afn = lambda t, d: load_audio(video, t, d)  # noqa: E731
                        for j in range(qp[0] + 1, i1):
                            if words[i1]["e"] - words[j]["s"] > 3.5:
                                continue
                            if silence_runs(afn, words[j - 1]["e"] - 0.05, words[j]["s"] + 0.05):
                                i0 = j
                                break
                    if words[i1]["e"] - words[i0]["s"] > 5.0:
                        log(f"{clip['id']}: payoff too long for a cold open")
                        i0 = None
                    if i0 is None:
                        clip["payoff_start"] = clip["payoff_end"] = None
                        continue_payoff = False
                    else:
                        continue_payoff = True
                        clip["payoff_start"], clip["payoff_end"] = words[i0]["s"], words[i1]["e"]
                    # Strong profanity in the first seconds risks limited ads: no cold open then.
                    from .render import _PROFANITY
                    if continue_payoff and any(_PROFANITY.search(w["w"]) for w in words[i0:i1 + 1]):
                        log(f"{clip['id']}: payoff has strong profanity, no cold open")
                        clip["payoff_start"] = clip["payoff_end"] = None
            if clip.get("end_pad"):  # manual tail for a slow word decay, survives re-runs
                clip["end"] = round(clip["end"] + clip["end_pad"], 3)
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


def snap_to_onsets(words: list[dict], audio: np.ndarray, lo: float, ref: list[dict] | None = None) -> list[dict]:
    """Forced alignment can start the first word after a pause 100-400 ms off. Move such a word
    onto the audio onset (mutual nearest pairing, onset preceded by >= 150 ms of silence)."""
    from .qa import _db_curve
    db = _db_curve(audio)
    speech = db >= -35.0
    onsets = [lo + i * 0.01 for i in range(15, len(speech)) if speech[i] and not speech[i - 15:i].any()]
    if not onsets:
        return words
    cands = list(range(len(words)))
    cand_t = [words[i]["s"] for i in cands]
    out = [dict(w) for w in words]
    for i in cands:
        s = words[i]["s"]
        o = min(onsets, key=lambda o: abs(o - s))
        if abs(o - s) <= 0.04 or abs(o - s) > 0.45:
            continue
        if min(cand_t, key=lambda c: abs(c - o)) != s:
            continue
        if i > 0 and o < out[i - 1]["s"] + 0.05:
            continue
        # The onset must fall inside this word as whisper heard it, and before the next aligned word:
        # it may belong to a word the aligner misplaced (the next speaker's "Jesus").
        # (The aligner may squeeze a word into 20 ms, so its own end is no limit.)
        if ref is not None and len(ref) == len(words) and o > ref[i]["e"] - 0.04:
            continue
        if i + 1 < len(words) and o > words[i + 1]["s"] - 0.04:
            continue
        if i > 0:
            out[i - 1]["e"] = round(min(out[i - 1]["e"], o), 3)
        log(f"  onset snap: '{words[i]['w']}' {s:.2f} -> {o:.2f}")
        out[i]["s"] = round(o, 3)
    return out
