"""Let Claude read the whole stream transcript and pick self-contained story moments with a cold-open payoff."""

import json
import re
import subprocess

from .common import find_source, job_dir, log, read_json, tool, write_json

PROMPT = """You are a top short-form editor who cuts viral streamer clips for YouTube Shorts.

Streamer: {streamer}
Below is the transcript of a stream VOD. Lines are "[absolute_seconds] text". Lines marked with
"<<SPIKE z=...>>" had a loudness/chat-activity spike right there (screaming, laughing, chat going crazy).

Find up to {n} moments that would perform as standalone Shorts. A good clip is a small story:
a clear setup the viewer understands within 2-3 seconds, rising tension, and a payoff (punchline, shocking
reaction, rage, fail, wholesome surprise, chaos). A stranger who has never seen this streamer must get it.

Hard rules:
- Reject anything that needs earlier context, is just talking/driving/gameplay without a payoff,
  is music, ads, reading donations, or has no clear payoff inside the clip.
- Clip length (start..end): {min_sec}-{max_sec} seconds. Absolute seconds from the transcript.
- payoff_start/payoff_end: the 1.0-2.5 s peak moment INSIDE the clip (the scream, the punchline, the reaction).
  It is shown first as a cold open before the clip plays from the setup.
- hook: on-screen text, 3-7 words, states the stakes or the question the clip answers
  (e.g. "He bet his car on this", "Chat lied to him"). No emojis, no lies, not just the streamer's name.
- title: under 70 characters, curiosity-driven, includes the streamer name.
- story: one sentence, "setup -> payoff".
- caption: one short line for TikTok/Instagram, under 100 characters, curiosity-driven, may differ from the title.
- hashtags: 3-5 relevant lowercase hashtags without "#" (streamer name first, no generic spam like fyp/viral).
- score: 1-10 for Shorts potential. Be harsh: 5 = average, 7 = clearly good, 9 = viral. Only return clips scoring 7+.
- If nothing qualifies, return [] and nothing else. Returning [] is a correct answer for a boring stream.

Return ONLY a JSON array:
[{{"start": s, "end": s, "payoff_start": s, "payoff_end": s, "hook": "...", "title": "...", "story": "...",
"caption": "...", "hashtags": ["..."], "score": n, "reason": "..."}}]

Transcript:
{transcript}
"""

MAX_PROMPT_CHARS = 400_000


def format_transcript(segments: list[dict], spikes: list[dict]) -> str:
    marks = {}
    for sp in spikes:
        # Attach each spike to the segment that contains it (or the last one before it).
        idx = max((i for i, s in enumerate(segments) if s["s"] <= sp["peak"]), default=None)
        if idx is not None:
            marks[idx] = max(marks.get(idx, 0), sp["score"])
    lines = []
    for i, s in enumerate(segments):
        tag = f" <<SPIKE z={marks[i]:.1f}>>" if i in marks else ""
        lines.append(f"[{s['s']:.0f}] {s['text']}{tag}")
    return "\n".join(lines)


def extract_json(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON array in ranker output: {text[:300]}")
    return json.loads(m.group(0))


def ask_claude(cfg: dict, prompt: str, cwd) -> str:
    res = subprocess.run(
        [tool("claude"), "-p", "--model", cfg["rank"]["model"], "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=1800, cwd=cwd,
    )
    if res.returncode != 0:
        raise RuntimeError(f"claude failed: {res.stderr[:500]}")
    return res.stdout


def rank(cfg: dict, job_id: str, min_score: int | None = None) -> list[dict]:
    rc, cc = cfg["rank"], cfg["clip"]
    min_score = rc["min_score"] if min_score is None else min_score
    jd = job_dir(cfg, job_id)
    source = find_source(cfg, read_json(jd / "job.json")["source"])
    segments = read_json(jd / "transcript.json", {}).get("segments")
    if not segments:
        raise SystemExit("run transcribe first (or the VOD has no speech)")
    spikes = [c for c in read_json(jd / "candidates.json", []) if c["score"] >= cfg["detect"]["spike_z"]]

    # Long VODs are split into overlapping parts so each prompt stays within a comfortable context size.
    lines = format_transcript(segments, spikes).split("\n")
    parts, cur, size = [], [], 0
    for line in lines:
        if size + len(line) > MAX_PROMPT_CHARS and cur:
            parts.append(cur)
            cur = cur[-40:]
            size = sum(len(x) + 1 for x in cur)
        cur.append(line)
        size += len(line) + 1
    parts.append(cur)

    picks, raw = [], []
    for i, part in enumerate(parts):
        log(f"ranking part {i + 1}/{len(parts)} with claude...")
        out = ask_claude(cfg, PROMPT.format(
            streamer=source["streamer"], n=rc["clips_per_vod"], min_sec=cc["min_sec"],
            max_sec=cc["max_sec"] - cfg["render"]["cold_open_max"], transcript="\n".join(part),
        ), jd)
        raw.append(out)
        picks += extract_json(out)
    (jd / "rank_raw.txt").write_text("\n\n".join(raw))

    clips = []
    for p in sorted(picks, key=lambda p: -p.get("score", 0)):
        if p.get("score", 0) < min_score or len(clips) >= rc["clips_per_vod"]:
            continue
        start, end = float(p["start"]), float(p["end"])
        if any(start < c["end"] and end > c["start"] for c in clips):
            continue  # overlapping pick from two transcript parts
        ps = min(max(float(p.get("payoff_start", start)), start), end)
        pe = min(float(p.get("payoff_end", ps + 1.5)), end, ps + cfg["render"]["cold_open_max"])
        clips.append({
            "id": f"{job_id}-c{len(clips) + 1}",
            "start": start,
            "end": end,
            "payoff_start": round(ps, 2),
            "payoff_end": round(pe, 2),
            "hook": p["hook"].strip(),
            "title": p["title"].strip()[:95],
            "story": p.get("story", ""),
            "caption": p.get("caption", p["title"]).strip(),
            "hashtags": [h.strip().lstrip("#").replace(" ", "") for h in p.get("hashtags", []) if h.strip()][:5],
            "score": p["score"],
            "reason": p.get("reason", ""),
        })
    write_json(jd / "clips.json", clips)
    log(f"{len(clips)} clips selected (min score {min_score})")
    return clips
