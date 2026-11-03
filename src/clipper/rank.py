"""Let Claude read the whole stream transcript and pick self-contained story moments with a cold-open payoff."""

import json
import re
import subprocess

from .common import find_source, job_dir, log, read_json, tool, write_json

PROMPT = """You are a top short-form editor who cuts viral streamer clips for YouTube Shorts.

Streamer: {streamer}
{notes}Below is the transcript of a stream VOD. Lines are "[absolute_seconds] text". Lines marked with
"<<SPIKE z=...>>" had a loudness/chat-activity spike right there (screaming, laughing, chat going crazy).

Find up to {n} moments that would perform as standalone Shorts. A good clip is a small story:
a clear setup the viewer understands within 2-3 seconds, rising tension, and a payoff (punchline, shocking
reaction, rage, fail, wholesome surprise, chaos). A stranger who has never seen this streamer must get it.

Hard rules:
- Reject anything that needs earlier context, is just talking/driving/gameplay without a payoff,
  is music, ads, reading donations, or has no clear payoff inside the clip.
- Also reject a clip if any of these holds: the key point is mumbled or unintelligible; sarcasm that flips the
  literal meaning (a viewer without the tone would take it the opposite way); it ends without a payoff
  (trails off, cuts before the answer); it cannot be understood without outside context (earlier events,
  inside jokes, unnamed references); a disputed claim is stated as fact and the clip does not attribute it.
- Clip length (start..end): {min_sec}-{max_sec} seconds. Absolute seconds from the transcript.
- payoff_start/payoff_end: the 1.0-2.5 s peak moment INSIDE the clip (the scream, the punchline, the reaction).
  It is shown first as a cold open before the clip plays from the setup.
- The clip must START at the beginning of a sentence and END at the end of a sentence (never mid-sentence,
  never on a dangling "and"/"so"/"I mean"). The first sentence must make sense to a stranger on its own.
- start_quote: the first 5-8 words of the clip, copied EXACTLY from the transcript (same words, same order).
  It must be the beginning of a full sentence: the transcript text right before it ends with . ? or !
- end_quote: the last 5-8 words of the clip, copied EXACTLY from the transcript, ending a sentence.
  End as soon as the moment resolves (the answer, the punchline, the reaction): at most one short beat after it,
  no trailing chatter, no "anyway", no wind-down. Watching to the end and replays are ranking signals on every
  platform, so a clip that stops right at its peak beats a longer one that fades out. If one natural ending
  flows back into the opening line (a question answered by the start, a callback), prefer it: it invites a rewatch.
- payoff_quote: the payoff as a short complete phrase or sentence (3-10 words), copied EXACTLY from the
  transcript, inside the clip. It must be understandable as the first thing a viewer hears.
  No strong profanity (f-word, s-word...) in the payoff_quote: it plays in the first seconds.
  The quotes decide the cut; the numbers are only a hint. A clip whose quotes are not in the transcript is dropped.
- hook: on-screen text, 3-5 words and at most {hook_max} characters (spaces included; a hard limit, longer
  hooks are rejected), shown in one line over the cold open. It must create tension or a question that only
  watching the clip answers. Good: "He bet his car on this", "Chat lied to him", "She was not ready for this",
  "This cost him $10,000". Bad (descriptive, no tension): "Kai's honest review of the greens",
  "Funny moment on stream", "Speed at the zoo". No emojis, no lies, not just the streamer's name.
- If the clip only uses pronouns (he/she) for the people it is about, the hook must name them
  ("Who pooped in Depp's bed?"), so a stranger knows who it is about.
- Never state allegations about real people as fact in hook or title (crimes, abuse, disputed events):
  attribute them to the speaker ("Mark Manson on ...") or phrase as the speaker's story.
- title: under {title_max} characters (the campaign prefix is added later and already counted), curiosity-driven, the streamer/speaker name (or the main keyword) within the
  first two words. Title and hook must be about the same moment and promise the same payoff, so the title,
  the on-screen hook and the first spoken line all point at one topic. Never promise more than the clip shows.
- claim: ONE sentence stating what the clip says or shows and its punchline/point (what a viewer takes away).
  If you cannot write it, the clip has nothing to say: do not return it. Empty claim = rejected.
- story: one sentence, "setup -> payoff".
- caption: one short line for TikTok/Instagram, under 100 characters, curiosity-driven, may differ from the title.
  The same rule as for hook/title: no allegations about real people stated as fact.
- Prefer moments where the streamer/speaker reacts, explains or comments in their own words over raw footage
  with little narration: platforms demote reused clips "with little or no narrative" (YouTube reused/inauthentic
  content, TikTok unoriginal content, Instagram originality rules 2026).
- hashtags: 3-5 relevant lowercase hashtags without "#" (streamer name first, no generic spam like fyp/viral).
- score: 1-10 for Shorts potential. Be harsh: 5 = average, 7 = clearly good, 9 = viral. Only return clips scoring 7+.
  Test: would a stranger scrolling past stop within 1 second, and would they send it to a friend in a DM?
  (Sends per reach is one of the strongest ranking signals on Reels; a clip nobody would send is a 6 at most.) Mild jokes, food reviews,
  casual banter and inside jokes are 5-6 at most. Loud genuine reactions, fails, shocking reveals, wins/losses
  with stakes, and chaos involving other people score higher.
- If nothing qualifies, return [] and nothing else. Returning [] is a correct answer for a boring stream.

Return ONLY a JSON array:
[{{"start": s, "end": s, "payoff_start": s, "payoff_end": s, "start_quote": "...", "end_quote": "...",
"payoff_quote": "...", "hook": "...", "title": "...", "story": "...",
"claim": "...", "caption": "...", "hashtags": ["..."], "score": n, "reason": "..."}}]

Transcript:
{transcript}
"""

MAX_PROMPT_CHARS = 400_000

CHECK_PROMPT = """You are a strict editorial checker for short-form clips. For each numbered transcript below,
decide whether the clip must be REJECTED. Reject if ANY holds:
1. mumble: the key point is mumbled, garbled or unintelligible in the transcript
2. sarcasm: sarcasm flips the literal meaning, so a viewer without tone would understand the opposite
3. no_payoff: the clip ends without a payoff (trails off, cuts before the answer or punchline)
4. context: it cannot be understood by a stranger without outside context (earlier events, inside jokes)
5. disputed: a disputed or unproven claim is stated as fact without attribution to the speaker
If none holds, keep it. Be conservative: reject only when clearly true.

Return ONLY a JSON array: [{{"n": 1, "reject": false, "rule": "", "reason": ""}}, ...] with one entry per clip.

{items}
"""


def clip_text(segments: list[dict], start: float, end: float) -> str:
    return " ".join(s["text"].strip() for s in segments if s["e"] > start and s["s"] < end)


def content_check(cfg: dict, jd, clips: list[dict], segments: list[dict], ask=None) -> list[dict]:
    """Second, cheap LLM pass over the selected transcripts: drop clips that hit a content exclusion.

    Fails open: if the check itself errors, the clips are kept. Verdicts go to <job>/content_check.json.
    """
    if not clips or not cfg["rank"].get("content_check", True):
        return clips
    items = "\n\n".join(f"Clip {i + 1}:\n{clip_text(segments, c['start'], c['end'])}" for i, c in enumerate(clips))
    cfg_c = {**cfg, "rank": {**cfg["rank"], "model": cfg["rank"].get("check_model", "haiku")}}
    try:
        out = (ask or ask_claude)(cfg_c, CHECK_PROMPT.format(items=items), jd)
        verdicts = {int(v["n"]): v for v in extract_json(out)}
    except Exception as exc:
        log(f"content check failed ({exc}), keeping all clips")
        return clips
    write_json(jd / "content_check.json", [{"id": c["id"], **verdicts.get(i + 1, {})} for i, c in enumerate(clips)])
    keep = []
    for i, c in enumerate(clips):
        v = verdicts.get(i + 1, {})
        if v.get("reject"):
            log(f"content check DROP {c['id']} ({v.get('rule', '?')}): {v.get('reason', '')}")
        else:
            keep.append(c)
    return keep


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


def title_budget(source: dict, limit: int = 60) -> int:
    """Characters left for the model's title once the mandatory prefix ("Ad: Scape - ") is prepended."""
    prefix = source.get("title_prefix", "")
    return limit - (len(prefix) + 1 if prefix else 0)


HOOK_MAX_CHARS = 28  # one line of the hook font at >= 80 px inside the safe zone
HOOK_MAX_WORDS = 5


def hook_budget(source: dict) -> int:
    """Characters the model may use for the hook once the mandatory "Name: " mention is prepended."""
    name = source.get("hook_mention", "")
    return HOOK_MAX_CHARS - (len(name) + 2 if name else 0)


def hook_ok(hook: str) -> bool:
    words = [w for w in hook.split() if not w.endswith(":")]
    return len(hook) <= HOOK_MAX_CHARS and len(words) <= HOOK_MAX_WORDS


def shorten_hooks(cfg: dict, picks: list[dict], source: dict, cwd) -> None:
    """One rewrite round for hooks over the limit; the renderer never shrinks a hook below 80 px."""
    long = [i for i, p in enumerate(picks)
            if not hook_ok(_hook_with_mention(p.get("hook", "").strip(), source.get("hook_mention", "")))]
    if not long:
        return
    budget = hook_budget(source)
    items = "\n".join(f'{n}. hook: "{picks[i]["hook"]}" | story: {picks[i].get("story", "")}'
                       for n, i in enumerate(long))
    prompt = (f"Rewrite each on-screen hook to at most {budget} characters (spaces included) and at most "
              f"{HOOK_MAX_WORDS} words. Keep the same tension and meaning, no lies, no emojis, no allegations "
              f"stated as fact.\n{items}\nReturn ONLY a JSON array of strings, one per hook, same order.")
    try:
        new = extract_json(ask_claude(cfg, prompt, cwd))
    except Exception as e:  # noqa: BLE001
        log(f"hook rewrite failed: {e}")
        return
    for i, h in zip(long, new):
        if isinstance(h, str) and h.strip():
            log(f"hook rewritten: {picks[i]['hook']!r} -> {h.strip()!r}")
            picks[i]["hook"] = h.strip()


def _hook_with_mention(hook: str, name: str) -> str:
    """Campaigns that require the creator's name in the on-screen text get it in the hook."""
    if name and name.lower() not in hook.lower():
        return f"{name}: {hook}"
    return hook


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


def _earlier_spans(jd, job_id: str) -> list[dict]:
    """Spans of clips already cut from the same video by other jobs (same input URL): never cut them twice."""
    try:
        me = read_json(jd / "job.json")["input"]
    except Exception:
        return []
    out = []
    for other in jd.parent.glob("*/job.json"):
        if other.parent == jd:
            continue
        try:
            if read_json(other)["input"] != me:
                continue
            out += [c for c in read_json(other.parent / "clips.json") if "start" in c]
        except Exception:
            continue
    return out


GAP_OK = 0.3  # seconds of silence that counts as a clean cut point


def boundary_problems(words: list[dict], i0: int, i1: int, gap: float = GAP_OK) -> list[str]:
    """Hard cut gate on the transcript: the clip must start at a sentence start (or after >= gap of
    silence) and end at a sentence end (or before >= gap of silence). words = segment_words output."""
    bad = []
    if i1 + 1 < len(words) and not words[i1]["w"].rstrip("\"')]").endswith((".", "?", "!", "…")) \
            and words[i1 + 1]["s"] - words[i1]["e"] < gap:
        bad.append(f"ends mid-sentence at '{words[i1]['w']}'")
    if i0 > 0 and not words[i0 - 1]["w"].rstrip("\"')]").endswith((".", "?", "!", "…")) \
            and words[i0]["s"] - words[i0 - 1]["e"] < gap:
        bad.append(f"starts mid-sentence at '{words[i0]['w']}'")
    return bad


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
            streamer=source["streamer"], n=rc["clips_per_vod"],
            notes=(f"Campaign rules (hard): {source['rank_notes']}\n" if source.get("rank_notes") else ""), min_sec=cc["min_sec"],
            max_sec=cc["max_sec"] - cfg["render"]["cold_open_max"], transcript="\n".join(part),
            title_max=title_budget(source), hook_max=hook_budget(source),
        ), jd)
        raw.append(out)
        picks += extract_json(out)
    (jd / "rank_raw.txt").write_text("\n\n".join(raw))
    shorten_hooks(cfg, [p for p in picks if p.get("score", 0) >= min_score], source, jd)

    from .anchors import find_quote, segment_words
    seg_words = segment_words(segments)
    clips = []
    earlier = _earlier_spans(jd, job_id)
    for p in sorted(picks, key=lambda p: -p.get("score", 0)):
        if p.get("score", 0) < min_score or len(clips) >= rc["clips_per_vod"]:
            continue
        start, end = float(p["start"]), float(p["end"])
        # Times come from the quotes, not the ranker's numbers (those can be far off).
        qs = find_quote(seg_words, p.get("start_quote", ""), near=start)
        qe = find_quote(seg_words, p.get("end_quote", ""), near=end,
                        lo=seg_words[qs[0]]["s"] if qs else None) if qs else None
        if not qs or not qe:
            log(f"dropping pick '{p.get('title', '')[:50]}': start/end quote not found in transcript")
            continue
        if not str(p.get("claim", "")).strip():
            log(f"dropping pick '{p.get('title', '')[:50]}': no claim (nothing to say)")
            continue
        bp = boundary_problems(seg_words, qs[0], qe[1])
        if bp:
            log(f"dropping pick '{p.get('title', '')[:50]}': {'; '.join(bp)}")
            continue
        start, end = seg_words[qs[0]]["s"], seg_words[qe[1]]["e"]
        qp = find_quote(seg_words, p.get("payoff_quote", ""), near=float(p.get("payoff_start", start)),
                        lo=start - 1, hi=end + 1)
        if qp:
            p["payoff_start"], p["payoff_end"] = seg_words[qp[0]]["s"], seg_words[qp[1]]["e"]
        else:
            log(f"pick '{p.get('title', '')[:50]}': payoff quote not found, no cold open")
            p["payoff_start"] = p["payoff_end"] = None
        if not cc["min_sec"] - 3 <= end - start <= cc["max_sec"] + 5:
            log(f"dropping pick '{p.get('title', '')[:50]}': quoted span is {end - start:.0f} s")
            continue
        if (jd / "source.mp4").exists():
            from .blackwin import black_fraction, black_max
            black = black_fraction(jd / "source.mp4", start, end)
            log(f"black_ratio {black:.3f} for '{p.get('title', '')[:50]}' ({start:.0f}-{end:.0f} s), limit {black_max(cfg):.2f}")
            if black > black_max(cfg):
                log(f"dropping pick '{p.get('title', '')[:50]}': {black:.0%} of the window is black (title cards)")
                continue
        if any(start < c["end"] and end > c["start"] for c in clips):
            continue  # overlapping pick from two transcript parts
        if any(start < c["end"] and end > c["start"] for c in earlier):
            log(f"dropping pick '{p.get('title', '')[:50]}': overlaps a clip from an earlier job of the same video")
            continue
        hook = _hook_with_mention(p["hook"].strip(), source.get("hook_mention", ""))
        if not hook_ok(hook):
            log(f"dropping pick '{p.get('title', '')[:50]}': hook too long after rewrite ({hook!r})")
            continue
        ps, pe = p["payoff_start"], p["payoff_end"]
        clips.append({
            "id": f"{job_id}-c{len(clips) + 1}",
            "start": round(start, 2),
            "end": round(end, 2),
            "payoff_start": round(ps, 2) if ps is not None else None,
            "payoff_end": round(pe, 2) if pe is not None else None,
            "quotes": {k: p.get(k, "") for k in ("start_quote", "end_quote", "payoff_quote")},
            "hook": hook,
            "title": p["title"].strip()[:95],
            "claim": p["claim"].strip(),
            "story": p.get("story", ""),
            "caption": p.get("caption", p["title"]).strip(),
            "hashtags": [h.strip().lstrip("#").replace(" ", "") for h in p.get("hashtags", []) if h.strip()][:5],
            "score": p["score"],
            "reason": p.get("reason", ""),
        })
    clips = content_check(cfg, jd, clips, segments)
    write_json(jd / "clips.json", clips)
    log(f"{len(clips)} clips selected (min score {min_score})")
    return clips
