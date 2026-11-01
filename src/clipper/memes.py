"""Meme trend library: weekly refresh from trending Shorts + per-clip effect selection."""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .common import log, path_of, probe, run, tool, write_json

EFFECTS = ["zoom", "shake", "flash", "red", "bw"]

ANALYZE_PROMPT = """You are a viral short-form video editor.
Below is the transcript and audio analysis of a recent trending YouTube Short.

Transcript:
{transcript}

Audio features (0-1 scale):
- Reaction sounds (laughter, screaming, cheering): {reaction:.2f}
- Music: {music:.2f}
- Clip score: {score}/10

Which ONE visual effect would make this clip MORE viral if applied subtly?
Choose exactly one from: zoom, shake, flash, red, bw, none

Rules:
- zoom: slow push-in, good for reaction/surprise moments
- shake: camera shake, good for explosive/rage moments
- flash: white flash cut, good for punchlines
- red: red tint, good for dramatic/intense moments
- bw: black-and-white, good for cinematic/emotional moments
- none: the clip is strong without any effect

Reply with ONLY the single word (e.g. "zoom").
"""


def memes_dir(cfg: dict) -> Path:
    d = path_of(cfg, "data") / "memes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def library_path(cfg: dict) -> Path:
    return memes_dir(cfg) / "library.json"


def load_library(cfg: dict) -> dict:
    p = library_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return {"updated": None, "scores": {e: 0.0 for e in EFFECTS}, "threshold_count": 0}


def save_library(cfg: dict, lib: dict) -> None:
    library_path(cfg).write_text(json.dumps(lib, indent=2, ensure_ascii=False))


def fetch_channel_shorts(channel_url: str, n: int = 15) -> list[str]:
    """Most viewed of the channel's newest Shorts (looks at the newest 3*n, keeps the top n by views).

    The Shorts tab is used directly: flat-playlist entries have no duration, so a duration
    match-filter would silently drop everything.
    """
    res = subprocess.run(
        [tool("yt-dlp"), "--flat-playlist", "--playlist-items", f"1-{n * 3}",
         "--print", "%(id)s %(view_count)s", channel_url.rstrip("/") + "/shorts"],
        capture_output=True, text=True, timeout=120,
    )
    rows = []
    for line in res.stdout.splitlines():
        vid, _, views = line.strip().partition(" ")
        if vid:
            rows.append((vid, int(views) if views.isdigit() else 0))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [f"https://www.youtube.com/shorts/{vid}" for vid, _ in rows[:n]]


def discover_channels(queries: list[str], k: int, per_query: int = 25) -> list[str]:
    """Channel search: the k channels with the most total views across the search results."""
    from collections import Counter
    views: Counter = Counter()
    for q in queries:
        res = subprocess.run(
            [tool("yt-dlp"), "--flat-playlist", "--print", "%(channel_url)s %(view_count)s %(duration)s",
             f"ytsearch{per_query}:{q}"],
            capture_output=True, text=True, timeout=120,
        )
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) != 3 or not parts[0].startswith("http"):
                continue
            url, v, dur = parts
            # Only Shorts-length results count: long-form clip channels often have no Shorts tab.
            try:
                if float(dur) > 90:
                    continue
            except ValueError:
                continue
            views[url] += int(v) if v.isdigit() else 0
    return [url for url, _ in views.most_common(k)]


def _download_preview(url: str, out_dir: Path) -> Path | None:
    """Download first 30 s of a Short into out_dir/preview.mp4."""
    out = out_dir / "preview.mp4"
    res = subprocess.run(
        [tool("yt-dlp"),
         "--no-playlist", "-f", "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/b[height<=720]/b",
         "--merge-output-format", "mp4",
         "--download-sections", "*0-30",
         "--force-keyframes-at-cuts",
         "-o", str(out), url],
        capture_output=True, text=True, timeout=120,
    )
    return out if out.exists() else None


_MODELS: dict = {}


def _audio(video: Path):
    """16 kHz mono float32, decoded with the pipeline's own ffmpeg helper (no soundfile/PyAV)."""
    from .transcribe import load_audio
    return load_audio(video, 0.0, 30.0)


def _quick_transcript(audio) -> str:
    """Fast Whisper tiny transcript of a short clip (model loaded once per refresh)."""
    try:
        if "whisper" not in _MODELS:
            from faster_whisper import WhisperModel
            _MODELS["whisper"] = WhisperModel("tiny", device="cpu", compute_type="int8")
        segs, _ = _MODELS["whisper"].transcribe(audio, language=None, beam_size=1)
        return " ".join(s.text.strip() for s in segs)[:1000]
    except Exception as exc:
        log(f"memes: transcript failed: {exc}")
        return ""


def _yamnet_scores(audio) -> tuple[float, float]:
    """Mean (reaction, music) scores for the preview clip."""
    try:
        import numpy as np
        from .audio_ai import Yamnet, WINDOW
        if "yamnet" not in _MODELS:
            _MODELS["yamnet"] = Yamnet(threads=2)
        ym = _MODELS["yamnet"]
        reactions, musics = [], []
        for start in range(0, max(len(audio) - WINDOW, 0) + 1, 16000):
            r, m = ym.tag(audio[start: start + WINDOW].astype(np.float32))
            reactions.append(r)
            musics.append(m)
        if not reactions:
            return 0.0, 0.0
        return float(np.mean(reactions)), float(np.mean(musics))
    except Exception as exc:
        log(f"memes: yamnet failed: {exc}")
        return 0.0, 0.0


def _ask_claude_effect(cfg: dict, transcript: str, reaction: float, music: float, tmp: Path) -> str | None:
    prompt = ANALYZE_PROMPT.format(
        transcript=transcript or "(no speech detected)",
        reaction=reaction, music=music, score=5,
    )
    model = cfg.get("memes", {}).get("claude_model", "haiku")
    res = subprocess.run(
        [tool("claude"), "-p", "--model", model, "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=120, cwd=tmp,
    )
    if res.returncode != 0:
        log(f"memes: claude failed: {res.stderr[:200]}")
        return None
    answer = res.stdout.strip().lower().split()[0] if res.stdout.strip() else None
    return answer if answer in EFFECTS + ["none"] else None


def _cluster(labels: list[str], n_total: int) -> dict[str, float]:
    """Effect → normalised score; only effects that pass min(3, 10%) threshold survive."""
    from collections import Counter
    counts = Counter(l for l in labels if l in EFFECTS)
    threshold = max(3, int(n_total * 0.1))
    scores = {}
    total = sum(counts.values()) or 1
    for effect in EFFECTS:
        c = counts.get(effect, 0)
        if c >= threshold:
            scores[effect] = round(c / total, 3)
    return scores


def refresh(cfg: dict) -> dict:
    """Scrape, analyse, cluster, save library.json. Returns the library."""
    mc = cfg.get("memes", {})
    sources = list(mc.get("sources", []))
    n_shorts = mc.get("shorts_per_source", 12)
    if mc.get("search_queries"):
        found = discover_channels(mc["search_queries"], mc.get("discover_channels", 4))
        log(f"memes: discovered channels: {found}")
        sources += [u for u in found if u not in sources]

    if not sources:
        log("memes: no sources configured, skipping refresh")
        lib = load_library(cfg)
        lib["updated"] = datetime.now().isoformat(timespec="seconds")
        save_library(cfg, lib)
        return lib

    labels: list[str] = []
    with tempfile.TemporaryDirectory(dir=memes_dir(cfg)) as tmp_str:
        tmp = Path(tmp_str)
        for ch_url in sources:
            urls = fetch_channel_shorts(ch_url, n_shorts)
            log(f"memes: {ch_url} -> {len(urls)} shorts")
            for url in urls:
                vid = url.rstrip("/").split("/")[-1]
                item_dir = tmp / vid
                item_dir.mkdir(parents=True, exist_ok=True)
                video = _download_preview(url, item_dir)
                if not video:
                    continue
                try:
                    audio = _audio(video)
                except Exception as exc:
                    log(f"memes: audio decode failed for {vid}: {exc}")
                    continue
                transcript = _quick_transcript(audio)
                reaction, music = _yamnet_scores(audio)
                if not transcript and reaction == 0.0 and music == 0.0:
                    continue  # no signal: a blind label would only skew the clusters
                label = _ask_claude_effect(cfg, transcript, reaction, music, item_dir)
                if label:
                    labels.append(label)
                    log(f"memes: {vid} -> {label}")

    scores = _cluster(labels, len(labels))
    lib = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "scores": scores,
        "threshold_count": len(labels),
        "sample_size": len(labels),
    }
    save_library(cfg, lib)
    log(f"memes: library updated, {len(labels)} samples, scores: {scores}")
    return lib


def pick_effects(cfg: dict, clip: dict) -> list[str]:
    """Return up to max_per_clip effects for this clip from the library."""
    max_n = cfg.get("memes", {}).get("max_per_clip", 2)
    lib = load_library(cfg)
    scores = lib.get("scores", {})
    if not scores:
        return []

    # Bias: high reaction → prefer zoom/shake; high music → skip (Content ID caution)
    reaction = clip.get("reaction_peak", 0.0)
    music = clip.get("music") or 0.0
    if music > 0.3:
        scores = {k: v for k, v in scores.items() if k not in ("flash",)}

    if reaction > 0.5:
        for k in ("zoom", "shake"):
            if k in scores:
                scores[k] = scores[k] * 1.4

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [e for e, _ in ranked[:max_n]]


def meme_filter_chain(effects: list[str], node_in: str, node_out: str, W: int, H: int,
                      window: tuple[float, float] | None = None) -> str:
    """ffmpeg filter graph applying the effects only inside `window` (output-timeline seconds).

    Effects are meme accents on the payoff moment, never on the whole clip. Without a window
    nothing is applied.
    """
    if not effects or window is None:
        return f"[{node_in}]null[{node_out}]"
    a, b = window
    b = max(b, a + 0.8)
    on = f"between(t,{a:.2f},{b:.2f})"
    parts, cur = [], node_in
    for i, ef in enumerate(effects):
        nxt = node_out if i == len(effects) - 1 else f"fx{i}"
        if ef == "zoom":
            parts.append(f"[{cur}]split[fxm{i}][fxe{i}]")
            parts.append(f"[fxe{i}]scale={int(W * 1.12) // 2 * 2}:{int(H * 1.12) // 2 * 2},crop={W}:{H}[fxz{i}]")
            parts.append(f"[fxm{i}][fxz{i}]overlay=0:0:enable='{on}'[{nxt}]")
        elif ef == "shake":
            parts.append(f"[{cur}]split[fxm{i}][fxe{i}]")
            parts.append(f"[fxe{i}]scale={W + 48}:{H + 48},crop={W}:{H}:24+16*sin(t*41):24+16*cos(t*33)[fxz{i}]")
            parts.append(f"[fxm{i}][fxz{i}]overlay=0:0:enable='{on}'[{nxt}]")
        elif ef == "flash":
            parts.append(f"[{cur}]drawbox=x=0:y=0:w=iw:h=ih:color=white@0.85:t=fill"
                         f":enable='between(t,{a:.2f},{a + 0.12:.2f})'[{nxt}]")
        elif ef == "red":
            parts.append(f"[{cur}]colorchannelmixer=rr=1.4:rg=0.1:rb=0.1:gg=0.6:bb=0.6:enable='{on}'[{nxt}]")
        elif ef == "bw":
            parts.append(f"[{cur}]hue=s=0:enable='{on}'[{nxt}]")
        else:
            parts.append(f"[{cur}]null[{nxt}]")
        cur = nxt
    return ";".join(parts)
