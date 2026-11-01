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
    """Return up to n video URLs from a channel's Shorts feed."""
    for suffix in ("/shorts", "/videos"):
        res = subprocess.run(
            [tool("yt-dlp"), "--flat-playlist", "--playlist-items", f"1-{n * 2}",
             "--print", "id", "--match-filter", "duration<90",
             channel_url + suffix],
            capture_output=True, text=True, timeout=90,
        )
        ids = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if ids:
            return [f"https://www.youtube.com/shorts/{vid}" for vid in ids[:n]]
    return []


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


def _quick_transcript(video: Path) -> str:
    """Fast Whisper tiny transcript of a short clip."""
    try:
        import faster_whisper
        model = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")
        segs, _ = model.transcribe(str(video), language=None, beam_size=1)
        return " ".join(s.text.strip() for s in segs)[:1000]
    except Exception as exc:
        log(f"memes: transcript failed: {exc}")
        return ""


def _yamnet_scores(video: Path) -> tuple[float, float]:
    """Mean (reaction, music) scores for the preview clip."""
    try:
        import numpy as np
        import soundfile as sf
        from .audio_ai import Yamnet, WINDOW
        wav = video.with_suffix(".wav")
        subprocess.run(
            [tool("ffmpeg"), "-v", "error", "-y", "-i", str(video),
             "-ac", "1", "-ar", "16000", str(wav)],
            check=True, capture_output=True,
        )
        audio, _ = sf.read(str(wav))
        ym = Yamnet(threads=2)
        reactions, musics = [], []
        for start in range(0, len(audio) - WINDOW, 16000):
            r, m = ym.tag(audio[start: start + WINDOW].astype(np.float32))
            reactions.append(r)
            musics.append(m)
        wav.unlink(missing_ok=True)
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
    sources = mc.get("sources", [])
    n_shorts = mc.get("shorts_per_source", 12)

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
                item_dir = tmp / url.split("=")[-1]
                item_dir.mkdir(exist_ok=True)
                video = _download_preview(url, item_dir)
                if not video:
                    continue
                transcript = _quick_transcript(video)
                reaction, music = _yamnet_scores(video)
                label = _ask_claude_effect(cfg, transcript, reaction, music, item_dir)
                if label:
                    labels.append(label)
                    log(f"memes: {url.split('=')[-1]} -> {label}")

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


def meme_filter_chain(effects: list[str], node_in: str, node_out: str, W: int, H: int) -> str:
    """Build an ffmpeg filter chain string for the given effects."""
    if not effects:
        return f"[{node_in}]null[{node_out}]"
    filters = []
    for ef in effects:
        if ef == "zoom":
            filters.append(f"scale={int(W * 1.05)}:{int(H * 1.05)},crop={W}:{H}:{int(W * 0.025)}:{int(H * 0.025)}")
        elif ef == "shake":
            filters.append(f"crop=iw-20:ih-20:10+5*sin(t*25):5+5*cos(t*17),scale={W}:{H}")
        elif ef == "flash":
            filters.append("fade=t=in:st=0:d=0.12:color=white")
        elif ef == "red":
            filters.append("colorchannelmixer=rr=1.5:rg=0.1:rb=0.1:gr=0:gg=0.65:gb=0:br=0:bg=0:bb=0.6")
        elif ef == "bw":
            filters.append("hue=s=0")
    chain = ",".join(filters)
    return f"[{node_in}]{chain}[{node_out}]"
