"""Caption sync meter: finished audio vs ASS highlight starts, word by word.

One script for everyone (QA, reviewer, developer):
    uv run python -m clipper.syncmeter <short.mp4> <captions.ass>

Per word it prints the highlight start, the whisper word start on the FINISHED audio, the audio-energy
onset (50 ms envelope) near it, and the delta in ms (+ = caption late, - = caption early) against each.
Whisper starts can be wrong before digital silence (FEELS 1.76: whisper 1.20, speech starts 1.85), so when
whisper and energy onset differ by more than UNCERTAIN_MS the word is marked 'uncertain', not pass/fail.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

UNCERTAIN_MS = 150
FAIL_MS = 150
CAP_NEAR_MS = 200  # an energy onset this close to the caption backs it over whisper
HOP = 0.01
QUIET_STEPS = 8  # 80 ms of quiet before an energy onset counts
ENV_WIN = 0.05


@dataclass
class Row:
    word: str
    cap: float
    whisper: float
    energy: float | None
    energy_at_cap: float | None = None  # onset after a pause found near the CAPTION time

    @property
    def uncertain(self) -> bool:
        if self.energy is not None and abs(self.whisper - self.energy) * 1000 > UNCERTAIN_MS:
            return True
        # whisper says off, but a real onset sits right at the caption: whisper is the likelier error
        # (FEELS: whisper 1.18 inside an 'um', true onset 1.90, caption 1.76)
        return (abs(self.d_whisper) > FAIL_MS and self.energy_at_cap is not None
                and abs(self.cap - self.energy_at_cap) * 1000 <= CAP_NEAR_MS)

    @property
    def d_whisper(self) -> float:
        return (self.cap - self.whisper) * 1000

    @property
    def d_energy(self) -> float | None:
        return None if self.energy is None else (self.cap - self.energy) * 1000

    @property
    def verdict(self) -> str:
        if self.uncertain:
            return "uncertain"
        return "late" if self.d_whisper > FAIL_MS else "early" if self.d_whisper < -FAIL_MS else "ok"


def envelope_db(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """RMS in dB over a 50 ms window, one value per 10 ms."""
    w, h = int(ENV_WIN * sr), int(HOP * sr)
    if len(audio) < w:
        return np.array([])
    return np.array([20 * np.log10(np.sqrt(np.mean(audio[i:i + w] ** 2)) + 1e-9) for i in range(0, len(audio) - w, h)])


def energy_onset(env: np.ndarray, t: float, before: float = 0.45, after: float = 0.45) -> float | None:
    """Start of the speech burst nearest to t: the first 10 ms step in [t-before, t+after] where the
    envelope exceeds max(-40 dB, that window's peak - 25 dB) after a quieter stretch.
    Envelope index k describes the window [k*10 ms, +50 ms], so a burst that just reaches it starts at k*10 ms + 50 ms."""
    i0, i1 = max(0, int((t - before) / HOP)), min(len(env), int((t + after) / HOP))
    if i1 - i0 < 5:
        return None
    seg = env[i0:i1]
    if seg.max() < -40.0:
        return None
    thr = max(-40.0, float(seg.max()) - 25.0)
    # Only a burst that follows a real pause (>= 80 ms below thr) is an onset; inside continuous speech
    # the envelope has no word starts, and the word is judged by whisper alone (energy None).
    cands = [i for i in range(max(i0, QUIET_STEPS), i1)
             if env[i] >= thr and not (env[i - QUIET_STEPS:i] >= thr).any()]
    if not cands:
        return None
    i = min(cands, key=lambda i: abs(i * HOP - t))
    return i * HOP + ENV_WIN  # index i is a window START: a hit means the burst is at its far end


def caption_words(ass: Path) -> list[tuple[str, float]]:
    """(word, highlight start); one word split over two Dialogue lines counts once."""
    out, prev = [], None
    for line in ass.read_text().splitlines():
        if not line.startswith("Dialogue: 0,"):
            continue
        p = line.split(",", 9)
        m = re.search(r"\\c&H0000E0FF\}([^{]+)", p[9])
        if not m:
            continue
        h, mi, s = p[1].split(":")
        t0 = int(h) * 3600 + int(mi) * 60 + float(s)
        h, mi, s = p[2].split(":")
        t1 = int(h) * 3600 + int(mi) * 60 + float(s)
        key = re.sub(r"\{\\pos[^}]*\}", "", p[9])
        if prev and prev[0] == key and t0 - prev[1] < 0.03:
            prev = (key, t1)
            continue
        prev = (key, t1)
        out.append((m.group(1).strip(), t0))
    return out


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("*", ""))


def whisper_words(audio: np.ndarray) -> list[tuple[str, float]]:
    from faster_whisper import WhisperModel
    segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(
        audio, language="en", word_timestamps=True)
    return [(w.word, float(w.start)) for s in segs for w in (s.words or [])]


def pair(cap: list[tuple[str, float]], heard: list[tuple[str, float]]) -> list[tuple[int, int]]:
    """Index pairs (caption word, whisper word) of words present in both, in order."""
    a = [_norm(w) for w, _ in cap]
    b = [_norm(w) for w, _ in heard]
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return [(i1 + k, j1 + k) for tag, i1, i2, j1, _j2 in sm.get_opcodes() if tag == "equal" for k in range(i2 - i1)]


def assign_group_onsets(rows: list[Row]) -> list[Row]:
    """An energy onset after a pause belongs to ONE word: the burst's first word. Rows that found the same
    onset (within 20 ms) keep it only on the row whose whisper start is nearest to it; the others get None
    and are judged by whisper alone (clip-biralo #367: 1.31 x2, 24.74 x3, 16.74 x2, 21.24 x2)."""
    for attr in ("energy", "energy_at_cap"):
        ref = (lambda r: r.whisper) if attr == "energy" else (lambda r: r.cap)
        groups: list[list[Row]] = []
        for r in sorted((r for r in rows if getattr(r, attr) is not None), key=lambda r: getattr(r, attr)):
            if groups and abs(getattr(groups[-1][-1], attr) - getattr(r, attr)) <= 0.02:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:
            keep = min(g, key=lambda r: abs(ref(r) - getattr(g[0], attr)))
            for r in g:
                if r is not keep:
                    setattr(r, attr, None)
    return rows


def measure(video: Path, ass: Path) -> list[Row]:
    from .transcribe import load_audio
    from .qa import _duration
    audio = load_audio(video, 0.0, _duration(video))
    env = envelope_db(audio)
    cap, heard = caption_words(ass), whisper_words(audio)
    rows = []
    for i, j in pair(cap, heard):
        rows.append(Row(cap[i][0], cap[i][1], heard[j][1], energy_onset(env, heard[j][1]),
                        energy_onset(env, cap[i][1], 0.2, 0.2)))
    return assign_group_onsets(rows)


def summary(rows: list[Row]) -> str:
    sure = [r for r in rows if not r.uncertain]
    if not sure:
        return "no certain words"
    med = float(np.median([r.d_whisper for r in sure]))
    bad = [r for r in sure if r.verdict != "ok"]
    return (f"{len(rows)} words, {len(sure)} certain, {len(rows) - len(sure)} uncertain; median {med:+.0f} ms; "
            f"{len(bad)} beyond +-{FAIL_MS} ms")


def main(argv: list[str]) -> int:
    rows = measure(Path(argv[1]), Path(argv[2]))
    print("cap_s   whisper_s energy_s  d_whisper_ms d_energy_ms  verdict    word")
    for r in rows:
        en = "   -  " if r.energy is None else f"{r.energy:6.2f}"
        de = "    -" if r.d_energy is None else f"{r.d_energy:+5.0f}"
        print(f"{r.cap:6.2f}  {r.whisper:6.2f}    {en}    {r.d_whisper:+6.0f}       {de}      {r.verdict:9s}  {r.word}")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
