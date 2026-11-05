"""Forced alignment of the transcript words to the audio (torchaudio MMS_FA, offline, CPU).

Whisper gives good text but unstable word times (0.2-0.6 s off, both directions). The caption text
stays Whisper's; the times come from CTC forced alignment of exactly those words on the audio.
"""

import re

import numpy as np

SR = 16000
_MODEL = {}

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num(n % 100))
    if n < 1_000_000:
        return _num(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _num(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))


def spoken(word: str) -> list[str]:
    """A transcript token as the letter sequences the aligner can match ('50%' -> fifty percent)."""
    w = word.lower().replace("’", "'")
    w = w.replace("%", " percent").replace("&", " and ")
    cur = "$" in w
    w = w.replace("$", " ")
    w = re.sub(r"\d[\d,]*\d|\d", lambda m: " " + _num(int(m.group().replace(",", ""))) + " ", w)
    if cur:  # "$20,000" is said "twenty thousand dollars"
        w += " dollars"
    parts = [re.sub(r"[^a-z']", "", p).strip("'") for p in w.split()]
    return [p for p in parts if p]


def _load():
    if not _MODEL:
        import torch
        from torchaudio.pipelines import MMS_FA as bundle
        torch.set_num_threads(8)
        _MODEL["model"] = bundle.get_model(with_star=True).eval()
        _MODEL["tok"] = bundle.get_tokenizer()
        _MODEL["align"] = bundle.get_aligner()
    return _MODEL


def align_words(words: list[dict], audio: np.ndarray, t0: float) -> list[dict] | None:
    """New word list with times from forced alignment of `words` on `audio` (16 kHz, starting at t0).

    Tokens without letters (e.g. '-') keep their neighbours' times. Returns None if alignment fails.
    """
    import torch
    m = _load()
    pieces, owner = [], []
    texts = [w["w"] for w in words]
    for i in range(len(texts) - 1, 0, -1):  # Whisper splits "$20,000." into "$20" + ",000."
        if re.match(r"^,\d", texts[i]) and re.search(r"\d$", texts[i - 1]):
            texts[i - 1] += texts[i]
            texts[i] = ""
    for i, w in enumerate(words):
        for p in spoken(texts[i]):
            pieces.append(p)
            owner.append(i)
    if not pieces:
        return None
    try:
        with torch.inference_mode():
            emission, _ = m["model"](torch.from_numpy(audio.astype(np.float32)).unsqueeze(0))
            spans = m["align"](emission[0], m["tok"](pieces))
    except Exception:
        return None
    ratio = len(audio) / emission.shape[1] / SR
    out = [dict(w) for w in words]
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    for p_idx, sp in enumerate(spans):
        i = owner[p_idx]
        s, e = t0 + sp[0].start * ratio, t0 + sp[-1].end * ratio
        first[i] = min(first.get(i, s), s)
        last[i] = max(last.get(i, e), e)
    for i, w in enumerate(out):
        if i in first:
            w["s"], w["e"] = round(first[i], 3), round(last[i], 3)
        else:  # no letters: sit between the neighbours
            prev_e = out[i - 1]["e"] if i > 0 else w["s"]
            w["s"] = w["e"] = prev_e
    return out


def outliers(aligned: list[dict], ref: list[dict], limit: float = 1.0) -> list[dict]:
    """Words whose aligned start is more than `limit` s from Whisper's start for the same word.

    Whisper drifts 0.2-0.6 s, so a bigger gap means one of the two is wrong (e.g. the aligner put
    'when he' 2.2 s late). It cannot tell which, so callers only report it.
    """
    if len(aligned) != len(ref):
        return []
    return [{"w": a["w"], "aligned": a["s"], "whisper": r["s"], "diff": round(a["s"] - r["s"], 2)}
            for a, r in zip(aligned, ref) if abs(a["s"] - r["s"]) > limit]
