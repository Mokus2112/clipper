"""Caption words not heard in the audio: 3+ or >10% fail, fewer is a WARN (Codee #1310)."""
import sys
import types
from pathlib import Path

from clipper import qa


class _W:
    def __init__(self, word, start):
        self.word, self.start, self.end = word, start, start + 0.3


class _Seg:
    def __init__(self, words):
        self.words = words


def _run(monkeypatch, heard_words, caption_words):
    class FakeModel:
        def __init__(self, *a, **k): pass
        def transcribe(self, *a, **k):
            return [_Seg([_W(w, t) for w, t in heard_words])], None

    fake = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr("clipper.transcribe.load_audio", lambda *a, **k: [0.0])
    monkeypatch.setattr(qa, "_duration", lambda v: 5.0)
    monkeypatch.setattr(qa, "_highlights", lambda ass: [(t, t + 0.3, w) for w, t in caption_words])
    return qa.check_caption_words(Path("x.mp4"), Path("x.ass"))


def test_invented_word_in_a_short_caption_fails(monkeypatch):
    heard = [("the", 0.0), ("senator", 0.5), ("said", 1.0)]
    out = _run(monkeypatch, heard, [("senator", 0.5), ("MOSSAD", 1.0)])
    assert out and "MOSSAD" in out[0]


def test_all_words_heard_passes(monkeypatch):
    heard = [("the", 0.0), ("senator", 0.5), ("said", 1.0)]
    assert _run(monkeypatch, heard, [("senator", 0.5)]) == []


def test_split_number_is_not_missing(monkeypatch):
    heard = [("agents", 0.0), ("1", 0.5), (",200", 0.56), ("sent", 1.0), ("70", 1.5), (",000", 1.9)]
    assert _run(monkeypatch, heard, [("1,200", 0.5), ("70,000", 1.5)]) == []


def test_one_unheard_word_in_a_long_caption_is_only_a_warning(monkeypatch):
    names = [f"word{i}x" for i in range(20)]
    heard = [(n, i * 0.5) for i, n in enumerate(names)]
    out = _run(monkeypatch, heard, [(n, i * 0.5) for i, n in enumerate(names)] + [("MOSSAD", 11.0)])
    assert len(out) == 1 and out[0].startswith(qa.WARN) and "MOSSAD" in out[0]


def test_three_unheard_words_fail(monkeypatch):
    names = [f"word{i}x" for i in range(40)]
    heard = [(n, i * 0.5) for i, n in enumerate(names)]
    caps = [(n, i * 0.5) for i, n in enumerate(names)] + [("AAAA", 21.0), ("BBBB", 22.0), ("CCCC", 23.0)]
    out = _run(monkeypatch, heard, caps)
    assert len(out) == 1 and not out[0].startswith(qa.WARN)
