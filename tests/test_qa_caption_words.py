"""One caption word that is not heard in the audio must fail QA (mlx can invent words)."""
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


def test_single_invented_word_fails(monkeypatch):
    heard = [("the", 0.0), ("senator", 0.5), ("said", 1.0)]
    out = _run(monkeypatch, heard, [("senator", 0.5), ("MOSSAD", 1.0)])
    assert out and "MOSSAD" in out[0]


def test_all_words_heard_passes(monkeypatch):
    heard = [("the", 0.0), ("senator", 0.5), ("said", 1.0)]
    assert _run(monkeypatch, heard, [("senator", 0.5)]) == []
