import sys
import types

import numpy as np

from clipper import transcribe as T


def test_mlx_words_use_window_offset_and_skip_blank_words(monkeypatch):
    calls = {}

    def fake_transcribe(audio, **kw):
        calls.update(kw)
        return {"segments": [{"words": [{"word": " Hello", "start": 0.5, "end": 0.9},
                                        {"word": " ", "start": 1.0, "end": 1.1},
                                        {"word": " world.", "start": 1.2, "end": 1.7}]}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    m = T.MlxClipModel("repo/x")
    words = m.words(np.zeros(16000, np.float32), 10.0, prompt="p", language="en", vad=False, condition=False)
    assert words == [{"w": "Hello", "s": 10.5, "e": 10.9}, {"w": "world.", "s": 11.2, "e": 11.7}]
    assert calls["word_timestamps"] and calls["initial_prompt"] == "p" and calls["language"] == "en"
    assert calls["condition_on_previous_text"] is False and calls["path_or_hf_repo"] == "repo/x"


def test_faster_words_pass_vad_flag():
    seen = {}

    class W:
        def __init__(self, w, s, e):
            self.word, self.start, self.end = w, s, e

    class M:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return [types.SimpleNamespace(words=[W(" Hi", 1.0, 1.2), W(" ", 1.3, 1.4)])], None

    words = T.FasterClipModel(M()).words(np.zeros(10, np.float32), 5.0, prompt="p", language=None, vad=False)
    assert words == [{"w": "Hi", "s": 6.0, "e": 6.2}] and seen["vad_filter"] is False


def test_clip_model_falls_back_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)  # import raises ImportError
    monkeypatch.setattr(T, "model", lambda cfg, name: ("faster", name))
    m = T.clip_model({"whisper": {"clip_backend": "mlx", "clip_model": "medium"}})
    assert isinstance(m, T.FasterClipModel) and m.m == ("faster", "medium")


def test_clip_model_default_is_faster(monkeypatch):
    monkeypatch.setattr(T, "model", lambda cfg, name: ("faster", name))
    assert isinstance(T.clip_model({"whisper": {"clip_model": "medium"}}), T.FasterClipModel)
