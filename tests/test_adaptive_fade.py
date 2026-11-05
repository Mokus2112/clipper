"""The fade-in must finish before the first word starts, not cut into it (alliance c1: 'I' lost its onset)."""
import numpy as np
import pytest

from clipper.render import AUDIO_FADE_IN, FADE_IN_MIN, adaptive_fade_in

SR = 16000


def audio_with_onset(onset: float, total: float = 0.3, quiet_amp: float = 1e-4, loud_amp: float = 0.1):
    def fn(t0, d):
        n = int(d * SR)
        t = t0 + np.arange(n) / SR
        return np.where(t >= onset, loud_amp, quiet_amp).astype(np.float32) * np.sin(2 * np.pi * 200 * t)
    return fn


def test_silent_start_keeps_the_default_fade():
    assert adaptive_fade_in(audio_with_onset(0.10), 0.0) == AUDIO_FADE_IN


def test_speech_from_the_first_sample_gets_the_minimum_fade():
    assert adaptive_fade_in(audio_with_onset(-1.0), 0.0) == FADE_IN_MIN


def test_onset_inside_the_fade_ends_the_fade_at_the_onset():
    f = adaptive_fade_in(audio_with_onset(0.03), 0.0)
    assert FADE_IN_MIN < f <= 0.03 and f < AUDIO_FADE_IN


def test_fade_never_below_minimum_never_above_default():
    for onset in (-1.0, 0.0, 0.005, 0.02, 0.05, 0.2):
        f = adaptive_fade_in(audio_with_onset(onset), 0.0)
        assert FADE_IN_MIN <= f <= AUDIO_FADE_IN


def test_no_audio_falls_back_to_default():
    assert adaptive_fade_in(None, 0.0) == AUDIO_FADE_IN



def test_onset_exactly_on_a_10_ms_window_edge():
    assert adaptive_fade_in(audio_with_onset(0.01), 0.0) == pytest.approx(0.01)
    assert adaptive_fade_in(audio_with_onset(0.04), 0.0) == pytest.approx(AUDIO_FADE_IN)


def test_audio_shorter_than_one_window_gives_the_default():
    for n in (0, 1, 100, 159):
        assert adaptive_fade_in(lambda t0, d, n=n: np.zeros(n, dtype=np.float32), 0.0) == AUDIO_FADE_IN
