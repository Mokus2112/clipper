import numpy as np

from clipper.render import snap_to_words


def test_start_in_connected_speech_moves_to_dip():
    """Word start is late (inside speech, no 60 ms pause): the cut moves to the last real dip before it."""
    sr = 16000
    a = np.zeros(int(3 * sr), dtype=np.float32)
    rng = np.random.default_rng(0)
    a[:] = 0.0001 * rng.standard_normal(len(a))
    a[int(0.2 * sr):int(1.5 * sr)] = 0.1 * rng.standard_normal(int(1.3 * sr))  # speech from 0.2 s
    a[int(0.66 * sr):int(0.70 * sr)] = 0.0003 * rng.standard_normal(int(0.04 * sr))  # 40 ms dip
    audio_fn = lambda t0, d: a[int(t0 * sr):int((t0 + d) * sr)]  # noqa: E731
    words = [{"w": "Um,", "s": 0.24, "e": 0.52}, {"w": "so", "s": 0.88, "e": 0.92},
             {"w": "one", "s": 0.94, "e": 1.1}, {"w": "thing.", "s": 1.1, "e": 1.4}]
    s0, _ = snap_to_words(words, 0.88, 1.4, sentence_end=True, audio_fn=audio_fn)
    assert 0.6 < s0 < 0.72, s0
    db = 20 * np.log10(np.sqrt(np.mean(audio_fn(s0, 0.03) ** 2)) + 1e-9)
    assert db < -30.0, db


def test_start_snap_reaches_a_dip_half_a_second_back():
    # Cotra cold open cut at 60.48 sat inside a word: the nearest silence was 0.45 s away (Codee #929: +-0.5 s)
    sr = 16000
    rng = np.random.default_rng(1)
    a = 0.0001 * rng.standard_normal(3 * sr).astype(np.float32)
    a[int(0.2 * sr):int(1.8 * sr)] = 0.1 * rng.standard_normal(int(1.6 * sr))
    a[int(0.5 * sr):int(0.54 * sr)] = 0.0003 * rng.standard_normal(int(0.04 * sr))  # dip 0.45 s before the start
    audio_fn = lambda t0, d: a[int(t0 * sr):int((t0 + d) * sr)]  # noqa: E731
    words = [{"w": "so", "s": 0.95, "e": 1.1}, {"w": "one", "s": 1.1, "e": 1.4}]
    s0, _ = snap_to_words(words, 0.95, 1.5, sentence_end=True, audio_fn=audio_fn)
    assert 0.48 < s0 < 0.56, s0


def test_end_snap_falls_back_to_a_quiet_point_up_to_half_a_second_back():
    # speech runs on past the last word's reported end (late aligner end): cut at the nearest quiet point behind it
    sr = 16000
    rng = np.random.default_rng(2)
    a = 0.1 * rng.standard_normal(4 * sr).astype(np.float32)
    a[int(1.25 * sr):int(1.32 * sr)] = 0.0001 * rng.standard_normal(int(0.07 * sr))
    audio_fn = lambda t0, d: a[int(t0 * sr):int((t0 + d) * sr)]  # noqa: E731
    words = [{"w": "so", "s": 0.3, "e": 0.5}, {"w": "thing.", "s": 0.5, "e": 1.6}]
    _, e1 = snap_to_words(words, 0.3, 1.6, sentence_end=True, audio_fn=audio_fn)
    assert 1.25 < e1 < 1.34, e1
