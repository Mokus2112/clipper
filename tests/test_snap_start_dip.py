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
