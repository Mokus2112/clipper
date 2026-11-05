"""Clip end: about 0.5 s of room after the speech, never into the next word, never in sound (klip template)."""
import numpy as np

from clipper.render import END_QUIET_DB, END_TAIL, extend_tail, snap_to_words

SR = 16000


def audio(speech, total=4.0, noise=1e-4, amp=0.1, seed=3):
    rng = np.random.default_rng(seed)
    a = noise * rng.standard_normal(int(total * SR)).astype(np.float32)
    for t0, t1 in speech:
        a[int(t0 * SR):int(t1 * SR)] = amp * rng.standard_normal(int((t1 - t0) * SR))
    return lambda t0, d: a[int(t0 * SR):int((t0 + d) * SR)]


def test_end_gets_about_half_a_second_after_the_speech():
    fn = audio([(0.2, 1.4)])
    words = [{"w": "so", "s": 0.2, "e": 0.6}, {"w": "thing.", "s": 0.6, "e": 1.4}]
    _, e0 = snap_to_words(words, 0.2, 1.4, sentence_end=True, audio_fn=fn)
    _, e1 = snap_to_words(words, 0.2, 1.4, sentence_end=True, audio_fn=fn, tail=END_TAIL)
    assert 1.4 + 0.45 <= e1 <= 1.4 + 0.55 and e1 > e0 + 0.2


def test_tail_stops_before_the_next_word():
    fn = audio([(0.2, 1.4), (1.7, 2.5)])
    words = [{"w": "so", "s": 0.2, "e": 0.6}, {"w": "thing.", "s": 0.6, "e": 1.4}, {"w": "And", "s": 1.7, "e": 2.0}]
    _, e1 = snap_to_words(words, 0.2, 1.4, sentence_end=True, audio_fn=fn, tail=END_TAIL)
    assert e1 <= 1.7 - 0.05 + 1e-9


def test_tail_never_ends_in_sound():
    fn = audio([(0.2, 1.4), (1.7, 2.5)])
    e = extend_tail(fn, 1.43, 0.5, limit=None)  # 1.9 would be inside the next sentence
    a = fn(e - 0.06, 0.06)
    assert 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9) <= END_QUIET_DB
    assert e < 1.7


def test_no_quiet_point_keeps_the_end():
    fn = audio([(0.0, 4.0)])
    assert extend_tail(fn, 1.0, 0.5) == 1.0


def test_default_snap_has_no_tail():
    fn = audio([(0.2, 1.4)])
    words = [{"w": "so", "s": 0.2, "e": 0.6}, {"w": "thing.", "s": 0.6, "e": 1.4}]
    assert snap_to_words(words, 0.2, 1.4, sentence_end=True, audio_fn=fn) == \
        snap_to_words(words, 0.2, 1.4, sentence_end=True, audio_fn=fn, tail=0.0)
