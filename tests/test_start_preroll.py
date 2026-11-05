"""The first sound comes at least 60 ms after the first frame (klip template 60-100 ms), through silence only."""
import numpy as np

from clipper.render import START_PRE, extend_preroll, snap_to_words

SR = 16000


def audio(speech, total=3.0, noise=1e-4, amp=0.1, seed=5):
    rng = np.random.default_rng(seed)
    a = noise * rng.standard_normal(int(total * SR)).astype(np.float32)
    for t0, t1 in speech:
        i0, i1 = round(t0 * SR), round(t1 * SR)
        a[i0:i1] = amp * rng.standard_normal(i1 - i0)
    return lambda t0, d: a[int(t0 * SR):int((t0 + d) * SR)]


def onset_of(fn, s):
    a = fn(s, 0.3)
    for k in range(len(a) // 160):
        if 20 * np.log10(np.sqrt(np.mean(a[k * 160:(k + 1) * 160] ** 2)) + 1e-9) > -35.0:
            return k * 0.01
    return None


def test_late_cut_moves_earlier_until_the_onset_is_60_ms_in():
    fn = audio([(1.03, 2.0)])
    s = extend_preroll(fn, 1.0, START_PRE, floor=0.5)
    assert 0.0 < 1.0 - s <= 0.04 and abs(onset_of(fn, s) - START_PRE) <= 0.011


def test_enough_room_already_changes_nothing():
    fn = audio([(1.1, 2.0)])
    assert extend_preroll(fn, 1.0, START_PRE, floor=0.5) == 1.0


def test_never_before_the_previous_word():
    fn = audio([(1.03, 2.0)])
    assert extend_preroll(fn, 1.0, START_PRE, floor=0.995) >= 0.995 - 1e-9


def test_never_into_sound():
    fn = audio([(0.9, 0.99), (1.03, 2.0)])  # the previous word's tail sits 10 ms before the cut
    s = extend_preroll(fn, 1.0, START_PRE, floor=0.5)
    assert s >= 0.99 - 1e-9


def test_snap_to_words_keeps_60_ms_before_the_first_sound():
    fn = audio([(0.62, 2.0)])
    words = [{"w": "I", "s": 0.62, "e": 0.7}, {"w": "think.", "s": 0.7, "e": 1.9}]
    s0, _ = snap_to_words(words, 0.62, 1.9, sentence_end=True, audio_fn=fn)
    assert onset_of(fn, s0) >= START_PRE - 0.011


def test_snap_floor_is_the_word_before_the_cut_not_the_word_before_the_first_kept_one():
    # "I" (0.72-0.74) is already inside the segment, only "saying." (ends 0.5) precedes the cut
    fn = audio([(0.72, 2.0)])
    words = [{"w": "saying.", "s": 0.2, "e": 0.5}, {"w": "I", "s": 0.72, "e": 0.74}, {"w": "think.", "s": 0.78, "e": 1.9}]
    s0, _ = snap_to_words(words, 0.76, 1.9, sentence_end=True, audio_fn=fn)
    assert onset_of(fn, s0) >= START_PRE - 0.011 and s0 >= 0.5
