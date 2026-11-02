import numpy as np

from clipper import syncmeter as sm


def _burst(dur, t0, t1, sr=16000):
    a = np.zeros(int(dur * sr), np.float32)
    n = int((t1 - t0) * sr)
    a[int(t0 * sr):int(t0 * sr) + n] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)
    return a


def test_energy_onset_finds_burst_after_digital_silence():
    env = sm.envelope_db(_burst(4.0, 1.85, 2.6))
    assert abs(sm.energy_onset(env, 1.60) - 1.85) < 0.03


def test_feels_case_is_uncertain_not_late():
    # clip-biralo #350: FEELS caption 1.76, whisper 1.20 (+560 ms), but speech starts at 1.85
    env = sm.envelope_db(_burst(4.0, 1.85, 2.6))
    r = sm.Row("FEELS", 1.76, 1.20, sm.energy_onset(env, 1.20, after=0.8))
    assert r.energy is not None and abs(r.energy - 1.85) < 0.03
    assert r.uncertain and r.verdict == "uncertain"
    assert round(r.d_energy) in range(-110, -70)  # the caption is really ~90 ms EARLY


def test_agreeing_whisper_and_energy_gives_late_verdict():
    env = sm.envelope_db(_burst(4.0, 1.20, 2.0))
    r = sm.Row("FEELS", 1.76, 1.20, sm.energy_onset(env, 1.20))
    assert not r.uncertain and r.verdict == "late" and 540 < r.d_whisper < 580


def test_ok_within_tolerance():
    assert sm.Row("A", 1.05, 1.0, 1.0).verdict == "ok"


def test_pairing_skips_words_missing_from_audio():
    cap = [("IS", 1.0), ("THAT", 1.2), ("THIS", 1.5)]
    heard = [("this", 0.9)]
    assert sm.pair(cap, heard) == [(2, 0)]


def test_whisper_error_inside_um_is_uncertain():
    # post-G alliance c1: whisper put FEELS at 1.18 (inside the 'um'), the real onset is 1.90, caption 1.76
    env = sm.envelope_db(_burst(4.0, 1.90, 2.6))
    r = sm.Row("FEELS", 1.76, 1.18, None, sm.energy_onset(env, 1.76, 0.2, 0.2))
    assert r.energy_at_cap is not None and r.uncertain and r.verdict == "uncertain"
