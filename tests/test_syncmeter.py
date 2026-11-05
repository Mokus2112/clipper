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
    assert sm.Row("A", 1.20, 1.0, 1.0).verdict == "ok"


def test_pairing_skips_words_missing_from_audio():
    cap = [("IS", 1.0), ("THAT", 1.2), ("THIS", 1.5)]
    heard = [("this", 0.9)]
    assert sm.pair(cap, heard) == [(2, 0)]


def test_whisper_error_inside_um_is_uncertain():
    # post-G alliance c1: whisper put FEELS at 1.18 (inside the 'um'), the real onset is 1.90, caption 1.76
    env = sm.envelope_db(_burst(4.0, 1.90, 2.6))
    r = sm.Row("FEELS", 1.76, 1.18, None, sm.energy_onset(env, 1.76, 0.2, 0.2))
    assert r.energy_at_cap is not None and r.uncertain and r.verdict == "uncertain"


def _row(word, cap, wh, en, ec=None):
    return sm.Row(word, cap, wh, en, ec)


def test_group_onset_goes_to_one_word_only():
    # clip-biralo #367: 24.74 repeated on THEY / SEEMED / COMPLETELY, 1.31 on THAT / FEELS
    rows = [_row("THEY", 24.73, 24.78, 24.74), _row("SEEMED", 24.87, 24.92, 24.74),
            _row("COMPLETELY", 25.02, 25.10, 24.74),
            _row("THAT", 0.93, 0.92, 1.31), _row("FEELS", 1.76, 1.18, 1.31),
            _row("OF", 16.35, 16.26, 16.74), _row("THEY2", 16.60, 16.56, 16.74)]
    sm.assign_group_onsets(rows)
    kept = {r.word: r.energy for r in rows}
    assert kept["THEY"] == 24.74 and kept["SEEMED"] is None and kept["COMPLETELY"] is None
    assert kept["FEELS"] == 1.31 and kept["THAT"] is None
    assert kept["THEY2"] == 16.74 and kept["OF"] is None
    values = [r.energy for r in rows if r.energy is not None]
    assert len(values) == len(set(values))


def test_word_without_onset_is_judged_by_whisper_alone():
    r = _row("SEEMED", 24.87, 24.92, None)
    assert not r.uncertain and r.verdict == "ok"
    late = _row("LOSS", 11.11, 10.60, None)
    assert late.verdict == "late"


def test_unmatched_caption_word_keeps_its_row(monkeypatch):
    """LOSS (alliance c1) is not found in the audio: it stays in the table as 'unmatched', not dropped."""
    import numpy as np
    from clipper import qa, transcribe
    monkeypatch.setattr(transcribe, "load_audio", lambda *a, **k: np.zeros(16000 * 3, dtype=np.float32))
    monkeypatch.setattr(qa, "_duration", lambda v: 3.0)
    monkeypatch.setattr(sm, "caption_words", lambda a: [("THE", 0.5), ("LOSS", 1.0), ("OF", 1.4)])
    monkeypatch.setattr(sm, "whisper_words", lambda a: [("the", 0.5), ("of", 1.4)])
    rows = sm.measure(None, None)
    assert [r.word for r in rows] == ["THE", "LOSS", "OF"]
    assert [r.verdict for r in rows] == ["ok", "unmatched", "ok"]
    assert not rows[1].uncertain and rows[1].d_energy is None
    assert "1 unmatched" in sm.summary(rows)


def test_qa_counts_unmatched_apart_from_late_and_early(monkeypatch):
    from clipper import qa
    rows = [sm.Row("A", 1.0, 1.0, None), sm.Row("LOSS", 2.0, None, None),
            sm.Row("B", 3.5, 3.0, 3.0), sm.Row("C", 4.0, None, None)]  # 50% unmatched
    monkeypatch.setattr(sm, "measure", lambda v, a: rows)
    out = qa.check_caption_sync_words(None, None)
    assert len(out) == 2
    assert out[0].startswith("caption sync (whisper+energy): 1 words") and "LOSS" not in out[0]
    assert out[1].startswith("caption sync unmatched: 2 of 4") and "LOSS@2.00" in out[1] and "C@4.00" in out[1]
    monkeypatch.setattr(sm, "measure", lambda v, a: rows[:1])
    assert qa.check_caption_sync_words(None, None) == []


def _rows(pattern):
    return [sm.Row(f"W{i}", float(i), None if c == "u" else float(i), None) for i, c in enumerate(pattern)]


def test_unmatched_threshold_is_ratio_only():
    assert not sm.unmatched_alarming(_rows("o" * 19 + "u"))        # 5%
    assert not sm.unmatched_alarming(_rows("o" * 9 + "u"))         # exactly 10% is not over
    assert sm.unmatched_alarming(_rows("oouoooooouoo"))             # 2 of 12, 17%
    assert not sm.unmatched_alarming(_rows("o" * 40 + "uuu" + "o" * 40))  # 3 in a row, 3.6%: no run rule
    assert not sm.unmatched_alarming([])


def test_qa_stays_silent_below_unmatched_threshold(monkeypatch):
    from clipper import qa
    monkeypatch.setattr(sm, "measure", lambda v, a: _rows("o" * 30 + "u"))
    assert qa.check_caption_sync_words(None, None) == []


def test_whisper_only_word_uses_noise_floor_threshold():
    from clipper.syncmeter import Row
    # no energy onset: 230 ms is inside whisper's own +-200 ms jitter, not a slip (Codee #880)
    assert Row("SEPARATE", 3.73, 4.10, None).verdict == "ok"
    assert Row("SEPARATE", 3.73, 4.20, None).verdict == "early"


def test_energy_confirmed_word_limit_is_250ms():
    from clipper.syncmeter import Row
    assert Row("X", 5.00, 5.30, 5.29).verdict == "early"
    assert Row("X", 5.00, 5.20, 5.19).verdict == "ok"


def test_few_off_words_are_a_warning_not_a_failure(monkeypatch):
    from clipper import qa
    rows = _rows("o" * 30)
    rows[3] = sm.Row("OFF", 3.5, 3.0, 3.0)  # 1 of 30 words 500 ms late, median stays 0
    monkeypatch.setattr(sm, "measure", lambda v, a: rows)
    out = qa.check_caption_sync_words(None, None)
    assert len(out) == 1 and out[0].startswith(qa.WARN)


def test_many_off_words_still_fail(monkeypatch):
    from clipper import qa
    rows = _rows("o" * 20)
    for i in range(4):  # 20% of the words
        rows[i] = sm.Row(f"OFF{i}", i + 0.5, float(i), float(i))
    monkeypatch.setattr(sm, "measure", lambda v, a: rows)
    out = qa.check_caption_sync_words(None, None)
    assert len(out) == 1 and not out[0].startswith(qa.WARN)
