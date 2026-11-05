from clipper.captionfix import retime_words

SEGS = [(14.64, 20.40, 0.0)]


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


def test_feels_after_um_is_moved_to_the_heard_start():
    # alliance c1 (clip-biralo #367): aligned FEELS 16.52 -> output 1.88; heard at 1.30
    words = [w("that,", 15.66, 16.0), w("feels", 16.52, 16.92), w("especially", 16.92, 17.8)]
    heard = [("that", 0.95), ("feels", 1.30), ("especially", 1.88)]
    out, moved = retime_words(words, SEGS, heard)
    assert abs(out[1]["s"] - (14.64 + 1.30)) < 1e-6 and out[1]["e"] > out[1]["s"]
    assert any(m.startswith("feels") for m in moved)
    assert out[2]["s"] == 16.92 - (2.28 - 1.88) + 0.0 or abs(out[2]["s"] - (14.64 + 1.88)) < 1e-6


def test_small_differences_and_mismatches_untouched():
    words = [w("this", 15.0, 15.3), w("whole", 16.0, 16.4)]
    heard = [("this", 0.30), ("whole", 3.0)]  # 60 ms off: noise; 1.6 s off: wrong match
    out, moved = retime_words(words, SEGS, heard)
    assert out == words and moved == []


def test_input_not_mutated():
    words = [w("feels", 16.52, 16.92)]
    retime_words(words, SEGS, [("feels", 1.30)])
    assert words[0]["s"] == 16.52


def test_bias_places_word_where_build_ass_shift_lands_it_on_the_heard_start():
    words = [w("feels", 16.52, 16.92)]
    out, _ = retime_words(words, SEGS, [("feels", 1.30)], bias=0.07)
    assert abs((out[0]["s"] - 0.07) - (14.64 + 1.30)) < 1e-6


def test_moved_word_is_held_until_the_next_word():
    words = [w("incident", 20.98, 21.30), w("is", 21.50, 21.80)]
    # incident heard at 5.46 (caption 6.34 -> moved), "is" heard 6.22
    out, _ = retime_words(words, [(14.64, 30.0, 0.0)], [("incident", 5.46), ("is", 6.22)])
    assert abs(out[0]["e"] - out[1]["s"]) < 1e-6


def test_word_moved_into_another_segment_is_left_alone():
    # alliance 162635 c2: 'And' (segment 1) was heard in the tail of segment 0's output span and moved there,
    # outside its own segment: build_ass then dropped it and QA saw a caption gap.
    segs = [(10.0, 11.0, 0.0), (11.5, 12.5, 1.0)]
    words = [w("and", 11.52, 11.8)]
    out, moved = retime_words(words, segs, [("and", 0.80)])
    assert out == words and moved == []


def test_held_word_stays_inside_its_own_segment():
    # a word held until the next moved word must not get its middle past the segment end (dropped by build_ass)
    segs = [(10.0, 11.0, 0.0), (11.1, 12.5, 1.0)]
    words = [w("necessary", 10.7, 10.95), w("and", 11.12, 11.4)]
    out, moved = retime_words(words, segs, [("necessary", 0.70), ("and", 1.25)])
    assert moved and out[1]["s"] > 11.3
    assert out[0]["e"] <= 11.0 and (out[0]["s"] + out[0]["e"]) / 2 <= 11.0


def test_cold_open_replay_is_retimed_per_occurrence():
    # alliance-secure-ai 000937 c1: "a completely universal way" is the cold open AND replayed in the main
    # segment. The same word entry has two output times; each occurrence must follow its own heard time
    # (the main "A" captioned 380 ms late left a caption gap under speech).
    segs = [(59.0, 60.0, 0.0), (50.0, 70.0, 1.0)]
    words = [w("a", 59.1, 59.3), w("completely", 59.3, 59.8)]
    # cold open: a heard 0.10 (caption 0.10 -> fine), completely 0.30. Main: a at 1.0+9.1=10.1 heard 9.6.
    heard = [("a", 0.10), ("completely", 0.30), ("a", 9.6), ("completely", 10.35)]
    out, moved = retime_words(words, segs, heard)
    a = out[0]
    assert abs(a["ov"][1][0] - (59.1 - (10.1 - 9.6) + 0.0)) < 1e-6
    assert abs(a["ov"][0][0] - 59.1) < 1e-6  # cold open occurrence untouched
    assert moved and moved[0].startswith("a@10.10")


def test_one_letter_word_skipped_by_the_sequence_match_is_retimed():
    # alliance cotra c2: caption "A" lit at 11.20, spoken at 10.82; the neighbour changed, so SequenceMatcher skipped it
    segs = [(0.0, 20.0, 0.0)]
    words = [w("proposes", 9.8, 10.39), w("a", 11.20, 11.33), w("completely", 11.33, 11.74), w("way", 12.38, 12.85)]
    heard = [("proposes", 9.84), (" a", 10.82), ("complete", 11.36), ("way", 12.4)]
    out, moved = retime_words(words, segs, heard)
    assert abs(out[1]["s"] - 10.82) < 0.01 and any(m.startswith("a@") for m in moved)


def test_leftover_pass_never_pairs_beyond_the_mismatch_window():
    segs = [(0.0, 20.0, 0.0)]
    words = [w("a", 5.0, 5.1), w("x", 6.0, 6.2)]
    heard = [("a", 8.0), ("y", 6.0)]
    out, moved = retime_words(words, segs, heard)
    assert out[0]["s"] == 5.0 and moved == []


# --- energy-onset retime (Codee #908) ---
import numpy as np

from clipper.captionfix import HOLE_MAX, MIN_WORD, retime_words as _retime


def _env(n_s, speech):
    """10 ms dB envelope: -60 dB silence, -20 dB inside the given (start, end) output spans."""
    env = np.full(int(n_s * 100), -60.0)
    for a, b in speech:
        env[int((a - 0.05) * 100):int((b - 0.05) * 100)] = -20.0
    return env


def _w(w, s, e):
    return {"w": w, "s": s, "e": e}


def test_hole_with_speech_energy_is_closed():
    # 010609-c1 gap 12.48-12.85 (0.37 s, under HOLE_MAX): speech runs through it, the caption must hold on
    words = [_w("this", 12.10, 12.48), _w("phase", 12.85, 13.30)]
    env = _env(14, [(12.0, 13.4)])
    out, _ = _retime(words, [(0.0, 14.0, 0.0)], [], env=env)
    assert out[0]["e"] == 12.85


def test_hole_in_silence_stays_open():
    # THEN case: no speech energy in the gap, so no caption may be drawn there
    words = [_w("and", 29.20, 29.50), _w("then", 29.85, 30.20)]
    env = _env(31, [(29.1, 29.5), (29.85, 30.3)])
    out, _ = _retime(words, [(0.0, 31.0, 0.0)], [], env=env)
    assert out[0]["e"] == 29.50


def test_long_gap_with_full_speech_is_closed():
    words = [_w("a", 19.0, 19.93), _w("b", 20.50, 21.0)]  # 0.57 s > HOLE_MAX, untouched words
    env = _env(22, [(19.0, 21.1)])
    out, _ = _retime(words, [(0.0, 22.0, 0.0)], [], env=env)
    assert 0.57 > HOLE_MAX and out[0]["e"] == 20.50  # speech fills it (drawn-out word tail): closed up to 0.8 s


def test_long_gap_mostly_silent_stays_open():
    words = [_w("a", 19.0, 19.93), _w("b", 20.50, 21.0)]
    env = _env(22, [(19.0, 20.1), (20.45, 21.1)])  # ~0.17 s of quiet in the gap: not filled
    out, _ = _retime(words, [(0.0, 22.0, 0.0)], [], env=env)
    assert out[0]["e"] == 19.93


def test_snap_to_onset_within_window_and_flag_outside():
    words = [_w("hello", 5.0, 5.4), _w("world", 6.0, 6.4)]
    # speech burst for 'hello' starts at 5.7; whisper said 5.55 (150 ms off, inside the window)
    env = _env(8, [(5.7, 6.1)])
    out, moved = _retime(words, [(0.0, 8.0, 0.0)], [("hello", 5.55)], env=env)
    assert abs(out[0]["s"] - (5.7 + 0.0)) < 0.06 and "~" not in moved[0]
    # nearest onset is 0.7 s from the whisper start: whisper time kept and flagged
    out2, moved2 = _retime(words, [(0.0, 8.0, 0.0)], [("hello", 5.6)], env=_env(8, [(6.3, 6.7)]))
    assert moved2 and moved2[0].endswith("~")


def test_snap_never_swaps_words():
    words = [_w("one", 2.0, 2.3), _w("two", 2.35, 2.6)]
    env = _env(4, [(2.7, 3.0)])  # single onset: both words would snap onto it
    out, _ = _retime(words, [(0.0, 4.0, 0.0)], [("one", 2.7), ("two", 2.72)], env=env)
    assert out[0]["s"] < out[1]["s"] and out[1]["s"] - out[0]["s"] >= MIN_WORD - 1e-9


def test_word_spoken_before_a_join_moves_into_the_previous_segment():
    # 011812-c1: "a" aligned at source 9.83 (segment 2) but spoken at output 9.74, the tail of segment 1
    segs = [(0.0, 10.0, 0.0), (10.0, 14.0, 10.0)]
    words = [_w("proposes", 8.0, 9.0), _w("a", 10.3, 10.45), _w("completely", 10.6, 11.2)]
    heard = [("proposes", 8.0), ("a", 9.55), ("completely", 10.6)]
    env = _env(15, [(7.9, 9.0), (9.5, 9.8), (10.55, 11.3)])
    out, moved = _retime(words, segs, heard, bias=0.07, env=env)
    assert out[1]["s"] < 10.0
    assert any(m.startswith("a@") for m in moved)
