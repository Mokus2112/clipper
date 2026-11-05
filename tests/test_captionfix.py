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
    env = _env(22, [(19.0, 19.9), (20.45, 21.1)])  # the gap is quiet: not filled
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


def test_replayed_word_moves_into_previous_segment_and_is_dropped_from_old_one():
    # 012313-c1: "a" is in the cold open (segment 0) and aligned past the join into segment 2, but spoken
    # at the tail of segment 1: only the segment-2 occurrence moves, ov[2] = None (not drawn there)
    segs = [(9.0, 10.5, 0.0), (0.0, 10.0, 1.5), (10.0, 14.0, 11.5)]
    words = [{"w": "a", "s": 10.3, "e": 10.45}]
    env = _env(16, [(1.25, 1.5), (11.3, 11.5)])
    out, moved = _retime(words, segs, [("a", 1.3), ("a", 11.35)], bias=0.07, env=env)
    ov = out[0]["ov"]
    assert ov[2] is None and ov[1][0] < 10.0 and 0 in ov
    assert len(moved) == 1


def test_word_tail_run_is_covered_but_silence_after_it_is_not():
    # 012313-c1 7.46-7.83: speech runs 0.3 s on from the caption end, then 0.1+ s of quiet before the next word
    words = [_w("established", 6.0, 7.45), _w("an", 8.20, 8.40)]
    env = _env(10, [(5.9, 7.80), (8.15, 8.5)])
    env[int(7.80 * 100):int(8.15 * 100)] = -60.0
    out, _ = _retime(words, [(0.0, 10.0, 0.0)], [], env=env)
    assert 7.74 <= out[0]["e"] < 8.0


def test_hole_across_a_segment_join_is_closed():
    # cotra 012943-c1 holes 8.07-8.55, 15.69-16.17, 32.30-32.99: the join sits inside the hole, speech runs through
    segs = [(10.0, 18.5, 0.0), (30.0, 45.0, 8.5)]
    words = [_w("a", 17.0, 18.07), _w("b", 30.05, 30.6)]  # a ends out 8.07, b starts out 8.55
    env = _env(24, [(7.0, 9.5)])
    out, _ = _retime(words, segs, [], env=env)
    assert out[0]["e"] >= 18.5 - 1e-6 and out[1]["s"] <= 30.0 + 0.021


def test_join_hole_in_silence_stays_open():
    segs = [(10.0, 18.5, 0.0), (30.0, 45.0, 8.5)]
    words = [_w("a", 17.0, 18.07), _w("b", 30.05, 30.6)]
    env = _env(24, [(7.0, 8.1), (8.5, 9.5)])
    out, _ = _retime(words, segs, [], env=env)
    assert out[0]["e"] < 18.3 and out[1]["s"] > 30.0


# --- onset pass on the finished audio (Codee #953) ---
def _audio(n_s, speech, sr=16000):
    """White-ish noise at -20 dB inside the (start, end) output spans, near-silence elsewhere."""
    rng = np.random.default_rng(0)
    a = (1e-4 * rng.standard_normal(int(n_s * sr))).astype(np.float32)
    for s0, e0 in speech:
        a[int(s0 * sr):int(e0 * sr)] = 0.1 * rng.standard_normal(int(e0 * sr) - int(s0 * sr))
    return a


def test_caption_after_a_join_waits_for_the_onset_150ms_later():
    # cut at output 8.5, speech starts 150 ms later: the first caption of segment 1 lights at the onset, not the cut
    segs = [(10.0, 18.5, 0.0), (30.0, 45.0, 8.5)]
    words = [_w("a", 17.0, 18.07), _w("b", 30.0, 30.6)]
    out, moved = _retime(words, segs, [], bias=0.07, audio=_audio(24, [(7.0, 8.3), (8.65, 10.0)]))
    assert abs((out[1]["s"] - 0.07 - 30.0 + 8.5) - 8.65) < 0.02, out[1]
    assert any(m.startswith("onset:b") for m in moved)


def test_no_onset_within_300ms_keeps_the_original_start_and_logs():
    segs = [(10.0, 18.5, 0.0), (30.0, 45.0, 8.5)]
    words = [_w("a", 17.0, 18.07), _w("b", 30.0, 30.6)]
    out, moved = _retime(words, segs, [], bias=0.07, audio=_audio(24, [(7.0, 8.3), (9.0, 10.0)]))
    assert out[1]["s"] == 30.0
    assert any(m.startswith("onset-none:b") for m in moved)


def test_caption_already_on_speech_is_not_touched():
    segs = [(10.0, 18.5, 0.0)]
    words = [_w("a", 10.5, 11.0)]
    out, moved = _retime(words, segs, [], bias=0.07, audio=_audio(9, [(0.4, 3.0)]))
    assert out[0]["s"] == 10.5 and moved == []


# Chip Roy c1 (alliance-secure-ai-20261105-035221): numbers taken from its retime.json.
from clipper.captionfix import repair_words, _spread_collapsed


def test_gonna_replaces_going_to_when_the_audio_has_gonna():
    segs = [(0.0, 40.0, 0.0)]
    words = [w("not", 32.49, 32.7), w("going", 32.56, 32.7), w("to", 32.7, 32.8), w("flip", 32.87, 33.0)]
    heard = [("not", 32.49), ("gonna", 32.58), ("flip", 32.87)]
    out, log = repair_words(words, segs, heard, None)
    assert [x["w"] for x in out] == ["not", "gonna", "flip"]
    assert any("gonna" in m for m in log)


def test_words_missing_from_the_transcript_are_filled_from_the_audio():
    segs = [(0.0, 30.0, 0.0)]
    env = _env(30, [(13.8, 15.0)])
    words = [w("off.", 13.86, 14.09), w("Oh,", 14.68, 14.78), w("you're", 14.78, 14.84)]
    heard = [("off", 13.86), ("If", 14.12), ("I", 14.25), ("could", 14.38), ("say", 14.51),
             ("Oh", 14.68), ("you're", 14.78)]
    out, log = repair_words(words, segs, heard, env)
    assert [x["w"] for x in out][:5] == ["off.", "If", "I", "could", "say"]
    assert abs(out[1]["s"] - 14.12) < 1e-6 and out[1].get("fill")


def test_collapsed_pair_is_spread_before_the_later_word():
    # "I" 9.76 and "would" 9.80 (40 ms apart); the energy rise of "I" is at ~9.64.
    cur = {(0, 0): [9.36, 9.76], (1, 0): [9.76, 10.01], (2, 0): [9.80, 10.05]}
    occ = [(0, 0, 9.36), (1, 0, 9.76), (2, 0, 9.80)]
    _spread_collapsed(cur, occ)
    assert abs(cur[(1, 0)][0] - 9.64) < 1e-6 and cur[(2, 0)][0] == 9.80


def test_duplicate_transcript_word_without_heard_counterpart_is_dropped():
    # chip-roy c1: transcript "And," + "and" (0.4 s apart), the audio has one "and" then "I would"
    segs = [(0.0, 40.0, 0.0)]
    words = [w("kill", 8.0, 8.3), w("And,", 8.84, 9.1), w("and", 9.26, 9.6), w("I", 9.6, 9.7), w("would", 9.7, 9.9)]
    heard = [("kill", 8.0), ("and", 8.9), ("I", 9.36), ("would", 9.64)]
    out, log = repair_words(words, segs, heard, None)
    assert [x["w"] for x in out] == ["kill", "and", "I", "would"]
    assert any(m.startswith("dup") for m in log)


def test_repeated_word_that_is_heard_twice_is_kept():
    segs = [(0.0, 40.0, 0.0)]
    words = [w("very", 1.0, 1.3), w("very", 1.4, 1.7), w("good", 1.8, 2.0)]
    heard = [("very", 1.0), ("very", 1.4), ("good", 1.8)]
    out, _log = repair_words(words, segs, heard, None)
    assert [x["w"] for x in out] == ["very", "very", "good"]


def test_stacked_word_is_placed_on_the_burst_between_the_two_words():
    # chip-roy c1: "I" stacked 40 ms before "would"; the audio has "I" as a separate burst 0.3 s earlier
    # (after a dip), the blind gap would put it at would - 0.16.
    env = _env(30, [(9.0, 9.2), (9.35, 9.55), (9.75, 10.0)])
    segs = [(0.0, 30.0, 0.0)]
    cur = {(0, 0): [9.0, 9.2], (1, 0): [9.76, 10.01], (2, 0): [9.80, 10.05]}
    occ = [(0, 0, 9.0), (1, 0, 9.76), (2, 0, 9.80)]
    _spread_collapsed(cur, occ, segs, env, 0.0)
    assert 9.30 <= cur[(1, 0)][0] <= 9.45 and cur[(2, 0)][0] == 9.80
