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
