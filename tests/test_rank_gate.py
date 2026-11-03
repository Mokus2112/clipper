from clipper.rank import boundary_problems


def W(*items):
    return [{"w": w, "s": s, "e": e} for w, s, e in items]


def test_clean_sentence_passes():
    w = W(("Hi.", 0, .4), ("We", .45, .6), ("win.", .6, 1.0), ("Next", 1.05, 1.3))
    assert boundary_problems(w, 1, 2) == []


def test_mid_sentence_end_and_start_fail():
    w = W(("so", 0, .3), ("we", .31, .5), ("won", .5, .8), ("and", .81, 1.0), ("then", 1.0, 1.3))
    bp = boundary_problems(w, 1, 2)
    assert len(bp) == 2


def test_silence_gap_is_clean_cut():
    w = W(("so", 0, .3), ("we", .8, 1.0), ("won", 1.0, 1.3), ("and", 1.8, 2.0))
    assert boundary_problems(w, 1, 2) == []
