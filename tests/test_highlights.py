from clipper.render import MIN_WORD_SHOW, balance_highlights


def test_edge_words_keep_their_highlight():
    # 3-word line: first and last word only 2 frames long, the middle one long
    evs = [[0.00, 0.07, 0], [0.07, 0.60, 1], [0.60, 0.67, 2]]
    out = balance_highlights(evs, MIN_WORD_SHOW)
    assert [e[2] for e in out] == [0, 1, 2]
    assert all(e[1] - e[0] >= MIN_WORD_SHOW - 1e-9 for e in out)
    assert out[0][0] == 0.0 and abs(out[-1][1] - 0.67) < 1e-9  # line start/end unchanged


def test_nothing_to_spare_still_merges():
    out = balance_highlights([[0.0, 0.05, 0], [0.05, 0.15, 1]], MIN_WORD_SHOW)
    assert len(out) == 1 and out[0][0] == 0.0 and out[0][1] == 0.15


def test_long_words_untouched():
    evs = [[0.0, 0.3, 0], [0.3, 0.6, 1], [0.6, 0.9, 2]]
    assert balance_highlights(evs, MIN_WORD_SHOW) == evs
