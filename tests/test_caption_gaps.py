import numpy as np

from clipper import qa


def _curve(dur, speech, hop=0.01):
    """dB curve: -20 dB inside the speech spans, -60 elsewhere."""
    db = np.full(int(dur / hop) + 1, -60.0)
    for a, b in speech:
        db[int(round(a / hop)):int(round(b / hop))] = -20.0
    return db


def _covered(dur, caps, hop=0.01):
    cov = np.zeros(int(dur / hop) + 1, bool)
    for a, b in caps:
        cov[int(round(a / hop)):int(round(b / hop)) + 1] = True
    return cov


def test_fragmented_speech_without_caption_is_one_gap():
    # duo (091333-c2) 36.9-37.45: three short speech runs 0.1 s apart, each under the old 0.3 s limit
    db = _curve(40, [(36.90, 37.23), (37.35, 37.70), (37.80, 37.95)])
    gaps = qa.caption_gaps(db, _covered(40, [(30, 36.9), (38.5, 40)]))
    assert len(gaps) == 1 and abs(gaps[0][0] - 36.90) < 0.02 and abs(gaps[0][1] - 37.95) < 0.02


def test_close_clip_gap_found():
    # close (091333-c1) 33.11-33.97
    db = _curve(42, [(33.11, 33.40), (33.49, 33.76), (33.88, 33.97)])
    gaps = qa.caption_gaps(db, _covered(42, [(30, 33.1), (34.2, 36)]))
    assert len(gaps) == 1 and gaps[0][1] - gaps[0][0] > 0.8


def test_dark_clip_gap_found():
    # dark (alliance c1) gap: Codee #1310 raised the minimum to 0.6 s, so this one is 0.7 s
    db = _curve(28, [(1.29, 1.99)])
    gaps = qa.caption_gaps(db, _covered(28, [(0.2, 1.2), (2.1, 3)]))
    assert len(gaps) == 1


def test_short_gap_and_covered_speech_are_fine():
    db = _curve(10, [(2.0, 2.2), (5.0, 6.0)])
    assert qa.caption_gaps(db, _covered(10, [(4.9, 6.1)])) == []


def test_far_apart_blips_are_not_joined():
    db = _curve(10, [(2.0, 2.1), (6.0, 6.1)])
    assert qa.caption_gaps(db, _covered(10, [])) == []


def test_gap_under_the_new_minimum_is_ignored():
    # 0.3 s of uncaptioned speech used to fail; since Codee #1310 only gaps over 0.6 s do
    db = _curve(10, [(5.0, 5.3)])
    assert qa.caption_gaps(db, _covered(10, [(4.0, 4.9), (5.4, 6)])) == []
