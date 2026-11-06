"""Felulbiralo #552: no <0.3 s final cut, no black stretch at the clip edge."""
from clipper.render import drop_short_final, trim_black_edges

FPS = 30.0


def test_short_final_segment_is_dropped():
    segs = [(0.0, 5.0, 0.0), (7.0, 7.2, 5.0)]
    s, n = drop_short_final(segs, [150, 6], FPS, [])
    assert s == segs[:1] and n == [150]


def test_long_enough_or_speech_or_only_segment_is_kept():
    segs = [(0.0, 5.0, 0.0), (7.0, 7.5, 5.0)]
    assert drop_short_final(segs, [150, 15], FPS, [])[0] == segs
    short = [(0.0, 5.0, 0.0), (7.0, 7.2, 5.0)]
    assert drop_short_final(short, [150, 6], FPS, [{"w": "x", "s": 7.0, "e": 7.1}])[0] == short
    assert drop_short_final(short[1:], [6], FPS, [])[0] == short[1:]


def test_black_edges_are_trimmed():
    assert trim_black_edges(10.0, 20.0, [(10.0, 10.6), (19.3, 20.0)], FPS) == (10.6, 19.3)
    assert trim_black_edges(10.0, 20.0, [(14.0, 15.0)], FPS) == (10.0, 20.0)  # interior black is not an edge


def test_lula_voiceover_over_black_is_not_cut():
    words = [{"w": "voice", "s": 1226.0, "e": 1229.0}]
    assert trim_black_edges(1225.6, 1240.5, [(1223.33, 1230.75)], FPS, words) == (1225.6, 1240.5)
    assert trim_black_edges(1225.6, 1240.5, [(1223.33, 1230.75)], FPS) == (1225.6, 1240.5)  # > max_edge even without words


def test_long_edge_cap_and_no_stub():
    assert trim_black_edges(10.0, 20.0, [(10.0, 12.0)], FPS) == (10.0, 20.0)  # 2 s > max_edge 1.5
    assert trim_black_edges(10.0, 13.0, [(10.0, 11.0)], FPS) == (10.0, 13.0)  # would leave 2 s < min_left
