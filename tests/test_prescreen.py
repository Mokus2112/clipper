from clipper import prescreen

OK = {"identity_ratio": 0.9, "multi_face_ratio": 0.0, "cuts_per_40s": 3, "top_text_ratio": 0.0, "silence_ratio": 0.0}


def test_keep():
    assert prescreen.decide(OK, prescreen.DEFAULTS, True) == []


def test_drops():
    thr = prescreen.DEFAULTS
    assert prescreen.decide({**OK, "identity_ratio": 0.3}, thr, True)
    assert prescreen.decide({**OK, "identity_ratio": 0.3}, thr, False) == []
    assert prescreen.decide({**OK, "multi_face_ratio": 0.8}, thr, False)
    assert prescreen.decide({**OK, "multi_face_ratio": 0.8}, thr, True) == []
    assert prescreen.decide({**OK, "multi_face_ratio": 0.8}, thr, False, tracks_faces=False) == []
    assert prescreen.decide({**OK, "cuts_per_40s": 14}, thr, False)
    assert prescreen.decide({**OK, "top_text_ratio": 0.5}, thr, False)
    assert prescreen.decide({**OK, "silence_ratio": 0.5}, thr, False)


def test_top_banner_without_text_drops_candidate():
    import numpy as np
    assert prescreen.decide({**OK, "top_edge_ratio": 1.0}, prescreen.DEFAULTS, True)
    assert prescreen.decide({**OK, "top_edge_ratio": 0.05}, prescreen.DEFAULTS, True) == []
    flat = np.full((1080, 1920, 3), 90, np.uint8)
    assert not prescreen.top_band_has_edges(flat)
    banner = flat.copy()
    banner[10:60, 100:1800:40] = 255  # vertical bars in the top strip, no letters
    assert prescreen.top_band_has_edges(banner)


# Codee #1799: animated source (lula-packed-lunch). Measurements of the 258 s sample (candidate 249-274 s).
LULA = {**OK, "cuts_per_40s": 17.9, "top_edge_ratio": 0.8}


def test_lula_sample_without_animated_flag_is_unchanged():
    why = prescreen.decide(LULA, prescreen.DEFAULTS, False)
    assert why == ["17.9 cuts per 40 s", "burned-in banner/graphic in the top strip in 80% of samples"]
    assert prescreen.decide_full(LULA, prescreen.DEFAULTS, False) == (why, [])


def test_animated_flag_raises_cuts_limit_and_downgrades_banner_to_warning():
    why, warn = prescreen.decide_full(LULA, prescreen.DEFAULTS, False, animated=True)
    assert why == []
    assert warn == ["burned-in banner/graphic in the top strip in 80% of samples"]
    assert prescreen.decide_full({**LULA, "cuts_per_40s": 31}, prescreen.DEFAULTS, False, animated=True)[0]
    # other gates stay blocking on animated sources
    assert prescreen.decide_full({**LULA, "top_text_ratio": 0.5}, prescreen.DEFAULTS, False, animated=True)[0]
    assert prescreen.decide_full({**LULA, "silence_ratio": 0.5}, prescreen.DEFAULTS, False, animated=True)[0]
