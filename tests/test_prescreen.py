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
