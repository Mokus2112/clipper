from clipper.align import outliers


def test_flags_words_far_from_whisper():
    ref = [{"w": "when", "s": 10.0, "e": 10.2}, {"w": "he", "s": 10.3, "e": 10.4}, {"w": "did", "s": 11.0, "e": 11.2}]
    al = [{"w": "when", "s": 12.2, "e": 12.4}, {"w": "he", "s": 12.5, "e": 12.6}, {"w": "did", "s": 11.3, "e": 11.5}]
    bad = outliers(al, ref)
    assert [b["w"] for b in bad] == ["when", "he"]
    assert bad[0]["diff"] == 2.2


def test_normal_drift_is_not_flagged():
    ref = [{"w": "a", "s": 1.0, "e": 1.1}]
    assert outliers([{"w": "a", "s": 1.5, "e": 1.6}], ref) == []
