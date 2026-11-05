

def test_spoken_currency_thousands():
    from clipper.align import spoken
    assert spoken("$20,000.") == ["twenty", "thousand", "dollars"]
    assert spoken("2,400") == ["two", "thousand", "four", "hundred"]


def test_merge_number_fragments():
    from clipper.transcribe import merge_number_fragments
    ws = [{"w": "with", "s": 1, "e": 1.5}, {"w": "$20", "s": 2, "e": 2.5}, {"w": ",000.", "s": 2.5, "e": 2.6}]
    out = merge_number_fragments(ws)
    assert [w["w"] for w in out] == ["with", "$20,000."] and out[1]["e"] == 2.6
