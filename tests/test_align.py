

def test_spoken_currency_thousands():
    from clipper.align import spoken
    assert spoken("$20,000.") == ["twenty", "thousand", "dollars"]
    assert spoken("2,400") == ["two", "thousand", "four", "hundred"]
