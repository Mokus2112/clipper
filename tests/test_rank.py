

def test_style_notes_loads_reference():
    from clipper.rank import style_notes
    text = style_notes()
    assert "Style reference" in text and "Hook" in text and len(text) < 2500
