from clipper.render import chunk_words


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


def test_short_chunk_never_merges_past_max_words():
    # "It was this" lasts 0.24 s (< MIN_CAPTION_SEC); merging it made a 6-word line.
    words = [w("It", 0.0, 0.06), w("was", 0.06, 0.12), w("this", 0.12, 0.24),
             w("guy,", 0.28, 0.40), w("MacArthur", 0.46, 0.86), w("Wheeler.", 0.92, 1.26)]
    chunks = chunk_words(words, 3, 1.4)
    assert all(len(c) <= 3 for c in chunks)
    assert [x["w"] for c in chunks for x in c] == [x["w"] for x in words]


def test_short_chunk_still_merges_when_it_fits():
    words = [w("Yes.", 0.0, 0.1), w("Exactly", 0.5, 1.0), w("right", 1.0, 1.3)]
    chunks = chunk_words(words, 3, 1.4)
    assert len(chunks) == 1


def test_wide_line_splits_by_pixels():
    from pathlib import Path
    from clipper.textfit import drawn_width, font_file
    font = font_file(Path(__file__).parent.parent / "fonts", "Bebas Neue")
    fits = lambda ch: drawn_width(font, " ".join(x["w"].upper() for x in ch), 116, 1, 4, 3) <= 760  # noqa: E731
    words = [w("psychological", 0.0, 0.5), w("argument", 0.5, 0.9), w("for", 0.9, 1.0),
             w("psychologically", 1.4, 2.0), w("yes", 2.7, 3.0)]
    chunks = chunk_words(words, 3, 1.4, fits)
    assert all(fits(c) or len(c) == 1 for c in chunks)
    assert [len(c) for c in chunks][:1] == [1] or [len(c) for c in chunks][:1] == [2]
    assert [x["w"] for c in chunks for x in c] == [x["w"] for x in words]


def test_flash_chunk_of_0_13s_merges_into_a_neighbour_even_at_max_words():
    w = lambda t, s, e: {"w": t, "s": s, "e": e}  # noqa: E731
    words = [w("a", 0.0, 0.3), w("b", 0.3, 0.6), w("c", 0.6, 0.9), w("d", 0.95, 1.08), w("e", 1.5, 1.9)]
    chunks = chunk_words(words, 3, 1.4)
    assert all(ch[-1]["e"] - ch[0]["s"] >= 0.25 for ch in chunks), chunks
    assert [x["w"] for ch in chunks for x in ch] == list("abcde")


def test_short_last_chunk_takes_a_word_from_a_full_neighbour():
    # Chip Roy c1: "I WOULD KILL | IT ALL." left a 0.23 s chunk because both neighbours were full.
    words = [w("I", 1.00, 1.20), w("would", 1.20, 1.40), w("kill", 1.40, 1.66),
             w("it", 1.66, 1.78), w("all.", 1.78, 1.89)]
    chunks = chunk_words(words, 3, 1.4)
    assert [[x["w"] for x in c] for c in chunks] == [["I", "would"], ["kill", "it", "all."]]
