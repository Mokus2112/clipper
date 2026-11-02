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
