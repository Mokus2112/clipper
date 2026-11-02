from clipper.render import merge_punct


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


def texts(words):
    return [x["w"] for x in merge_punct(words)]


def test_number_with_separator_token_is_glued():
    assert texts([w("17", 0, 0.3), w(",500", 0.3, 0.6)]) == ["17,500"]
    assert texts([w("3", 0, 0.2), w(".5", 0.2, 0.4), w("million", 0.4, 0.8)]) == ["3.5", "million"]


def test_number_split_after_separator_is_glued():
    assert texts([w("17,", 0, 0.3), w("500", 0.3, 0.6)]) == ["17,500"]


def test_number_glue_extends_end_time():
    out = merge_punct([w("17", 0, 0.3), w(",500", 0.3, 0.6)])
    assert out[0]["s"] == 0 and out[0]["e"] == 0.6


def test_normal_words_untouched():
    assert texts([w("I", 0, .1), w("had", .1, .2), w("17,", .2, .4), w("then", .4, .6)]) == ["I", "had", "17,", "then"]
    assert texts([w("50", 0, .2), w("%", .2, .3)]) == ["50%"]


def test_year_followed_by_number_not_glued():
    assert texts([w("1999,", 0, .3), w("200", .3, .6)]) == ["1999,", "200"]
