from clipper.shotvar import longest_shot, broll_cards, MAX_SHOT_SEC


def test_longest_shot_no_cuts():
    assert longest_shot([], 29.0) == 29.0


def test_longest_shot_with_cuts():
    assert longest_shot([8.0, 15.0, 40.0], 30.0) == 15.0  # cut at 40 is outside the window


def test_broll_cards_counts_number_phrase():
    words = [{"w": "We", "s": 100.0, "e": 100.2}] + [{"w": "raised", "s": 105.0, "e": 105.3},
             {"w": "$20", "s": 105.4, "e": 105.8}, {"w": ",000.", "s": 105.8, "e": 106.2}]
    assert broll_cards(words, 100.0, 130.0) == 1
    assert broll_cards(words[:2], 100.0, 130.0) == 0


def test_limit_value():
    assert MAX_SHOT_SEC == 12.0
