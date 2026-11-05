from clipper.qa import drop_false_faces


def test_hand_under_face_dropped():
    real = (343, 403, 367, 461, 0.93)
    false = (122, 1015, 382, 493, 0.81)
    assert drop_false_faces([real, false]) == [real]


def test_two_people_side_by_side_kept():
    a, b = (100, 400, 300, 380, 0.92), (600, 420, 300, 380, 0.90)
    assert drop_false_faces([a, b]) == [a, b]


def test_two_real_faces_stacked_kept():
    a, b = (340, 200, 360, 450, 0.93), (330, 1000, 360, 450, 0.89)
    assert drop_false_faces([a, b]) == [a, b]
