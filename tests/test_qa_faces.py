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


def test_chip_roy_c1_27s_frame_hand_is_not_dropped_by_the_x_overlap_rule():
    # Real detections at 27.0 s: the hand (x3-202, score 0.821) does not overlap the face in x and the score gap is 0.09,
    # so the shared rule keeps it. Documents why that frame still needs a different decision (Codee).
    face, hand = (479.4, 438.9, 307.5, 430.7, 0.911), (3.5, 1349.6, 198.8, 283.5, 0.821)
    assert drop_false_faces([face, hand]) == [face, hand]
