from clipper.render import PUNCH_AMP, PUNCH_PERIOD, punch_expr, punch_filter, punch_schedule


def test_step_at_least_every_3_s_and_first_after_the_hook():
    st = punch_schedule(29.2, [])
    assert st[0] >= 2.5 and all(b - a <= PUNCH_PERIOD + 1e-9 for a, b in zip(st, st[1:]))
    assert 29.2 - st[-1] <= PUNCH_PERIOD + 0.6


def test_step_snaps_earlier_onto_a_cut_never_later():
    st = punch_schedule(20.0, [5.2, 5.9])
    assert 5.2 in st and 5.9 not in st
    assert all(b - a <= PUNCH_PERIOD + 1e-9 for a, b in zip(st, st[1:]))


def test_steps_keep_min_spacing_when_snapping():
    st = punch_schedule(20.0, [0.5, 2.7, 3.0, 5.0, 5.4])
    assert all(b - a >= 2.0 - 1e-9 for a, b in zip(st, st[1:]))


def test_no_step_near_the_clip_end():
    assert all(t < 10 - 0.6 for t in punch_schedule(10.0, []))


def test_zoom_expression_alternates_and_reaches_the_amplitude():
    e = punch_expr([3.0, 6.0])
    t = lambda x: eval(e.replace("min", "min").replace("max", "max"), {"t": x, "min": min, "max": max})  # noqa: E731
    assert t(0) == 1 and abs(t(4.5) - (1 + PUNCH_AMP)) < 1e-9 and abs(t(8) - 1) < 1e-9
    assert PUNCH_AMP + 1 >= 1.06


def test_filter_keeps_the_anchor_fixed():
    f = punch_filter([3.0], 30, (540, 400), 1080, 1920)
    assert f.startswith("perspective=") and "400.0*" in f and "interpolation=cubic" in f and "(in/30.00000)" in f


from clipper.render import FACE_BOTTOM_MAX, punch_plan


def _edges(top, bottom, plan):
    ay, a = plan
    z = 1 + a
    return ay + (top - ay) * z, ay + (bottom - ay) * z


def test_plan_keeps_face_top_and_chin_inside_the_safe_band():
    for top, bottom in ((353, 760), (260, 900), (230, 980), (400, 1000)):
        plan = punch_plan(top, bottom)
        assert plan is not None and plan[1] >= 0.06
        t2, b2 = _edges(top, bottom, plan)
        assert t2 >= 220 - 1e-6 and b2 <= FACE_BOTTOM_MAX + 1e-6


def test_plan_anchors_at_the_chin_so_it_does_not_move_towards_the_captions():
    ay, a = punch_plan(520, 1062)
    assert ay == 1062 and a == PUNCH_AMP
    assert _edges(520, 1062, (ay, a))[1] == 1062


def test_plan_moves_the_anchor_up_when_the_forehead_would_hit_the_top_zone():
    ay, _ = punch_plan(260, 1040)
    assert ay < 1040


def test_plan_gives_up_when_the_face_is_too_tall_for_a_1_06_zoom():
    assert punch_plan(225, 1070) is None
    assert punch_plan(230, 1100) is None


def test_punch_anchor_uses_the_main_range_face_bottom():
    """face_boxes.last_bottom is overwritten by per-segment calls; the punch block must use the value read next to face_top."""
    import inspect
    from clipper import render as r
    src = inspect.getsource(r.render)
    top = src.index("face_top = getattr(face_boxes")
    bottom = src.index('face_bottom = getattr(face_boxes, "last_bottom"')
    assert 0 < bottom - top < 400  # read right next to face_top, before per-segment calls
    assert src.count('getattr(face_boxes, "last_bottom"') == 1 and "fb = face_bottom" in src
