from clipper.render import cap_pan_speed


def _speed(track, cw, out_w, fps):
    return max(abs(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(track, track[1:])) / fps * out_w / cw


def test_fast_pan_is_capped():
    # Fox c1 z1 keyframes: ~7.5 output px/frame
    raw = [(4.5, 257), (5.0, 327), (5.5, 387), (6.0, 438), (6.5, 482), (7.0, 519)]
    out = cap_pan_speed(raw, 420, 1080, 30, 1.5, max_lag_frac=1.0)
    assert _speed(out, 420, 1080, 30) <= 1.5 + 0.1


def test_cap_released_when_face_would_leave():
    raw = [(0.0, 0), (0.5, 300)]
    out = cap_pan_speed(raw, 420, 1080, 30, 1.5, 0.25)
    assert abs(out[-1][1] - 300) <= 420 * 0.25 + 1  # face stays within the allowed lag


def test_slow_pan_unchanged():
    raw = [(0.0, 100), (0.5, 105), (1.0, 110)]
    assert cap_pan_speed(raw, 420, 1080, 30) == raw
