"""Eye-line framing (video #207): punch pivot at the eyes, eye target y 640, zone overlay."""
import numpy as np

from clipper import qa
from clipper.render import EYELINE_Y, FACE_TOP_PAD_MAX, center_pad, eyeline_pad, punch_plan


def test_punch_pivot_is_the_eye_line():
    a, _ = punch_plan(300, 700, pivot=600)
    assert a == 600


def test_punch_pivot_is_clamped_into_the_feasible_band():
    a, amp = punch_plan(300, 700, pivot=1900)
    z = 1 + amp
    assert a + (700 - a) * z <= 1070 + 1e-6 and a + (300 - a) * z >= 220 - 1e-6


def test_eyeline_pad_lowers_a_high_eye():
    pad = eyeline_pad(0.2, 0.5, 1080, 1080, 1920)
    assert 0 < pad <= FACE_TOP_PAD_MAX and pad % 2 == 0


def test_eyeline_pad_zero_when_eye_already_low_or_unknown():
    assert eyeline_pad(0.6, 0.9, 1080, 1080, 1920) == 0
    assert eyeline_pad(None, 0.5, 1080, 1080, 1920) == 0


def test_center_pad_keeps_the_larger_source_pad():
    assert center_pad(180, None, None, None, 1080, 1080, 1920) == 180


def test_zone_overlay_writes_image(tmp_path):
    import cv2
    v = tmp_path / "short.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1080, 1920))
    for _ in range(10):
        w.write(np.zeros((1920, 1080, 3), np.uint8))
    w.release()
    out = tmp_path / "z.png"
    qa.zone_overlay(v, out, [(0.0, (100, 300, 900, 400))], [])
    assert out.exists()
