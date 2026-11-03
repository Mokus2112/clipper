import subprocess
from types import SimpleNamespace

from clipper import render


def test_leftover_flash_before_the_end_is_dropped():
    # CoD c2 (reviewer): cuts at 20.133 and 20.300 s, clip ends 20.767 s: the 0.17 s shot at 20.133 is a flash
    cuts = [14, 99, 156, 209, 604, 609]
    assert render.trailing_flash(cuts, 623, 30.0) == 604


def test_normal_end_shots_are_kept():
    assert render.trailing_flash([100, 560], 623, 30.0) == 623   # 2.1 s last shot
    assert render.trailing_flash([], 623, 30.0) == 623
    assert render.trailing_flash([3, 100], 623, 30.0) == 623     # a short FIRST shot is the edge-lead logic's job


def test_short_shot_far_from_the_end_is_left_alone():
    assert render.trailing_flash([200, 205, 300], 623, 30.0) == 623


def test_black_runs_parse_blackdetect(monkeypatch):
    err = ("[blackdetect @ 0x1] black_start:3.3 black_end:3.86667 black_duration:0.566667\n"
           "[blackdetect @ 0x1] black_start:17.0333 black_end:19.7667 black_duration:2.73333\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stderr=err, stdout="", returncode=0))
    runs = render.black_runs("x.mp4", 74.0, 95.0)
    assert runs == [(74.0 + 3.3, 74.0 + 3.86667), (74.0 + 17.0333, 74.0 + 19.7667)]


def test_montage_fades_are_long_only_at_the_clip_edges():
    assert render.montage_fades(0, 6) == (0.4, 0.05)
    assert render.montage_fades(3, 6) == (0.05, 0.05)
    assert render.montage_fades(5, 6) == (0.05, 0.4)
    assert render.montage_fades(0, 1) == (0.4, 0.4)
