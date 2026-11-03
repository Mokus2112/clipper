import subprocess
from types import SimpleNamespace

from clipper import blackwin


ERR = ("[blackdetect @ 0x1] black_start:3.3 black_end:3.86667 black_duration:0.566667\n"
       "[blackdetect @ 0x1] black_start:17.0333 black_end:19.7667 black_duration:2.73333\n")


def test_black_seconds_sums_the_runs():
    assert abs(blackwin.black_seconds(ERR) - (0.56667 + 2.73333)) < 1e-4
    assert blackwin.black_seconds("no black here") == 0.0


def test_black_fraction_of_the_cod_c3_window(monkeypatch):
    # CoD c3: ~30% title cards in the 21 s window -> over the 20% gate
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stderr=ERR, stdout="", returncode=0))
    f = blackwin.black_fraction("x.mp4", 74.0, 95.0)
    assert abs(f - 3.3 / 21.0) < 1e-3          # these two runs alone: 16%
    monkeypatch.setattr(blackwin, "black_seconds", lambda e: 6.3)
    assert blackwin.black_fraction("x.mp4", 74.0, 95.0) > blackwin.BLACK_MAX


def test_clean_window_is_under_the_gate(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stderr="", stdout="", returncode=0))
    assert blackwin.black_fraction("x.mp4", 42.0, 63.0) == 0.0


def test_render_and_rank_use_the_same_black_gate():
    from pathlib import Path
    root = Path(__file__).parents[1] / "src/clipper"
    for name in ("render.py", "rank.py"):
        src = (root / name).read_text()
        assert "from .blackwin import BLACK_MAX, black_fraction" in src and "black > BLACK_MAX" in src
    render = (root / "render.py").read_text()
    assert render.index('source.get("montage"):  # title-card windows') < render.index("# Frame-exact segments")
