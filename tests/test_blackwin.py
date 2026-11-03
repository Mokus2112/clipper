import subprocess
from types import SimpleNamespace

from clipper import blackwin


ERR = ("[blackdetect @ 0x1] black_start:3.3 black_end:3.86667 black_duration:0.566667\n"
       "[blackdetect @ 0x1] black_start:17.0333 black_end:19.7667 black_duration:2.73333\n")


def test_black_seconds_sums_the_runs():
    assert abs(blackwin.black_seconds(ERR) - (0.56667 + 2.73333)) < 1e-4
    assert blackwin.black_seconds("no black here") == 0.0


def test_black_fraction_of_the_cod_c3_window(monkeypatch):
    # CoD c3: ~30% title cards in the 21 s window -> over the 15% gate
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
        assert "from .blackwin import black_fraction, black_max" in src and "black > black_max(cfg)" in src
        assert "black_ratio" in src
    render = (root / "render.py").read_text()
    assert render.index('source.get("montage"):  # title-card windows') < render.index("# Frame-exact segments")


def test_blackdetect_uses_the_calibrated_pixel_threshold(monkeypatch):
    seen = {}
    def fake(cmd, **k):
        seen["vf"] = cmd[cmd.index("-vf") + 1]
        return SimpleNamespace(stderr="", stdout="", returncode=0)
    monkeypatch.setattr(subprocess, "run", fake)
    blackwin.black_fraction("x.mp4", 0.0, 10.0)
    assert "pix_th=0.15" in seen["vf"] and "pic_th=0.9" in seen["vf"]


def test_black_limit_comes_from_config_with_a_default():
    assert blackwin.black_max({}) == 0.15
    assert blackwin.black_max({"rank": {"black_max": 0.25}}) == 0.25


def test_cod_c3_is_over_the_configured_limit():
    # window black shares measured at pix_th 0.15: good windows 0%, CoD c3 19%
    assert 0.19 > blackwin.black_max({}) and 0.0 <= blackwin.black_max({})


def test_shipped_config_sets_the_limit():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    assert cfg["rank"]["black_max"] == 0.15
