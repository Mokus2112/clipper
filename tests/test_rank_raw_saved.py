import json

import pytest

from clipper import rank as rk


def _setup(monkeypatch, tmp_path, reply):
    (tmp_path / "job.json").write_text(json.dumps({"source": "s"}))
    (tmp_path / "transcript.json").write_text(json.dumps(
        {"segments": [{"s": 0, "e": 2, "text": "hello there"}]}))
    monkeypatch.setattr(rk, "job_dir", lambda cfg, jid: tmp_path)
    monkeypatch.setattr(rk, "find_source", lambda cfg, name: {"streamer": "x", "name": "s"})
    monkeypatch.setattr(rk, "title_budget", lambda s: 60)
    monkeypatch.setattr(rk, "hook_budget", lambda s: 40)

    def fake(cfg, prompt, cwd):
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rk, "ask_claude", fake)
    cfg = {"rank": {"clips_per_vod": 3, "min_score": 7, "model": "m"},
           "clip": {"min_sec": 10, "max_sec": 60}, "render": {"cold_open_max": 3},
           "detect": {"spike_z": 3}}
    return cfg


def test_unparseable_reply_is_saved(monkeypatch, tmp_path):
    cfg = _setup(monkeypatch, tmp_path, "sorry, no clips here")
    with pytest.raises(ValueError):
        rk.rank(cfg, "j")
    assert (tmp_path / "rank_raw.txt").read_text() == "sorry, no clips here"
    assert "ValueError" in (tmp_path / "rank_error.txt").read_text()


def test_claude_failure_is_saved(monkeypatch, tmp_path):
    cfg = _setup(monkeypatch, tmp_path, RuntimeError("claude failed: boom"))
    with pytest.raises(RuntimeError):
        rk.rank(cfg, "j")
    assert "boom" in (tmp_path / "rank_error.txt").read_text()
