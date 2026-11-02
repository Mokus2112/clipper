import json
import pytest
from clipper import fetch as F


def cfg_for(tmp_path):
    return {"paths": {"data": str(tmp_path)}, "fetch": {"rate_limit": "1M"}}


def test_video_id():
    assert F.video_id("https://www.youtube.com/watch?v=FuPe6ZfrgBg&t=5") == "FuPe6ZfrgBg"
    assert F.video_id("https://youtu.be/FuPe6ZfrgBg") == "FuPe6ZfrgBg"
    assert F.video_id("/some/local.mp4") is None


def test_used_video(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "path_of", lambda cfg, k: tmp_path)
    j = tmp_path / "jobs" / "camp-1"
    j.mkdir(parents=True)
    (j / "job.json").write_text(json.dumps({"input": "https://www.youtube.com/watch?v=FuPe6ZfrgBg"}))
    assert F.used_video({}, "camp", "https://youtu.be/FuPe6ZfrgBg") == "camp-1"
    assert F.used_video({}, "other", "https://youtu.be/FuPe6ZfrgBg") is None
    assert F.used_video({}, "camp", "https://youtu.be/AAAAAAAAAAA") is None
