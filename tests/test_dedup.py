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


def _job(root, name, url, clips):
    d = root / "jobs" / name
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"input": url}))
    (d / "clips.json").write_text(json.dumps(clips))
    return d


def test_earlier_spans_filters_same_video_only(tmp_path):
    from clipper.rank import _earlier_spans
    url = "https://www.youtube.com/watch?v=FuPe6ZfrgBg"
    _job(tmp_path, "camp-1", url, [{"start": 100, "end": 130}])
    _job(tmp_path, "camp-2", "https://www.youtube.com/watch?v=AAAAAAAAAAA", [{"start": 500, "end": 530}])
    me = _job(tmp_path, "camp-3", url, [])
    spans = _earlier_spans(me, "camp-3")
    assert [(c["start"], c["end"]) for c in spans] == [(100, 130)]  # other video and own job excluded
    overlaps = lambda s, e: any(s < c["end"] and e > c["start"] for c in spans)  # noqa: E731
    assert overlaps(110, 140)       # overlapping period -> dropped
    assert not overlaps(130, 160)   # adjacent, not overlapping -> kept
    assert not overlaps(500, 530)   # only the other video had it -> kept
