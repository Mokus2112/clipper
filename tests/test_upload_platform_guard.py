import pytest

from clipper import upload


CFG = {"sources": [{"name": "cod", "platforms": ["tiktok", "instagram"]}, {"name": "jason"}],
       "upload": {"daily_limit": 3}}


def test_youtube_allowed():
    assert not upload.youtube_allowed(CFG, {"source": "cod"})
    assert upload.youtube_allowed(CFG, {"source": "jason"})
    assert upload.youtube_allowed(CFG, {"source": "unknown"})


def test_upload_refuses_explicit_non_youtube_clip(monkeypatch):
    monkeypatch.setattr(upload.review_queue, "uploaded_today", lambda cfg: 0)
    monkeypatch.setattr(upload.review_queue, "load", lambda cfg: [{"id": "c3", "status": "approved", "source": "cod"}])
    with pytest.raises(SystemExit, match="does not allow YouTube"):
        upload.upload(CFG, clip_ids=["c3"])


def test_upload_skips_non_youtube_in_batch(monkeypatch):
    monkeypatch.setattr(upload.review_queue, "uploaded_today", lambda cfg: 0)
    monkeypatch.setattr(upload.review_queue, "load", lambda cfg: [{"id": "c3", "status": "approved", "source": "cod"}])
    assert upload.upload(CFG) == []
