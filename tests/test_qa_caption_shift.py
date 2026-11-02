from pathlib import Path

from clipper import qa


def _patch(monkeypatch, boxes):
    monkeypatch.setattr(qa, "burned_text_conflicts", lambda *a, **k: [(1.0, (0, 0, 10, 10), "Solved")])
    monkeypatch.setattr(qa, "busy_caption_times", lambda *a, **k: set())
    monkeypatch.setattr(qa, "_duration", lambda p: 5.0)
    monkeypatch.setattr(qa, "text_boxes", lambda *a, **k: boxes)
    monkeypatch.setattr(qa, "face_boxes", lambda *a, **k: [])


def test_empty_inside_returns_no_shift(monkeypatch):
    # OCR conflict at t=1.0 but the only caption box is in the upper band (bottom <= 1100): nothing to shift
    _patch(monkeypatch, [(1.0, (100, 200, 900, 300))])
    assert qa.plan_caption_shift(Path("."), Path(".")) == []


def test_no_boxes_at_all_returns_no_shift(monkeypatch):
    _patch(monkeypatch, [])
    assert qa.plan_caption_shift(Path("."), Path(".")) == []
