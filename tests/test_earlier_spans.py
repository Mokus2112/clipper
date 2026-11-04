import json

from clipper.rank import _earlier_spans, _section_offset


def _job(root, name, section, clips):
    d = root / name
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"input": "x.mp4", "section": section}))
    (d / "clips.json").write_text(json.dumps(clips))
    return d


def test_section_offset_formats():
    assert _section_offset({"section": "00:32:50-00:34:20"}) == 1970
    assert _section_offset({"section": "8150-8220"}) == 8150
    assert _section_offset({"section": None}) == 0


def test_earlier_spans_are_absolute(tmp_path):
    _job(tmp_path, "a", "8150-8220", [{"start": 10, "end": 50}])
    me = _job(tmp_path, "b", "00:32:50-00:34:20", [])
    spans = _earlier_spans(me, "b")
    assert spans == [{"start": 8160, "end": 8200}]
