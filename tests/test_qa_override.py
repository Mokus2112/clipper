import json

from clipper.qa import apply_overrides

FACE = "safe zone: face at 27.0s box x4-205 y1347-1635"


def entry(**kw):
    e = {"check": "face_box", "t": 27.0, "box": [4, 1347, 205, 1635], "by": "codee", "basis": "frame",
         "date": "2026-11-05", "reason": "gesturing hand, not a face"}
    e.update(kw)
    return e


def run(tmp_path, entries, problems):
    (tmp_path / "override.json").write_text(json.dumps({"overrides": entries}))
    return apply_overrides(tmp_path, problems)


def test_valid_override_passes_the_point(tmp_path):
    left, notes = run(tmp_path, [entry()], [FACE])
    assert left == []
    assert notes and notes[0].startswith("OVERRIDDEN by codee: gesturing hand")


def test_time_within_tolerance_ok_outside_not(tmp_path):
    assert run(tmp_path, [entry(t=27.15)], [FACE])[0] == []
    assert run(tmp_path, [entry(t=27.5)], [FACE])[0] == [FACE]


def test_wrong_box_does_not_pass(tmp_path):
    assert run(tmp_path, [entry(box=[400, 300, 900, 700])], [FACE])[0] == [FACE]


def test_missing_field_is_invalid(tmp_path):
    for k in ("by", "basis", "date", "reason", "box"):
        e = entry()
        del e[k]
        assert run(tmp_path, [e], [FACE])[0] == [FACE], k


def test_who_must_be_codee_or_reviewer(tmp_path):
    assert run(tmp_path, [entry(by="clipper")], [FACE])[0] == [FACE]
    assert run(tmp_path, [entry(by="felulbiralo")], [FACE])[0] == []


def test_content_check_types_never_overridable(tmp_path):
    p = "caption words not heard in the audio: COULD"
    assert run(tmp_path, [entry(check="caption_words")], [p])[0] == [p]
    assert run(tmp_path, [entry(check="content")], [p])[0] == [p]


def test_does_not_cover_other_points_or_checks(tmp_path):
    other = "safe zone: face at 12.0s box x4-205 y1347-1635"
    sync = "caption sync: word lights up at 27.0s, audio onset 26.8s (+200 ms)"
    left, _ = run(tmp_path, [entry()], [other, sync])
    assert left == [other, sync]


def test_caption_sync_silence_word(tmp_path):
    p = "caption sync: 'AND' lights up at 1.09s during silence"
    e = {"check": "caption_sync_word", "t": 1.09, "word": "and", "by": "felulbiralo", "basis": "listening",
         "date": "2026-11-05", "reason": "heard"}
    assert run(tmp_path, [e], [p])[0] == []
    assert run(tmp_path, [dict(e, word="the")], [p])[0] == [p]


def test_no_file(tmp_path):
    assert apply_overrides(tmp_path, [FACE]) == ([FACE], [])
