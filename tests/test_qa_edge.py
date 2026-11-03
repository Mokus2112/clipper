from clipper.qa import edge_problems


def test_c3_trailing_off_fails():
    w = [("No,", 0.0, 0.3), ("deadly", 8.88, 9.12), ("tech...", 9.12, 9.32)]
    assert any("ends on" in p for p in edge_problems(w, 9.41, -60))


def test_sentence_end_ok():
    assert edge_problems([("Hi", 0.1, 0.3), ("there.", 0.3, 0.9)], 0.95, -60) == []


def test_silence_tail_ok():
    assert edge_problems([("Hi", 0.1, 0.3), ("and", 0.3, 0.6)], 1.0, -60) == []


def test_clipped_start_fails():
    assert any("starts inside" in p for p in edge_problems([("ing", 0.0, 0.2), ("is.", 0.2, 0.5)], 1.0, -12))


def test_hook_length_gate(tmp_path):
    import json
    from clipper.qa import check_hook_length
    (tmp_path / "clip.json").write_text(json.dumps({"hook": "Agassi's dad built a ball machine monster"}))
    assert check_hook_length(tmp_path)
    (tmp_path / "clip.json").write_text(json.dumps({"hook": "Jason: Dad's monster"}))
    assert check_hook_length(tmp_path) == []
