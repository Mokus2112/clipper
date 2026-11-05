import json
from clipper import qa


def test_log_qa(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "ROOT", tmp_path)
    job = tmp_path / "jobs" / "j1"
    cdir = job / "clips" / "j1-c2-v2"
    cdir.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"input": "https://y/x", "source": "camp"}))
    qa.log_qa(cdir, ["caption/hook on the speaker's face at 0.4s (overlap 119px)", "speech without caption at 4.49-4.62"])
    rec = json.loads((tmp_path / "data" / "qa_log.jsonl").read_text())
    assert rec["version"] == 2 and rec["campaign"] == "camp" and not rec["ok"]
    assert len(rec["failed_checks"]) == 2


def test_log_qa_records_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "ROOT", tmp_path)
    cdir = tmp_path / "jobs" / "j1" / "clips" / "j1-c1"
    cdir.mkdir(parents=True)
    qa.log_qa(cdir, [], [qa.WARN + "safe zone: face at 1.0s box x1-2 y3-4"])
    rec = json.loads((tmp_path / "data" / "qa_log.jsonl").read_text())
    assert rec["ok"] and rec["failed_checks"] == [] and rec["warnings"] == ["safe zone: face at s box x- y-"]
