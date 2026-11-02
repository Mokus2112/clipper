import json
from clipper import rank

SEG = [{"s": 0, "e": 5, "text": "Hello there."}, {"s": 5, "e": 10, "text": "And then it trails"}]
CFG = {"rank": {"model": "sonnet", "content_check": True}}


def test_drops_rejected(tmp_path):
    clips = [{"id": "j-c1", "start": 0, "end": 5}, {"id": "j-c2", "start": 5, "end": 10}]
    ask = lambda cfg, prompt, cwd: json.dumps([{"n": 1, "reject": False}, {"n": 2, "reject": True, "rule": "no_payoff", "reason": "x"}])
    assert [c["id"] for c in rank.content_check(CFG, tmp_path, clips, SEG, ask)] == ["j-c1"]
    assert (tmp_path / "content_check.json").exists()


def test_fail_open(tmp_path):
    clips = [{"id": "j-c1", "start": 0, "end": 5}]
    def boom(*a): raise RuntimeError("down")
    assert rank.content_check(CFG, tmp_path, clips, SEG, boom) == clips
