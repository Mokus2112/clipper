import os, time
from clipper.common import prune_sources


def _cfg(tmp_path, **fetch):
    return {"paths": {"data": str(tmp_path)}, "fetch": fetch}


def _job(tmp_path, name, age_days):
    jd = tmp_path / "jobs" / name
    jd.mkdir(parents=True)
    f = jd / "source.mp4"
    f.write_bytes(b"x" * 10)
    t = time.time() - age_days * 86400
    os.utime(f, (t, t))
    return f


def test_prunes_only_old_sources(tmp_path):
    old, new = _job(tmp_path, "a", 8), _job(tmp_path, "b", 2)
    assert prune_sources(_cfg(tmp_path)) == 10
    assert not old.exists() and new.exists()


def test_keep_source_disables(tmp_path):
    old = _job(tmp_path, "a", 30)
    assert prune_sources(_cfg(tmp_path, keep_source=True)) == 0
    assert old.exists()
