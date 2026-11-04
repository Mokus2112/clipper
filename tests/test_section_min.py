import pytest

from clipper.fetch import check_section


def test_short_section_refused():
    with pytest.raises(SystemExit):
        check_section("00:13:19-00:14:41")


def test_long_section_ok_and_none_ok():
    check_section("00:13:19-00:16:00")
    check_section(None)


def test_override(monkeypatch):
    monkeypatch.setenv("CLIPPER_ALLOW_SHORT_SECTION", "1")
    check_section("00:00:00-00:00:30")
