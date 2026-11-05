"""qa/syncmeter find the active word by the configured captions.highlight, not a hard-coded yellow."""
import pytest

from clipper import qa
from clipper.qa import LEGACY_HIGHLIGHT, highlight_re

SENT = "&H00C8C8C8"


def word(line, color):
    m = highlight_re(color).search(line)
    return m.group(1) if m else None


def test_legacy_yellow_still_matches():
    assert word(r"{\3c&H00000000\c&H0000E0FF}HELLO{\3c&H00000000\c&H00FFFFFF} WORLD", LEGACY_HIGHLIGHT) == "HELLO"


def test_white_highlight_matches_the_active_word_not_the_grey_sentence():
    line = r"{\3c&H00000000\c&H00FFFFFF}THIS{\3c&H00000000\c&H00C8C8C8} IS FINE"
    assert word(line, "&H00FFFFFF") == "THIS"
    assert word(line, SENT) == " IS FINE"  # the sentence colour only "matches" if it is configured as highlight


def test_sentence_colour_text_is_never_an_active_word():
    assert word(r"THIS {\c&H00C8C8C8}IS", "&H00FFFFFF") is None


@pytest.mark.parametrize("cfg", ["&H00FFFFFF", "&HFFFFFF&", "&hffffff", "&H00ffffff", "FFFFFF"])
def test_colour_format_and_case_are_normalised(cfg):
    for tag in (r"{\c&H00FFFFFF}A", r"{\c&HFFFFFF&}A", r"{\c&H00ffffff}A"):
        assert word(tag, cfg) == "A"


def test_alpha_is_respected():
    assert word(r"{\c&H80FFFFFF}A", "&H00FFFFFF") is None
    assert word(r"{\c&H80FFFFFF}A", "&H80FFFFFF") == "A"


def test_yellow_does_not_match_white_config():
    assert word(r"{\c&H0000E0FF}A", "&H00FFFFFF") is None


def test_bad_or_missing_colour_falls_back_to_legacy_yellow():
    for bad in ("", "red", "&H12", None):
        if bad is None:
            continue
        assert word(r"{\c&H0000E0FF}A", bad) == "A"


def test_default_reads_config(monkeypatch):
    import clipper.common as common
    monkeypatch.setattr(common, "load_config", lambda: {"captions": {"highlight": "&H00FFFFFF"}})
    assert highlight_re().search(r"{\c&H00FFFFFF}A")
    monkeypatch.setattr(common, "load_config", lambda: {})
    assert highlight_re().search(r"{\c&H0000E0FF}A")
