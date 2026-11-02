import inspect

from clipper import qa


def test_run_all_has_montage_flag():
    assert "montage" in inspect.signature(qa.run_all).parameters
