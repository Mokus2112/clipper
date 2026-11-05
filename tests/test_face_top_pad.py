from clipper.render import CROP_FACE_TOP_MIN, face_top_pad, layout_filter

QA_TOP = 220  # qa.py TOP: must never be relaxed

CFG = {"render": {"width": 1080, "height": 1920, "track_face": False}}
INFO = {"width": 1920, "height": 1080}
SRC = {"campaign_url": "x"}


def out_top(face_top_frac, pad, sh=1080, ch=1080, H=1920):
    t = face_top_frac * sh / ch
    return pad + t * (H - pad)


def test_fox_c1_samples_lifted_above_qa_limit():
    # Fox c1 faces sat at y170-216 in the 1920 output (QA flagged ~45 samples).
    for y_out in (170, 190, 216):
        frac = y_out / 1920
        pad = face_top_pad(frac, 1080, 1080, 1920)
        assert pad > 0
        assert out_top(frac, pad) >= QA_TOP + 20 - 1  # 20 px reserve (even rounding)
        assert out_top(frac, pad) >= CROP_FACE_TOP_MIN - 1


def test_no_pad_when_face_already_low_enough():
    assert face_top_pad(300 / 1920, 1080, 1080, 1920) == 0
    assert face_top_pad(None, 1080, 1080, 1920) == 0


def test_layout_filter_applies_pad_only_with_source():
    f = layout_filter(CFG, INFO, "center", 0.5, None, "z0", "l0", source=SRC, face_top=170 / 1920)
    assert "boxblur" in f and "overlay=0:" in f
    plain = layout_filter(CFG, INFO, "center", 0.5, None, "z0", "l0", source=SRC, face_top=300 / 1920)
    assert "boxblur" not in plain
