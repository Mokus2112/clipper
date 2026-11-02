from clipper.render import AUDIO_FADE_IN, AUDIO_FADE_OUT, audio_chain


def test_both_edges_fade_at_least_40_ms():
    assert AUDIO_FADE_IN >= 0.04 and AUDIO_FADE_OUT >= 0.04


def test_chain_fades_in_at_start_and_out_at_end():
    c = audio_chain(3, AUDIO_FADE_IN, AUDIO_FADE_OUT, 1.0, 1.0333)
    assert c.startswith("[3:a:0]afade=t=in:st=0:d=0.04,")
    assert "afade=t=out:st=0.9500:d=0.05" in c and c.endswith("atrim=end=1.033300")
