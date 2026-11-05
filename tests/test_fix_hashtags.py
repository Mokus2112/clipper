from clipper.rank import fix_hashtags

SRC = {"streamer": "Alliance for Secure AI", "tags": ["aisafety", "ai", "airisk"], "required_text": ""}


def test_typo_snaps_to_campaign_tag():
    assert fix_hashtags(["allitforsecureai", "aisafety"], SRC)[0] == "allianceforsecureai"
    assert fix_hashtags(["alliCforsecureai", "airisks"], SRC)[0] == "allianceforsecureai"


def test_streamer_tag_added_when_missing_and_unrelated_kept():
    out = fix_hashtags(["aiagents"], SRC)
    assert out == ["allianceforsecureai", "aiagents"]


def test_required_text_tag_is_canonical():
    src = {"streamer": "Lacy", "tags": [], "required_text": "#LacyStream"}
    assert fix_hashtags(["lacystreem"], src) == ["lacystream"]
