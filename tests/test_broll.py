from pathlib import Path

from clipper.broll import (CARD_BOX, CARD_SEC, HOOK_KEEP, LICENSE, MAX_INSERTS, TAIL_KEEP, card_frame,
                           number_phrases, overlay_graph, plan_inserts)

FONT = Path(__file__).resolve().parents[1] / "fonts" / "Montserrat-ExtraBold.ttf"


def W(w, s, e):
    return {"w": w, "s": s, "e": e}


def test_number_phrases_joins_unit():
    ws = [W("paid", 3, 3.2), W("$1", 3.3, 3.6), W("million", 3.6, 4.0), W("to", 4, 4.1), W("2,400", 5, 5.5), W("stores,", 5.5, 6)]
    ph = number_phrases(ws)
    assert [p["text"] for p in ph] == ["$1 million", "2,400 stores"]
    assert ph[0]["s"] == 3.3 and ph[0]["e"] == 4.0


def test_plan_respects_hook_tail_gap_and_max():
    ph = [{"text": f"{i}", "s": s, "e": s + .4} for i, s in enumerate([1.0, 3.0, 4.0, 7.0, 10.0, 13.0, 19.5])]
    ins = plan_inserts(ph, 21.0)
    assert len(ins) <= MAX_INSERTS
    for a, b in zip(ins, ins[1:]):
        assert b["t0"] >= a["t1"]
    for x in ins:
        assert x["t0"] >= HOOK_KEEP and x["t1"] <= 21.0 - TAIL_KEEP
        assert abs(x["t1"] - x["t0"] - CARD_SEC) < 1e-6 and x["license"] == LICENSE


def test_plan_avoids_spans():
    ph = [{"text": "5", "s": 5.0, "e": 5.4}]
    assert plan_inserts(ph, 30.0, avoid=[(4.0, 8.0)]) == []


def test_card_inside_box_and_motion():
    n = 40
    mid = card_frame("$1 million", "", n // 2, n, FONT)
    first = card_frame("$1 million", "", 0, n, FONT)
    x0, y0, x1, y1 = CARD_BOX
    assert mid.size == (1080, 1920)
    bb = mid.getchannel("A").getbbox()
    assert bb[0] >= x0 and bb[2] <= x1 + 1 and bb[1] >= y0 and bb[3] <= y1 + 1
    assert first.getchannel("A").getbbox() is None or first.getchannel("A").getextrema()[1] < 255  # fades in
    assert y1 <= 1250


def test_overlay_graph_chain():
    g = overlay_graph("cap_a", [{"t0": 3.0}, {"t0": 9.5}], 4, "cap_b", 30)
    assert "[4:v]" in g and "[5:v]" in g and g.endswith("[cap_b]") and "+9.5/TB" in g


def test_number_phrases_glues_split_thousands():
    ws = [W("with", 1, 1.1), W("$20", 2, 2.5), W(",000.", 2.5, 3.0), W("It", 3.1, 3.2)]
    assert [p["text"] for p in number_phrases(ws)] == ["$20,000"]
