from clipper.rank import PROMPT


def test_prompt_requires_natural_closure_and_sentence_end():
    assert "NATURAL CLOSURE" in PROMPT
    assert "END at the end of a sentence" in PROMPT
