import re
from kelly_bot import respond


def test_respond_returns_multiline_poem():
    # Force the intent checker to return True during tests
    import kelly_intent
    from unittest.mock import patch
    with patch.object(kelly_intent, "intent_is_technical", return_value=True):
        out = respond("Will AI solve climate change?")
    # must be a non-empty multiline string
    assert isinstance(out, str)
    assert "\n" in out
    lines = out.strip().splitlines()
    assert len(lines) >= 8


def test_poem_includes_skepticism_and_suggestions():
    import kelly_intent
    from unittest.mock import patch
    with patch.object(kelly_intent, "intent_is_technical", return_value=True):
        out = respond("Can models be perfect?")
    lowered = out.lower()
    # Skeptical phrasing
    assert any(w in lowered for w in ("doubt", "question", "uncertainty", "skept")) or "not" in lowered
    # Suggestions: look for a few recommended keywords
    assert "measure" in lowered
    assert "audit" in lowered or "monitor" in lowered


def test_empty_prompt_falls_back():
    import kelly_intent
    from unittest.mock import patch
    # Simulate LLM classifier returning False for empty prompt
    with patch.object(kelly_intent, "intent_is_technical", return_value=False):
        out = respond("")
        assert out.startswith("[Out of scope]")
