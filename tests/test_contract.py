from __future__ import annotations

from jev_coreml.api import _criteria, _state_text


def test_criteria_mapping_preserves_return_keys() -> None:
    keys, descriptions = _criteria({"billing": "payments", "technical": "bugs"}, "choice")
    assert keys == ("billing", "technical")
    assert descriptions == ("payments", "bugs")


def test_noul_has_explicit_default_branches() -> None:
    keys, descriptions = _criteria(None, "noul")
    assert keys == ("true", "false")
    assert len(descriptions) == 2


def test_state_mapping_is_stable_json() -> None:
    assert _state_text({"b": 1, "a": "x"}) == '{"a": "x","b": 1}'
