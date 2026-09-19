from __future__ import annotations

from jev_coreml.laya import CoreMLLayaAgent


def test_laya_choice_keeps_criteria_labels() -> None:
    internal = CoreMLLayaAgent._internal(
        {"type": "choice", "instructions": "route it", "criteria": {"billing": "money", "technical": "bugs"}}
    )
    assert internal == {"t": "choice", "ins": "route it", "crit": {"billing": "money", "technical": "bugs"}}


def test_laya_noul_accepts_omitted_criteria() -> None:
    internal = CoreMLLayaAgent._internal({"type": "noul", "instructions": "refund?"})
    assert internal == {"t": "noul", "ins": "refund?", "crit": {}}
