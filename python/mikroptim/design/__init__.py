"""Entwurfswerkzeuge mit Mensch oder Sprachmodell als Entscheider (M10, ADR-0012).

Grundgerüst (M10-T6a): Zustand ``mikroptim.design.state/1``, Aktionen
``mikroptim.design.action/1`` über die Kommandozeile von Mikroptim, Kandidatenbaum,
Protokoll ``mikroptim.design.log/1`` und Wiedereinspielen. Beschreibung:
docs/entwurfssteuerung.md; Kommandozeile: ``python -m mikroptim.design --help``.

Beispiel::

    from mikroptim import design
    s = design.Session.create("start.json", "sitzung", spec={"tube_focal_length": 164.5})
    print(design.format_state(s.state(0)))
    s.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                "actions": [{"action": "free_gaps"}, {"action": "optimize", "cycles": 50}]},
            decider={"kind": "mensch"})
"""

from .actions import ActionError, SCHEMA_ACTIONS
from .deciders import Decider, HumanDecider, ReplayDecider, replay
from .session import Session
from .state import build_state, format_state

__all__ = ["ActionError", "Decider", "HumanDecider", "ReplayDecider", "SCHEMA_ACTIONS",
           "Session", "build_state", "format_state", "replay"]
