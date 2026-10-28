"""Entscheider (docs/entwurfssteuerung.md, Abschnitt 2.1): Mensch, Wiedereinspieler.

Alle Entscheider liefern eine Antwort ``mikroptim.design.action/1`` zu einem Zustand
``mikroptim.design.state/1``. Das Sprachmodell (OllamaDecider) und die feste Regel
(RuleDecider) folgen als eigene Aufgaben.
"""

from __future__ import annotations

import json
import sys

from .state import format_state


class Decider:
    kind = "abstrakt"

    def decide(self, state: dict, session) -> dict:  # pragma: no cover - Schnittstelle
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind}


class HumanDecider(Decider):
    """Mensch auf der Kommandozeile: sieht den Zustand als Text und gibt die Antwort als JSON
    ein (eine Zeile oder ein Dateiname mit vorangestelltem ``@``)."""

    kind = "mensch"

    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def decide(self, state, session):
        print(format_state(state), file=self.stdout)
        print("Antwort (JSON mikroptim.design.action/1, oder @datei.json):", file=self.stdout)
        line = self.stdin.readline().strip()
        if line.startswith("@"):
            with open(line[1:], encoding="utf-8") as f:
                return json.load(f)
        return json.loads(line)


class ReplayDecider(Decider):
    """Liefert die Antworten eines Protokolls in ihrer Reihenfolge (Wiedereinspielen)."""

    kind = "wiedereinspielen"

    def __init__(self, log_entries):
        self.entries = list(log_entries)
        self.pos = 0

    def next_candidate(self):
        return self.entries[self.pos]["candidate"] if self.pos < len(self.entries) else None

    def decide(self, state, session):
        e = self.entries[self.pos]
        self.pos += 1
        return e["response"]

    def describe(self):
        d = self.entries[self.pos - 1]["decider"] if self.pos else {}
        return {"kind": self.kind, "original": d}


def replay(log_entries, start_system, directory, *, spec=None, budget=None):
    """Spielt ein Protokoll in einer neuen Sitzung ab; liefert die neue Sitzung."""
    from .session import Session

    s = Session.create(start_system, directory, spec=spec, budget=budget)
    d = ReplayDecider(log_entries)
    while d.next_candidate() is not None:
        s.step(d, d.next_candidate())
    return s
