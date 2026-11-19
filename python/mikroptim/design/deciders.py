"""Entscheider (docs/entwurfssteuerung.md, Abschnitt 2.1): Mensch, Regel, Bestätigung,
Wiedereinspieler; das Sprachmodell in ``ollama.py``.

Alle Entscheider liefern eine Antwort ``mikroptim.design.action/1`` zu einem Zustand
``mikroptim.design.state/1``.
"""

from __future__ import annotations

import json
import sys

from .files import read_system
from .state import format_state


class DecisionError(RuntimeError):
    """Ein Entscheider konnte keine gültige Antwort liefern (z. B. Modell nach Nachbesserungen)."""


def _response(candidate, assessment, *actions):
    return {"format": "mikroptim.design.action/1", "candidate": candidate,
            "assessment": assessment, "actions": list(actions)}


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


class RuleDecider(Decider):
    """Feste Vergleichsstrategie ohne Sprachmodell (V3, docs/entwurfssteuerung.md, Abschnitt 8),
    auf denselben Aktionen und Diagnosen. Bewusst einfach (Ausbau M10-T7: Apertur).

    Reihenfolge:

    1. Ziel erreicht (aktuelle NA ≥ Soll-NA der Spezifikation und RMS ≤ Ziel) → abschließen.
    2. Apertur offen (``homotopy_point.aperture`` < ``spec.na``): Homotopie-Abschnitt
       ``synth_segment`` zur Soll-NA mit höchstens ``max_new_elements`` neuen Elementen; erreicht
       er die Soll-NA nicht, zuerst Spalte freigeben und optimieren (wie das M6-T6-Skript, F2),
       dann erneut; höchstens ``max_synth`` Abschnitte pro Sitzung.
    3. Nach einem strukturellen Schritt: Spalte freigeben → optimieren.
    4. Farbe (größter |dW020|-Beitrag einer Gruppe über ``color_limit``) mit split_lens im Element
       dieser Gruppe → sphärische Aberration (|W040| der Summe über ``spherical_limit``) mit
       insert_null_meniscus im Luftspalt der größten Randstrahlhöhe → abschließen, sobald keine
       Regel greift.
    """

    kind = "regel"

    def __init__(self, rms_target=0.0714, color_limit=0.25, spherical_limit=0.25,
                 glass="N-BK7", cycles=100, min_gap=1.0, max_new_elements=4, max_synth=4):
        self.min_gap = min_gap
        self.rms_target = rms_target
        self.color_limit = color_limit
        self.spherical_limit = spherical_limit
        self.glass = glass
        self.cycles = cycles
        self.max_new_elements = max_new_elements
        self.max_synth = max_synth

    def describe(self):
        return {"kind": self.kind, "rms_target": self.rms_target, "color_limit": self.color_limit,
                "spherical_limit": self.spherical_limit, "glass": self.glass,
                "max_new_elements": self.max_new_elements, "max_synth": self.max_synth}

    def decide(self, state, session):
        c = state["candidate"]
        last = session.candidates()[c].get("action")
        rms = state["merit"]["rms_max_waves"]
        ap = (state.get("homotopy_point") or {}).get("aperture")
        target_na = state["spec"].get("na")
        open_ap = ap is not None and target_na is not None and ap < target_na - 1e-6
        if not open_ap and rms is not None and rms <= self.rms_target:
            return _response(c, "Ziel erreicht", {"action": "finish", "reason": "RMS ≤ Ziel"})
        synths = sum(1 for e in session.log() for r in e.get("results", [])
                     if r.get("action") == "synth_segment" and r.get("status") == "ok")
        # ein gescheiterter Abschnitt wird nicht wiederholt (Fehler stehen in last_errors)
        synth_failed = any(str(e).startswith("synth_segment")
                           for e in state.get("last_errors") or [])
        if open_ap and synths < self.max_synth and not synth_failed:
            if last == "synth_segment":
                return _response(c, f"NA {ap:.3f} < {target_na:.3f} nach dem Abschnitt: Spalte "
                                 "freigeben und optimieren", {"action": "free_gaps"},
                                 {"action": "optimize", "cycles": self.cycles})
            if last != "free_gaps":
                return _response(c, f"Apertur öffnen: NA {ap:.3f} → {target_na:.3f}",
                                 {"action": "synth_segment", "target": {"aperture": target_na},
                                  "max_new_elements": self.max_new_elements,
                                  "reason": "Soll-NA der Spezifikation"})
        if last in (None, "insert_null_meniscus", "split_lens"):
            return _response(c, "Spalte freigeben und optimieren", {"action": "free_gaps"},
                             {"action": "optimize", "cycles": self.cycles})
        if last == "free_gaps":
            return _response(c, "optimieren", {"action": "optimize", "cycles": self.cycles})
        groups = state["groups"]
        color = max(((g, v.get("dW020") or 0.0) for g, v in groups.items() if v.get("elements")),
                    key=lambda t: abs(t[1]), default=(None, 0.0))
        if abs(color[1]) > self.color_limit:
            element = groups[color[0]]["elements"][0]
            return _response(c, f"Farblängsfehler {color[1]:.2f} Wellen in {color[0]}",
                             {"action": "split_lens", "element": element, "free_part": 1,
                              "reason": f"dW020 der Gruppe {color[0]}"})
        total_w040 = sum((v.get("W040") or 0.0) for v in groups.values())
        if abs(total_w040) > self.spherical_limit:
            table = read_surfaces(session, c)
            data = read_system(session.system_path(c))
            # Luftspalte mit Platz für den Meniskus (die Hälften müssen über dem kleinsten
            # Luftspalt der Variablen liegen)
            gaps = [s for s in table if s["n_after"] == 1.0 and 0 < s["surface"] < len(table) - 1
                    and data["surfaces"][s["surface"]]["thickness"] >= self.min_gap]
            if gaps:
                gap = max(gaps, key=lambda s: abs(s["y"]))
                return _response(c, f"sphärische Aberration {total_w040:.2f} Wellen",
                                 {"action": "insert_null_meniscus", "after_surface": gap["surface"],
                                  "glass": {"name": self.glass}, "saddle_branches": True,
                                  "reason": "W040 der Summe"})
        return _response(c, "keine Regel greift", {"action": "finish", "reason": "keine Regel"})


def read_surfaces(session, candidate):
    from .actions import read_action

    return read_action({"action": "get_surface_table"}, session.system_path(candidate))["surfaces"]


class ConfirmDecider(Decider):
    """Ein Entscheider schlägt vor, ein Mensch bestätigt: leere Eingabe = annehmen, ``x`` =
    verwerfen (Antwort ``abandon``), sonst eine geänderte Antwort als JSON."""

    kind = "bestaetigung"

    def __init__(self, proposer: Decider, stdin=None, stdout=None):
        self.proposer = proposer
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.last_choice = None

    def describe(self):
        return {"kind": self.kind, "proposer": self.proposer.describe(), "choice": self.last_choice}

    def decide(self, state, session):
        proposal = self.proposer.decide(state, session)
        print(format_state(state), file=self.stdout)
        print("Vorschlag:", json.dumps(proposal, ensure_ascii=False, indent=2), file=self.stdout)
        print("Annehmen (Enter), verwerfen (x) oder geänderte Antwort (JSON):", file=self.stdout)
        line = self.stdin.readline().strip()
        if not line:
            self.last_choice = "angenommen"
            return proposal
        if line.lower() == "x":
            self.last_choice = "verworfen"
            return _response(state["candidate"], "vom Menschen verworfen",
                             {"action": "abandon", "reason": "Vorschlag verworfen"})
        self.last_choice = "geändert"
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
