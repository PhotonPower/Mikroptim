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


#: Stufen der Regel bei Stillstand der Homotopie (M10-T7b), in dieser Reihenfolge je einmal
ESCALATION = ("free_gaps", "aplanatic_shell", "ghost_dictionary", "null_meniscus")
#: Kennung der Stufen in der Begründung der ersten Aktion (Protokoll)
_STALL_TAG = "Stillstand:"


class RuleDecider(Decider):
    """Feste Vergleichsstrategie ohne Sprachmodell (V3, docs/entwurfssteuerung.md, Abschnitt 8),
    auf denselben Aktionen und Diagnosen. Bewusst einfach.

    Reihenfolge:

    1. Ziel erreicht (aktuelle NA ≥ Soll-NA der Spezifikation und RMS ≤ Ziel) → abschließen.
    2. Apertur offen (``homotopy_point.aperture`` < ``spec.na``): Homotopie-Abschnitt
       ``synth_segment`` zur Soll-NA mit höchstens ``max_new_elements`` neuen Elementen
       (``synth_options``: zusätzliche Parameter des Abschnitts, z. B. ``augmented_lagrangian``,
       ``sine_condition``). Erreicht er die Soll-NA nicht (Stillstand, meist ``step_too_small``)
       oder scheitert er, folgt die nächste noch nicht benutzte Stufe aus ``escalation`` und
       danach ein neuer Abschnitt (M10-T7b), höchstens ``max_synth`` Abschnitte pro Sitzung:

       - ``free_gaps``: Spalte freigeben und optimieren (wie das M6-T6-Skript, F2),
       - ``aplanatic_shell``: aplanatisch-konzentrische Schale (``front.shell_ca``) im
         Arbeitsabstand aus dem hochbrechendsten Katalogglas; sie übernimmt einen Teil der
         Apertur ohne sphärische Aberration aller Ordnungen, danach Spalte freigeben und
         optimieren (die Brennweitenbedingung verteilt die Brechkraft neu). Die Schale
         verkürzt den Arbeitsabstand etwa auf (w − p)/n′ − t; die Stufe ist nur anwendbar, wenn
         dabei mindestens 1,05 w_min und t ≥ 0,1 mm bleiben. (Platz schaffen, indem die
         Schranke des Arbeitsabstands erst gehoben wird, scheiterte im Vorlauf M10-T7b an der
         Falle F12 und ist deshalb nicht Teil der Regel.)
       - ``ghost_dictionary``: Geisterplatten-Wörterbuch in der Mitte des Luftspalts mit der
         größten Randstrahlhöhe (Bedarf sichtbar machen, F3), danach sofort der Abschnitt,
       - ``null_meniscus``: Null-Meniskus mit Dicken-Homotopie im selben Luftspalt (Sphärik),
         danach Spalte freigeben und optimieren.

       Eine Stufe, die am Kandidaten nicht anwendbar ist (kein passender Spalt), wird
       übersprungen; eine Stufe, deren Aktion scheitert, gilt als benutzt. Sind alle Stufen
       benutzt, geht es mit ``free_gaps`` + ``optimize`` und einem neuen Abschnitt weiter,
       solange der letzte Abschnitt die NA um mindestens ``PROGRESS_NA`` geöffnet hat. Ein
       gescheiterter Abschnitt wird auf demselben Kandidaten nicht wiederholt.
    3. Nach einem strukturellen Schritt: Spalte freigeben → optimieren.
    4. Farbe (nur bei mehreren Wellenlängen; größter |dW020|-Beitrag einer Gruppe über
       ``color_limit``) mit split_lens im Element dieser Gruppe → sphärische Aberration (|W040|
       der Summe über ``spherical_limit``) mit insert_null_meniscus im Luftspalt der größten
       Randstrahlhöhe → abschließen, sobald keine Regel greift.
    """

    kind = "regel"

    def __init__(self, rms_target=0.0714, color_limit=0.25, spherical_limit=0.25,
                 glass="N-BK7", cycles=100, min_gap=1.0, max_new_elements=4, max_synth=5,
                 escalation=ESCALATION, shell_glass=None, synth_options=None):
        self.min_gap = min_gap
        self.rms_target = rms_target
        self.color_limit = color_limit
        self.spherical_limit = spherical_limit
        self.glass = glass
        self.cycles = cycles
        self.max_new_elements = max_new_elements
        self.max_synth = max_synth
        unknown = [e for e in escalation if e not in ESCALATION]
        if unknown:
            raise ValueError(f"unbekannte Stufen {unknown} (bekannt: {list(ESCALATION)})")
        self.escalation = tuple(escalation)
        self.shell_glass = shell_glass
        self.synth_options = dict(synth_options or {})

    def describe(self):
        return {"kind": self.kind, "rms_target": self.rms_target, "color_limit": self.color_limit,
                "spherical_limit": self.spherical_limit, "glass": self.glass,
                "max_new_elements": self.max_new_elements, "max_synth": self.max_synth,
                "escalation": list(self.escalation), "shell_glass": self.shell_glass,
                "synth_options": self.synth_options}

    def _synth(self, target_na, reason="Soll-NA der Spezifikation"):
        return {"action": "synth_segment", "target": {"aperture": target_na},
                "max_new_elements": self.max_new_elements, **self.synth_options,
                "reason": reason}

    def decide(self, state, session):
        c = state["candidate"]
        last = session.candidates()[c].get("action")
        rms = state["merit"]["rms_max_waves"]
        ap = (state.get("homotopy_point") or {}).get("aperture")
        target_na = state["spec"].get("na")
        open_ap = ap is not None and target_na is not None and ap < target_na - 1e-6
        if not open_ap and rms is not None and rms <= self.rms_target:
            return _response(c, "Ziel erreicht", {"action": "finish", "reason": "RMS ≤ Ziel"})
        log = session.log()
        synths = sum(1 for e in log for r in e.get("results", [])
                     if r.get("action") == "synth_segment" and r.get("status") == "ok")
        # ein gescheiterter Abschnitt wird nicht wiederholt (Fehler stehen in last_errors)
        synth_failed = any(str(e).startswith("synth_segment")
                           for e in state.get("last_errors") or [])
        if open_ap and synths < self.max_synth:
            if last == "synth_segment" or synth_failed:
                used = _stall_steps(log)
                for step in self.escalation:
                    if step in used:
                        continue
                    resp = self._stall_step(step, state, session, ap, target_na)
                    if resp is not None:
                        return resp
                # alle Stufen benutzt: weiter, solange der letzte Abschnitt die NA merklich
                # geöffnet hat (Spalte freigeben, optimieren, nächster Abschnitt)
                points = _synth_apertures(log)
                if last == "synth_segment" and len(points) >= 2 and \
                        points[-1] - points[-2] >= PROGRESS_NA:
                    return _response(
                        c, f"NA {points[-2]:.3f} → {points[-1]:.3f} im letzten Abschnitt: weiter",
                        {"action": "free_gaps", "reason": f"{_STALL_TAG} repeat"},
                        {"action": "optimize", "cycles": self.cycles})
            elif last != "free_gaps":
                return _response(c, f"Apertur öffnen: NA {ap:.3f} → {target_na:.3f}",
                                 self._synth(target_na))
        if last in (None, "insert_null_meniscus", "split_lens"):
            return _response(c, "Spalte freigeben und optimieren", {"action": "free_gaps"},
                             {"action": "optimize", "cycles": self.cycles})
        if last == "free_gaps":
            return _response(c, "optimieren", {"action": "optimize", "cycles": self.cycles})
        groups = state["groups"]
        color = max(((g, v.get("dW020") or 0.0) for g, v in groups.items() if v.get("elements")),
                    key=lambda t: abs(t[1]), default=(None, 0.0))
        if not state["spec"].get("monochromatic") and abs(color[1]) > self.color_limit:
            element = groups[color[0]]["elements"][0]
            return _response(c, f"Farblängsfehler {color[1]:.2f} Wellen in {color[0]}",
                             {"action": "split_lens", "element": element, "free_part": 1,
                              "reason": f"dW020 der Gruppe {color[0]}"})
        total_w040 = sum((v.get("W040") or 0.0) for v in groups.values())
        if abs(total_w040) > self.spherical_limit:
            gap = _widest_height_gap(session, c, self.min_gap)
            if gap is not None:
                return _response(c, f"sphärische Aberration {total_w040:.2f} Wellen",
                                 {"action": "insert_null_meniscus", "after_surface": gap,
                                  "glass": {"name": self.glass}, "saddle_branches": True,
                                  "reason": "W040 der Summe"})
        return _response(c, "keine Regel greift", {"action": "finish", "reason": "keine Regel"})

    # --- Stufen bei Stillstand ------------------------------------------------------------------

    def _stall_step(self, step, state, session, ap, target_na):
        """Antwort für eine Stufe bei Stillstand, ``None`` wenn am Kandidaten nicht anwendbar."""
        c = state["candidate"]
        tag = f"{_STALL_TAG} {step}"
        where = f"NA {ap:.3f} < {target_na:.3f}"
        optimize = {"action": "optimize", "cycles": self.cycles}
        if step == "free_gaps":
            return _response(c, f"{where} nach dem Abschnitt: Spalte freigeben und optimieren",
                             {"action": "free_gaps", "reason": tag}, optimize)
        path = session.system_path(c)
        data = read_system(path)
        n = len(data["surfaces"])
        if step == "aplanatic_shell":
            wd_entry = _wd_entry(data)
            if wd_entry is None or not data["surfaces"][n - 1]["thickness"] > 0:
                return None
            return self._shell(state, session, ap, target_na, float(wd_entry.get("min", 0.0)))
        gap = _widest_height_gap(session, c, STALL_MIN_GAP)
        if gap is None:
            return None
        width = data["surfaces"][gap]["thickness"]
        if step == "ghost_dictionary":
            glasses = [g["glass"] for g in data.get("ghost_plates", [])
                       if isinstance(g.get("glass"), str)]
            hi = self.shell_glass or high_index_glass(path)
            glasses = list(dict.fromkeys(([hi] if hi else []) + glasses))[:4] or [self.glass]
            return _response(
                c, f"{where}: Wörterbuch an der größten Randstrahlhöhe macht Bedarf sichtbar",
                {"action": "place_ghost_dictionary", "after_surface": gap,
                 "distance": width / 2, "glasses": glasses,
                 "reason": f"{tag} Spalt nach Fläche {gap} (F3)"},
                self._synth(target_na, "nach dem Wörterbuch"))
        if step == "null_meniscus":
            # Lage in der Mitte; der Luftspalt dahinter (T/2 − t) muss die untere Schranke der
            # Dickenvariable des Spalts halten (Vorlauf M10-T7b: sonst lehnte optimize ab)
            lb = max([float(it.get("min", 0.0)) for it in (data.get("variables") or {})
                      .get("items", []) if it.get("type") == "thickness"
                      and (it.get("surface") == gap or gap in (it.get("surfaces") or []))]
                     or [0.0])
            thickness = round(min(1.0, 0.4 * width, 0.5 * width - lb - 0.01), 4)
            if thickness < 0.05:
                return None
            glass = self.shell_glass or high_index_glass(path) or self.glass
            return _response(
                c, f"{where}: Null-Meniskus mit Dicken-Homotopie an der größten Randstrahlhöhe",
                {"action": "insert_module", "module": "generic.null_meniscus",
                 "after_surface": gap, "glasses": [{"name": glass}],
                 "thickness": thickness, "stages": 4, "cycles": 50,
                 "group": "middle", "reason": f"{tag} sphärische Aberration der Mittelgruppe"},
                {"action": "free_gaps"}, optimize)
        return None

    def _shell_glass(self, path):
        """Glas der Schale und seine Brechzahl n₀ (für die Abschätzung des Arbeitsabstands)."""
        table = glass_indices(path)
        name = self.shell_glass or (max(table, key=lambda k: (table[k], k)) if table
                                    else self.glass)
        return name, table.get(name, 1.5)

    def _shell(self, state, session, ap, target_na, wd_min):
        """Schale im Arbeitsabstand: Lage p, Dicke t so, dass der nachfokussierte Arbeitsabstand
        (w − p)/n′ − t etwa 1,05 w_min ist; ``None``, wenn dafür t < 0,1 mm wäre."""
        c = state["candidate"]
        path = session.system_path(c)
        data = read_system(path)
        n = len(data["surfaces"])
        w = data["surfaces"][n - 1]["thickness"]
        glass, n_glass = self._shell_glass(path)
        t = (w - SHELL_POSITION) / n_glass - 1.05 * wd_min
        if t < SHELL_MIN_THICKNESS:
            return None
        t = round(min(t, SHELL_MAX_THICKNESS), 4)
        return _response(
            c, f"NA {ap:.3f} < {target_na:.3f}: aplanatische Schale vor dem Objekt übernimmt "
               "Apertur",
            {"action": "insert_module", "module": "front.shell_ca", "after_surface": n - 1,
             "position": SHELL_POSITION, "thickness": t, "glasses": [{"name": glass}],
             "group": "front",
             "reason": f"{_STALL_TAG} aplanatic_shell Frontgruppe, W040 aller Ordnungen "
                       "unverändert"},
            {"action": "free_gaps"}, {"action": "optimize", "cycles": self.cycles})


#: kleinster Luftspalt [mm] für Wörterbuch und Null-Meniskus der Stufen bei Stillstand
STALL_MIN_GAP = 0.4
#: aplanatische Schale im Arbeitsabstand: Lage der ersten Fläche und Grenzen der Dicke [mm]
SHELL_POSITION = 0.05
SHELL_MIN_THICKNESS = 0.1
SHELL_MAX_THICKNESS = 2.0
#: kleinster Fortschritt der NA in einem Abschnitt, mit dem die Regel nach allen Stufen weitermacht
PROGRESS_NA = 0.01


def _synth_apertures(log) -> list:
    """Erreichte NA (Punkt des Controllers) aller erfolgreichen Abschnitte, in Reihenfolge."""
    out = []
    for e in log:
        for r in e.get("results", []):
            if r.get("action") == "synth_segment" and r.get("status") == "ok":
                for rep in r.get("reports") or []:
                    ap = ((rep.get("result") or {}).get("point") or {}).get("aperture")
                    if ap is not None:
                        out.append(float(ap))
    return out


def _wd_entry(data):
    return next((it for it in (data.get("variables") or {}).get("items", [])
                 if it.get("type") == "working_distance"), None)


def _widest_height_gap(session, c, min_gap):
    """Luftspalt (nicht der Arbeitsabstand) mit der größten Randstrahlhöhe und Dicke ≥
    ``min_gap``; ``None``, wenn es keinen gibt."""
    table = read_surfaces(session, c)
    data = read_system(session.system_path(c))
    gaps = [s for s in table if s["n_after"] == 1.0 and 0 < s["surface"] < len(table) - 1
            and data["surfaces"][s["surface"]]["thickness"] >= min_gap]
    if not gaps:
        return None
    return max(gaps, key=lambda s: abs(s["y"]))["surface"]


def _stall_steps(log) -> set:
    """Bereits versuchte Stufen (Begründung der ersten Aktion trägt die Kennung), auch wenn ihre
    Aktion scheiterte – sonst wiederholte die Regel eine scheiternde Stufe bis zum Budgetende."""
    used = set()
    for e in log:
        acts = (e.get("response") or {}).get("actions") or []
        reason = str((acts[0] if acts else {}).get("reason") or "")
        if reason.startswith(_STALL_TAG):
            used.add(reason[len(_STALL_TAG):].split()[0])
    return used


def glass_indices(system_path) -> dict:
    """Brechzahl n₀ (Buchdahl-Fit bei λ_ref) aller Kataloggläser mit Fit aus der Glasvorauswahl
    (``mikroptim glasses``); leer ohne Kataloggläser."""
    from ._cli import ToolError, cli_json

    try:
        rep = cli_json(["glasses", "--system", system_path, "--order", 1, "--top", 5000,
                        "--max-power-sum", 1e12])
    except ToolError:
        return {}
    n0 = {}
    for r in rep.get("results", []):
        for name, d in zip(r["glasses"], r["dispersion"]):
            n0[name] = d["n0"]
    return n0


def high_index_glass(system_path):
    """Katalogglas mit der größten Brechzahl n₀; ``None`` ohne Kataloggläser."""
    n0 = glass_indices(system_path)
    return max(n0, key=lambda k: (n0[k], k)) if n0 else None


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
