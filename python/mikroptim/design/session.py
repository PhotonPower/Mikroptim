"""Entwurfssitzung: Kandidatenbaum, Ausführung von Entscheidungen, Protokoll.

Verzeichnis einer Sitzung::

    <dir>/session.json             Start, Spezifikation, Budget
    <dir>/candidates/<id>.json     Systemdatei jedes Kandidaten
    <dir>/log.jsonl                Protokoll mikroptim.design.log/1, eine Entscheidung pro Zeile

Kandidatennummern werden fortlaufend vergeben; dieselben Entscheidungen in derselben
Reihenfolge ergeben denselben Baum (Wiedereinspielen, ReplayDecider).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

from ._cli import ToolError
from .actions import (CONTROL_ACTIONS, MODIFY_ACTIONS, READ_ACTIONS, ActionError, modify_action,
                      read_action, validate_response)
from .files import read_system, resolve_catalogs, write_system
from .state import build_state

LOG_FORMAT = "mikroptim.design.log/1"
SESSION_FORMAT = "mikroptim.design.session/1"


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


class Session:
    def __init__(self, directory):
        self.dir = pathlib.Path(directory)
        self.meta = json.loads((self.dir / "session.json").read_text(encoding="utf-8"))

    # --- Anlegen und Ablage -------------------------------------------------------------------

    @classmethod
    def create(cls, start_system, directory, *, spec=None, budget=None) -> "Session":
        """Neue Sitzung mit Kandidat 0 = Startsystem (Glaskataloge als absolute Pfade)."""
        d = pathlib.Path(directory)
        if (d / "session.json").exists():
            raise FileExistsError(f"{d}: Sitzung existiert bereits")
        (d / "candidates").mkdir(parents=True, exist_ok=True)
        data = resolve_catalogs(read_system(start_system), pathlib.Path(start_system).parent)
        write_system(data, d / "candidates" / "0.json")
        meta = {
            "format": SESSION_FORMAT, "start": str(pathlib.Path(start_system).resolve()), "spec": spec or {},
            "budget": {"actions": 50, **(budget or {})},
            "candidates": {"0": {"parent": None, "status": "open", "action": None}},
            "next_id": 1, "decisions": 0,
        }
        (d / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        (d / "log.jsonl").write_text("", encoding="utf-8")
        return cls(d)

    def _save(self):
        (self.dir / "session.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def system_path(self, candidate: int) -> pathlib.Path:
        return self.dir / "candidates" / f"{candidate}.json"

    def candidates(self) -> dict:
        return {int(k): v for k, v in self.meta["candidates"].items()}

    def open_candidates(self) -> list[int]:
        return [k for k, v in self.candidates().items() if v["status"] == "open"]

    def _new_candidate(self, source_file, parent, action) -> int:
        cid = self.meta["next_id"]
        self.meta["next_id"] += 1
        shutil.move(str(source_file), self.system_path(cid))
        self.meta["candidates"][str(cid)] = {"parent": parent, "status": "open",
                                            "action": action.get("action")}
        return cid

    def history(self) -> list[str]:
        out = []
        for e in self.log():
            for r in e.get("results", []):
                if r.get("new_candidates"):
                    out.append(f"#{e['candidate']}: {r['action']} → {r['new_candidates']}")
        return out

    def log(self) -> list[dict]:
        text = (self.dir / "log.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    # --- Zustand und Entscheidung -------------------------------------------------------------

    def state(self, candidate: int, last_errors=None) -> dict:
        c = self.candidates()[candidate]
        budget = {"actions_left": self.meta["budget"]["actions"] - self.meta["decisions"],
                  **{k: v for k, v in self.meta["budget"].items() if k != "actions"}}
        return build_state(self.system_path(candidate), candidate=candidate, parent=c["parent"],
                           spec=self.meta["spec"], budget=budget, history=self.history(),
                           last_errors=last_errors)

    def apply(self, candidate: int, response: dict, decider: dict | None = None,
              state: dict | None = None) -> dict:
        """Führt eine Antwort ``mikroptim.design.action/1`` auf einem Kandidaten aus.

        Aktionen laufen nacheinander; jede verändernde Aktion erzeugt einen neuen Kandidaten,
        auf dem die nächste aufsetzt. ``branch`` führt Alternativen getrennt ab dem
        Ausgangskandidaten aus. Fehler werden pro Aktion gemeldet, die folgenden Aktionen der
        Antwort laufen dann nicht mehr. Rückgabe und Protokolleintrag: Ergebnisse pro Aktion.
        """
        if candidate not in self.candidates():
            raise KeyError(f"Kandidat {candidate} existiert nicht")
        results = []
        try:
            actions = validate_response(response)
        except ActionError as e:
            results.append({"action": None, "status": "error", **e.as_dict()})
            actions = []
        current = candidate
        for a in actions:
            res = self._apply_one(current, a)
            results.append(res)
            if res["status"] != "ok":
                break
            if res.get("new_candidates") and a["action"] != "branch":
                current = res["new_candidates"][-1]
        self.meta["decisions"] += 1
        entry = {"format": LOG_FORMAT, "seq": self.meta["decisions"], "candidate": candidate,
                 "decider": decider or {"kind": "unbekannt"},
                 "state_sha256": _sha256(state) if state is not None else None,
                 "response": response, "results": results}
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._save()
        return entry

    def _apply_one(self, current: int, a: dict) -> dict:
        name = a["action"]
        try:
            if name in READ_ACTIONS:
                return {"action": name, "status": "ok",
                        "result": read_action(a, self.system_path(current))}
            if name in MODIFY_ACTIONS:
                outs = modify_action(a, self.system_path(current), self.dir / "tmp")
                new = [self._new_candidate(path, current, a) for path, _ in outs]
                return {"action": name, "status": "ok", "new_candidates": new,
                        "reports": [rep for _, rep in outs]}
            if name in CONTROL_ACTIONS:
                return self._control(current, a)
            raise ActionError("not_implemented", f"{name} ist noch nicht umgesetzt")
        except ActionError as e:
            return {"status": "error", **e.as_dict(name)}
        except ToolError as e:
            return {"status": "error", "action": name, "code": "tool_failed", "message": str(e)}

    def _control(self, current: int, a: dict) -> dict:
        name = a["action"]
        if name in ("abandon", "finish"):
            self.meta["candidates"][str(current)]["status"] = \
                "abandoned" if name == "abandon" else "finished"
            self.meta["candidates"][str(current)]["reason"] = a.get("reason")
            return {"action": name, "status": "ok", "candidate": current}
        # branch: jede Alternative ab demselben Kandidaten
        new_all, sub = [], []
        for alt in a.get("alternatives", []):
            cur = current
            for b in alt:
                if b.get("action") in CONTROL_ACTIONS:
                    sub.append({"status": "error", "action": b.get("action"),
                                "code": "invalid_parameter",
                                "message": "in branch nur Lese- und verändernde Aktionen"})
                    break
                r = self._apply_one(cur, b)
                sub.append(r)
                if r["status"] != "ok":
                    break
                if r.get("new_candidates"):
                    cur = r["new_candidates"][-1]
            if cur != current:
                new_all.append(cur)
        return {"action": "branch", "status": "ok", "new_candidates": new_all, "results": sub}

    def step(self, decider, candidate: int, last_errors=None) -> dict:
        """Zustand bilden, Entscheider fragen, Antwort ausführen und protokollieren."""
        state = self.state(candidate, last_errors=last_errors)
        response = decider.decide(state, self)
        return self.apply(candidate, response, decider=decider.describe(), state=state)
