"""Aktionen ``mikroptim.design.action/1``: Prüfung und Ausführung (docs/entwurfssteuerung.md,
Abschnitt 4). Jeder Entscheider (Mensch, Sprachmodell, Regel) benutzt genau diese Aktionen.

Umgesetzt (M10-T6a), jeweils über die Kommandozeile von Mikroptim:

- lesen: ``get_surface_table``, ``get_ghost_diagnosis``, ``evaluate``
- Läufe: ``optimize``, ``synth_segment``, ``snap_glasses``
- Variablen: ``free_gaps``
- Topologie (Merit unverändert): ``insert_null_meniscus``, ``split_lens``
- Steuerung: ``branch``, ``abandon``, ``finish``

Die übrigen Aktionen des Schemas melden ``not_implemented`` (strukturierter Fehler).
"""

from __future__ import annotations

import pathlib
import tempfile

from .. import aberrations
from ._cli import ToolError, cli_json, cli_text
from .files import elements, is_air, read_system, write_system

READ_ACTIONS = {"get_surface_table", "get_ghost_diagnosis", "evaluate"}
MODIFY_ACTIONS = {"optimize", "synth_segment", "snap_glasses", "free_gaps",
                  "insert_null_meniscus", "split_lens"}
CONTROL_ACTIONS = {"branch", "abandon", "finish"}
SCHEMA_ACTIONS = {
    "get_surface_table", "get_ghost_diagnosis", "evaluate", "sensitivity", "glass_query",
    "module_lookup", "optimize", "synth_segment", "penalty_continuation", "snap_glasses",
    "free_gaps", "freeze", "set_bounds", "set_constraint", "set_homotopy_plan",
    "place_ghost_dictionary", "scale_ghost_penalty", "insert_null_meniscus", "split_lens",
    "insert_module", "remove_element", "replace_glass", "replace_group", "start_from_skeleton",
    "branch", "abandon", "finish",
}
#: Abstand der beiden Sattelzweige von Operator A [1/mm] (wie synth --saddle-delta)
SADDLE_DELTA = 0.002


class ActionError(Exception):
    """Aktion abgelehnt; ``code`` maschinenlesbar, Meldung für Mensch und Modell."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self, action: str | None = None) -> dict:
        return {"action": action, "code": self.code, "message": str(self)}


def validate_response(resp: dict) -> list[dict]:
    """Prüft den Rahmen einer Antwort und liefert die Aktionsliste."""
    if not isinstance(resp, dict) or resp.get("format") != "mikroptim.design.action/1":
        raise ActionError("format", "erwartet ein Objekt mit format mikroptim.design.action/1")
    acts = resp.get("actions")
    if not isinstance(acts, list) or not 1 <= len(acts) <= 4:
        raise ActionError("format", "actions: 1 bis 4 Aktionen erwartet")
    for a in acts:
        if not isinstance(a, dict) or a.get("action") not in SCHEMA_ACTIONS:
            raise ActionError("unknown_action", f"unbekannte Aktion {a!r}")
    return acts


def _require(a: dict, *keys):
    for k in keys:
        if k not in a:
            raise ActionError("missing_parameter", f"{a['action']}: Parameter {k} fehlt")


def _tmp_out(workdir: pathlib.Path) -> pathlib.Path:
    workdir.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(dir=workdir, suffix=".json", delete=False)
    fd.close()
    return pathlib.Path(fd.name)


# --- Lesen ------------------------------------------------------------------------------------


def read_action(a: dict, system_path) -> dict:
    name = a["action"]
    if name == "get_surface_table":
        rep = aberrations(system_path, field_angle_deg=a.get("field_angle_deg"))
        rng = a.get("surfaces")
        surfs = rep["surfaces"]
        if rng:
            surfs = [s for s in surfs if rng[0] <= s["surface"] <= rng[1]]
        return {"surfaces": surfs, "totals": rep["totals"], "lagrange": rep["lagrange"]}
    if name == "get_ghost_diagnosis":
        data = read_system(system_path)
        if not data.get("ghost_plates"):
            return {"ghost_diagnosis": None, "note": "keine Geisterplatten im System"}
        if not data.get("variables"):
            raise ActionError("precondition", "Geisterdiagnose braucht Variablen im System")
        opt = cli_json(["optimize", system_path, "--max-iterations", "0", "--quiet"])
        return {"ghost_diagnosis": opt.get("ghost_diagnosis")}
    if name == "evaluate":
        args = ["eval", system_path]
        if a.get("grid"):
            args += ["--grid", a["grid"]]
        if a.get("psf"):
            args += ["--psf"]
        return cli_json(args)
    raise ActionError("not_implemented", f"{name} ist noch nicht umgesetzt")


# --- Verändern --------------------------------------------------------------------------------


def modify_action(a: dict, system_path, workdir) -> list[tuple[pathlib.Path, dict]]:
    """Führt eine verändernde Aktion aus. Rückgabe: Liste (neue Systemdatei, Bericht); mehr
    als ein Eintrag bei Verzweigung (Sattelzweige von Operator A)."""
    name = a["action"]
    workdir = pathlib.Path(workdir)
    if name == "optimize":
        variables = a.get("variables", "current")
        if variables != "current":
            raise ActionError("not_implemented", f"optimize variables={variables}: nur 'current'")
        out = _tmp_out(workdir)
        args = ["optimize", system_path, "-o", out, "--quiet",
                "--max-iterations", int(a.get("cycles", 100))]
        if a.get("jacobian"):
            args += ["--jacobian", a["jacobian"]]
        rep = cli_json(args)
        return [(out, {"merit": (rep.get("result") or {}), "runs": rep.get("runs")})]
    if name == "synth_segment":
        _require(a, "target")
        t = a["target"]
        if "spectrum_stage" in t:
            raise ActionError("not_implemented", "synth_segment: spectrum_stage noch nicht unterstützt")
        out = _tmp_out(workdir)
        args = ["synth", system_path, "-o", out, "--quiet"]
        if "aperture" in t:
            args += ["--target-na", t["aperture"]]
        if "field" in t:
            args += ["--target-field", t["field"]]
        if a.get("beam_width"):
            args += ["--beam", a["beam_width"]]
            if a.get("alternatives"):
                args += ["--alternatives", a["alternatives"]]
        if "max_new_elements" in a:
            args += ["--max-elements", len(elements(read_system(system_path))) + int(a["max_new_elements"])]
        rep = cli_json(args)
        summary = {k: rep.get(k) for k in ("status", "goal_reached", "result", "summary") if k in rep}
        return [(out, summary)]
    if name == "snap_glasses":
        out = _tmp_out(workdir)
        args = ["snap", system_path, "-o", out]
        if a.get("k"):
            args += ["--neighbors", a["k"]]
        if a.get("max_combinations"):
            args += ["--max-combinations", a["max_combinations"]]
        rep = cli_json(args)
        return [(out, {k: rep.get(k) for k in ("best", "result") if k in rep})]
    if name == "free_gaps":
        out = _free_gaps(a, system_path, workdir)
        return [] if out is None else [(out, {})]
    if name == "insert_null_meniscus":
        return _insert_meniscus(a, system_path, workdir)
    if name == "split_lens":
        return [_split_lens(a, system_path, workdir)]
    raise ActionError("not_implemented", f"{name} ist noch nicht umgesetzt")


def _glass_arg(glass) -> str:
    if isinstance(glass, dict) and "name" in glass:
        return glass["name"]
    if isinstance(glass, dict) and "filter" in glass:
        raise ActionError("not_implemented", "Glasfilter brauchen glass_query (M10-T5)")
    if isinstance(glass, (int, float)):
        return f"n={float(glass)!r}"
    raise ActionError("invalid_parameter", f"Glas {glass!r}: erwartet {{'name': ...}}")


def _insert_meniscus(a, system_path, workdir):
    _require(a, "after_surface", "glass")
    data = read_system(system_path)
    s = int(a["after_surface"])
    if not 0 <= s < len(data["surfaces"]) - 1 or not is_air(data["surfaces"][s].get("material")):
        raise ActionError("precondition", f"nach Fläche {s} liegt kein Luftspalt")
    gap = data["surfaces"][s]["thickness"]
    pos = float(a.get("position", gap / 2))
    if not 0 < pos < gap:
        raise ActionError("precondition", f"Lage {pos} mm liegt nicht im Spalt (0, {gap})")
    curv = a.get("curvature", {"from_surface": s})
    if isinstance(curv, dict):
        ref = int(curv["from_surface"])
        c0 = float(data["surfaces"][ref].get("curvature", 0.0))
    else:
        c0 = float(curv)
    glass = _glass_arg(a["glass"])
    c_list = [c0 - SADDLE_DELTA, c0 + SADDLE_DELTA] if a.get("saddle_branches") else [c0]
    out = []
    for c in c_list:
        target = _tmp_out(workdir)
        cli_text(["insert", system_path, "-o", target, "--meniscus", f"{s},{pos!r},{c!r},{glass}"])
        out.append((target, {"after_surface": s, "position": pos, "curvature": c, "glass": glass}))
    return out


def _split_lens(a, system_path, workdir):
    _require(a, "element")
    data = read_system(system_path)
    elems = elements(data)
    k = int(a["element"])
    if not 0 <= k < len(elems):
        raise ActionError("precondition", f"Element {k} existiert nicht ({len(elems)} Elemente)")
    e = elems[k]
    c_front = data["surfaces"][e["front"]].get("curvature", 0.0)
    c_back = data["surfaces"][e["back"]].get("curvature", 0.0)
    cs = a.get("cement_curvature", "mean")
    if cs == "mean":
        cs = 0.5 * (c_front + c_back)
    elif cs == "achromat":
        raise ActionError("not_implemented", "cement_curvature achromat folgt mit M11-T5")
    spec = f"{e['front']},{float(cs)!r}"
    if "fraction" in a:
        spec += f",{float(a['fraction'])!r}"
    target = _tmp_out(workdir)
    args = ["insert", system_path, "-o", target, "--split", spec]
    if a.get("free_part") is not None or a.get("target_glass") is not None:
        args.append("--free-glass")
    if a.get("target_glass") is not None:
        raise ActionError("not_implemented", "target_glass folgt mit M10-T5/M11-T5")
    cli_text(args)
    return target, {"element": k, "surface": e["front"], "cement_curvature": cs}


def _free_gaps(a, system_path, workdir):
    """Dicken als Variablen freigeben: Luftspalte und, mit include_glass_thickness, Glasdicken;
    Schranken [min, max_gap], der Startwert liegt immer im Bereich."""
    data = read_system(system_path)
    surfaces = data["surfaces"]
    sel = a.get("surfaces", "all")
    candidates = range(len(surfaces) - 1) if sel == "all" else [int(s) for s in sel]
    include_glass = bool(a.get("include_glass_thickness", False))
    max_gap = float(a.get("max_gap", 20.0))
    variables = data.setdefault("variables", {"items": []})
    items = variables.setdefault("items", [])
    covered = set()
    for it in items:
        if it.get("type") == "thickness":
            if "surface" in it:
                covered.add(int(it["surface"]))
            elif it.get("surfaces") == "all":
                covered.update(range(len(surfaces) - 1))
            else:
                covered.update(int(x) for x in it.get("surfaces", []))
    added = []
    for s in candidates:
        if s in covered or not 0 <= s < len(surfaces) - 1:
            continue
        air = is_air(surfaces[s].get("material", "AIR"))
        if not air and not include_glass:
            continue
        t = surfaces[s]["thickness"]
        lo = min(0.1 if air else 0.5, t)
        hi = max(max_gap, t)
        # Geisterplatten im Spalt: untere Schranke mindestens die letzte Plattenlage
        for g in data.get("ghost_plates", []):
            if g["after_surface"] == s:
                lo = max(lo, g["distance"])
        items.append({"type": "thickness", "surface": s, "min": lo, "max": hi})
        added.append(s)
    if not added:
        return None  # nichts Neues: kein neuer Kandidat, die Folgeaktion läuft auf demselben
    out = _tmp_out(pathlib.Path(workdir))
    write_system(data, out)
    return out
