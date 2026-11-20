"""Aktionen ``mikroptim.design.action/1``: Prüfung und Ausführung (docs/entwurfssteuerung.md,
Abschnitt 4). Jeder Entscheider (Mensch, Sprachmodell, Regel) benutzt genau diese Aktionen.

Umgesetzt (M10-T6a), jeweils über die Kommandozeile von Mikroptim:

- lesen: ``get_surface_table``, ``get_ghost_diagnosis``, ``evaluate``, ``glass_preselect``,
  ``sensitivity`` (M10-T3), ``glass_query`` und ``module_lookup`` (Wissensbasis, M10-T5)
- Läufe: ``optimize``, ``synth_segment``, ``snap_glasses``
- Variablen: ``free_gaps``, ``set_bounds`` (M10-T7b: Arbeitsabstand ``wd`` und Dicken ``t[i]``),
  ``set_constraint`` (M10-T7b: Gewicht und an/aus vorhandener Einträge, ``paraxial_focus`` neu)
- Topologie (Merit unverändert): ``insert_null_meniscus``, ``split_lens``
- Steuerung: ``branch``, ``abandon``, ``finish``

Seit M10-T4 (``operators.py``): ``place_ghost_dictionary`` (Merit unverändert),
``insert_module``, ``remove_element``, ``replace_glass``, ``replace_group``,
``start_from_skeleton`` (neuer Wurzelkandidat).

Die übrigen Aktionen des Schemas melden ``not_implemented`` (strukturierter Fehler).
"""

from __future__ import annotations

import pathlib
import tempfile

from .. import aberrations
from ._cli import cli_json, cli_text
from .files import catalog_names, elements, is_air, read_system, write_system

READ_ACTIONS = {"get_surface_table", "get_ghost_diagnosis", "evaluate", "glass_preselect",
                "sensitivity", "glass_query", "module_lookup"}
MODIFY_ACTIONS = {"optimize", "synth_segment", "snap_glasses", "free_gaps",
                  "insert_null_meniscus", "split_lens", "place_ghost_dictionary", "insert_module",
                  "remove_element", "replace_glass", "replace_group", "start_from_skeleton",
                  "set_bounds", "set_constraint"}
#: strukturelle Aktionen aus operators.py (M10-T4)
OPERATOR_ACTIONS = {"place_ghost_dictionary", "insert_module", "remove_element", "replace_glass",
                    "replace_group", "start_from_skeleton"}
CONTROL_ACTIONS = {"branch", "abandon", "finish"}
SCHEMA_ACTIONS = {
    "get_surface_table", "get_ghost_diagnosis", "evaluate", "glass_preselect", "sensitivity",
    "glass_query",
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
    if name == "glass_preselect":
        args = ["glasses", "--system", system_path, "--order", int(a.get("order", 1)),
                "--top", int(a.get("top", 10))]
        if a.get("spectrum"):
            args.append("--spectrum")
        if a.get("bending"):
            args += ["--bending", a["bending"]]
        fixed = None
        if "element" in a:
            data = read_system(system_path)
            elems = elements(data)
            k = int(a["element"])
            if not 0 <= k < len(elems):
                raise ActionError("precondition", f"Element {k} existiert nicht")
            glass = data["surfaces"][elems[k]["front"]].get("material")
            if not isinstance(glass, str):
                raise ActionError("precondition",
                                  f"Element {k} hat kein Katalogglas (Glasvariable oder Modell)")
            args += ["--fixed", glass]
            fixed = glass
        if "filter" in a:
            # derselbe Glasfilter wie glass_query (M11-T5g): Kandidaten = Treffer der Abfrage,
            # das feste Glas bleibt
            from .glass_query import QueryError, glass_query
            flt = {k: v for k, v in a["filter"].items() if k != "max_results"}
            if "partner_of_element" in flt:
                raise ActionError("invalid_parameter",
                                  "glass_preselect: partner_of_element im Filter nicht erlaubt "
                                  "(element hält das Glas fest)")
            try:
                hits = glass_query(system_path, flt, limit=None)
            except QueryError as e:
                raise ActionError(e.code, str(e)) from None
            names = [g["name"] for g in hits["glasses"]]
            if fixed is not None and fixed not in names:
                names.append(fixed)
            if not names:
                return {"order": int(a.get("order", 1)), "results": [], "candidates": 0}
            args += ["--only", ",".join(names)]
        rep = cli_json(args)
        keys = ("glasses", "powers", "power_sum", "residual", "spectral_residual", "curvatures")
        return {"order": rep["order"], "candidates": rep["glasses"], "results": [
            {key: r[key] for key in keys if key in r} for r in rep["results"]]}
    if name == "glass_query":
        _require(a, "filter")
        from .glass_query import QueryError, glass_query
        try:
            return glass_query(system_path, a["filter"], limit=int(a.get("limit", 20)))
        except QueryError as e:
            raise ActionError(e.code, str(e)) from None
    if name == "module_lookup":
        from . import knowledge
        try:
            mods = knowledge.module_lookup(
                group=a.get("group"), corrects=a.get("corrects"), zone=a.get("zone"),
                na=a.get("na"), immersion=a.get("immersion"),
                insertable_only=bool(a.get("insertable_only", False)))
        except knowledge.KnowledgeError as e:
            raise ActionError("invalid_parameter", str(e)) from None
        return {"modules": mods}
    if name == "sensitivity":
        return sensitivity_report(system_path, kinds=a.get("kinds"), elements=a.get("elements"))
    if name == "evaluate":
        args = ["eval", system_path]
        if a.get("grid"):
            args += ["--grid", a["grid"]]
        if a.get("psf"):
            args += ["--psf"]
        return cli_json(args)
    raise ActionError("not_implemented", f"{name} ist noch nicht umgesetzt")


#: Arten der Aktion ``sensitivity`` → Toleranztypen von ``mikroptim sensitivity``
SENSITIVITY_KINDS = {
    "decenter": ("element_decenter", "decenter"),
    "tilt": ("element_tilt", "tilt"),
    "radius": ("radius",),
    "thickness": ("thickness",),
    "index": ("index", "abbe"),
}


def _covers(entry: dict, layers: list[dict]) -> bool:
    """Gehört die Toleranz zu einem der Glaselemente ``layers`` (``files.elements``)?"""
    if entry["element"] is not None:
        first, last = entry["element"]
        return any(first <= e["front"] < last for e in layers)
    return any(entry["surface"] in (e["front"], e["back"]) for e in layers)


def sensitivity_report(system_path, kinds=None, elements=None) -> dict:
    """Empfindlichkeiten und Herstellbarkeit (``mikroptim sensitivity``, M10-T3): Toleranzsatz der
    Datei, sonst Mikroptim-Standardsatz. ``kinds`` und ``elements`` (Glaselemente wie
    ``files.elements``) filtern die Ausgabe; Reihenfolge nach größter Wirkung."""
    types = None
    if kinds:
        unknown = [k for k in kinds if k not in SENSITIVITY_KINDS]
        if unknown:
            raise ActionError("invalid_parameter", f"sensitivity: unbekannte Arten {unknown}")
        types = {t for k in kinds for t in SENSITIVITY_KINDS[k]}
    layers = None
    if elements is not None:
        elems = _elements_of(system_path)
        bad = [k for k in elements if not 0 <= int(k) < len(elems)]
        if bad:
            raise ActionError("precondition",
                              f"Elemente {bad} existieren nicht ({len(elems)} Elemente)")
        layers = [elems[int(k)] for k in elements]
    rep = cli_json(["sensitivity", system_path])
    rms0 = rep["nominal"]["rms_waves"]
    rows = []
    for i in rep["ranking"]:
        t = rep["tolerances"][i]
        if types is not None and t["type"] not in types:
            continue
        if layers is not None and not _covers(t, layers):
            continue
        rows.append({"type": t["type"], "surface": t["surface"], "element": t["element"],
                     "limit": t["limit"], "unit": t["unit"],
                     "delta_rms_waves": (None if t["worst_rms_waves"] is None or rms0 is None
                                         else t["worst_rms_waves"] - rms0),
                     "worst_delta_q_waves2": t["worst_delta_q_waves2"],
                     "jacobian_quadratic_waves2": t["mean_jacobian_quadratic_waves2"],
                     "failed_rays": t["failed_rays"]})
    man = rep["manufacturability"]
    return {"set": rep["settings"]["set"], "rms_nominal_waves": rms0, "tolerances": rows,
            "manufacturability": {
                "warnings": man["warnings"], "errors": man["errors"],
                "findings": [c for c in man["checks"] if c["severity"] != "ok"]}}


def _elements_of(system_path) -> list[dict]:
    return elements(read_system(system_path))


# --- Verändern --------------------------------------------------------------------------------


def modify_action(a: dict, system_path, workdir) -> list[tuple[pathlib.Path, dict]]:
    """Führt eine verändernde Aktion aus. Rückgabe: Liste (neue Systemdatei, Bericht); mehr
    als ein Eintrag bei Verzweigung (Sattelzweige von Operator A)."""
    name = a["action"]
    workdir = pathlib.Path(workdir)
    check_glass_names(a, read_system(system_path))
    if name in OPERATOR_ACTIONS:
        from . import operators
        return getattr(operators, name)(a, system_path, workdir)
    if name == "optimize":
        variables = a.get("variables", "current")
        if variables not in OPTIMIZE_VARIABLES:
            raise ActionError("invalid_parameter",
                              f"optimize variables={variables!r}: erlaubt {OPTIMIZE_VARIABLES}")
        src, added = _widen_variables(system_path, variables, workdir)
        out = _tmp_out(workdir)
        args = ["optimize", src, "-o", out, "--quiet",
                "--max-iterations", int(a.get("cycles", 100))]
        if a.get("jacobian"):
            args += ["--jacobian", a["jacobian"]]
        try:
            rep = cli_json(args)
        finally:
            if src != system_path:
                pathlib.Path(src).unlink(missing_ok=True)
        report = {"merit": (rep.get("result") or {}), "runs": rep.get("runs")}
        if added:
            report["variables"] = {"selection": variables, **added}
        return [(out, report)]
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
        if a.get("augmented_lagrangian"):
            args += ["--augmented-lagrangian", int(a["augmented_lagrangian"])]
        if "sine_condition" in a:
            args += ["--sine-condition", float(a["sine_condition"])]
        rep = cli_json(args)
        summary = {k: rep.get(k) for k in ("status", "goal_reached", "result", "summary") if k in rep}
        return [(out, summary)]
    if name == "snap_glasses":
        data = read_system(system_path)
        if not any(it.get("type") == "glass"
                   for it in (data.get("variables") or {}).get("items", [])):
            # Vorprüfung wie glass_check im Zustand (Befund E7 aus M10-T7)
            raise ActionError("precondition", "snap_glasses braucht Glasvariablen (Typ glass); "
                              "das System hat keine (siehe glass_check im Zustand)")
        out = _tmp_out(workdir)
        args = ["snap", system_path, "-o", out]
        if a.get("k"):
            args += ["--neighbors", a["k"]]
        if a.get("max_combinations"):
            args += ["--max-combinations", a["max_combinations"]]
        rep = cli_json(args)
        return [(out, {k: rep.get(k) for k in ("best", "result") if k in rep})]
    if name == "set_bounds":
        return [_set_bounds(a, system_path, workdir)]
    if name == "set_constraint":
        return [_set_constraint(a, system_path, workdir)]
    if name == "free_gaps":
        out = _free_gaps(a, system_path, workdir)
        return [] if out is None else [(out, {})]
    if name == "insert_null_meniscus":
        return _insert_meniscus(a, system_path, workdir)
    if name == "split_lens":
        return [_split_lens(a, system_path, workdir)]
    raise ActionError("not_implemented", f"{name} ist noch nicht umgesetzt")


def resolve_filter(flt, system_path) -> str:
    """Glasangabe ``{"filter": …}``: erstes Glas der Glasabfrage (M10-T5); ein leerer Filter
    ist ein Fehler (docs/entwurfssteuerung.md, Abschnitt 4.7)."""
    from .glass_query import QueryError, resolve_glass
    if not isinstance(flt, dict):
        raise ActionError("invalid_parameter", "filter: Objekt erwartet")
    try:
        return resolve_glass(system_path, flt)
    except QueryError as e:
        raise ActionError(e.code, str(e)) from None


def _named_glasses(a: dict) -> list[str]:
    """Glasnamen einer Aktion (``glass``, ``target_glass``, ``glasses``)."""
    out = []
    for key in ("glass", "target_glass"):
        g = a.get(key)
        if isinstance(g, dict) and isinstance(g.get("name"), str):
            out.append(g["name"])
        elif isinstance(g, str):
            out.append(g)
    for g in a.get("glasses") or []:
        if isinstance(g, dict) and isinstance(g.get("name"), str):
            out.append(g["name"])
        elif isinstance(g, str):
            out.append(g)
    return out


def check_glass_names(a: dict, data: dict) -> None:
    """Glasnamen müssen in den geladenen Katalogen stehen (Befund E7 aus M10-T7: Modelle
    erfanden Namen wie ``F2`` oder ``buchdahl(...)``); Fehler ``unknown_glass`` mit Hinweis auf
    ``glass_check.catalog_names`` statt einer Fehlermeldung des Laders."""
    names = _named_glasses(a)
    if not names:
        return
    known = set(catalog_names(data))
    if not known:  # Kataloge nicht lesbar: das Laden des Systems meldet es
        return
    unknown = [g for g in names if g not in known]
    if unknown:
        raise ActionError("unknown_glass", f"Gläser {unknown} stehen nicht in den Katalogen "
                          "des Systems (Namen aus glass_check.catalog_names im Zustand)")


def _glass_arg(glass, system_path=None) -> str:
    if isinstance(glass, dict) and "name" in glass:
        return glass["name"]
    if isinstance(glass, dict) and "filter" in glass:
        return resolve_filter(glass["filter"], system_path)
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
    glass = _glass_arg(a["glass"], system_path)
    c_list = [c0 - SADDLE_DELTA, c0 + SADDLE_DELTA] if a.get("saddle_branches") else [c0]
    out = []
    for c in c_list:
        target = _tmp_out(workdir)
        cli_text(["insert", system_path, "-o", target, "--meniscus", f"{s},{pos!r},{c!r},{glass}"])
        out.append((target, {"after_surface": s, "position": pos, "curvature": c, "glass": glass}))
    return out


def _split_lens(a, system_path, workdir):
    """Operator B. cement_curvature: Zahl, "mean" oder "achromat" (dünnes Achromat in η aus der
    Glasvorauswahl, M11-T5; Partner = target_glass oder der beste der Vorauswahl).
    target_glass: das freigegebene (hintere) Glas startet dort (insert --start-glass)."""
    _require(a, "element")
    data = read_system(system_path)
    elems = elements(data)
    k = int(a["element"])
    if not 0 <= k < len(elems):
        raise ActionError("precondition", f"Element {k} existiert nicht ({len(elems)} Elemente)")
    e = elems[k]
    c_front = data["surfaces"][e["front"]].get("curvature", 0.0)
    c_back = data["surfaces"][e["back"]].get("curvature", 0.0)
    free_part = a.get("free_part")
    target = a.get("target_glass")
    if (free_part == 0) and target is not None:
        raise ActionError("not_implemented", "target_glass nur mit free_part 1 (hinterer Teil)")
    target_name = _glass_arg(target, system_path) if target is not None else None
    cs = a.get("cement_curvature", "mean")
    report = {"element": k, "surface": e["front"]}
    if cs == "mean":
        cs = 0.5 * (c_front + c_back)
    elif cs == "achromat":
        pre = read_action({"action": "glass_preselect", "order": 1, "element": k, "top": 50},
                          system_path)["results"]
        if target_name is not None:
            pre = [r for r in pre if r["glasses"][1] == target_name]
        if not pre:
            raise ActionError("precondition", "kein passender Partner in der Glasvorauswahl"
                              + (f" für {target_name}" if target_name else ""))
        best = pre[0]
        front_fraction = best["powers"][0] if free_part in (None, 1) else best["powers"][1]
        cs = c_front - front_fraction * (c_front - c_back)
        report.update({"partner": best["glasses"][1], "secondary_spectrum": best["residual"],
                       "power_sum": best["power_sum"]})
    spec = f"{e['front']},{float(cs)!r}"
    if "fraction" in a:
        spec += f",{float(a['fraction'])!r}"
    # Dickenvariable des Elements: untere Schranke höchstens die dünnere Hälfte, sonst lehnt
    # insert das geteilte System ab (Startwert unter der Schranke; Befund M10-T7)
    frac = float(a.get("fraction", 0.5))
    part = data["surfaces"][e["front"]]["thickness"] * min(frac, 1.0 - frac)
    relaxed = False
    for it in (data.get("variables") or {}).get("items", []):
        on_element = it.get("surface") == e["front"] or (
            isinstance(it.get("surfaces"), list) and e["front"] in it["surfaces"])
        if it.get("type") == "thickness" and on_element and it.get("min", 0.0) > part:
            it["min"] = part
            relaxed = True
    if relaxed:
        system_path = _tmp_out(workdir)
        write_system(data, system_path)
        report["relaxed_thickness_min"] = part
    target_file = _tmp_out(workdir)
    args = ["insert", system_path, "-o", target_file, "--split", spec]
    if free_part is not None or target_name is not None:
        if free_part == 0:
            raise ActionError("not_implemented", "free_part 0 (vorderer Teil) über die CLI noch nicht")
        args.append("--free-glass")
        if target_name is not None:
            args += ["--start-glass", target_name]
    cli_text(args)
    report["cement_curvature"] = cs
    return target_file, report


#: set_constraint: Standardgewicht eines neuen Eintrags paraxial_focus [Wellen²/mm²] (wie im
#: Versuch M11-T4: 0,1 mm Randstrahlhöhe kostet eine Welle², die Falle F12 einige hundert)
PARAXIAL_FOCUS_WEIGHT = 100.0


def _set_constraint(a, system_path, workdir):
    """Nebenbedingung ändern (M10-T7b, Teilumfang): Gewicht ``weight`` oder ``enabled: false``
    (Eintrag entfernen) für alle Einträge des Typs; ``paraxial_focus`` (Bildebene im paraxialen
    Brennpunkt, Schutz gegen die Falle F12) wird angelegt, wenn es fehlt. ``limit`` und ``mode``
    sind noch nicht umgesetzt."""
    _require(a, "type")
    if "limit" in a or "mode" in a:
        raise ActionError("not_implemented", "set_constraint: limit und mode noch nicht")
    kind = a["type"]
    data = read_system(system_path)
    items = data.setdefault("constraints", {"items": []}).setdefault("items", [])
    found = [c for c in items if c.get("type") == kind]
    if a.get("enabled") is False:
        if not found:
            raise ActionError("precondition", f"keine Nebenbedingung {kind} im System")
        data["constraints"]["items"] = [c for c in items if c.get("type") != kind]
        change = "entfernt"
    elif found:
        if "weight" not in a:
            raise ActionError("missing_parameter", "set_constraint: weight oder enabled fehlt")
        for c in found:
            c["weight"] = float(a["weight"])
        change = "Gewicht"
    elif kind == "paraxial_focus":
        items.append({"type": "paraxial_focus",
                      "weight": float(a.get("weight", PARAXIAL_FOCUS_WEIGHT)),
                      "comment": "mikroptim.design set_constraint: gegen F12"})
        change = "angelegt"
    else:
        raise ActionError("precondition", f"keine Nebenbedingung {kind} im System (neu anlegen "
                          "nur paraxial_focus)")
    out = _tmp_out(pathlib.Path(workdir))
    write_system(data, out)
    return out, {"type": kind, "change": change, "entries": len(found) or 1}


def _set_bounds(a, system_path, workdir):
    """Schranken einer Variablen setzen (M10-T7b): ``wd`` (Arbeitsabstand) oder ``t[i]`` (Dicke
    nach Fläche i). Ein Eintrag mit mehreren Flächen wird geteilt (Fläche i bekommt einen eigenen
    Eintrag). Liegt der Startwert außerhalb der neuen Schranken, wird er auf sie geklemmt (die
    Merit ändert sich dann; Nachoptimierung mit ``optimize``)."""
    _require(a, "variable")
    name = str(a["variable"]).strip()
    data = read_system(system_path)
    items = (data.get("variables") or {}).get("items", [])
    surfaces = data["surfaces"]
    if name in ("wd", "working_distance"):
        entry = next((it for it in items if it.get("type") == "working_distance"), None)
        if entry is None:
            raise ActionError("precondition", "keine Variable working_distance im System")
        surface = len(surfaces) - 1
    elif name.startswith("t[") and name.endswith("]"):
        try:
            surface = int(name[2:-1])
        except ValueError:
            raise ActionError("invalid_parameter", f"Variable {name!r}: erwartet t[i]") from None
        if not 0 <= surface < len(surfaces) - 1:
            raise ActionError("precondition", f"t[{surface}]: keine Dicke (Arbeitsabstand: wd)")
        entry = None
        for it in items:
            if it.get("type") != "thickness":
                continue
            if it.get("surface") == surface:
                entry = it
                break
            sel = it.get("surfaces")
            if sel == "all" or (isinstance(sel, list) and surface in sel):
                if sel == "all":
                    raise ActionError("not_implemented",
                                      f"t[{surface}] liegt in einem Eintrag mit surfaces all")
                sel.remove(surface)
                if not sel:
                    items.remove(it)
                entry = {k: v for k, v in it.items() if k not in ("surfaces", "comment")}
                entry["surface"] = surface
                items.append(entry)
                break
        if entry is None:
            raise ActionError("precondition", f"t[{surface}] ist keine Variable (free_gaps)")
    else:
        raise ActionError("not_implemented", f"set_bounds: nur wd und t[i], nicht {name!r}")
    lo = float(a["min"]) if "min" in a else entry.get("min")
    hi = float(a["max"]) if "max" in a else entry.get("max")
    if lo is not None and hi is not None and not lo < hi:
        raise ActionError("invalid_parameter", f"min {lo} muss kleiner als max {hi} sein")
    for key, v in (("min", lo), ("max", hi)):
        if v is not None:
            entry[key] = v
    value = surfaces[surface]["thickness"]
    clamped = min(max(value, lo if lo is not None else value), hi if hi is not None else value)
    surfaces[surface]["thickness"] = clamped
    out = _tmp_out(pathlib.Path(workdir))
    write_system(data, out)
    return out, {"variable": name, "surface": surface, "min": lo, "max": hi,
                 "value_before": value, "value": clamped}


def _free_gaps(a, system_path, workdir):
    """Dicken als Variablen freigeben: Luftspalte und, mit include_glass_thickness, Glasdicken;
    Schranken [min, max_gap], der Startwert liegt immer im Bereich."""
    data = read_system(system_path)
    if not _add_gap_variables(data, a):
        return None  # nichts Neues: kein neuer Kandidat, die Folgeaktion läuft auf demselben
    out = _tmp_out(pathlib.Path(workdir))
    write_system(data, out)
    return out


def ghost_plate_gaps(data) -> set[int]:
    """Spalte (Fläche davor), in denen Geisterplatten liegen; sie bleiben fest (E14)."""
    return {int(g["after_surface"]) for g in data.get("ghost_plates", [])}


#: Variablenauswahl von ``optimize`` (Schema ``optimize.variables``); jede Stufe umfasst die
#: vorige. Gläser bleiben immer fest (Glasvariablen über split_lens/free_part).
OPTIMIZE_VARIABLES = ("current", "all_curvatures", "curvatures_and_gaps", "everything")


def _widen_variables(system_path, variables, workdir):
    """Variablen für ``optimize`` erweitern (M10-T7c; vorher ``not_implemented``, von Opus und
    Sonnet in 18 Antworten verlangt): ``all_curvatures`` macht jede sphärische Fläche ohne
    Krümmungsvariable variabel (ohne Schranken, wie die Menisken von Operator A),
    ``curvatures_and_gaps`` gibt zusätzlich die Luftspalte ohne Geisterplatten frei wie
    ``free_gaps``,
    ``everything`` auch die Glasdicken. Rückgabe: (Systemdatei für optimize, neu freigegebene
    Flächen); bei ``current`` oder ohne Neues die unveränderte Datei."""
    if variables == "current":
        return system_path, {}
    data = read_system(system_path)
    surfaces = data["surfaces"]
    items = data.setdefault("variables", {"items": []}).setdefault("items", [])
    spheres = [i for i, s in enumerate(surfaces) if s.get("type") == "sphere"]
    covered = set()
    for it in items:
        if it.get("type") != "curvature":
            continue
        if "surface" in it:
            covered.add(int(it["surface"]))
        elif it.get("surfaces") == "all":
            covered.update(spheres)
        else:
            covered.update(int(x) for x in it.get("surfaces", []))
    curv = [i for i in spheres if i not in covered]
    if curv:
        items.append({"type": "curvature", "surfaces": curv})
    gaps = []
    if variables in ("curvatures_and_gaps", "everything"):
        # Spalte mit Geisterplatten bleiben fest wie bei free_gaps (E14, M10-T7c/M10-T7d)
        gaps = _add_gap_variables(data, {"include_glass_thickness": variables == "everything"})
    if not curv and not gaps:
        return system_path, {}
    out = _tmp_out(pathlib.Path(workdir))
    write_system(data, out)
    return out, {"curvatures": curv, "thicknesses": gaps}


def _add_gap_variables(data, a) -> list[int]:
    """Dickenvariablen nach ``free_gaps`` in ``data`` eintragen; Rückgabe: neue Flächen."""
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
    # Spalte mit Geisterplatten bleiben fest, auch mit ausdrücklicher Flächenliste (Befund E14,
    # M10-T7c; behoben in M10-T7d): Operator A fügt dort ein, und beide Teilstücke erben die
    # untere Schranke der Spaltdicke. Frei schob der Optimierer den Spalt auf diese Schranke
    # (die Plattenlage), danach war er zu kurz für zwei Teilstücke und synth lehnte den Start ab.
    # Eine Schranke allein kann das nicht verhindern, weil Operator A Spaltdicke ≥ 2 × Schranke
    # braucht und die Schranke selbst vererbt.
    plates = ghost_plate_gaps(data)
    added = []
    for s in candidates:
        if s in covered or s in plates or not 0 <= s < len(surfaces) - 1:
            continue
        air = is_air(surfaces[s].get("material", "AIR"))
        if not air and not include_glass:
            continue
        t = surfaces[s]["thickness"]
        lo = min(0.1 if air else 0.5, t)
        hi = max(max_gap, t)
        items.append({"type": "thickness", "surface": s, "min": lo, "max": hi})
        added.append(s)
    return added
