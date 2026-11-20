"""Zustand eines Kandidaten im Format ``mikroptim.design.state/1`` (docs/entwurfssteuerung.md,
Abschnitt 3.5). Grundlage der Ansicht für den Menschen und der Eingabe für das Sprachmodell."""

from __future__ import annotations

import math

from .. import aberrations
from ._cli import ToolError, cli_json
from .classes import best_focus_field_shift, color_class, depth_of_focus, field_class
from .files import catalog_names, elements, read_system

FORMAT = "mikroptim.design.state/1"
WAVE_KEYS = ("W040", "W131", "W222", "W220", "W311", "dW020", "dW111", "dW040", "W040_full",
             "W040_higher", "W040_zone")
#: Spalten der Textansicht
TEXT_KEYS = ("W040", "W040_full", "W131", "W222", "W220", "dW020", "dW111", "dW040")
#: Gruppengrenze: Mittelgruppe = Elemente mit Randstrahlhöhe ≥ MIDDLE_FRACTION · h_max
MIDDLE_FRACTION = 0.8
#: Etendue-Zonen nach Zhang, Abb. 4.4 (Näherung über G bei 22 mm Zwischenbild) [mm²]
ZONE_LIMITS = (0.051, 0.086, 0.95)
#: höchstens so viele Katalognamen im Zustand (Rest über glass_preselect)
CATALOG_NAMES_MAX = 60


def reference_sphere_state(data) -> dict:
    """Konvention der Referenzsphäre, mit der Merit und RMS des Zustands gerechnet sind
    (ADR-0017): ``kind`` aus dem Feld ``reference_sphere`` der Systemdatei, ohne Feld die alte
    Konvention ``last_surface`` (``explicit`` = False; die CLI warnt dann auf stderr)."""
    kind = data.get("reference_sphere")
    return {"kind": kind or "last_surface", "explicit": kind is not None}


def etendue_zone(G, magnification, na):
    """Etendue-Zone 1–6 (Zhang, Dissertation 2019, Abschnitt 4.2.2; Näherung)."""
    if magnification is not None and magnification < 4:
        return 5
    if magnification is not None and magnification > 100:
        return 6
    if na is not None and na > 1.5:
        return 4
    if G is None:
        return None
    for zone, limit in enumerate(ZONE_LIMITS, start=1):
        if G < limit:
            return zone
    return 4


def assign_groups(elems, heights):
    """Front-, Mittel-, Hintergruppe pro Element. Tracerichtung: Fläche 0 ist die Blende
    (Hintergruppe), die letzten Flächen liegen am Objekt (Frontgruppe)."""
    if not elems:
        return {}
    h = [max(abs(heights[e["front"]]), abs(heights[e["back"]])) for e in elems]
    k = max(range(len(h)), key=lambda i: h[i])
    lo = hi = k
    while lo > 0 and h[lo - 1] >= MIDDLE_FRACTION * h[k]:
        lo -= 1
    while hi < len(h) - 1 and h[hi + 1] >= MIDDLE_FRACTION * h[k]:
        hi += 1
    return {e["element"]: ("rear" if i < lo else "front" if i > hi else "middle")
            for i, e in enumerate(elems)}


def element_groups(data, elems, heights):
    """Gruppe pro Element: das Feld ``group`` der Vorderfläche (sonst der Rückfläche) der
    Systemdatei überschreibt die automatische Zuordnung nach der Randstrahlhöhe
    (``assign_groups``; Entscheidung Maintainer 2026-11-19, docs/entwurfssteuerung.md,
    Abschnitt 3.3)."""
    groups = assign_groups(elems, heights)
    surfs = data["surfaces"]
    for e in elems:
        g = surfs[e["front"]].get("group") or surfs[e["back"]].get("group")
        if g:
            groups[e["element"]] = g
    return groups


def _surface_groups(n_surfaces, elems, groups, data=None):
    out = ["rear"] * n_surfaces
    for e in elems:
        g = groups[e["element"]]
        out[e["front"]] = g
        out[e["back"]] = g
    # Flächen hinter dem letzten Element (Luft) zur Frontgruppe, vor dem ersten zur Hintergruppe
    if elems:
        for s in range(elems[-1]["back"] + 1, n_surfaces):
            out[s] = "front"
    # ausdrückliche Gruppen der Datei haben Vorrang (auch an Flächen ohne Element)
    if data is not None:
        for s, surf in enumerate(data["surfaces"][:n_surfaces]):
            if surf.get("group"):
                out[s] = surf["group"]
    return out


def build_state(system_path, *, candidate=0, parent=None, spec=None, budget=None,
                history=None, last_errors=None, with_merit=True, with_sensitivity=True) -> dict:
    """Zustand eines Kandidaten aus seiner Systemdatei.

    ``spec``: Sollwerte (u. a. ``tube_focal_length``, ``magnification``, ``na``,
    ``field_number``, ``target_color_class``, ``target_field_class``); fehlende Angaben werden,
    wo möglich, aus dem System bestimmt, sonst ``None``.
    """
    spec = dict(spec or {})
    data = read_system(system_path)
    trace = cli_json(["trace", system_path])
    ab = aberrations(system_path)
    par = trace["paraxial"]
    f = par["focal_length"]
    a = data["surfaces"][0]["semi_diameter"]
    theta = max([abs(x) for x in data["source"].get("field_angles_deg", [0.0])] or [0.0])

    # Spezifikation: was fehlt, aus dem System
    tube = spec.get("tube_focal_length")
    # aktuelle NA wie synth: Pupillenradius / Soll-Brennweite (sonst paraxiale Brennweite)
    f_target = (data.get("target") or {}).get("focal_length") or f
    na_now = a / f_target if f_target else None
    na = spec.get("na", na_now)    # Soll-NA der Spezifikation
    mag = spec.get("magnification", (tube / f) if (tube and f) else None)
    y_obj = abs(f) * math.tan(math.radians(theta)) if f else None
    G = math.pi * (y_obj * na) ** 2 if (y_obj is not None and na is not None) else None
    field_number = spec.get("field_number", (2 * y_obj * mag) if (y_obj and mag) else None)
    # monochromatisch (eine Wellenlänge): keine Ziel-Farbklasse, Farbe ist nicht Teil der
    # Aufgabe (Befund E8 aus M10-T7: ein Modell verfolgte sonst die Farbklasse)
    mono = len(set(trace["wavelengths_nm"])) == 1
    spec_out = {
        "magnification": mag, "na": na, "field_number": field_number,
        "tube_focal_length": tube, "parfocal_max": spec.get("parfocal_max"),
        "immersion": spec.get("immersion", False),
        "spectrum_nm": [min(trace["wavelengths_nm"]), max(trace["wavelengths_nm"])],
        "monochromatic": mono,
        "target_color_class": None if mono else spec.get("target_color_class"),
        "target_field_class": spec.get("target_field_class"),
        "etendue": G, "zone": etendue_zone(G, mag, na),
    }

    # Merit (RMS pro Tupel) aus trace, optional F und Geisterdiagnose aus optimize (0 Zyklen)
    rms = [[w["field_angle_deg"] / theta if theta else 0.0, w["wavelength_nm"], w["rms_waves"]]
           for w in trace["wavefront"]]
    merit = {"F": None, "F_B": None, "F_G": None,
             "rms_max_waves": max((r[2] for r in rms if r[2] is not None), default=None),
             "rms_by_tuple": rms, "failed_rays": trace["failed_rays"], "lm_status": None}
    ghosts = None
    if with_merit and data.get("variables"):
        try:
            opt = cli_json(["optimize", system_path, "--max-iterations", "0", "--quiet"])
            res = opt.get("result") or {}
            merit["F"] = res.get("merit", res.get("merit_final"))
            ghosts = _ghost_summary(opt.get("ghost_diagnosis"))
        except ToolError:
            pass

    # Gruppen und Bilanzen aus dem Aberrationsbericht
    elems = elements(data)
    heights = [s["y"] for s in ab["surfaces"]]
    groups = element_groups(data, elems, heights)
    sgroups = _surface_groups(len(ab["surfaces"]), elems, groups, data)
    balance = {}
    for name in ("front", "middle", "rear"):
        surfs = [i for i, g in enumerate(sgroups) if g == name]
        if not surfs:
            continue
        b = {"surfaces": [min(surfs), max(surfs)]}
        for key in WAVE_KEYS:
            vals = [ab["surfaces"][i]["waves"][key] for i in surfs]
            b[key] = None if any(v is None for v in vals) else sum(vals)
        b["elements"] = [e["element"] for e in elems if groups[e["element"]] == name]
        balance[name] = b
    top = {}
    for key in ("W040", "W040_higher", "W131", "W220", "dW020", "dW111", "dW040"):
        vals = [(s["surface"], s["waves"][key]) for s in ab["surfaces"] if s["waves"].get(key)]
        vals.sort(key=lambda t: -abs(t[1]))
        top[key] = [[i, v] for i, v in vals[:8]]

    h_max = max(abs(h) for h in heights) if heights else None
    wd = data["surfaces"][-1]["thickness"]
    paraxial = {"f": f, "working_distance": wd, "petzval_radius": par["petzval_radius"],
                "retrofocus_factor": (abs(heights[0]) / h_max) if h_max else None,
                "kappa": wd / f if f else None}

    hemi = [(i, abs(s.get("curvature", 0.0)) * s.get("semi_diameter", 0.0))
            for i, s in enumerate(data["surfaces"]) if s.get("type") == "sphere"]
    mechanics = {"hemisphere_max": list(max(hemi, key=lambda t: t[1])) if hemi else None}
    sensitivity = None
    if with_sensitivity:
        try:
            sensitivity, manufacturability = sensitivity_summary(system_path)
            mechanics["manufacturability"] = manufacturability
        except ToolError:
            pass
    glasses = [{"element": e["element"], "surface": e["front"], "name": e["material"],
                "group": groups.get(e["element"])} for e in elems]
    glass_check = glass_summary(data, elems)

    # Farb- und Feldklassen (M10-T2)
    classes = None
    last_s = ab["surfaces"][-1]
    if na:
        dof = depth_of_focus(na, last_s["n_after"])
        shifts = {f["line"]: f["shift_mm"] for f in ab.get("focal_shifts", [])}
        col = color_class(shifts, dof)
        S = ab["totals"]["S"]
        # Seidel bei vollem Feld θ_max (Bericht mit Standard-Feldwinkel = θ_max)
        fshift = best_focus_field_shift(S[2], S[3], last_s["n_after"], last_s["u_after"])
        edge = [w for w in trace["wavefront"] if theta and abs(w["field_angle_deg"]) == theta]
        edge_ref = [w["rms_waves"] for w in edge
                    if abs(w["wavelength_nm"] - trace["reference_nm"]) < 1e-9]
        edge_rms = edge_ref[0] if edge_ref else (max(w["rms_waves"] for w in edge) if edge else None)
        fld = field_class(field_shift=fshift, dof=dof, na=na, etendue=G, edge_rms=edge_rms,
                          edge_failed=sum(w["failed_rays"] for w in edge))
        classes = {"color": col["color"], "field": fld["field"], "dof_mm": dof,
                   "color_detail": col, "field_detail": fld}

    return {
        "format": FORMAT, "candidate": candidate, "parent": parent,
        "elements": len(elems), "surfaces": len(data["surfaces"]),
        "spec": spec_out, "homotopy_point": {"field": 1.0, "aperture": na_now},
        "merit": merit, "classes": classes, "paraxial": paraxial, "groups": balance,
        "top_contributors": top,
        "sine_condition": {"osc": ab["totals"].get("osc"), "osc_zone": ab["totals"].get("osc_zone")},
        "ghosts": ghosts, "mechanics": mechanics,
        "sensitivity": sensitivity, "glasses": glasses, "glass_check": glass_check,
        "aberration_settings": {k: ab[k] for k in ("reference_nm", "short_nm", "long_nm",
                                                   "field_angle_deg", "sign_convention")},
        "reference_sphere": reference_sphere_state(data),
        "budget": budget or {}, "history": list(history or [])[-10:],
        "last_errors": list(last_errors or []),
    }


def glass_summary(data: dict, elems=None) -> dict:
    """Vorprüfung für Glasaktionen (Befund E7 aus M10-T7): ob ``snap_glasses`` möglich ist
    (braucht Variablen vom Typ ``glass``), welche Elemente ein Katalogglas bzw. ein Glasmodell
    tragen, und die Namen der geladenen Kataloggläser (höchstens ``CATALOG_NAMES_MAX``)."""
    elems = elements(data) if elems is None else elems
    glass_vars = [it for it in (data.get("variables") or {}).get("items", [])
                  if it.get("type") == "glass"]
    names = catalog_names(data)
    model = [e["element"] for e in elems
             if not isinstance(data["surfaces"][e["front"]].get("material"), str)]
    if glass_vars:
        reason = (f"{len(glass_vars)} Glasvariable(n): snap_glasses rastet sie auf "
                  "Kataloggläser ein")
    else:
        # nur der Befund, keine Handlungsempfehlung: ein Glastausch ist für eine monochromatische
        # Aperturaufgabe kein Ziel (Vorlauf M10-T7b: qwen3 tauschte daraufhin nur noch Gläser)
        reason = "keine Glasvariablen: snap_glasses nicht möglich"
    return {"snappable": bool(glass_vars), "glass_variables": len(glass_vars),
            "model_glass_elements": model, "reason": reason,
            "catalog_glass_count": len(names), "catalog_names": names[:CATALOG_NAMES_MAX],
            "catalog_names_truncated": len(names) > CATALOG_NAMES_MAX}


#: Einträge pro Liste in der Zusammenfassung der Empfindlichkeiten
SENSITIVITY_TOP = 3


def sensitivity_summary(system_path):
    """Kurzfassung von ``mikroptim sensitivity`` für den Zustand (M10-T3): die empfindlichsten
    Elemente für Dezentrierung und Verkippung und die empfindlichsten übrigen Toleranzen, je mit
    der Zunahme des RMS bei ±δ [Wellen]; dazu die Befunde der Herstellbarkeit."""
    rep = cli_json(["sensitivity", system_path])
    rms0 = rep["nominal"]["rms_waves"]

    def rows(types, key):
        out = []
        for i in rep["ranking"]:
            t = rep["tolerances"][i]
            if t["type"] in types and t["worst_rms_waves"] is not None and rms0 is not None:
                out.append([key(t), t["worst_rms_waves"] - rms0])
            if len(out) == SENSITIVITY_TOP:
                break
        return out

    def where(t):
        return t["element"] or [t["surface"], t["surface"]]

    summary = {
        "set": rep["settings"]["set"], "rms_nominal_waves": rms0,
        "decenter_worst": rows({"element_decenter", "decenter"}, where),
        "tilt_worst": rows({"element_tilt", "tilt"}, where),
        "parameter_worst": rows({"radius", "thickness", "index", "abbe"},
                                lambda t: [t["type"], t["surface"]]),
    }
    man = rep["manufacturability"]
    manufacturability = {"warnings": man["warnings"], "errors": man["errors"],
                         "findings": [[c["check"], c["surface"], c["value"], c["severity"],
                                      c.get("message", "")]
                                      for c in man["checks"] if c["severity"] != "ok"]}
    return summary, manufacturability


def _ghost_summary(diag):
    if not diag:
        return None
    active = [{"plate": p["plate"], "after_surface": p["after_surface"],
               "sum_abs": p["sum_abs"], "dominant_orders": p.get("dominant_orders"),
               "merit_gain": p.get("merit_gain")}
              for p in diag.get("plates", []) if p.get("active")]
    return {"active": active, "recommendation": diag.get("recommendation")}


def _reference_line(ref) -> str:
    if not ref:
        return "  Referenzsphäre: –"
    note = "" if ref["explicit"] else " (ohne Feld in der Datei: alte Konvention, ADR-0017)"
    return f"  Referenzsphäre: {ref['kind']}{note}"


def format_state(state: dict) -> str:
    """Kurze Textansicht des Zustands (für den Menschen auf der Kommandozeile)."""
    sp, m, p = state["spec"], state["merit"], state["paraxial"]
    lines = [
        f"Kandidat {state['candidate']} (Eltern {state['parent']}): {state['elements']} Elemente, "
        f"{state['surfaces']} Flächen",
        f"  NA {_n(sp['na'])}, Vergrößerung {_n(sp['magnification'])}, Etendue {_n(sp['etendue'])} mm², "
        f"Zone {sp['zone']}",
        f"  f {_n(p['f'])} mm, Arbeitsabstand {_n(p['working_distance'])} mm, "
        f"Retrofokus r {_n(p['retrofocus_factor'])}, κ {_n(p['kappa'])}",
        f"  Klassen: Farbe {(state.get('classes') or {}).get('color', '–')}, "
        f"Feld {(state.get('classes') or {}).get('field', '–')} (Zhang)",
        f"  RMS max {_n(m['rms_max_waves'])} Wellen, Merit F {_n(m['F'])}, "
        f"versagte Strahlen {m['failed_rays']}",
        _reference_line(state.get("reference_sphere")),
        f"  OSC Rand {_n(state['sine_condition']['osc'])}, Zone {_n(state['sine_condition']['osc_zone'])}",
        "  Gruppenbilanzen [Wellen]:  " + "  ".join(f"{k:>7}" for k in TEXT_KEYS),
    ]
    for name in ("front", "middle", "rear"):
        g = state["groups"].get(name)
        if g:
            lines.append(f"    {name:<8} Fl. {g['surfaces'][0]:>2}–{g['surfaces'][1]:<2}"
                         + "".join(f"{_n(g[k]):>9}" for k in TEXT_KEYS))
    sens = state.get("sensitivity")
    if sens:
        dec = ", ".join(f"Fl. {e[0]}-{e[1]}: +{d:.3g}" for e, d in sens["decenter_worst"])
        tilt = ", ".join(f"Fl. {e[0]}-{e[1]}: +{d:.3g}" for e, d in sens["tilt_worst"])
        lines.append(f"  Empfindlichkeit (RMS-Zunahme [Wellen], Satz {sens['set']}): "
                     f"Dezentrierung {dec or '-'}; Verkippung {tilt or '-'}")
    man = (state.get("mechanics") or {}).get("manufacturability")
    if man and (man["warnings"] or man["errors"]):
        lines.append(f"  Herstellbarkeit: {man['warnings']} Warnungen, {man['errors']} Fehler")
    gc = state.get("glass_check")
    if gc:
        lines.append(f"  Gläser: einrastbar {'ja' if gc['snappable'] else 'nein'} – {gc['reason']}; "
                     f"{gc['catalog_glass_count']} Kataloggläser")
    if state["spec"].get("monochromatic"):
        lines.append("  monochromatisch: keine Ziel-Farbklasse")
    if state.get("ghosts") and state["ghosts"].get("active"):
        lines.append(f"  aktive Geisterplatten: {len(state['ghosts']['active'])}")
    if state.get("last_errors"):
        lines.append(f"  letzte Fehler: {state['last_errors']}")
    return "\n".join(lines)


def _n(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)
