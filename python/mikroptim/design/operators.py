"""Strukturelle Aktionen der Entwurfssteuerung (M10-T4; docs/entwurfssteuerung.md, Abschnitte
4.3–4.5): ``place_ghost_dictionary``, ``insert_module``, ``remove_element``, ``replace_glass``,
``replace_group`` und ``start_from_skeleton``.

Die Geometrie bauen die Operatoren des C++-Kerns (``mikroptim insert``, docs/optimierung.md,
Abschnitt 13); hier werden Vorbedingungen geprüft, Vorlagen ausgewählt, Homotopien in Stufen
(Dicke, Rückbau) mit Nachoptimierung über ``mikroptim optimize`` gefahren und die Berichte
zusammengestellt. Jede Funktion liefert eine Liste (neue Systemdatei, Bericht) wie
``actions.modify_action``.
"""

from __future__ import annotations

import math
import pathlib

from .. import aberrations
from ._cli import ToolError, cli_json
from .actions import ActionError, _require, _tmp_out, read_action
from .files import elements, is_air, read_system, write_system
from .modules import NULL_TEMPLATES, module
from .state import assign_groups

#: Stufen der Dicken- bzw. Rückbau-Homotopie und LM-Zyklen pro Stufe (Standard)
DEFAULT_STAGES = 4
DEFAULT_CYCLES = 50
#: Anteil des Spalts für Lage und Dicke der Schale, wenn nicht angegeben
SHELL_POSITION = 0.1
SHELL_THICKNESS = 0.3
#: Abstand der Menisken des Gauss-Paars als Anteil des Spalts
GAUSS_SPACING = 0.4
#: größter Halbkugelgrad |c|·h der Vorlagen
MAX_HEMISPHERE = 0.9


# --- Hilfen -----------------------------------------------------------------------------------


def _insert(system_path, workdir, args) -> tuple[pathlib.Path, dict]:
    """``mikroptim insert <system> -o <neu> <args> --json``; Rückgabe (neue Datei, Bericht)."""
    out = _tmp_out(pathlib.Path(workdir))
    try:
        rep = cli_json(["insert", system_path, "-o", out, *args])
    except ToolError:
        out.unlink(missing_ok=True)
        raise
    return out, rep


def _has_variables(path) -> bool:
    return bool((read_system(path).get("variables") or {}).get("items"))


def _optimize(path, workdir, cycles) -> tuple[pathlib.Path, dict | None]:
    """Nachoptimierung mit den Variablen der Datei (nichts ohne Variablen oder cycles = 0)."""
    if cycles <= 0 or not _has_variables(path):
        return pathlib.Path(path), None
    out = _tmp_out(pathlib.Path(workdir))
    rep = cli_json(["optimize", path, "-o", out, "--quiet", "--max-iterations", int(cycles)])
    res = rep.get("result") or {}
    return out, {"merit": res.get("merit"), "status": res.get("status"),
                 "iterations": res.get("iterations")}


def _drop(paths, keep):
    """Zwischendateien einer Aktion löschen (außer ``keep``)."""
    for p in paths:
        p = pathlib.Path(p)
        if p != pathlib.Path(keep):
            p.unlink(missing_ok=True)


def _glass_name(glass) -> str:
    if isinstance(glass, dict) and "name" in glass:
        return glass["name"]
    if isinstance(glass, dict) and "filter" in glass:
        raise ActionError("not_implemented", "Glasfilter hier noch nicht (glass_query, M10-T5)")
    if isinstance(glass, str):
        return glass
    raise ActionError("invalid_parameter", f"Glas {glass!r}: erwartet {{'name': ...}}")


def _block_start(data, element: int) -> tuple[int, list[int]]:
    """Erste Fläche des Glasblocks (Kittglied) mit Element ``element`` und alle Elemente des
    Blocks."""
    elems = elements(data)
    if not 0 <= element < len(elems):
        raise ActionError("precondition",
                          f"Element {element} existiert nicht ({len(elems)} Elemente)")
    k = element
    while k > 0 and elems[k - 1]["cemented_to_next"]:
        k -= 1
    first = elems[k]["front"]
    members = [k]
    while elems[members[-1]]["cemented_to_next"]:
        members.append(members[-1] + 1)
    return first, members


def _air_gap(data, s: int, *, allow_last: bool = False) -> float:
    n = len(data["surfaces"])
    last = n - 1 if allow_last else n - 2
    if not 0 <= s <= last or not is_air(data["surfaces"][s].get("material", "AIR")):
        raise ActionError("precondition", f"nach Fläche {s} liegt kein Luftspalt"
                          + ("" if allow_last else " (Arbeitsabstand ausgenommen)"))
    return float(data["surfaces"][s]["thickness"])


def _partner_glass(system_path, glass: str) -> str:
    """Bester Partner von ``glass`` aus der Glasvorauswahl (Achromat, M11-T5)."""
    rep = cli_json(["glasses", "--system", system_path, "--order", 1, "--fixed", glass,
                    "--top", 1])
    if not rep["results"]:
        raise ActionError("precondition", f"kein Partner für {glass} in der Glasvorauswahl")
    return rep["results"][0]["glasses"][1]


# --- place_ghost_dictionary -------------------------------------------------------------------


def place_ghost_dictionary(a, system_path, workdir):
    """Geisterplatten mit Koeffizienten 0 (Merit unverändert); Gläser: Parameter ``glasses``,
    sonst die Katalognamen der vorhandenen Platten. ``weight``: eigenes Strafgewicht μ."""
    _require(a, "after_surface", "distance")
    data = read_system(system_path)
    s = int(a["after_surface"])
    gap = _air_gap(data, s)
    dist = float(a["distance"])
    if not 0 <= dist <= gap:
        raise ActionError("precondition", f"Lage {dist} mm liegt nicht im Spalt [0, {gap}]")
    glasses = a.get("glasses")
    if not glasses:
        glasses = list(dict.fromkeys(g["glass"] for g in data.get("ghost_plates", [])
                                     if isinstance(g.get("glass"), str)))
    if not glasses:
        raise ActionError("missing_parameter",
                          "place_ghost_dictionary: glasses fehlt (keine Platten im System)")
    args = ["--ghost-dictionary", f"{s},{dist!r}"]
    for g in glasses:
        args += ["--ghost-glass", g]
    args += ["--ghost-order", int(a.get("max_order", 6))]
    if "weight" in a:
        args += ["--ghost-weight", float(a["weight"])]
    out, rep = _insert(system_path, workdir, args)
    return [(out, {"after_surface": s, "distance": dist, "glasses": glasses,
                   "new_plates": rep["details"]["new_plates"],
                   "merit_before": rep["merit_before"], "merit_after": rep["merit_after"],
                   "relative_change": rep["relative_change"]})]


# --- insert_module ----------------------------------------------------------------------------


def _thickness_homotopy(path, workdir, glass_gaps, air_gap, target, stages, cycles, files):
    """Dicke der Glasspalte ``glass_gaps`` in ``stages`` Stufen auf ``target`` bringen, der
    Luftspalt ``air_gap`` dahinter wird entsprechend kürzer (Lage der folgenden Scheitel bleibt);
    nach jeder Stufe Nachoptimierung."""
    log = []
    cur = pathlib.Path(path)
    for k in range(1, stages + 1):
        data = read_system(cur)
        t = target * k / stages
        grow = sum(t - data["surfaces"][g]["thickness"] for g in glass_gaps)
        rest = data["surfaces"][air_gap]["thickness"] - grow
        if rest < 0:
            raise ActionError("precondition",
                              f"Dicke {t} mm passt nicht in den Spalt (fehlen {-rest:.4g} mm)")
        for g in glass_gaps:
            data["surfaces"][g]["thickness"] = t
        data["surfaces"][air_gap]["thickness"] = rest
        nxt = _tmp_out(pathlib.Path(workdir))
        write_system(data, nxt)
        files.append(nxt)
        cur, opt = _optimize(nxt, workdir, cycles)
        files.append(cur)
        log.append({"thickness": t, "optimize": opt})
    return cur, log


def insert_module(a, system_path, workdir):
    """Modul aus der Vorlage einfügen. Vorlagen mit Dicke 0 lassen die Merit unverändert; mit
    ``thickness`` folgt die Dicken-Homotopie (``stages`` Stufen, ``cycles`` LM-Zyklen)."""
    _require(a, "module", "after_surface")
    try:
        mod = module(a["module"])
    except KeyError as e:
        raise ActionError("invalid_parameter", str(e.args[0])) from None
    kind = mod["template"]
    data = read_system(system_path)
    s = int(a["after_surface"])
    gap = _air_gap(data, s, allow_last=(kind == "aplanatic_shell"))
    glasses = [_glass_name(g) for g in a.get("glasses", [])]
    report = {"module": a["module"], "template": kind, "after_surface": s,
              "group": a.get("group", mod["group"])}
    files = []
    if kind == "aplanatic_shell":
        if not glasses:
            raise ActionError("missing_parameter", "aplanatic_shell: glasses fehlt (Glas der Schale)")
        pos = float(a.get("position", SHELL_POSITION * gap))
        t = float(a.get("thickness", SHELL_THICKNESS * (gap - pos)))
        out, rep = _insert(system_path, workdir,
                           ["--aplanatic-shell", f"{s},{pos!r},{t!r},{glasses[0]}"])
        report.update({"position": pos, "thickness": t, "glass": glasses[0],
                       "surfaces": [rep["first_new_surface"], rep["first_new_surface"] + 1],
                       **{k: rep["details"][k] for k in ("intercept", "image_intercept",
                                                          "c_aplanatic", "c_concentric",
                                                          "working_distance")},
                       "merit_before": rep["merit_before"], "merit_after": rep["merit_after"]})
        return [(out, report)]
    if kind not in NULL_TEMPLATES:
        raise ActionError("invalid_parameter", f"Vorlage {kind} nicht für insert_module")
    if not glasses:
        raise ActionError("missing_parameter", f"{kind}: glasses fehlt")
    pos = float(a.get("position", gap / 2))
    if not 0 < pos < gap:
        raise ActionError("precondition", f"Lage {pos} mm liegt nicht im Spalt (0, {gap})")
    c0 = float(data["surfaces"][s].get("curvature", 0.0))
    if kind == "null_meniscus":
        out, rep = _insert(system_path, workdir,
                           ["--meniscus", f"{s},{pos!r},{c0!r},{glasses[0]}"])
        first, glass_gaps, air_after = rep["first_new_surface"], [rep["first_new_surface"]], \
            rep["first_new_surface"] + 1
        report["glasses"] = glasses[:1]
    elif kind == "null_doublet":
        g2 = glasses[1] if len(glasses) > 1 else _partner_glass(system_path, glasses[0])
        out, rep = _insert(system_path, workdir,
                           ["--null-doublet", f"{s},{pos!r},{c0!r},{glasses[0]},{g2}"])
        first = rep["first_new_surface"]
        glass_gaps, air_after = [first, first + 1], first + 2
        report["glasses"] = [glasses[0], g2]
    else:  # gauss_pair: zwei Null-Menisken, Hohlseiten einander zugewandt
        semi = max(data["surfaces"][s].get("semi_diameter", 0.0),
                   data["surfaces"][s + 1].get("semi_diameter", 0.0))
        spacing = GAUSS_SPACING * min(pos, gap - pos) * 2
        p1, p2 = pos - spacing / 2, pos + spacing / 2
        c = 2.0 / spacing
        if semi > 0:
            c = min(c, MAX_HEMISPHERE / semi)
        g2 = glasses[1] if len(glasses) > 1 else glasses[0]
        mid, rep1 = _insert(system_path, workdir, ["--meniscus", f"{s},{p1!r},{-c!r},{glasses[0]}"])
        files.append(mid)
        out, rep = _insert(mid, workdir, ["--meniscus", f"{s + 2},{p2 - p1!r},{c!r},{g2}"])
        first = s + 1
        glass_gaps, air_after = [s + 1, s + 3], s + 4
        rep["merit_before"] = rep1["merit_before"]
        report.update({"glasses": [glasses[0], g2], "curvatures": [-c, c], "positions": [p1, p2]})
    report.update({"position": pos, "curvature": c0,
                   "surfaces": list(range(first, air_after + 1)),
                   "merit_before": rep["merit_before"], "merit_after": rep["merit_after"]})
    if "thickness" in a:
        if kind == "gauss_pair":
            # beide Menisken wachsen, jeder Luftspalt dahinter wird kürzer
            cur = out
            log = []
            for g, air in ((s + 1, s + 2), (s + 3, s + 4)):
                cur, part = _thickness_homotopy(cur, workdir, [g], air, float(a["thickness"]),
                                                int(a.get("stages", DEFAULT_STAGES)),
                                                int(a.get("cycles", DEFAULT_CYCLES)), files)
                log += part
        else:
            cur, log = _thickness_homotopy(out, workdir, glass_gaps, air_after,
                                           float(a["thickness"]),
                                           int(a.get("stages", DEFAULT_STAGES)),
                                           int(a.get("cycles", DEFAULT_CYCLES)), files)
        files.append(out)
        _drop(files, cur)
        report["homotopy"] = log
        return [(pathlib.Path(cur), report)]
    _drop(files, out)
    return [(out, report)]


# --- remove_element ---------------------------------------------------------------------------


def _remove_block(system_path, workdir, first, stages, cycles, files):
    """Rückbau des Glasblocks ab ``first``: Rückbau-Homotopie (falls kein Null-Element) und
    Entfernen. Rückgabe (Datei, Bericht)."""
    log = []
    cur = pathlib.Path(system_path)
    for k in range(stages):
        frac = 1.0 / (stages - k)
        nxt, rep = _insert(cur, workdir, ["--collapse", f"{first},{frac!r}"])
        files.append(nxt)
        cur, opt = _optimize(nxt, workdir, cycles)
        files.append(cur)
        log.append({"fraction": frac, "merit": rep["merit_after"], "optimize": opt})
    out, rep = _insert(cur, workdir, ["--remove", first])
    return out, {"first_surface": first, "removed_surfaces": rep["details"]["removed_surfaces"],
                 "homotopy": log,
                 "merit_before_removal": rep["merit_before"], "merit_after": rep["merit_after"],
                 "relative_change": rep["relative_change"]}


def _is_null(data, first) -> bool:
    s = data["surfaces"]
    k = first
    c = s[first].get("curvature", 0.0)
    while not is_air(s[k].get("material", "AIR")):
        if s[k]["thickness"] != 0.0 or s[k + 1].get("curvature", 0.0) != c:
            return False
        k += 1
    return True


def remove_element(a, system_path, workdir):
    """Element (bei Kittgliedern der ganze Block) entfernen: Glasdicken und Krümmungsunterschiede
    in ``stages`` Stufen gegen 0, nach jeder Stufe ``cycles`` LM-Zyklen mit den übrigen Variablen,
    dann Flächen entfernen. Ein Null-Element (z. B. nach insert_null_meniscus) wird sofort
    entfernt: das Ergebnis ist das System vor dem Einfügen."""
    _require(a, "element")
    data = read_system(system_path)
    first, members = _block_start(data, int(a["element"]))
    null = _is_null(data, first)
    stages = 0 if null else int(a.get("stages", DEFAULT_STAGES))
    if stages < 0:
        raise ActionError("invalid_parameter", "stages < 0")
    files = []
    out, rep = _remove_block(system_path, workdir, first, stages,
                             int(a.get("cycles", DEFAULT_CYCLES)), files)
    _drop(files, out)
    return [(out, {"element": int(a["element"]), "removed_elements": members,
                   "null_element": null, **rep})]


# --- replace_glass ----------------------------------------------------------------------------


def replace_glass(a, system_path, workdir):
    """Glas eines Elements tauschen. Mit Name: ein Kandidat (``cycles`` LM-Zyklen, Standard 0).
    Mit Filter ``partner_of_element``: die besten Partner aus der Glasvorauswahl (M11-T5) als
    getrennte Kandidaten, jeder nachoptimiert (Standard 50 Zyklen), nach Merit sortiert."""
    _require(a, "element", "glass")
    data = read_system(system_path)
    elems = elements(data)
    k = int(a["element"])
    if not 0 <= k < len(elems):
        raise ActionError("precondition", f"Element {k} existiert nicht ({len(elems)} Elemente)")
    surface = elems[k]["front"]
    glass = a["glass"]
    if isinstance(glass, dict) and "filter" in glass:
        flt = dict(glass["filter"])
        partner = flt.pop("partner_of_element", None)
        limit = int(flt.pop("max_results", 4))
        if flt or partner is None:
            raise ActionError("not_implemented",
                              f"Glasfilter {sorted(flt) or ['ohne partner_of_element']}: "
                              "glass_query folgt in M10-T5")
        pre = read_action({"action": "glass_preselect", "order": 1, "element": int(partner),
                           "top": 50}, system_path)["results"]
        names = list(dict.fromkeys(r["glasses"][1] for r in pre))[:limit]
        if not names:
            raise ActionError("precondition", "Glasfilter: keine passenden Gläser (leer)")
        cycles = int(a.get("cycles", DEFAULT_CYCLES))
    else:
        names = [_glass_name(glass)]
        cycles = int(a.get("cycles", 0))
    outs = []
    for name in names:
        out, rep = _insert(system_path, workdir, ["--replace-glass", f"{surface},{name}"])
        final, opt = _optimize(out, workdir, cycles)
        if final != out:
            out.unlink(missing_ok=True)
        outs.append((final, {"element": k, "surface": surface, "glass": name,
                             "merit_before": rep["merit_before"],
                             "merit_after": rep["merit_after"], "optimize": opt}))
    if len(outs) > 1:
        def key(o):
            m = (o[1]["optimize"] or {}).get("merit")
            return (math.inf if m is None else m, o[1]["glass"])
        outs.sort(key=key)
    return outs


# --- replace_group ----------------------------------------------------------------------------


def group_elements(system_path) -> dict:
    """Elemente pro Gruppe wie im Zustand (state.assign_groups, Randstrahlhöhen)."""
    data = read_system(system_path)
    elems = elements(data)
    heights = [s["y"] for s in aberrations(system_path)["surfaces"]]
    groups = assign_groups(elems, heights)
    out = {"front": [], "middle": [], "rear": []}
    for e in elems:
        out[groups[e["element"]]].append(e["element"])
    return out


def replace_group(a, system_path, workdir):
    """Gruppe ersetzen: alle Glasblöcke der Gruppe zurückbauen (wie remove_element), dann die
    Module in den vereinigten Spalt einfügen, gleichmäßig verteilt. ``glasses`` gilt für alle
    Module (Standard: Katalogglas des ersten entfernten Elements)."""
    _require(a, "group", "modules")
    group = a["group"]
    mods = list(a["modules"])
    if not 1 <= len(mods) <= 4:
        raise ActionError("invalid_parameter", "modules: 1 bis 4 Module")
    for m in mods:
        try:
            module(m)
        except KeyError as e:
            raise ActionError("invalid_parameter", str(e.args[0])) from None
    data = read_system(system_path)
    members = group_elements(system_path).get(group)
    if not members:
        raise ActionError("precondition", f"Gruppe {group} hat keine Elemente")
    elems = elements(data)
    firsts = sorted({_block_start(data, k)[0] for k in members}, reverse=True)
    glasses = [_glass_name(g) for g in a.get("glasses", [])]
    if not glasses:
        m = data["surfaces"][elems[members[0]]["front"]].get("material")
        glasses = [m] if isinstance(m, str) else []
    if not glasses:
        raise ActionError("missing_parameter", "glasses fehlt (erstes Element ohne Katalogglas)")
    stages = int(a.get("stages", DEFAULT_STAGES))
    cycles = int(a.get("cycles", DEFAULT_CYCLES))
    files = []
    cur = pathlib.Path(system_path)
    removed = []
    for first in firsts:  # von hinten, damit die Flächennummern davor gültig bleiben
        d = read_system(cur)
        stg = 0 if _is_null(d, first) else stages
        cur, rep = _remove_block(cur, workdir, first, stg, cycles, files)
        files.append(cur)
        removed.append(rep)
    gap_surface = min(firsts) - 1
    gap = float(read_system(cur)["surfaces"][gap_surface]["thickness"])
    inserted = []
    n = len(mods)
    for j in reversed(range(n)):  # hinterstes Modul zuerst: der Spalt nach gap_surface bleibt vorn
        pos = gap * (j + 1) / (n + 1)
        act = {"action": "insert_module", "module": mods[j], "after_surface": gap_surface,
               "position": pos, "glasses": [{"name": g} for g in glasses]}
        if module(mods[j])["template"] == "aplanatic_shell":
            act["thickness"] = SHELL_THICKNESS * (gap * (j + 2) / (n + 1) - pos)
        [(cur, rep)] = insert_module(act, cur, workdir)
        files.append(cur)
        inserted.append(rep)
    _drop(files, cur)
    return [(pathlib.Path(cur), {"group": group, "removed": removed, "inserted": inserted,
                                 "gap_surface": gap_surface})]


# --- start_from_skeleton ----------------------------------------------------------------------

#: Vorzeichen der Brechkräfte pro Gruppe (Frontgruppe, Mittelgruppe, Hintergruppe), innerhalb
#: einer Gruppe von der Objektseite aus; Typen nach Zhang 2019, Tab. 6.2
SKELETONS = {
    "PPN": ("P", "P", "N"),
    "PPN_strong_retrofocus": ("P", "P", "N"),
    "PP(NP)": ("P", "P", "NP"),
    "P(PN)0": ("P", "PN", "0"),
    "P(P0)N": ("P", "P0", "N"),
    "(P0)PN": ("P0", "P", "N"),
    "P(P0P)N": ("P", "P0P", "N"),
    "PNP": ("P", "N", "P"),
}
#: relative Brechkraft der Zeichen (N: Hintergruppe als Retrofokus-Glied)
SIGN_WEIGHT = {"P": 1.0, "N": -0.5, "0": 0.0}
STRONG_RETROFOCUS_N = -1.0
#: Geometrie des Gerüsts in Einheiten der Brennweite f
SKELETON_THICKNESS = 0.3   # Mittendicke einer Linse (Kittglied: je Teil 0,6 davon)
SKELETON_GAP = 0.15        # Luftspalt zwischen Elementen
SKELETON_STOP_GAP = 0.1    # Blende → erstes Element
SKELETON_SEMI = 1.5        # Halbdurchmesser in Einheiten des Pupillenradius


def _paraxial(curv, thick, ns):
    """y-nu-Trace mit y₀ = 1, u₀ = 0 (Luft vor Fläche 0): Brechkraft φ und hintere Schnittweite."""
    y, u, n = 1.0, 0.0, 1.0
    for c, t, n2 in zip(curv, thick, ns):
        u = (n * u - y * c * (n2 - n)) / n2
        n = n2
        last_y, last_u = y, u
        y += t * u
    phi = -n * last_u / 1.0
    bfd = -last_y / last_u if last_u != 0 else math.inf
    return phi, bfd


def _skeleton_elements(a, distribution):
    """Elemente in Tracerichtung (Blende → Objekt): Liste (Vorlage, Zeichen, Gruppe)."""
    signs = dict(zip(("front", "middle", "rear"), SKELETONS[distribution]))
    out = []
    for group in ("rear", "middle", "front"):
        kinds = []
        for m in a[group]:
            try:
                tpl = module(m)["template"]
            except KeyError as e:
                raise ActionError("invalid_parameter", str(e.args[0])) from None
            kinds += ["lens", "lens"] if tpl == "gauss_pair" else [
                "doublet" if tpl == "null_doublet" else "lens"]
        letters = signs[group]
        if len(letters) == 1:
            letters = letters * len(kinds)
        if len(letters) != len(kinds):
            raise ActionError("invalid_parameter",
                              f"{group}: {len(kinds)} Elemente, Verteilung {signs[group]} "
                              "(ein Zeichen für alle oder eines pro Element)")
        # Listen und Zeichen von der Objektseite: in Tracerichtung umkehren
        out += [(k, s, group) for k, s in reversed(list(zip(kinds, letters)))]
    return out


def start_from_skeleton(a, system_path, workdir):
    """Neuer Wurzelkandidat aus einer Brechkraftverteilung: pro Element eine Linse bzw. ein
    Kittglied (gleichseitig gebogen, Kittfläche eben), Brechkräfte nach den Zeichen, gemeinsam
    skaliert auf die Brennweite des Kandidaten (``target.focal_length``, sonst paraxial), Blende
    f · ``start_aperture``, Arbeitsabstand = hintere Schnittweite. Quelle, Spektrum, Kataloge und
    Nebenbedingungen ohne Flächenbezug kommen vom Kandidaten. Variablen: alle Krümmungen,
    Luftspalte und Arbeitsabstand."""
    _require(a, "power_distribution", "front", "middle", "rear")
    dist = a["power_distribution"]
    if dist not in SKELETONS:
        raise ActionError("not_implemented", f"Brechkraftverteilung {dist} noch nicht umgesetzt")
    data = read_system(system_path)
    els = _skeleton_elements(a, dist)
    trace = cli_json(["trace", system_path])
    f = ((data.get("target") or {}).get("focal_length") or trace["paraxial"]["focal_length"])
    if not (f and math.isfinite(f) and f > 0):
        raise ActionError("precondition", "keine positive Brennweite für das Gerüst")
    na = float(a.get("start_aperture", data["surfaces"][0]["semi_diameter"] / f))
    pupil = f * na
    glasses = [_glass_name(g) for g in a.get("glasses", [])]
    if not glasses:
        first = next((s.get("material") for s in data["surfaces"]
                      if isinstance(s.get("material"), str) and not is_air(s.get("material"))),
                     None)
        if first is None:
            raise ActionError("missing_parameter", "glasses fehlt (Kandidat ohne Katalogglas)")
        glasses = [first]
    partner = glasses[1] if len(glasses) > 1 else None
    if partner is None and any(k == "doublet" for k, _, _ in els):
        partner = _partner_glass(system_path, glasses[0])
    weight = dict(SIGN_WEIGHT)
    if dist == "PPN_strong_retrofocus":
        weight["N"] = STRONG_RETROFOCUS_N

    # Gerüst mit Brechkräften s · w: Flächen (Krümmungsfaktor, Dicke, Glas) bei s = 1
    t_lens = SKELETON_THICKNESS * f
    semi = SKELETON_SEMI * pupil
    proto = [{"type": "stop", "thickness": SKELETON_STOP_GAP * f, "material": "AIR",
              "semi_diameter": pupil}]
    for kind, sign, group in els:
        w = weight[sign]
        if kind == "lens":
            proto.append({"k": ("half", w, glasses[0]), "thickness": t_lens,
                          "material": glasses[0], "group": group})
            proto.append({"k": ("half_back", w, glasses[0]), "thickness": SKELETON_GAP * f,
                          "material": "AIR"})
        else:
            proto.append({"k": ("half", w, glasses[0]), "thickness": 0.6 * t_lens,
                          "material": glasses[0], "group": group})
            proto.append({"k": ("zero", w, None), "thickness": 0.6 * t_lens, "material": partner})
            proto.append({"k": ("half_back", w, partner), "thickness": SKELETON_GAP * f,
                          "material": "AIR"})
    # Brechzahlen bei λ_ref über den Aberrationsbericht einer Probe
    probe = []
    for p in proto:
        q = {k: v for k, v in p.items() if k not in ("k", "group")}
        if "k" in p:
            q.update({"type": "sphere", "curvature": 0.0, "semi_diameter": semi})
        probe.append(q)
    sample = {**{k: v for k, v in data.items()
                 if k in ("format", "units", "glass_catalogs", "spectrum", "source")},
              "name": f"Gerüst {dist}", "surfaces": probe}
    probe_path = _tmp_out(pathlib.Path(workdir))
    write_system(sample, probe_path)
    try:
        ns = [s["n_after"] for s in aberrations(probe_path)["surfaces"]]
    finally:
        probe_path.unlink(missing_ok=True)
    thick = [p["thickness"] for p in proto]

    def curvatures(scale):
        # Vorderfläche: Glas danach (ns[i]), Rückfläche: Glas davor (ns[i - 1]); dünn hat jedes
        # Element die Brechkraft scale · w (Kittglied: je Hälfte in einem Glas, Kittfläche eben)
        out = []
        for i, p in enumerate(proto):
            how, w = p["k"][:2] if "k" in p else ("none", 0.0)
            if how == "half":
                out.append(scale * w / (2.0 * (ns[i] - 1.0)))
            elif how == "half_back":
                out.append(-scale * w / (2.0 * (ns[i - 1] - 1.0)))
            else:
                out.append(0.0)
        return out

    target = 1.0 / f
    total = sum(weight[s] for _, s, _ in els)
    if total <= 0:
        raise ActionError("precondition", f"Verteilung {dist}: Summe der Brechkräfte ≤ 0")
    s0, s1 = target / total, 1.1 * target / total
    p0 = _paraxial(curvatures(s0), thick, ns)[0] - target
    p1 = _paraxial(curvatures(s1), thick, ns)[0] - target
    for _ in range(60):
        if p1 == p0:
            break
        s0, s1, p0 = s1, s1 - p1 * (s1 - s0) / (p1 - p0), p1
        p1 = _paraxial(curvatures(s1), thick, ns)[0] - target
        if abs(p1) <= 1e-14 * target:
            break
    phi, bfd = _paraxial(curvatures(s1), thick, ns)
    if not (math.isfinite(phi) and abs(phi - target) <= 1e-9 * target):
        raise ActionError("precondition", f"Verteilung {dist}: Brennweite {f} mm nicht erreichbar")
    if not bfd > 0:
        raise ActionError("precondition",
                          f"Verteilung {dist}: kein positiver Arbeitsabstand ({bfd:.4g} mm)")
    curv = curvatures(s1)
    surfaces = []
    for p, c in zip(proto, curv):
        q = {k: v for k, v in p.items() if k not in ("k", "group")}
        if "k" in p:
            sd = semi if c == 0 else min(semi, 0.95 / abs(c))
            q.update({"type": "sphere", "curvature": c, "semi_diameter": sd})
        surfaces.append(q)
    surfaces[-1]["thickness"] = bfd
    air = [i for i, q in enumerate(surfaces[:-1]) if is_air(q["material"])]
    constraints = [c for c in (data.get("constraints") or {}).get("items", [])
                   if not any(k in c for k in ("surface", "surfaces", "reference_surface",
                                               "plate", "plates"))
                   and c.get("type") not in ("ghost_penalty", "glass_hull", "max_semi_diameter",
                                             "min_center_thickness", "max_center_thickness",
                                             "min_edge_thickness", "max_incidence_angle",
                                             "parfocal_length")]
    target_block = dict(data.get("target") or {})
    target_block["focal_length"] = f
    if not any(c.get("type") == "focal_length" for c in constraints):
        constraints.append({"type": "focal_length", "weight": 1.0})
    new = {**{k: v for k, v in data.items()
              if k in ("format", "units", "glass_catalogs", "spectrum", "source")},
           "name": f"Gerüst {dist} ({len(els)} Elemente)",
           "description": "Gerüst aus start_from_skeleton (M10-T4): "
                          + ", ".join(f"{g} {a[g]}" for g in ("front", "middle", "rear")),
           "target": target_block, "surfaces": surfaces,
           "variables": {"items": [
               {"type": "curvature", "surfaces": "all"},
               {"type": "thickness", "surfaces": air, "min": 0.1, "max": max(20.0, 5 * f)},
               {"type": "working_distance", "min": 0.05}]},
           "constraints": {"items": constraints}}
    out = _tmp_out(pathlib.Path(workdir))
    write_system(new, out)
    try:
        check = cli_json(["trace", out])
    except ToolError as e:
        out.unlink(missing_ok=True)
        raise ActionError("precondition", f"Gerüst ungültig: {e}") from None
    return [(out, {"root": True, "power_distribution": dist, "elements": len(els),
                   "groups": {g: [k for k, _, gg in els if gg == g] for g in
                              ("front", "middle", "rear")},
                   "glasses": [glasses[0]] + ([partner] if partner else []),
                   "focal_length": check["paraxial"]["focal_length"], "start_aperture": na,
                   "working_distance": bfd, "power_scale": s1,
                   "failed_rays": check["failed_rays"]})]
