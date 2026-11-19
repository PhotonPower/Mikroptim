"""Strukturelle Aktionen der Entwurfssteuerung (M10-T4; docs/entwurfssteuerung.md, Abschnitte
4.3–4.5): ``place_ghost_dictionary``, ``insert_module``, ``remove_element``, ``replace_glass``,
``replace_group`` und ``start_from_skeleton``. Seit M10-T5 kommen Module und
Brechkraftverteilungen aus der Wissensbasis (``knowledge.py``), Glasangaben dürfen Filter der
Glasabfrage sein, und ``insert_module`` kennt die Vorlagen ``strong_doublet`` und
``petzval_group``.

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
from . import knowledge
from ._cli import ToolError, cli_json
from .actions import ActionError, _require, _tmp_out, resolve_filter
from .files import elements, is_air, read_system, write_system
from .modules import NULL_TEMPLATES, module
from .state import element_groups

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


#: erlaubte Werte des Felds ``group`` an Flächen (docs/dateiformat.md)
GROUPS = ("front", "middle", "rear")


def _set_group(path, surfaces, group):
    """Feld ``group`` an den Flächen ``surfaces`` der Datei setzen (Entscheidung Maintainer
    2026-11-19; ohne Gruppe nichts)."""
    if group is None:
        return
    data = read_system(path)
    for s in surfaces:
        data["surfaces"][s]["group"] = group
    write_system(data, path)


def _drop(paths, keep):
    """Zwischendateien einer Aktion löschen (außer ``keep``)."""
    for p in paths:
        p = pathlib.Path(p)
        if p != pathlib.Path(keep):
            p.unlink(missing_ok=True)


def _glass_name(glass, system_path=None) -> str:
    if isinstance(glass, dict) and "name" in glass:
        return glass["name"]
    if isinstance(glass, dict) and "filter" in glass:
        return resolve_filter(glass["filter"], system_path)
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
    glasses = [_glass_name(g, system_path) for g in a.get("glasses", [])]
    group = a.get("group", mod["group"])
    if group is not None and group not in GROUPS:
        raise ActionError("invalid_parameter", f"group {group}: erlaubt sind {', '.join(GROUPS)}")
    report = {"module": a["module"], "template": kind, "after_surface": s, "group": group}
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
        _set_group(out, report["surfaces"], group)
        return [(out, report)]
    if kind == "petzval_group":
        return _petzval_group(a, mod, system_path, workdir, data, s, gap, glasses, report)
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
    elif kind in ("null_doublet", "strong_doublet"):
        if kind == "strong_doublet":
            g2 = glasses[1] if len(glasses) > 1 else _merte_partner(system_path, glasses[0])
            target = float(a.get("incidence_deg", mod["template_params"].get("incidence_deg", 40)))
            c0, info = _strong_curvature(system_path, data, s, pos, target)
            report.update(info)
        else:
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
        _set_group(cur, report["surfaces"], group)
        return [(pathlib.Path(cur), report)]
    _drop(files, out)
    _set_group(out, report["surfaces"], group)
    return [(out, report)]


def _merte_partner(system_path, glass: str) -> str:
    """Partner nach der Glasregel ``merte_pair`` mit dem größten |Δν_d| (Glasabfrage, M10-T5)."""
    from .glass_query import QueryError, glass_query
    try:
        res = glass_query(system_path, {"rule": "merte_pair", "partner": glass}, limit=None)
    except QueryError as e:
        raise ActionError(e.code, str(e)) from None
    if not res["glasses"]:
        raise ActionError("precondition", f"kein Partner nach merte_pair für {glass}; "
                          "glasses mit zwei Gläsern angeben")
    ref = res["reference"]["vd"]
    return max(res["glasses"], key=lambda g: (abs(g["vd"] - ref), g["name"]))["name"]


def incidence_angle_deg(y: float, u: float, c: float) -> float | None:
    """Einfallswinkel [°] eines meridionalen Strahls (Höhe y in der Scheitelebene, Steigung u)
    auf eine Kugelfläche der Krümmung c mit Scheitel in dieser Ebene; None ohne Schnitt."""
    norm = math.hypot(1.0, u)
    dz, dy = 1.0 / norm, u / norm
    if c == 0.0:
        return math.degrees(math.acos(min(1.0, abs(dz))))
    # Kugel: (z − 1/c)² + y² = 1/c², Strahl (t dz, y + t dy)
    r = 1.0 / c
    b = -r * dz + y * dy
    disc = b * b - y * y
    if disc < 0.0:
        return None
    # Schnitt nahe am Scheitel: kleinere Wurzel von t² + 2 b t + y² = 0 (für |t| klein)
    t = -b - math.copysign(math.sqrt(disc), b) if b != 0.0 else math.sqrt(disc)
    t = y * y / t if t != 0.0 else 0.0
    pz, py = t * dz, y + t * dy
    nz, ny = (pz - r) * c, py * c
    cosi = abs(dz * nz + dy * ny) / math.hypot(nz, ny)
    return math.degrees(math.acos(min(1.0, cosi)))


def _strong_curvature(system_path, data, s, pos, target_deg):
    """Krümmung der Kittfläche des starken Kittglieds: der paraxiale Randstrahl (Höhe und
    Steigung hinter Fläche s, an der Lage pos) trifft sie unter ``target_deg``. Von beiden
    Biegungen die mit kleinerem |c|; Halbkugelgrad höchstens MAX_HEMISPHERE."""
    surf = aberrations(system_path)["surfaces"][s]
    u = float(surf["u_after"])
    y = float(surf["y"]) + pos * u
    semi = max(data["surfaces"][s].get("semi_diameter", 0.0),
               data["surfaces"][s + 1].get("semi_diameter", 0.0), abs(y))
    c_max = MAX_HEMISPHERE / semi
    best = None
    for sign in (1.0, -1.0):
        lo, hi = 0.0, c_max
        f = lambda c: (incidence_angle_deg(y, u, sign * c) or 90.0) - target_deg  # noqa: E731
        # Winkel wächst auf dem Ast, der den Strahl stärker ablenkt; Vorzeichenwechsel suchen
        grid = [c_max * k / 200 for k in range(201)]
        for c1, c2 in zip(grid, grid[1:]):
            if f(c1) <= 0.0 <= f(c2) or f(c1) >= 0.0 >= f(c2):
                lo, hi = c1, c2
                break
        else:
            continue
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if (f(lo) <= 0.0) == (f(mid) <= 0.0):
                lo = mid
            else:
                hi = mid
        c = sign * 0.5 * (lo + hi)
        if best is None or abs(c) < abs(best):
            best = c
    if best is None:
        reach = max(incidence_angle_deg(y, u, k * c_max) or 0.0 for k in (-1.0, 1.0))
        raise ActionError("precondition", f"Einfallswinkel {target_deg}° an der Kittfläche nicht "
                          f"erreichbar (höchstens {reach:.1f}° bei Halbkugelgrad "
                          f"{MAX_HEMISPHERE})")
    return best, {"incidence_deg": incidence_angle_deg(y, u, best), "marginal_height": y,
                  "marginal_slope": u, "cement_curvature": best}


def _petzval_group(a, mod, system_path, workdir, data, s, gap, glasses, report):
    """Petzval-Gruppe: zwei Null-Menisken einfügen, dann Dicke ``thickness`` geben und die
    Brechkräfte so setzen, dass das erste Glied (Tracerichtung) die Brechkraft
    ``power``/f hat und das Paar in Dünnlinsennäherung afokal ist: φ₂ = −φ₁/(1 − d φ₁),
    d = Scheitelabstand. Vorderflächen behalten die Krümmung c₀ von Fläche s; die Rückflächen
    folgen aus der Formel der dicken Linse. Ändert die Merit; Nachoptimierung mit ``cycles``."""
    if not glasses:
        raise ActionError("missing_parameter", "petzval_group: glasses fehlt")
    pos = float(a.get("position", gap / 2))
    if not 0 < pos < gap:
        raise ActionError("precondition", f"Lage {pos} mm liegt nicht im Spalt (0, {gap})")
    spacing = GAUSS_SPACING * min(pos, gap - pos) * 2
    p1 = pos - spacing / 2
    t = float(a.get("thickness", 0.25 * spacing))
    if not 0 < t < spacing:
        raise ActionError("precondition", f"Dicke {t} mm passt nicht zwischen die Glieder "
                          f"(Abstand {spacing:.4g} mm)")
    g2 = glasses[1] if len(glasses) > 1 else glasses[0]
    c0 = float(data["surfaces"][s].get("curvature", 0.0))
    files = []
    mid, rep1 = _insert(system_path, workdir, ["--meniscus", f"{s},{p1!r},{c0!r},{glasses[0]}"])
    files.append(mid)
    nul, _ = _insert(mid, workdir, ["--meniscus", f"{s + 2},{spacing!r},{c0!r},{g2}"])
    files.append(nul)
    f = cli_json(["trace", system_path])["paraxial"]["focal_length"]
    if not (f and math.isfinite(f) and f != 0):
        raise ActionError("precondition", "keine endliche Brennweite für die Petzval-Gruppe")
    ns = [x["n_after"] for x in aberrations(nul)["surfaces"]]
    phi1 = float(a.get("power", mod["template_params"].get("power", 0.5))) / abs(f)
    if spacing * phi1 >= 1.0:
        raise ActionError("precondition", "Petzval-Gruppe: Abstand · Brechkraft ≥ 1 (power kleiner)")
    phi2 = -phi1 / (1.0 - spacing * phi1)
    new = read_system(nul)
    sf = new["surfaces"]
    curv = []
    for front, phi in ((s + 1, phi1), (s + 3, phi2)):
        n = ns[front]
        # φ = (n−1)(c₁ − c₂) + t (n−1)² c₁ c₂ / n  →  c₂
        c2 = ((n - 1.0) * c0 - phi) / ((n - 1.0) * (1.0 - t * (n - 1.0) * c0 / n))
        sf[front]["thickness"] = t
        sf[front + 1]["curvature"] = c2
        sf[front + 1]["thickness"] -= t
        semi = max(sf[front].get("semi_diameter", 0.0), sf[front + 1].get("semi_diameter", 0.0))
        if semi and abs(c2) * semi > MAX_HEMISPHERE:
            raise ActionError("precondition", f"Petzval-Gruppe: Halbkugelgrad {abs(c2) * semi:.3g}"
                              f" > {MAX_HEMISPHERE} (power kleiner)")
        curv.append(c2)
    if min(sf[s + 2]["thickness"], sf[s + 4]["thickness"]) <= 0:
        raise ActionError("precondition", "Petzval-Gruppe: kein Luftspalt hinter den Gliedern")
    # Dickenvariablen der verkürzten Luftspalte: untere Schranke höchstens der neue Wert
    for it in (new.get("variables") or {}).get("items", []):
        if it.get("type") == "thickness" and it.get("surface") in (s + 2, s + 4):
            it["min"] = min(it.get("min", 0.0), sf[it["surface"]]["thickness"])
    out = _tmp_out(pathlib.Path(workdir))
    write_system(new, out)
    try:
        after = cli_json(["trace", out])
    except ToolError as e:
        out.unlink(missing_ok=True)
        _drop(files, out)
        raise ActionError("precondition", f"Petzval-Gruppe ungültig: {e}") from None
    files.append(out)
    final, opt = _optimize(out, workdir, int(a.get("cycles", DEFAULT_CYCLES)))
    _drop(files, final)
    report.update({"glasses": [glasses[0], g2], "positions": [p1, p1 + spacing],
                   "thickness": t, "powers": [phi1, phi2], "back_curvatures": curv,
                   "surfaces": list(range(s + 1, s + 5)),
                   "merit_before": rep1["merit_before"],
                   "focal_length_before": f,
                   "focal_length_after": after["paraxial"]["focal_length"],
                   "failed_rays": after["failed_rays"], "optimize": opt})
    _set_group(final, report["surfaces"], report.get("group"))
    return [(pathlib.Path(final), report)]


# --- remove_element ---------------------------------------------------------------------------


def _remove_block(system_path, workdir, first, stages, cycles, files):
    """Rückbau des Glasblocks ab ``first``: Rückbau-Homotopie (falls kein Null-Element) und
    Entfernen. Rückgabe (Datei, Bericht). ``warnings`` sammelt die Warnungen von
    ``mikroptim insert`` (angehobene obere Schranken von Dicke bzw. Arbeitsabstand, Entscheidung
    Maintainer 2026-11-19); sie stehen im Bericht und damit im Sitzungsprotokoll."""
    log = []
    warnings = []
    cur = pathlib.Path(system_path)
    for k in range(stages):
        frac = 1.0 / (stages - k)
        nxt, rep = _insert(cur, workdir, ["--collapse", f"{first},{frac!r}"])
        warnings += rep.get("warnings") or []
        files.append(nxt)
        cur, opt = _optimize(nxt, workdir, cycles)
        files.append(cur)
        log.append({"fraction": frac, "merit": rep["merit_after"], "optimize": opt})
    out, rep = _insert(cur, workdir, ["--remove", first])
    warnings += rep.get("warnings") or []
    return out, {"first_surface": first, "removed_surfaces": rep["details"]["removed_surfaces"],
                 "homotopy": log,
                 "merit_before_removal": rep["merit_before"], "merit_after": rep["merit_after"],
                 "relative_change": rep["relative_change"], "warnings": warnings}


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
        # Glasabfrage (M10-T5): mit partner_of_element die Partner der Glasvorauswahl in deren
        # Rangfolge, sonst Katalogreihenfolge; das bisherige Glas des Elements fällt heraus
        from .glass_query import QueryError, glass_query
        flt = dict(glass["filter"])
        limit = int(flt.pop("max_results", 4))
        current = data["surfaces"][surface].get("material")
        if isinstance(current, str):
            flt["exclude"] = list(flt.get("exclude", [])) + [current]
        try:
            res = glass_query(system_path, flt, limit=limit)
        except QueryError as e:
            raise ActionError(e.code, str(e)) from None
        names = [g["name"] for g in res["glasses"]]
        if not names:
            raise ActionError("precondition", f"Glasfilter {glass['filter']}: keine passenden "
                              "Gläser (leer)")
        cycles = int(a.get("cycles", DEFAULT_CYCLES))
    else:
        names = [_glass_name(glass, system_path)]
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
    """Elemente pro Gruppe wie im Zustand (state.element_groups: Feld ``group`` der Flächen,
    sonst Randstrahlhöhen)."""
    data = read_system(system_path)
    elems = elements(data)
    heights = [s["y"] for s in aberrations(system_path)["surfaces"]]
    groups = element_groups(data, elems, heights)
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
    glasses = [_glass_name(g, system_path) for g in a.get("glasses", [])]
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
                                 "gap_surface": gap_surface,
                                 "warnings": [w for r in removed for w in r["warnings"]]})]


# --- start_from_skeleton ----------------------------------------------------------------------

#: Vorzeichen der Brechkräfte pro Gruppe (Frontgruppe, Mittelgruppe, Hintergruppe), innerhalb
#: einer Gruppe von der Objektseite aus; aus der Wissensbasis (power_distributions.json)
SKELETONS = {k: (d["groups"]["front"], d["groups"]["middle"], d["groups"]["rear"])
             for k, d in knowledge.distributions().items()}
#: Geometrie des Gerüsts in Einheiten der Brennweite f
SKELETON_THICKNESS = 0.3   # Mittendicke einer Linse (Kittglied: je Teil 0,6 davon)
SKELETON_GAP = 0.15        # Luftspalt zwischen Elementen
SKELETON_STOP_GAP = 0.1    # Blende → erstes Element
SKELETON_SEMI = 1.5        # Halbdurchmesser in Einheiten des Pupillenradius
#: Faktoren für Dicken und Abstände, falls das Gerüst die Brennweite sonst nicht erreicht (M10-T5)
SKELETON_SHRINK = (1.0, 0.5, 0.25)


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
            if tpl not in knowledge.SKELETON_ELEMENTS:
                raise ActionError("invalid_parameter",
                                  f"Modul {m}: Vorlage {tpl} nicht für Gerüste")
            kinds += list(knowledge.SKELETON_ELEMENTS[tpl])
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
    glasses = [_glass_name(g, system_path) for g in a.get("glasses", [])]
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
    weight = knowledge.sign_weights(dist)

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

    def solve(thick):
        """Gemeinsamer Faktor der Brechkräfte für φ = 1/f (Sekantenverfahren); None ohne
        Lösung mit positivem Faktor und positivem Arbeitsabstand."""
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
        if not (math.isfinite(phi) and abs(phi - target) <= 1e-9 * target and s1 > 0):
            return None, "Brennweite"
        if not bfd > 0:
            return None, f"Arbeitsabstand {bfd:.4g} mm"
        return (s1, bfd), None

    # Zu lange Gerüste (viele dicke Glieder, z. B. double_gauss) erreichen die Brennweite nicht:
    # Dicken und Abstände dann halbieren (höchstens zweimal)
    for shrink in SKELETON_SHRINK:
        thick_k = [t * shrink for t in thick]
        sol, why = solve(thick_k)
        if sol is not None:
            break
    if sol is None:
        if why == "Brennweite":
            raise ActionError("precondition", f"Verteilung {dist}: Brennweite {f} mm nicht erreichbar")
        raise ActionError("precondition",
                          f"Verteilung {dist}: kein positiver Arbeitsabstand ({why})")
    s1, bfd = sol
    for p, t in zip(proto, thick_k):
        p["thickness"] = t
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
           # neues System: Referenzsphäre ausdrücklich, Empfehlung image_infinite für die
           # Synthese mit Geisterplatten (ADR-0017, Entscheidung Maintainer 2026-11-19); eine
           # ausdrückliche Wahl des Kandidaten bleibt
           "reference_sphere": data.get("reference_sphere") or "image_infinite",
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
                   "working_distance": bfd, "power_scale": s1, "length_scale": shrink,
                   "failed_rays": check["failed_rays"]})]
