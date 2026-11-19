"""Glasabfrage ``glass_query`` (M10-T5; docs/entwurfssteuerung.md, Abschnitt 5.3).

Kennzahlen n_d, ν_d, P_g,F und ΔP_g,F der geladenen Kataloge rechnet der Kern
(``mikroptim glasses --list``, docs/numerik.md, Abschnitt 17); die Reintransmission kommt aus dem
Feld ``transmission`` der Katalogdateien. Darauf wirken der Filter und die Glasregeln der
Wissensbasis (``knowledge.py``).

Filter (``mikroptim.design.action/1``, ``$defs/glass_filter``), alle Angaben müssen gelten:

- ``rule`` bzw. ``rules``: Glasregeln der Wissensbasis (Paarregeln brauchen ein Bezugsglas)
- ``nd``, ``vd``, ``pgf``, ``dPgF``: geschlossene Bereiche ``[min, max]``
- ``transmission``: ``{"nm": λ, "min": τ, "thickness_mm": d}`` – Reintransmission bei λ
  (linear interpoliert) auf die Dicke d umgerechnet (τ^(d/d₀)), mindestens τ; Gläser ohne Daten
  fallen heraus und stehen unter ``no_transmission_data``
- ``partner``: Bezugsglas (Name) der Paarregeln
- ``partner_of_element``: Bezugsglas ist das Glas dieses Elements, und die Kandidaten sind seine
  achromatischen Partner aus der algebraischen Glasvorauswahl (M11-T5) in deren Rangfolge
- ``exclude``: Namen, die nicht in Frage kommen

Reihenfolge der Treffer: Rang der Glasvorauswahl (mit ``partner_of_element``), sonst
Katalogreihenfolge.
"""

from __future__ import annotations

import json
import pathlib

from . import knowledge
from ._cli import cli_json
from .files import elements, read_system

RANGE_KEYS = {"nd": "nd", "vd": "vd", "pgf": "pgf", "dPgF": "dpgf"}
FILTER_KEYS = {"rule", "rules", "nd", "vd", "pgf", "dPgF", "transmission", "partner",
               "partner_of_element", "exclude", "max_results"}


class QueryError(ValueError):
    """Ungültiger Filter; ``code`` wie bei ActionError."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def glass_table(system_path) -> list[dict]:
    """Kennzahlen aller Gläser der Kataloge des Systems (Katalogreihenfolge), mit
    ``transmission`` aus den Katalogdateien, falls vorhanden."""
    rep = cli_json(["glasses", "--system", system_path, "--list"])
    trans = _transmission_data(system_path)
    return [{"name": g["name"], "nd": g["n_d"], "vd": g["v_d"], "pgf": g["p_gF"],
             "dpgf": g["dp_gF"], "transmission": trans.get(g["name"])} for g in rep["list"]]


def _transmission_data(system_path) -> dict:
    from .. import load_system
    out = {}
    for f in load_system(str(system_path)).catalog_files:
        doc = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        for g in doc.get("glasses", []):
            if isinstance(g.get("transmission"), dict):
                out[g["name"]] = g["transmission"]
    return out


def transmission_at(data: dict | None, nm: float, thickness_mm: float | None = None):
    """Reintransmission bei ``nm`` (linear zwischen den Stützstellen), auf ``thickness_mm``
    umgerechnet; None ohne Daten oder außerhalb des tabellierten Bereichs."""
    if not data:
        return None
    xs, ys = data.get("nm") or [], data.get("tau") or []
    if len(xs) != len(ys) or not xs or not xs[0] <= nm <= xs[-1]:
        return None
    tau = ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= nm <= xs[i + 1]:
            w = 0.0 if xs[i + 1] == xs[i] else (nm - xs[i]) / (xs[i + 1] - xs[i])
            tau = ys[i] + w * (ys[i + 1] - ys[i])
            break
    if thickness_mm is not None and data.get("thickness_mm"):
        tau = max(tau, 0.0) ** (float(thickness_mm) / float(data["thickness_mm"]))
    return tau


def _range(name, v):
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise QueryError("invalid_parameter", f"{name}: [min, max] erwartet")
    lo, hi = v
    if lo is not None and hi is not None and lo > hi:
        raise QueryError("invalid_parameter", f"{name}: min > max")
    return (lo, hi)


def _element_glass(system_path, k: int) -> str:
    data = read_system(system_path)
    elems = elements(data)
    if not 0 <= k < len(elems):
        raise QueryError("precondition", f"Element {k} existiert nicht ({len(elems)} Elemente)")
    glass = data["surfaces"][elems[k]["front"]].get("material")
    if not isinstance(glass, str):
        raise QueryError("precondition", f"Element {k} hat kein Katalogglas")
    return glass


def glass_query(system_path, flt: dict, limit: int | None = 20) -> dict:
    """Gläser der Kataloge des Systems, die den Filter ``flt`` erfüllen. Rückgabe mit
    ``count`` (alle Treffer), ``glasses`` (höchstens ``limit``; Kennzahlen und erfüllte
    Glasregeln), ``reference``, ``rules`` (Bedingung in Worten und Quelle)."""
    flt = dict(flt or {})
    unknown = set(flt) - FILTER_KEYS
    if unknown:
        raise QueryError("invalid_parameter", f"glass_query: unbekannte Filter {sorted(unknown)}")
    kb_rules = knowledge.rules()
    names = ([flt["rule"]] if "rule" in flt else []) + list(flt.get("rules", []))
    bad = [r for r in names if r not in kb_rules]
    if bad:
        raise QueryError("invalid_parameter",
                         f"unbekannte Glasregeln {bad} (bekannt: {sorted(kb_rules)})")
    ranges = {RANGE_KEYS[k]: _range(k, flt[k]) for k in RANGE_KEYS if k in flt}
    table = glass_table(system_path)
    by_name = {g["name"]: g for g in table}

    ref_name = None
    candidates = table
    preselect = None
    if "partner_of_element" in flt:
        ref_name = _element_glass(system_path, int(flt["partner_of_element"]))
        rep = cli_json(["glasses", "--system", system_path, "--order", 1, "--fixed", ref_name,
                        "--top", 50])
        preselect = {}
        for r in rep["results"]:
            preselect.setdefault(r["glasses"][1], {"secondary_spectrum": r["residual"],
                                                   "power_sum": r["power_sum"]})
        candidates = [by_name[n] for n in preselect if n in by_name]
    if "partner" in flt:
        if ref_name is not None and flt["partner"] != ref_name:
            raise QueryError("invalid_parameter", "partner und partner_of_element widersprechen sich")
        ref_name = flt["partner"]
    ref = None
    if ref_name is not None:
        if ref_name not in by_name:
            raise QueryError("precondition", f"Bezugsglas {ref_name} nicht in den Katalogen "
                             "(oder ohne die Linien g bis C)")
        ref = by_name[ref_name]
    pair_rules = [r for r in names if kb_rules[r].get("needs_partner")]
    if pair_rules and ref is None:
        raise QueryError("missing_parameter",
                         f"Paarregeln {pair_rules} brauchen partner oder partner_of_element")

    trans = flt.get("transmission")
    if trans is not None:
        if not isinstance(trans, dict) or "nm" not in trans or "min" not in trans:
            raise QueryError("invalid_parameter", "transmission: {nm, min[, thickness_mm]}")
    exclude = set(flt.get("exclude", [])) | ({ref_name} if ref_name else set())

    hits, no_data = [], []
    for g in candidates:
        if g["name"] in exclude:
            continue
        if any(not knowledge._in(rng, g[k]) for k, rng in ranges.items()):
            continue
        if any(not knowledge.evaluate(kb_rules[r]["condition"], g, ref) for r in names):
            continue
        entry = {"name": g["name"], "nd": g["nd"], "vd": g["vd"], "pgf": g["pgf"],
                 "dPgF": g["dpgf"]}
        if trans is not None:
            tau = transmission_at(g["transmission"], float(trans["nm"]), trans.get("thickness_mm"))
            if tau is None:
                no_data.append(g["name"])
                continue
            if tau < float(trans["min"]):
                continue
            entry["transmission"] = tau
        entry["rules"] = [r for r, d in kb_rules.items()
                          if (ref is not None or not d.get("needs_partner"))
                          and knowledge.evaluate(d["condition"], g, ref)]
        if preselect is not None:
            entry.update(preselect[g["name"]])
        hits.append(entry)
    shown = hits if limit is None else hits[:int(limit)]
    return {"count": len(hits), "glasses": shown,
            "reference": None if ref is None else {k: ref[k] for k in ("name", "nd", "vd", "pgf")}
            | {"dPgF": ref["dpgf"]},
            "rules": {r: {"title": kb_rules[r]["title"], "notes": kb_rules[r]["notes"],
                          "source": kb_rules[r]["source"],
                          **({"assumption": kb_rules[r]["assumption"]}
                             if "assumption" in kb_rules[r] else {})} for r in names},
            "no_transmission_data": no_data,
            "catalog_glasses": len(table)}


def resolve_glass(system_path, flt: dict) -> str:
    """Erstes Glas, das den Filter erfüllt (für Glasangaben ``{"filter": …}`` in Aktionen);
    QueryError ``precondition``, wenn der Filter leer ausgeht (wird nicht aufgeweicht)."""
    flt = {k: v for k, v in flt.items() if k != "max_results"}
    res = glass_query(system_path, flt, limit=1)
    if not res["glasses"]:
        raise QueryError("precondition", f"Glasfilter {flt} ergibt kein Glas (leer)")
    return res["glasses"][0]["name"]
