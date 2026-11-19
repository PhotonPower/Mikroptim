"""Wissensbasis der Entwurfssteuerung (M10-T5; docs/entwurfssteuerung.md, Abschnitt 5):
Linsenmodule, Brechkraftverteilungen und Glasregeln als JSON (``mikroptim.design.knowledge/1``,
Schema ``data/design/schemas/design_knowledge.schema.json``), in eigenen Worten mit Quellenangabe.

Die Dateien liegen im Repository unter ``data/design/knowledge/``; der CMake-Build und das Wheel
kopieren sie nach ``mikroptim/design/knowledge/``. Gesucht wird in dieser Reihenfolge: Umgebungs-
variable ``MIKROPTIM_KNOWLEDGE_DIR``, Paketverzeichnis, ``data/design/knowledge`` oberhalb des
Pakets (Entwicklung im Quellbaum).

Glasregeln sind Bedingungen über den Kennzahlen eines Glases (``nd``, ``vd``, ``pgf``,
``dpgf``) und, bei Paarregeln, über den Beträgen der Differenzen zum Bezugsglas
(``abs_delta_nd`` …); geschlossene Bereiche ``[min, max]`` (``null`` offen), verknüpft mit
``all``, ``any``, ``not`` und Verweisen ``rule``. Ausgewertet von ``glass_query``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pathlib

FORMAT = "mikroptim.design.knowledge/1"
FILES = ("modules.json", "power_distributions.json", "glass_rules.json")
ABERRATIONS = ("SA", "ZSA", "coma", "AST", "FC", "ACA", "LCA", "SPHCRM")
#: Vorlagen, die insert_module baut (operators.py)
INSERT_TEMPLATES = ("aplanatic_shell", "null_meniscus", "null_doublet", "gauss_pair",
                    "strong_doublet", "petzval_group")
#: Vorlagen mit Merit-Invarianz beim Einfügen (Dicke 0)
NULL_TEMPLATES = ("null_meniscus", "null_doublet", "gauss_pair", "strong_doublet")
#: Elemente einer Vorlage im Gerüst (start_from_skeleton): Linsen bzw. Kittglieder
SKELETON_ELEMENTS = {
    "lens": ("lens",), "aplanatic_shell": ("lens",), "null_meniscus": ("lens",),
    "thick_meniscus": ("lens",), "amici_front": ("lens",),
    "null_doublet": ("doublet",), "strong_doublet": ("doublet",),
    "gauss_pair": ("lens", "lens"), "petzval_group": ("lens", "lens"),
}
GLASS_KEYS = ("nd", "vd", "pgf", "dpgf")
PAIR_KEYS = ("abs_delta_nd", "abs_delta_vd", "abs_delta_pgf", "abs_delta_dpgf")


class KnowledgeError(RuntimeError):
    """Wissensbasis fehlt oder ist inkonsistent."""


def knowledge_dir() -> pathlib.Path:
    env = os.environ.get("MIKROPTIM_KNOWLEDGE_DIR")
    candidates = [pathlib.Path(env)] if env else []
    here = pathlib.Path(__file__).resolve().parent
    candidates.append(here / "knowledge")
    candidates += [p / "data" / "design" / "knowledge" for p in here.parents]
    for c in candidates:
        if all((c / f).is_file() for f in FILES):
            return c
    raise KnowledgeError("Wissensbasis nicht gefunden (MIKROPTIM_KNOWLEDGE_DIR, "
                         f"{here / 'knowledge'} oder data/design/knowledge)")


@functools.lru_cache(maxsize=None)
def _load(directory: str) -> dict:
    d = pathlib.Path(directory)
    raw = {f: (d / f).read_bytes() for f in FILES}
    data = {f: json.loads(raw[f].decode("utf-8")) for f in FILES}
    for f, doc in data.items():
        if doc.get("format") != FORMAT:
            raise KnowledgeError(f"{f}: format {FORMAT} erwartet")
    kb = {
        "dir": str(d),
        "digest": hashlib.sha256(b"".join(raw[f] for f in FILES)).hexdigest(),
        "modules": {m["id"]: m for m in data["modules.json"]["modules"]},
        "distributions": {x["id"]: x for x in data["power_distributions.json"]["distributions"]},
        "sign_weights": data["power_distributions.json"]["sign_weights"],
        "rules": {r["id"]: r for r in data["glass_rules.json"]["rules"]},
        "references": {f: doc.get("references", {}) for f, doc in data.items()},
    }
    _check(kb, data)
    return kb


def _check(kb: dict, data: dict) -> None:
    """Konsistenz, die das Schema nicht ausdrückt: eindeutige IDs, Verweise, Quellen."""
    for f, key in (("modules.json", "modules"), ("power_distributions.json", "distributions"),
                   ("glass_rules.json", "rules")):
        ids = [e["id"] for e in data[f][key]]
        if len(ids) != len(set(ids)):
            raise KnowledgeError(f"{f}: doppelte IDs")
        for e in data[f][key]:
            if not str(e.get("source", "")).strip():
                raise KnowledgeError(f"{f}: {e['id']} ohne Quelle")
    for m in kb["modules"].values():
        if m.get("glass_rule") and m["glass_rule"] not in kb["rules"]:
            raise KnowledgeError(f"Modul {m['id']}: unbekannte Glasregel {m['glass_rule']}")
        bad = set(m["effect"]) - set(ABERRATIONS)
        if bad:
            raise KnowledgeError(f"Modul {m['id']}: unbekannte Aberrationen {sorted(bad)}")
    for r in kb["rules"].values():
        _check_condition(r["condition"], kb["rules"], r["id"], set())
        if bool(r.get("needs_partner")) != _uses_partner(r["condition"], kb["rules"]):
            raise KnowledgeError(f"Regel {r['id']}: needs_partner passt nicht zur Bedingung")


def _check_condition(cond, rules, rid, seen):
    if not isinstance(cond, dict) or not cond:
        raise KnowledgeError(f"Regel {rid}: leere Bedingung")
    for k, v in cond.items():
        if k in GLASS_KEYS or k in PAIR_KEYS:
            if not (isinstance(v, list) and len(v) == 2):
                raise KnowledgeError(f"Regel {rid}: {k} erwartet [min, max]")
        elif k in ("all", "any"):
            for c in v:
                _check_condition(c, rules, rid, seen)
        elif k == "not":
            _check_condition(v, rules, rid, seen)
        elif k == "rule":
            if v not in rules or v in seen | {rid}:
                raise KnowledgeError(f"Regel {rid}: Verweis auf {v} unbekannt oder zyklisch")
            _check_condition(rules[v]["condition"], rules, v, seen | {rid})
        else:
            raise KnowledgeError(f"Regel {rid}: unbekannter Schlüssel {k}")


def _uses_partner(cond, rules) -> bool:
    for k, v in cond.items():
        if k in PAIR_KEYS:
            return True
        if k in ("all", "any") and any(_uses_partner(c, rules) for c in v):
            return True
        if k == "not" and _uses_partner(v, rules):
            return True
        if k == "rule" and _uses_partner(rules[v]["condition"], rules):
            return True
    return False


def load() -> dict:
    """Die geladene Wissensbasis (zwischengespeichert)."""
    return _load(str(knowledge_dir()))


def digest() -> str:
    """SHA-256 der drei Dateien (für das Protokoll der Entscheider)."""
    return load()["digest"]


def modules() -> dict:
    return load()["modules"]


def distributions() -> dict:
    return load()["distributions"]


def rules() -> dict:
    return load()["rules"]


def sign_weights(distribution: str) -> dict:
    kb = load()
    return {**kb["sign_weights"], **kb["distributions"][distribution].get("weights", {})}


def insertable(module: dict) -> bool:
    return module["template"]["kind"] in INSERT_TEMPLATES


def skeleton_capable(module: dict) -> bool:
    return module["template"]["kind"] in SKELETON_ELEMENTS


# --- Glasregeln -------------------------------------------------------------------------------


def _in(rng, x) -> bool:
    lo, hi = rng
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def pair_values(glass: dict, ref: dict) -> dict:
    """Beträge der Differenzen zum Bezugsglas (Kennzahlen wie ``glass_query``)."""
    return {"abs_delta_" + k: abs(glass[k] - ref[k]) for k in GLASS_KEYS}


def evaluate(cond: dict, glass: dict, ref: dict | None = None) -> bool:
    """Bedingung ``cond`` für ``glass`` (dict mit nd, vd, pgf, dpgf); ``ref`` ist das Bezugsglas
    der Paarregeln (sonst KnowledgeError, wenn die Bedingung eines braucht)."""
    pair = None
    for k, v in cond.items():
        if k in GLASS_KEYS:
            ok = _in(v, glass[k])
        elif k in PAIR_KEYS:
            if ref is None:
                raise KnowledgeError("Paarregel ohne Bezugsglas")
            pair = pair or pair_values(glass, ref)
            ok = _in(v, pair[k])
        elif k == "all":
            ok = all(evaluate(c, glass, ref) for c in v)
        elif k == "any":
            ok = any(evaluate(c, glass, ref) for c in v)
        elif k == "not":
            ok = not evaluate(v, glass, ref)
        elif k == "rule":
            ok = evaluate(rules()[v]["condition"], glass, ref)
        else:
            raise KnowledgeError(f"unbekannter Schlüssel {k}")
        if not ok:
            return False
    return True


# --- Module nachschlagen ----------------------------------------------------------------------


def module_lookup(group=None, corrects=None, zone=None, na=None, immersion=None,
                  insertable_only=False) -> list[dict]:
    """Module nach Gruppe (``generic`` mit ``group`` null passt zu jeder Gruppe), korrigierbarer
    Aberration (Wirkung ≠ 0), Vorbedingungen (Etendue-Zone, NA, Immersion) und Einfügbarkeit.
    Reihenfolge: größte Summe |Wirkung| über ``corrects``, dann ID."""
    corrects = list(corrects or [])
    bad = [c for c in corrects if c not in ABERRATIONS]
    if bad:
        raise KnowledgeError(f"unbekannte Aberrationen {bad} (bekannt: {list(ABERRATIONS)})")
    out = []
    for m in modules().values():
        if group is not None and m["group"] not in (None, group):
            continue
        if corrects and not any(m["effect"].get(c) for c in corrects):
            continue
        req = m.get("requires", {})
        if zone is not None and "zones" in req and int(zone) not in req["zones"]:
            continue
        if na is not None and ("na_min" in req and na < req["na_min"]
                               or "na_max" in req and na > req["na_max"]):
            continue
        if immersion is not None and "immersion" in req and bool(immersion) != req["immersion"]:
            continue
        if insertable_only and not insertable(m):
            continue
        out.append(m)
    out.sort(key=lambda m: (-sum(abs(m["effect"].get(c, 0)) for c in corrects), m["id"]))
    return [{**m, "insertable": insertable(m)} for m in out]


# --- Kurzfassung für den System-Prompt --------------------------------------------------------


def _effect_text(effect: dict) -> str:
    return ", ".join(f"{k}{v:+d}" for k, v in effect.items()) or "neutral"


def prompt_summary() -> str:
    """Kurzfassung der Wissensbasis für den System-Prompt (Module, Verteilungen, Glasregeln);
    Einzelheiten holt der Entscheider mit module_lookup und glass_query."""
    lines = ["Wissensbasis (Kurzfassung; Einzelheiten mit module_lookup und glass_query):",
             "Module (Gruppe, Wirkung −2…+2; * = mit insert_module einfügbar):"]
    for m in modules().values():
        if m["template"]["kind"] == "lens":
            continue
        star = "*" if insertable(m) else " "
        lines.append(f"{star} {m['id']} ({m['group'] or 'jede'}): {_effect_text(m['effect'])} – "
                     f"{m['title']}")
    lines.append("Brechkraftverteilungen (front/middle/rear von der Objektseite): " + "; ".join(
        f"{d['id']} {d['groups']['front']}/{d['groups']['middle']}/{d['groups']['rear']}"
        for d in distributions().values()))
    lines.append("Glasregeln (glass_query filter.rule): " + "; ".join(
        f"{r['id']}{' (Paar)' if r.get('needs_partner') else ''}" for r in rules().values()))
    return "\n".join(lines) + "\n"
