"""Linsenmodule und Vorlagen für ``insert_module``, ``replace_group`` und ``start_from_skeleton``
(M10-T4; docs/entwurfssteuerung.md, Abschnitte 4.5, 5.1 und 5.2).

Seit M10-T5 stehen die Module in der Wissensbasis (``data/design/knowledge/modules.json``,
``knowledge.py``) mit Wirkungsvektor, Vorbedingungen, Glasregel und Quelle. Diese Datei bietet
die Sicht der Operatoren darauf: pro Modul Gruppe, Vorlage (``template`` = Art der Vorlage),
Quelle und Beschreibung. Keine Linsendaten realer Objektive (AGENTS.md, Abschnitt 8): Die
Vorlagen konstruieren die Geometrie aus den lokalen Strahldaten des Kandidaten.

Vorlagen, die ``insert_module`` baut:

- ``aplanatic_shell``: aplanatisch-konzentrische Schale aus dem realen Randstrahl
  (``mikroptim insert --aplanatic-shell``); ändert die sphärische Aberration aller Ordnungen der
  Gruppe nicht, macht das Bündel um n steiler.
- ``null_meniscus``: Meniskus ohne Brechkraft, Dicke 0 (Operator A).
- ``null_doublet``: Kittglied ohne Wirkung, drei Flächen gleicher Krümmung, Dicke 0.
- ``strong_doublet``: wie ``null_doublet``, Krümmung so, dass der paraxiale Randstrahl die
  Kittfläche unter ``incidence_deg`` (Standard 40°, Bereich 35–50°) trifft (M10-T5).
- ``gauss_pair``: zwei Menisken ohne Brechkraft mit einander zugewandten Hohlseiten
  (konzentrisch zur Mitte zwischen ihnen).
- ``petzval_group``: sammelndes und zerstreuendes Glied mit Dicke, zusammen ungefähr afokal
  (Dünnlinsennäherung), ändert die Merit (M10-T5).

Nur im Gerüst bzw. zum Nachschlagen: ``lens``, ``thick_meniscus``, ``amici_front``,
``cemented_triplet``.
"""

from __future__ import annotations

from . import knowledge

TEMPLATES = knowledge.INSERT_TEMPLATES + ("lens",)

#: Vorlagen mit Merit-Invarianz beim Einfügen (Dicke 0)
NULL_TEMPLATES = knowledge.NULL_TEMPLATES


def _view(m: dict) -> dict:
    return {"group": m["group"], "template": m["template"]["kind"],
            "template_params": {k: v for k, v in m["template"].items() if k != "kind"},
            "source": m["source"], "notes": m["notes"], "title": m["title"],
            "glass_rule": m.get("glass_rule"), "insertable": knowledge.insertable(m),
            "skeleton": knowledge.skeleton_capable(m)}


#: Modul-ID → Sicht der Operatoren (aus der Wissensbasis)
MODULES = {mid: _view(m) for mid, m in knowledge.modules().items()}


def module(module_id: str) -> dict:
    """Eintrag eines Moduls; KeyError mit der Liste der bekannten Module."""
    if module_id not in MODULES:
        raise KeyError(f"unbekanntes Modul {module_id!r} (bekannt: {sorted(MODULES)})")
    return MODULES[module_id]
