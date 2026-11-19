"""Linsenmodule und Vorlagen für ``insert_module``, ``replace_group`` und ``start_from_skeleton``
(M10-T4; docs/entwurfssteuerung.md, Abschnitte 4.5, 5.1 und 5.2).

Nur das, was die Operatoren brauchen: pro Modul die Gruppe, die Vorlage (Startgeometrie) und eine
kurze Beschreibung in eigenen Worten mit Quellenangabe. Die Wissensbasis mit Wirkungsvektoren,
Vorbedingungen und Glasregeln folgt in M10-T5 (``data/design/knowledge/``). Keine Linsendaten
realer Objektive (AGENTS.md, Abschnitt 8): Die Vorlagen konstruieren die Geometrie aus den
lokalen Strahldaten des Kandidaten.

Vorlagen (``template``):

- ``aplanatic_shell``: aplanatisch-konzentrische Schale aus dem realen Randstrahl
  (``mikroptim insert --aplanatic-shell``); ändert die sphärische Aberration aller Ordnungen der
  Gruppe nicht, macht das Bündel um n steiler.
- ``null_meniscus``: Meniskus ohne Brechkraft, Dicke 0 (Operator A).
- ``null_doublet``: Kittglied ohne Wirkung, drei Flächen gleicher Krümmung, Dicke 0.
- ``gauss_pair``: zwei Menisken ohne Brechkraft mit einander zugewandten Hohlseiten
  (konzentrisch zur Mitte zwischen ihnen).
- ``lens``: Einzellinse (nur in ``start_from_skeleton``; Brechkraft aus der Verteilung).

Modul-IDs: ``generic.<vorlage>`` für jede Vorlage und einige Module nach Zhang (Dissertation Jena
2019), die sich mit diesen Vorlagen beginnen lassen.
"""

from __future__ import annotations

TEMPLATES = ("aplanatic_shell", "null_meniscus", "null_doublet", "gauss_pair", "lens")

#: Vorlagen mit Merit-Invarianz beim Einfügen (Dicke 0)
NULL_TEMPLATES = ("null_meniscus", "null_doublet", "gauss_pair")

MODULES = {
    "generic.aplanatic_shell": {
        "group": None, "template": "aplanatic_shell",
        "source": "Welford, Aberrations of Optical Systems, aplanatische Flächen; Zhang 2019, §3.2",
        "notes": "erste Fläche aplanatisch zum Bildpunkt des realen Randstrahls, zweite "
                 "konzentrisch: erhöht die Apertur um n ohne sphärische Aberration"},
    "generic.null_meniscus": {
        "group": None, "template": "null_meniscus",
        "source": "Bociort, Nullelemente (docs/optimierung.md, Abschnitt 6)",
        "notes": "Meniskus ohne Brechkraft und Dicke; wächst über die Dicken-Homotopie"},
    "generic.null_doublet": {
        "group": None, "template": "null_doublet",
        "source": "docs/optimierung.md, Abschnitt 13",
        "notes": "Kittglied ohne Wirkung (zwei Gläser, Dicke 0); Ausgangspunkt für Farbkorrektur"},
    "generic.gauss_pair": {
        "group": None, "template": "gauss_pair",
        "source": "Zhang 2019, Tab. 6.2 (Typ d), Abb. 6.13b",
        "notes": "zwei Menisken ohne Brechkraft, Hohlseiten einander zugewandt"},
    "generic.lens": {
        "group": None, "template": "lens",
        "source": "-", "notes": "Einzellinse (Gerüst)"},
    "front.shell_ca": {
        "group": "front", "template": "aplanatic_shell",
        "source": "Zhang 2019, Tab. 6.4 Nr. 3, Abb. 6.7",
        "notes": "Schalenlinse hinter der Frontlinse: teilt die Brechkraft, ohne sphärische "
                 "Aberration hinzuzufügen"},
    "middle.merte_doublet": {
        "group": "middle", "template": "null_doublet",
        "source": "Zhang 2019, Tab. 6.6 Nr. 3, Tab. 6.7",
        "notes": "Kittglied mit stark gekrümmter Kittfläche und kleinem Brechzahlsprung (Start "
                 "als Null-Kittglied, die Kittfläche krümmt der Optimierer)"},
    "rear.gauss_pair": {
        "group": "rear", "template": "gauss_pair",
        "source": "Zhang 2019, Tab. 6.8 Nr. 2, Abb. 6.13b",
        "notes": "zwei dicke Menisken, quasisymmetrisch; korrigiert Koma, Astigmatismus und "
                 "Bildfeldwölbung"},
}


def module(module_id: str) -> dict:
    """Eintrag eines Moduls; KeyError mit der Liste der bekannten Module."""
    if module_id not in MODULES:
        raise KeyError(f"unbekanntes Modul {module_id!r} (bekannt: {sorted(MODULES)})")
    return MODULES[module_id]
