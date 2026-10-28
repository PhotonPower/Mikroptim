"""Farb- und Feldklassen nach Zhang (Dissertation Jena 2019, Abschnitt 4.2.1, Tab. 4.1, 4.2,
Abb. 4.2) aus Größen, die Mikroptim berechnet (M10-T2). Reine Funktionen ohne Aufruf der CLI.

Schärfentiefe DoF = n λ_d / NA² (Zhang Gl. 2.16, gesamte Tiefe; NA 0,8 trocken: 0,918 µm wie
in Zhangs Anhang IV), n = Brechzahl im Objektraum.

Farbklassen über den paraxialen chromatischen Fokusverlauf (Spanne der Foki der Linien):

- (a) Achromat: |F − C| ≤ 2 DoF und sekundäres Spektrum |d − (F + C)/2| ≤ 1,5 DoF
- (b) Fluorit (Halbapochromat): g … C innerhalb 2,5 DoF (drei Farben wie beim Apochromaten,
  Toleranz 2,5 DoF; mit F statt g wäre jeder Achromat schon ein Fluorit)
- (c) Apochromat: g … C innerhalb DoF
- (d) verbesserter VIS-Apochromat: g … A′ innerhalb DoF
- (e) IR-Superapochromat: g … t innerhalb DoF
- (f) UV-Superapochromat: i … C innerhalb DoF
- (g) UV-IR-Superapochromat: i … t innerhalb DoF

Die Klasse ist die höchste erfüllte in der Reihenfolge g > e, f > d > c > b > a; ``satisfied``
nennt alle erfüllten. Linien, die ein Glas nicht abdeckt (NaN), machen die Klasse unbestimmbar.
Der Fokusverlauf ist paraxial: Sphärochromasie und Zonen gehen nicht ein (Zhang bewertet die
Längsaberration über die Pupille; Näherung).

Feldklassen (Tab. 4.2) über die Bildfeldwölbung in Seidel-Näherung, den RMS am Feldrand und
versagte Strahlen als Ersatz für Vignettierung:

- Plan: Fokusabweichung der besten Bildschale am Feldrand ≤ 2,5 DoF; nicht plan → 1
- NA < 0,5 → 2
- NA ≥ 0,5, Feldrand nicht beugungsbegrenzt (RMS > Ziel) → 3
- NA ≥ 0,5, Etendue < 0,9503 mm²: mit versagten Strahlen am Feldrand → 4, ohne → 5
- Etendue ≥ 0,9503 mm²: mit versagten Strahlen → 6, ohne → 7
"""

from __future__ import annotations

import math

COLOR_ORDER = ("g", "e", "f", "d", "c", "b", "a")
COLOR_SPANS = {
    "c": ("g", "C"), "d": ("g", "A'"), "e": ("g", "t"), "f": ("i", "C"), "g": ("i", "t"),
}
LINE_ORDER = ("i", "h", "g", "F", "e", "d", "C", "A'", "s", "t")
HIGH_NA = 0.5
HIGH_ETENDUE = 0.9503
RMS_TARGET = 0.0714
LAMBDA_D_MM = 587.56e-6


def depth_of_focus(na: float, n_object: float = 1.0, lambda_mm: float = LAMBDA_D_MM) -> float:
    """DoF = n λ / NA² [mm]."""
    return n_object * lambda_mm / (na * na)


def _span(shifts: dict, first: str, last: str):
    names = LINE_ORDER[LINE_ORDER.index(first): LINE_ORDER.index(last) + 1]
    vals = [shifts.get(k) for k in names]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return None
    return max(vals) - min(vals)


def color_class(shifts: dict, dof: float) -> dict:
    """Farbklasse aus den Fokusverschiebungen pro Linie [mm] (Name → Wert)."""
    satisfied, spans = [], {}
    for cls, (first, last) in COLOR_SPANS.items():
        sp = _span(shifts, first, last)
        spans[cls] = sp
        if sp is not None and sp <= dof:
            satisfied.append(cls)
    spans["b"] = _span(shifts, "g", "C")
    if spans["b"] is not None and spans["b"] <= 2.5 * dof:
        satisfied.append("b")
    f, d, c = (shifts.get(k) for k in ("F", "d", "C"))
    if _span(shifts, "F", "C") is not None:
        secondary = abs(d - 0.5 * (f + c))
        spans["a"] = abs(f - c)
        if abs(f - c) <= 2.0 * dof and secondary <= 1.5 * dof:
            satisfied.append("a")
    best = next((k for k in COLOR_ORDER if k in satisfied), "none")
    return {"color": best, "satisfied": sorted(satisfied),
            "spans_in_dof": {k: (None if v is None else v / dof) for k, v in spans.items()}}


def best_focus_field_shift(S_III: float, S_IV: float, n_image: float, u_image: float) -> float:
    """Längsabweichung der besten (medialen) Bildschale am Feldrand in Seidel-Näherung [mm]:
    W020_eff = W220 + W222/2 = (2 S_III + S_IV)/4, Δz = 2 W020_eff / (n′ u′²)."""
    return (2.0 * S_III + S_IV) / (2.0 * n_image * u_image * u_image)


def field_class(*, field_shift: float, dof: float, na: float, etendue: float | None,
                edge_rms: float | None, edge_failed: int, rms_target: float = RMS_TARGET) -> dict:
    plan = abs(field_shift) <= 2.5 * dof
    corrected = edge_rms is not None and edge_rms <= rms_target
    if not plan:
        cls = 1
    elif na < HIGH_NA:
        cls = 2
    elif not corrected:
        cls = 3
    elif etendue is not None and etendue >= HIGH_ETENDUE:
        cls = 6 if edge_failed else 7
    else:
        cls = 4 if edge_failed else 5
    return {"field": cls, "plan": plan, "field_shift_in_dof": field_shift / dof,
            "edge_rms_waves": edge_rms, "edge_failed_rays": edge_failed,
            "approximation": "Seidel (Bildfeldwölbung), Vignettierung über versagte Strahlen"}
