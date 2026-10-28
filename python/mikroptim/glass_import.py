"""Import von Glaskatalogen nach ``mikroptim.glass/1`` (M9-T3).

Quellen:

- Zemax-AGF-Dateien der Glashersteller (Schott, Ohara, Hoya, CDGM, Sumita, Hikari, ...),
- YAML-Dateien der Datenbank refractiveindex.info (CC0; braucht PyYAML).

Mikroptim kennt seit M9-T4 die Modelle ``sellmeier`` (bis 8 Terme mit Konstante) und
``polynomial`` (n² = Σ a λ^p; ADR-0014). Formeln, die sich darin darstellen lassen (alle
Sellmeier- und Polynomformeln der Kataloge), werden exakt übernommen und gegen die
Originalformel nachgerechnet; alle übrigen (Herzberger, Conrady, Handbook, Tabellen,
Umrechnung auf Luft) werden mit drei Sellmeier-Termen angenähert und nur übernommen, wenn die
größte Abweichung der Brechzahl unter der Toleranz bleibt (Standard 5e-6). Jedes Glas mit
Katalogwert n_d wird zusätzlich gegen diesen geprüft; eine große Abweichung deutet auf eine
falsch gelesene Formel und führt zum Verwerfen.

Glasdateien gehören nie ins Repository (data/glass/README.md); das Ergebnis wird z. B. nach
``data/glass/private/`` geschrieben oder von der GUI im Arbeitsbereich abgelegt.

Beispiel::

    from mikroptim import glass_import as gi
    result = gi.convert_files(["SCHOTT.AGF"])
    gi.write_catalog(result, "data/glass/private/schott.json")
    print(gi.format_report(result))

Kommandozeile: ``python -m mikroptim.glass_import SCHOTT.AGF -o data/glass/private/schott.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
from typing import Callable, Iterable, Sequence

import numpy as np

FORMAT = "mikroptim.glass/1"

#: Standardtoleranz für die Abweichung der Brechzahl (Fit gegen Originalformel).
DEFAULT_TOLERANCE = 5e-6
#: Abweichung vom Katalogwert n_d, ab der ein Glas verworfen wird (Formel vermutlich falsch).
ND_REJECT = 1e-4
#: Abweichung vom Katalogwert n_d, ab der gewarnt wird (Rundung der Kataloge ca. 1e-5).
ND_WARN = 2e-5
#: Fitbereiche [µm], nacheinander versucht, jeweils geschnitten mit dem Gültigkeitsbereich.
FIT_RANGES_UM = ((0.0, math.inf), (0.365, 1.1), (0.4, 0.8))
#: Bevorzugte Probendicke der Reintransmission [mm].
PREFERRED_THICKNESS_MM = 10.0

LAMBDA_D = 0.5875618
LAMBDA_F = 0.4861327
LAMBDA_C = 0.6562725
LAMBDA_G = 0.4358343

AGF_STATUS = {0: "standard", 1: "preferred", 2: "obsolete", 3: "special", 4: "melt"}

AGF_FORMULAS = {
    1: "Schott",
    2: "Sellmeier 1",
    3: "Herzberger",
    4: "Sellmeier 2",
    5: "Conrady",
    6: "Sellmeier 3",
    7: "Handbook of Optics 1",
    8: "Handbook of Optics 2",
    9: "Sellmeier 4",
    10: "Extended",
    11: "Sellmeier 5",
    12: "Extended 2",
    13: "Extended 3",
}


class GlassImportError(Exception):
    """Fehler beim Lesen einer Quelldatei (Format, nicht unterstützte Formel)."""


# --------------------------------------------------------------------------------------------
# Datenmodell
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class SourceGlass:
    """Ein Glas, wie es in der Quelle steht (λ in µm, Brechzahl relativ zu Luft)."""

    name: str
    origin: str  # Datei (und Zeile) für Meldungen
    formula: str  # lesbarer Name der Formel
    index: Callable[[np.ndarray], np.ndarray]  # n(λ_µm), relativ zu Luft
    range_um: tuple[float, float]
    #: exakte Darstellung als Teil eines Katalogeintrags (Modell sellmeier oder polynomial)
    exact: dict | None = None
    #: exakte Darstellung mit höchstens drei Sellmeier-Termen ohne Konstante (ältere Versionen)
    exact3: dict | None = None
    nd: float | None = None
    vd: float | None = None
    status: str | None = None
    transmission: dict | None = None  # {"thickness_mm", "nm", "tau"}
    notes: list[str] = dataclasses.field(default_factory=list)
    #: Stützstellen λ_µm einer Tabelle: Fit und Prüfung dort statt auf der Interpolation
    samples: np.ndarray | None = None


@dataclasses.dataclass
class ConvertedGlass:
    source: SourceGlass
    entry: dict | None  # Eintrag für mikroptim.glass/1, None wenn verworfen
    method: str  # "exakt", "Fit" oder "verworfen"
    max_dn: float | None = None  # größte Abweichung gegen die Originalformel
    nd_error: float | None = None  # n_d (Ergebnis) − n_d (Katalog)
    reason: str | None = None  # bei verworfen
    warnings: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ImportResult:
    glasses: list[ConvertedGlass]
    files: list[str]
    warnings: list[str]
    tolerance: float

    @property
    def accepted(self) -> list[ConvertedGlass]:
        return [g for g in self.glasses if g.entry is not None]

    @property
    def rejected(self) -> list[ConvertedGlass]:
        return [g for g in self.glasses if g.entry is None]


# --------------------------------------------------------------------------------------------
# Dispersionsformeln (λ in µm)
# --------------------------------------------------------------------------------------------


def _sqrt_n2(n2: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sqrt(n2)


def _sellmeier_terms(lam2: np.ndarray, pairs: Sequence[tuple[float, float]]) -> np.ndarray:
    out = np.zeros_like(lam2)
    for k, l in pairs:
        out = out + k * lam2 / (lam2 - l)
    return out


def agf_index_function(formula: int, cd: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    """Brechzahlfunktion n(λ_µm) einer AGF-Formel (Nummerierung des Zemax-Formats)."""
    c = list(cd) + [0.0] * 12

    def f(lam):
        lam = np.asarray(lam, dtype=float)
        l2 = lam * lam
        if formula == 1:  # Schott
            n2 = c[0] + c[1] * l2 + c[2] / l2 + c[3] / l2**2 + c[4] / l2**3 + c[5] / l2**4
            return _sqrt_n2(n2)
        if formula == 2:  # Sellmeier 1
            return _sqrt_n2(1.0 + _sellmeier_terms(l2, [(c[0], c[1]), (c[2], c[3]), (c[4], c[5])]))
        if formula == 3:  # Herzberger
            L = 1.0 / (l2 - 0.028)
            return c[0] + c[1] * L + c[2] * L * L + c[3] * l2 + c[4] * l2**2 + c[5] * l2**3
        if formula == 4:  # Sellmeier 2
            n2 = 1.0 + c[0] + c[1] * l2 / (l2 - c[2] ** 2) + c[3] / (l2 - c[4] ** 2)
            return _sqrt_n2(n2)
        if formula == 5:  # Conrady
            return c[0] + c[1] / lam + c[2] / lam**3.5
        if formula == 6:  # Sellmeier 3
            pairs = [(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])]
            return _sqrt_n2(1.0 + _sellmeier_terms(l2, pairs))
        if formula == 7:  # Handbook of Optics 1
            return _sqrt_n2(c[0] + c[1] / (l2 - c[2]) - c[3] * l2)
        if formula == 8:  # Handbook of Optics 2
            return _sqrt_n2(c[0] + c[1] * l2 / (l2 - c[2]) - c[3] * l2)
        if formula == 9:  # Sellmeier 4
            return _sqrt_n2(c[0] + c[1] * l2 / (l2 - c[2]) + c[3] * l2 / (l2 - c[4]))
        if formula == 10:  # Extended
            n2 = (c[0] + c[1] * l2 + c[2] / l2 + c[3] / l2**2 + c[4] / l2**3 + c[5] / l2**4
                  + c[6] / l2**5 + c[7] / l2**6)
            return _sqrt_n2(n2)
        if formula == 11:  # Sellmeier 5
            pairs = [(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7]), (c[8], c[9])]
            return _sqrt_n2(1.0 + _sellmeier_terms(l2, pairs))
        if formula == 12:  # Extended 2
            n2 = (c[0] + c[1] * l2 + c[2] / l2 + c[3] / l2**2 + c[4] / l2**3 + c[5] / l2**4
                  + c[6] * l2**2 + c[7] * l2**3)
            return _sqrt_n2(n2)
        if formula == 13:  # Extended 3
            n2 = (c[0] + c[1] * l2 + c[2] * l2**2 + c[3] / l2 + c[4] / l2**2 + c[5] / l2**3
                  + c[6] / l2**4 + c[7] / l2**5 + c[8] / l2**6)
            return _sqrt_n2(n2)
        raise GlassImportError(f"AGF-Formel {formula} wird nicht unterstützt")

    if formula not in AGF_FORMULAS:
        raise GlassImportError(f"unbekannte AGF-Formel {formula}")
    return f


MAX_SELLMEIER_TERMS = 8  # kMaxSellmeierTerms (include/mikroptim/glass/catalog.hpp)
MAX_POLYNOMIAL_TERMS = 12  # kMaxPolynomialTerms


def _sellmeier_model(pairs: Sequence[tuple[float, float]], constant: float = 0.0):
    """Modell ``sellmeier`` (n² = 1 + A + Σ B λ²/(λ² − C)); None bei zu vielen Termen."""
    used = [(float(k), float(l)) for k, l in pairs if k != 0.0]
    if not used:
        used = [(0.0, 0.0)]
    if len(used) > MAX_SELLMEIER_TERMS:
        return None
    model = {"model": "sellmeier"}
    if constant != 0.0:
        model["A"] = float(constant)
    model["B"] = [k for k, _ in used]
    model["C"] = [l for _, l in used]
    return model


def _sellmeier3_model(model: dict | None):
    """Dasselbe Modell mit genau drei Termen ohne Konstante, sonst None (ältere Versionen)."""
    if model is None or model["model"] != "sellmeier" or model.get("A", 0.0) != 0.0:
        return None
    if len(model["B"]) > 3:
        return None
    pad = 3 - len(model["B"])
    return {"model": "sellmeier", "B": model["B"] + [0.0] * pad, "C": model["C"] + [0.0] * pad}


def _polynomial_model(terms: Sequence[tuple[float, float]]):
    """Modell ``polynomial`` (n² = Σ a λ^p); None bei zu vielen Termen."""
    used = [[float(a), float(p)] for a, p in terms if a != 0.0]
    if not used:
        return None
    if len(used) > MAX_POLYNOMIAL_TERMS:
        return None
    return {"model": "polynomial", "terms": used}


def model_index(model: dict, lam_um) -> np.ndarray:
    """Brechzahl eines Katalogeintrags (Modelle ``sellmeier`` und ``polynomial``), λ in µm."""
    lam = np.asarray(lam_um, float)
    if model["model"] == "polynomial":
        n2 = np.zeros_like(lam)
        for a, p in model["terms"]:
            n2 = n2 + a * lam**p
        return _sqrt_n2(n2)
    l2 = lam * lam
    n2 = 1.0 + model.get("A", 0.0) + sum(b * l2 / (l2 - c) for b, c in zip(model["B"], model["C"]))
    return _sqrt_n2(n2)


def _model_valid(model: dict, a: float, b: float) -> bool:
    """Wie der Leser von Mikroptim: kein Pol im Bereich, gültige Brechzahl auf 201 Punkten."""
    if model["model"] == "sellmeier" and not _poles_ok(model["C"], a, b):
        return False
    n = model_index(model, np.linspace(a, b, 201))
    return bool(np.all(np.isfinite(n)) and np.all(n > 0.0))


# Exponenten der Polynomformeln des AGF-Formats
_AGF_POLY_EXPONENTS = {
    1: [0, 2, -2, -4, -6, -8],
    10: [0, 2, -2, -4, -6, -8, -10, -12],
    12: [0, 2, -2, -4, -6, -8, 4, 6],
    13: [0, 2, 4, -2, -4, -6, -8, -10, -12],
}


def agf_exact_model(formula: int, cd: Sequence[float]) -> dict | None:
    """Exakte Darstellung einer AGF-Formel in den Modellen von Mikroptim, sonst None."""
    c = list(cd) + [0.0] * 12
    if formula in _AGF_POLY_EXPONENTS:
        exps = _AGF_POLY_EXPONENTS[formula]
        return _polynomial_model(list(zip(c[: len(exps)], exps)))
    if formula in (2, 6, 11):
        terms = {2: 3, 6: 4, 11: 5}[formula]
        return _sellmeier_model([(c[2 * i], c[2 * i + 1]) for i in range(terms)])
    if formula == 4:  # n² − 1 = A + B1 λ²/(λ² − λ1²) + B2/(λ² − λ2²)
        # B2/(λ² − c) = (B2/c)·λ²/(λ² − c) − B2/c
        c2 = c[4] ** 2
        if c2 == 0.0:
            return None
        return _sellmeier_model([(c[1], c[2] ** 2), (c[3] / c2, c2)], c[0] - c[3] / c2)
    if formula == 9:  # n² = A + B λ²/(λ² − C) + D λ²/(λ² − E)
        return _sellmeier_model([(c[1], c[2]), (c[3], c[4])], c[0] - 1.0)
    return None


def air_index(lam_um: np.ndarray) -> np.ndarray:
    """Brechzahl trockener Luft (15 °C, 101325 Pa) nach Edlén/Birch-Downs, λ in µm (Luft)."""
    sigma2 = 1.0 / np.asarray(lam_um, dtype=float) ** 2
    return 1.0 + 1e-8 * (8342.54 + 2406147.0 / (130.0 - sigma2) + 15998.0 / (38.9 - sigma2))


# --------------------------------------------------------------------------------------------
# AGF
# --------------------------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    if len(data) > 1 and data[1:2] == b"\x00":  # UTF-16 LE ohne BOM
        return data.decode("utf-16-le")
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _floats(fields: Sequence[str]) -> list[float]:
    out = []
    for f in fields:
        try:
            out.append(float(f))
        except ValueError:
            break
    return out


def _pick_transmission(rows: list[tuple[float, float, float]]) -> dict | None:
    """Reintransmission einer Probendicke (die häufigste, bei Gleichstand 10 mm)."""
    if not rows:
        return None
    counts: dict[float, int] = {}
    for _, _, d in rows:
        counts[d] = counts.get(d, 0) + 1
    best = max(counts, key=lambda d: (counts[d], d == PREFERRED_THICKNESS_MM, -abs(d - 10.0)))
    sel = sorted((lam, tau) for lam, tau, d in rows if d == best)
    return {
        "thickness_mm": best,
        "nm": [round(lam * 1000.0, 6) for lam, _ in sel],
        "tau": [tau for _, tau in sel],
    }


def read_agf(path: str | os.PathLike, text: str | None = None) -> tuple[list[SourceGlass], list[str]]:
    """Liest eine AGF-Datei. Rückgabe: Gläser und Warnungen (z. B. nicht unterstützte Formel)."""
    path = os.fspath(path)
    if text is None:
        text = _decode(pathlib.Path(path).read_bytes())
    label = os.path.basename(path)
    glasses: list[SourceGlass] = []
    warnings: list[str] = []
    current: dict | None = None

    def finish():
        if current is None:
            return
        try:
            glasses.append(_agf_glass(current, label))
        except GlassImportError as e:
            warnings.append(f"{label}:{current['line']}: {current['name']}: {e}")

    for lineno, raw in enumerate(text.splitlines(), start=1):
        fields = raw.split()
        if not fields:
            continue
        tag = fields[0].upper()
        if tag == "NM":
            finish()
            if len(fields) < 3:
                warnings.append(f"{label}:{lineno}: NM-Zeile unvollständig, übersprungen")
                current = None
                continue
            nums = _floats(fields[2:])
            current = {
                "line": lineno,
                "name": fields[1],
                "formula": int(nums[0]) if nums else -1,
                "nd": nums[2] if len(nums) > 2 else None,
                "vd": nums[3] if len(nums) > 3 else None,
                "status": int(nums[5]) if len(nums) > 5 else None,
                "cd": [],
                "ld": None,
                "it": [],
            }
        elif current is None:
            continue
        elif tag == "CD":
            current["cd"] = _floats(fields[1:])
        elif tag == "LD":
            ld = _floats(fields[1:])
            if len(ld) >= 2:
                current["ld"] = (ld[0], ld[1])
        elif tag == "IT":
            it = _floats(fields[1:])
            if len(it) >= 3 and it[2] > 0:
                current["it"].append((it[0], it[1], it[2]))
    finish()
    return glasses, warnings


def _agf_glass(rec: dict, label: str) -> SourceGlass:
    formula = rec["formula"]
    if formula not in AGF_FORMULAS:
        raise GlassImportError(f"unbekannte AGF-Formel {formula}")
    if rec["ld"] is None:
        raise GlassImportError("Gültigkeitsbereich (LD) fehlt")
    lo, hi = rec["ld"]
    if not (0 < lo < hi):
        raise GlassImportError(f"ungültiger Gültigkeitsbereich {lo}–{hi} µm")
    exact = agf_exact_model(formula, rec["cd"])
    status = rec["status"]
    return SourceGlass(
        name=rec["name"],
        origin=f"{label}:{rec['line']}",
        formula=f"AGF {formula} ({AGF_FORMULAS[formula]})",
        index=agf_index_function(formula, rec["cd"]),
        range_um=(lo, hi),
        exact=exact,
        exact3=_sellmeier3_model(exact),
        nd=rec["nd"],
        vd=rec["vd"],
        status=AGF_STATUS.get(status, str(status)) if status is not None else None,
        transmission=_pick_transmission(rec["it"]),
    )


# --------------------------------------------------------------------------------------------
# refractiveindex.info (YAML)
# --------------------------------------------------------------------------------------------


def _rii_formula(kind: int, c: list[float]) -> tuple[Callable[[np.ndarray], np.ndarray], tuple | None]:
    c = c + [0.0] * 20

    def pairs(start, squared):
        out = []
        i = start
        while i + 1 < len(c) and (c[i] != 0.0 or c[i + 1] != 0.0):
            out.append((c[i], c[i + 1] ** 2 if squared else c[i + 1]))
            i += 2
        return out

    if kind in (1, 2):
        pr = pairs(1, kind == 1)
        f = lambda lam: _sqrt_n2(1.0 + c[0] + _sellmeier_terms(np.asarray(lam, float) ** 2, pr))
        return f, _sellmeier_model(pr, c[0])
    if kind == 3:  # Polynom n² = C1 + Σ C_{2i} λ^{C_{2i+1}}
        poly = [(c[0], 0.0)] + [(c[i], c[i + 1]) for i in range(1, len(c) - 1, 2) if c[i] != 0.0]

        def f(lam):
            lam = np.asarray(lam, float)
            n2 = np.full_like(lam, c[0])
            for i in range(1, len(c) - 1, 2):
                if c[i] != 0.0:
                    n2 = n2 + c[i] * lam ** c[i + 1]
            return _sqrt_n2(n2)
        return f, _polynomial_model(poly)
    if kind == 5:  # Cauchy n = C1 + Σ C_{2i} λ^{C_{2i+1}}
        def f(lam):
            lam = np.asarray(lam, float)
            n = np.full_like(lam, c[0])
            for i in range(1, len(c) - 1, 2):
                if c[i] != 0.0:
                    n = n + c[i] * lam ** c[i + 1]
            return n
        return f, None
    if kind == 6:  # Gase n − 1 = C1 + Σ C_{2i} / (C_{2i+1} − λ⁻²)
        def f(lam):
            inv2 = 1.0 / np.asarray(lam, float) ** 2
            n = np.full_like(inv2, 1.0 + c[0])
            for i in range(1, len(c) - 1, 2):
                if c[i] != 0.0:
                    n = n + c[i] / (c[i + 1] - inv2)
            return n
        return f, None
    if kind == 7:  # Herzberger
        def f(lam):
            l2 = np.asarray(lam, float) ** 2
            L = 1.0 / (l2 - 0.028)
            return c[0] + c[1] * L + c[2] * L * L + c[3] * l2 + c[4] * l2**2 + c[5] * l2**3
        return f, None
    raise GlassImportError(f"refractiveindex.info-Formel {kind} wird nicht unterstützt")


def _parse_table(text: str) -> np.ndarray:
    rows = [list(map(float, line.split())) for line in str(text).strip().splitlines() if line.strip()]
    if not rows:
        raise GlassImportError("leere Tabelle")
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], dtype=float)


def read_rii_yaml(path: str | os.PathLike, name: str | None = None,
                  text: str | None = None, absolute: bool | None = None,
                  vacuum: bool | None = None) -> tuple[list[SourceGlass], list[str]]:
    """Liest eine Materialdatei der Datenbank refractiveindex.info (YAML, braucht PyYAML).

    ``absolute``/``vacuum``: Daten als absolute Brechzahl bzw. über der Vakuumwellenlänge
    behandeln und auf Luft umrechnen. ``None``: Angaben der Datei (``n_absolute``,
    ``wavelength_vacuum``), fehlen sie, gilt relativ zu Luft wie in den Herstellerkatalogen.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:  # pragma: no cover - abhängig von der Umgebung
        raise GlassImportError(
            "YAML-Dateien brauchen PyYAML: python -m pip install pyyaml"
        ) from e
    path = os.fspath(path)
    label = os.path.basename(path)
    if text is None:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "DATA" not in doc:
        raise GlassImportError(f"{label}: kein Abschnitt DATA")
    # Ältere Dateien: SPECS; aktuelle: PROPERTIES (nd, Vd) und CONDITIONS (Temperatur u. a.)
    specs = {}
    for key in ("SPECS", "CONDITIONS", "PROPERTIES"):
        part = doc.get(key)
        if isinstance(part, dict):
            specs.update(part)
    glass_name = name or pathlib.Path(path).stem.upper()

    n_func = None
    exact = None
    formula = None
    rng = None
    k_table = None
    samples = None
    for item in doc["DATA"]:
        kind = str(item.get("type", "")).strip()
        if kind.startswith("formula") and n_func is None:
            number = int(kind.split()[1])
            coeffs = [float(x) for x in str(item.get("coefficients", "")).split()]
            n_func, exact = _rii_formula(number, coeffs)
            wr = [float(x) for x in str(item.get("wavelength_range", "")).split()]
            if len(wr) != 2:
                raise GlassImportError(f"{label}: wavelength_range fehlt")
            rng = (wr[0], wr[1])
            formula = f"refractiveindex.info formula {number}"
        elif kind in ("tabulated n", "tabulated nk") and n_func is None:
            table = _parse_table(item.get("data", ""))
            lam, n = table[:, 0], table[:, 1]
            order = np.argsort(lam)
            lam, n = lam[order], n[order]
            n_func = (lambda lam_t, n_t: (lambda x: np.interp(np.asarray(x, float), lam_t, n_t)))(lam, n)
            rng = (float(lam[0]), float(lam[-1]))
            samples = lam
            formula = f"refractiveindex.info {kind}"
            if kind == "tabulated nk" and table.shape[1] >= 3:
                k_table = table[order][:, [0, 2]]
        elif kind == "tabulated k":
            table = _parse_table(item.get("data", ""))
            k_table = table[np.argsort(table[:, 0])][:, :2]
    if n_func is None:
        raise GlassImportError(f"{label}: keine Brechzahldaten (Formel oder Tabelle)")

    notes = []
    if absolute is None:
        absolute = bool(specs.get("n_absolute", False))
    if vacuum is None:
        vacuum = bool(specs.get("wavelength_vacuum", False))
    base = n_func
    if absolute or vacuum:
        exact = None  # Umrechnung auf Luft: nur als Fit darstellbar

        def n_func(lam, base=base):
            lam = np.asarray(lam, float)
            na = air_index(lam)
            n = base(lam * na) if vacuum else base(lam)
            return n / na if absolute else n

        notes.append(
            "umgerechnet auf Luft (Edlén/Birch-Downs, 15 °C): "
            + ("absolute Brechzahl" if absolute else "")
            + (", " if absolute and vacuum else "")
            + ("Vakuumwellenlänge" if vacuum else "")
        )

    transmission = None
    if k_table is not None and len(k_table) >= 2:
        d_mm = PREFERRED_THICKNESS_MM
        lam_k = k_table[:, 0]
        tau = np.exp(-4.0 * math.pi * k_table[:, 1] * (d_mm * 1e3) / lam_k)
        transmission = {
            "thickness_mm": d_mm,
            "nm": [round(float(x) * 1000.0, 6) for x in lam_k],
            "tau": [float(t) for t in tau],
        }

    nd = specs.get("nd")
    vd = specs.get("Vd")
    if samples is not None:
        notes.append("aus Tabelle, Fit an den Stützstellen")
    glass = SourceGlass(
        name=glass_name,
        origin=label,
        formula=formula,
        index=n_func,
        range_um=rng,
        exact=exact,
        exact3=_sellmeier3_model(exact),
        nd=float(nd) if nd is not None else None,
        vd=float(vd) if vd is not None else None,
        transmission=transmission,
        notes=notes,
        samples=samples,
    )
    return [glass], []


# --------------------------------------------------------------------------------------------
# Umwandlung in 3-Term-Sellmeier
# --------------------------------------------------------------------------------------------


def sellmeier_index(lam_um: np.ndarray, B: Sequence[float], C: Sequence[float]) -> np.ndarray:
    """Brechzahl des Modells ``sellmeier`` von Mikroptim (λ in µm, C in µm²)."""
    l2 = np.asarray(lam_um, float) ** 2
    return np.sqrt(1.0 + sum(b * l2 / (l2 - c) for b, c in zip(B, C)))


def _cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * math.pi / (2 * n))[::-1]


def _poles_ok(C: Sequence[float], a: float, b: float) -> bool:
    for c in C:
        if c > 0.0 and a * a <= c <= b * b:
            return False
    return True


def fit_sellmeier(func: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                  samples: int = 240, points: np.ndarray | None = None
                  ) -> tuple[list[float], list[float], float]:
    """Fit von n²−1 = Σ B_i λ²/(λ²−C_i) über [a, b] µm (zwei UV-Pole, ein IR-Pol).

    Variable Projektion: Für feste Pole ist das Modell linear in B (gewichtete kleinste
    Quadrate, Gewicht 1/(2n), also Fehler in der Brechzahl); die drei Pole werden mit einem
    gedämpften Gauß-Newton-Verfahren über unbeschränkte Parameter bestimmt, die die Pole
    außerhalb von [a, b] halten. Deterministisch, mehrere feste Startwerte.
    Rückgabe: B, C und die größte Abweichung |Δn| auf einem feinen Prüfgitter bzw., bei
    tabellierten Daten (``points``), an den Stützstellen der Tabelle im Bereich.
    """
    if points is not None:
        lam = np.asarray(points, float)
        lam = lam[(lam >= a) & (lam <= b)]
        if lam.size < 8:
            raise GlassImportError("zu wenige Stützstellen der Tabelle im Fitbereich")
    else:
        lam = _cheb_nodes(a, b, samples)
    n_t = func(lam)
    if not np.all(np.isfinite(n_t)) or np.any(n_t <= 1.0):
        raise GlassImportError("Originalformel liefert im Fitbereich keine gültige Brechzahl")
    y = n_t * n_t - 1.0
    w = 1.0 / (2.0 * n_t)
    l2 = lam * lam
    uv_max = 0.95 * a * a
    ir_min = 1.05 * b * b

    def sigmoid(x):
        x = min(max(float(x), -40.0), 40.0)
        return 1.0 / (1.0 + math.exp(-x))

    def poles(q):
        return [uv_max * sigmoid(q[0]), uv_max * sigmoid(q[1]),
                ir_min * (1.0 + math.exp(min(max(float(q[2]), -40.0), 40.0)))]

    def solve(q):
        C = poles(q)
        A = np.stack([l2 / (l2 - c) for c in C], axis=1)
        Bv, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = (A @ Bv - y) * w
        return r, Bv, C

    def logit(x):
        x = min(max(x, 1e-9), 1.0 - 1e-9)
        return math.log(x / (1.0 - x))

    starts = []
    for c1 in (0.004, 0.01):
        for c2 in (0.02, 0.05, 0.08):
            for c3 in (100.0, 200.0):
                if c1 < uv_max and c2 < uv_max and c3 > ir_min:
                    starts.append([logit(c1 / uv_max), logit(c2 / uv_max),
                                   math.log(c3 / ir_min - 1.0)])
    if not starts:  # sehr kleines a: Pole relativ zum Bereich setzen
        starts.append([logit(0.05), logit(0.3), math.log(max(100.0 / ir_min - 1.0, 1e-3))])

    best = None
    for q0 in starts:
        q = np.array(q0, float)
        r, Bv, C = solve(q)
        cost = float(r @ r)
        mu = 1e-3
        for _ in range(200):
            h = 1e-6
            J = np.empty((r.size, 3))
            for j in range(3):
                dq = np.zeros(3)
                dq[j] = h
                J[:, j] = (solve(q + dq)[0] - solve(q - dq)[0]) / (2 * h)
            g = J.T @ r
            H = J.T @ J
            improved = False
            for _ in range(12):
                step = np.linalg.solve(H + mu * np.diag(np.diag(H) + 1e-30), -g)
                q_new = q + step
                r_new, B_new, C_new = solve(q_new)
                cost_new = float(r_new @ r_new)
                if np.isfinite(cost_new) and cost_new < cost:
                    q, r, Bv, C = q_new, r_new, B_new, C_new
                    rel = (cost - cost_new) / max(cost, 1e-300)
                    cost = cost_new
                    mu = max(mu / 3.0, 1e-12)
                    improved = True
                    break
                mu *= 4.0
            if not improved or rel < 1e-12:
                break
        if best is None or cost < best[0]:
            best = (cost, [float(x) for x in Bv], [float(x) for x in C])

    _, Bv, C = best
    check = lam if points is not None else np.linspace(a, b, 2001)
    max_dn = float(np.max(np.abs(sellmeier_index(check, Bv, C) - func(check))))
    return Bv, C, max_dn


def _dispersion_summary(func) -> str:
    nd, nf, nc, ng = (float(func(np.array([x]))[0]) for x in (LAMBDA_D, LAMBDA_F, LAMBDA_C, LAMBDA_G))
    vd = (nd - 1.0) / (nf - nc)
    pgf = (ng - nf) / (nf - nc)
    return f"n_d {nd:.6f}, ν_d {vd:.2f}, P_g,F {pgf:.4f}"


def convert_glass(glass: SourceGlass, tolerance: float = DEFAULT_TOLERANCE,
                  legacy: bool = False) -> ConvertedGlass:
    """Wandelt ein Quellglas in einen Eintrag von ``mikroptim.glass/1`` um (oder verwirft es).

    Exakt darstellbare Formeln werden in den Modellen ``sellmeier`` (bis 8 Terme, Konstante)
    bzw. ``polynomial`` übernommen (ab M9-T4, ADR-0014); ``legacy`` beschränkt auf drei
    Sellmeier-Terme ohne Konstante (Mikroptim vor M9-T4), alles andere wird gefittet.
    """
    lo, hi = glass.range_um
    warnings: list[str] = []
    method = "exakt"
    max_dn = 0.0
    exact = glass.exact3 if legacy else glass.exact
    model_part = None
    range_um = (lo, hi)
    if exact is not None:
        if not _model_valid(exact, lo, hi):
            warnings.append("exakte Darstellung im Gültigkeitsbereich ungültig (Pol), deshalb Fit")
        else:
            check = np.linspace(lo, hi, 2001)
            max_dn = float(np.max(np.abs(model_index(exact, check) - glass.index(check))))
            if max_dn > 1e-9:  # Selbstkontrolle der Umrechnung
                warnings.append(f"exakte Darstellung weicht um {max_dn:.1e} ab, deshalb Fit")
            else:
                model_part = exact
    if model_part is None:
        method = "Fit"
        result = None
        last_error = None
        tried = set()
        for flo, fhi in FIT_RANGES_UM:
            a, b = max(lo, flo), min(hi, fhi)
            if not (a < b) or (a, b) in tried:
                continue
            tried.add((a, b))
            try:
                B, C, max_dn = fit_sellmeier(glass.index, a, b, points=glass.samples)
            except GlassImportError as e:
                last_error = str(e)
                continue
            if max_dn <= tolerance and _poles_ok(C, a, b):
                result = (B, C, max_dn, (a, b))
                break
            last_error = f"Fit über {a*1000:.0f}–{b*1000:.0f} nm: max |Δn| = {max_dn:.2e}"
        if result is None:
            return ConvertedGlass(glass, None, "verworfen", reason=last_error or "kein Fitbereich",
                                  warnings=warnings)
        B, C, max_dn, range_um = result
        model_part = {"model": "sellmeier", "B": B, "C": C}
        if range_um != (lo, hi):
            warnings.append(
                f"Gültigkeitsbereich auf {range_um[0]*1000:.0f}–{range_um[1]*1000:.0f} nm "
                f"eingeschränkt (Quelle {lo*1000:.0f}–{hi*1000:.0f} nm)"
            )

    def model(lam):
        return model_index(model_part, lam)

    nd_error = None
    if glass.nd is not None and range_um[0] <= LAMBDA_D <= range_um[1]:
        nd_error = float(model(np.array([LAMBDA_D]))[0] - glass.nd)
        if abs(nd_error) > ND_REJECT:
            return ConvertedGlass(
                glass, None, "verworfen", max_dn=max_dn, nd_error=nd_error,
                reason=f"n_d weicht um {nd_error:+.2e} vom Katalogwert ab (Formel prüfen)",
            )
        if abs(nd_error) > ND_WARN:
            warnings.append(f"n_d weicht um {nd_error:+.2e} vom Katalogwert ab")

    comment = [glass.formula]
    if method == "Fit":
        comment.append(f"3-Term-Fit, max |Δn| {max_dn:.1e}")
    elif model_part["model"] == "polynomial" or len(model_part.get("B", [])) > 3 or "A" in model_part:
        comment.append("exakt, Modell ab Mikroptim M9-T4")
    if glass.status:
        comment.append(f"Status {glass.status}")
    if range_um[0] <= LAMBDA_G and range_um[1] >= LAMBDA_C:
        comment.append(_dispersion_summary(model))
    comment += glass.notes

    entry = {"name": glass.name}
    entry.update(model_part)
    entry["range_nm"] = [round(range_um[0] * 1000.0, 6), round(range_um[1] * 1000.0, 6)]
    if glass.nd is not None:
        entry["reference"] = {"n_d": glass.nd}
    if glass.transmission:
        entry["transmission"] = glass.transmission
    entry["source"] = glass.origin
    entry["comment"] = "; ".join(comment)
    return ConvertedGlass(glass, entry, method, max_dn=max_dn, nd_error=nd_error, warnings=warnings)


# --------------------------------------------------------------------------------------------
# Mehrere Dateien, Ausgabe
# --------------------------------------------------------------------------------------------


def read_source(path: str | os.PathLike, name: str | None = None, absolute: bool | None = None,
                vacuum: bool | None = None) -> tuple[list[SourceGlass], list[str]]:
    """Liest eine Quelldatei; das Format folgt aus der Endung (.agf, .yml/.yaml)."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".agf":
        if name is not None or absolute or vacuum:
            raise GlassImportError("name, absolute und vacuum gelten nur für YAML-Dateien")
        return read_agf(path)
    if suffix in (".yml", ".yaml"):
        return read_rii_yaml(path, name=name, absolute=absolute, vacuum=vacuum)
    raise GlassImportError(f"{os.fspath(path)}: unbekanntes Format (erwartet .agf, .yml, .yaml)")


def _catalog_names(path: str | os.PathLike) -> set[str]:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {g["name"] for g in data.get("glasses", []) if "name" in g}


def convert_files(paths: Iterable[str | os.PathLike], *, tolerance: float = DEFAULT_TOLERANCE,
                  include_obsolete: bool = False, prefix: str = "",
                  exclude_names_from: Iterable[str | os.PathLike] = (),
                  name: str | None = None, absolute: bool | None = None,
                  vacuum: bool | None = None, legacy: bool = False) -> ImportResult:
    """Liest und wandelt alle Quelldateien um.

    - ``include_obsolete``: AGF-Gläser mit Status „obsolete“ übernehmen (Standard: nein).
    - ``prefix``: vor jeden Glasnamen setzen (gegen Namenskonflikte mehrerer Kataloge).
    - ``exclude_names_from``: Kataloge ``mikroptim.glass/1``, deren Namen übersprungen werden
      (z. B. ``data/glass/basis.json``; doppelte Namen lehnt der Leser beim Zusammenführen ab).
    - ``name``: Glasname für eine einzelne YAML-Datei (sonst der Dateiname).
    - ``legacy``: nur drei Sellmeier-Terme ohne Konstante ausgeben (für Mikroptim vor M9-T4);
      sonst werden Polynom- und Mehrterm-Formeln exakt übernommen.
    - ``absolute``, ``vacuum``: YAML-Daten (z. B. Kristalle aus der Literatur) als absolute
      Brechzahl bzw. über der Vakuumwellenlänge behandeln und auf Luft umrechnen.
    """
    paths = [os.fspath(p) for p in paths]
    if name is not None and len(paths) != 1:
        raise GlassImportError("name gilt nur für genau eine Quelldatei")
    excluded: set[str] = set()
    for p in exclude_names_from:
        excluded |= _catalog_names(p)
    out: list[ConvertedGlass] = []
    warnings: list[str] = []
    seen: dict[str, str] = {}
    for p in paths:
        glasses, w = read_source(p, name=name, absolute=absolute, vacuum=vacuum)
        warnings += w
        for g in glasses:
            g.name = prefix + g.name
            if g.status == "obsolete" and not include_obsolete:
                out.append(ConvertedGlass(g, None, "verworfen", reason="Status obsolete"))
                continue
            if g.name in excluded:
                out.append(ConvertedGlass(g, None, "verworfen",
                                          reason="Name in einem ausgeschlossenen Katalog"))
                continue
            if g.name.upper() == "AIR":
                out.append(ConvertedGlass(g, None, "verworfen", reason="Name AIR ist reserviert"))
                continue
            if g.name in seen:
                out.append(ConvertedGlass(g, None, "verworfen",
                                          reason=f"Name doppelt (zuerst in {seen[g.name]})"))
                continue
            converted = convert_glass(g, tolerance, legacy=legacy)
            if converted.entry is not None:
                seen[g.name] = g.origin
            out.append(converted)
    return ImportResult(out, paths, warnings, tolerance)


def catalog_dict(result: ImportResult, description: str | None = None) -> dict:
    """Katalog ``mikroptim.glass/1`` mit allen übernommenen Gläsern."""
    files = ", ".join(os.path.basename(f) for f in result.files)
    return {
        "format": FORMAT,
        "description": description or f"Importiert aus {files} (mikroptim.glass_import)",
        "source": files,
        "glasses": [g.entry for g in result.accepted],
    }


def write_catalog(result: ImportResult, path: str | os.PathLike, description: str | None = None):
    """Schreibt den Katalog als JSON (UTF-8)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog_dict(result, description), ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")


def report_dict(result: ImportResult) -> dict:
    """Bericht als ``dict`` (für JSON und die GUI)."""
    return {
        "format": "mikroptim.glass_import/1",
        "files": result.files,
        "tolerance": result.tolerance,
        "accepted": len(result.accepted),
        "rejected": len(result.rejected),
        "warnings": result.warnings,
        "glasses": [
            {
                "name": g.source.name,
                "origin": g.source.origin,
                "formula": g.source.formula,
                "method": g.method,
                "max_dn": g.max_dn,
                "nd_error": g.nd_error,
                "range_nm": g.entry["range_nm"] if g.entry else None,
                "reason": g.reason,
                "warnings": g.warnings,
            }
            for g in result.glasses
        ],
    }


def format_report(result: ImportResult) -> str:
    """Kurzer Textbericht."""
    lines = [
        f"Dateien: {', '.join(result.files)}",
        f"übernommen: {len(result.accepted)} "
        f"(exakt {sum(g.method == 'exakt' for g in result.accepted)}, "
        f"Fit {sum(g.method == 'Fit' for g in result.accepted)}), "
        f"verworfen: {len(result.rejected)}, Toleranz |Δn| ≤ {result.tolerance:.1e}",
    ]
    fits = [g.max_dn for g in result.accepted if g.method == "Fit"]
    if fits:
        lines.append(f"größte Fit-Abweichung: {max(fits):.2e}")
    for w in result.warnings:
        lines.append(f"Warnung: {w}")
    for g in result.glasses:
        if g.entry is None:
            lines.append(f"verworfen: {g.source.name} ({g.source.origin}): {g.reason}")
        for w in g.warnings:
            lines.append(f"Hinweis: {g.source.name}: {w}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mikroptim.glass_import",
        description="Glaskataloge (AGF, refractiveindex.info-YAML) nach mikroptim.glass/1 umwandeln.",
    )
    parser.add_argument("inputs", nargs="+", help="Quelldateien (.agf, .yml, .yaml)")
    parser.add_argument("-o", "--output", required=True, help="Ausgabe mikroptim.glass/1 (JSON)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="größte zulässige Abweichung |Δn| eines Fits (Standard 5e-6)")
    parser.add_argument("--include-obsolete", action="store_true",
                        help="AGF-Gläser mit Status obsolete übernehmen")
    parser.add_argument("--prefix", default="", help="Präfix für alle Glasnamen")
    parser.add_argument("--exclude-names-from", action="append", default=[], metavar="KATALOG",
                        help="Gläser überspringen, deren Name in diesem Katalog steht (mehrfach)")
    parser.add_argument("--name", help="Glasname (nur bei genau einer YAML-Datei)")
    parser.add_argument("--absolute", action="store_true", default=None,
                        help="YAML: absolute Brechzahl (Vakuum), auf Luft umrechnen")
    parser.add_argument("--vacuum", action="store_true", default=None,
                        help="YAML: Wellenlängen im Vakuum, auf Luft umrechnen")
    parser.add_argument("--legacy", action="store_true",
                        help="nur 3-Term-Sellmeier ausgeben (Mikroptim vor M9-T4), sonst exakt")
    parser.add_argument("--description", help="Beschreibung im Katalog")
    parser.add_argument("--report", help="Bericht als JSON schreiben")
    args = parser.parse_args(argv)
    try:
        result = convert_files(
            args.inputs, tolerance=args.tolerance, include_obsolete=args.include_obsolete,
            prefix=args.prefix, exclude_names_from=args.exclude_names_from, name=args.name,
            absolute=args.absolute, vacuum=args.vacuum, legacy=args.legacy,
        )
    except (GlassImportError, OSError, ValueError) as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 1
    print(format_report(result))
    if args.report:
        pathlib.Path(args.report).write_text(
            json.dumps(report_dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result.accepted:
        print("Fehler: kein Glas übernommen, keine Ausgabe geschrieben", file=sys.stderr)
        return 2
    write_catalog(result, args.output, args.description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
