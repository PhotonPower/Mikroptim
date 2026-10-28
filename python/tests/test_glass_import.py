"""Tests des Glasimports (M9-T3): AGF und refractiveindex.info-YAML nach mikroptim.glass/1.

Alle Testdaten sind synthetisch: die öffentlichen Sellmeier-Koeffizienten von N-BK7 aus
data/glass/basis.json und daraus abgeleitete Formeln. Es werden keine Herstellerkataloge
verwendet oder erzeugt (data/glass/README.md).
"""

import json
import math

import numpy as np
import pytest

import mikroptim
from mikroptim import glass_import as gi

from conftest import ROOT, SYSTEMS

BK7_B = [1.03961212, 0.231792344, 1.01046945]
BK7_C = [0.00600069867, 0.0200179144, 103.560653]
BK7_ND = 1.5168


def bk7(lam):
    return gi.sellmeier_index(lam, BK7_B, BK7_C)


def schott_coefficients(lo=0.365, hi=1.1):
    """Schott-Formel (AGF 1), linear an N-BK7 angepasst; eigene Formel, kein Katalogwert."""
    lam = np.linspace(lo, hi, 300)
    l2 = lam**2
    A = np.stack([np.ones_like(l2), l2, 1 / l2, 1 / l2**2, 1 / l2**3, 1 / l2**4], axis=1)
    a, *_ = np.linalg.lstsq(A, bk7(lam) ** 2, rcond=None)
    return [float(x) for x in a]


def schott_index(a, lam):
    l2 = np.asarray(lam, float) ** 2
    return np.sqrt(a[0] + a[1] * l2 + a[2] / l2 + a[3] / l2**2 + a[4] / l2**3 + a[5] / l2**4)


def agf_text(extra=""):
    b, c = BK7_B, BK7_C
    return (
        "CC Synthetischer Testkatalog\n"
        f"NM TEST-BK7 2 517642 {BK7_ND} 64.17 0 0\n"
        "ED 7.1 8.3 2.51 0 0\n"
        f"CD {b[0]} {c[0]} {b[1]} {c[1]} {b[2]} {c[2]}\n"
        "TD 0 0 0 0 0 0 20\n"
        "LD 0.3 2.5\n"
        "IT 0.35 0.5 10\n"
        "IT 0.40 0.98 10\n"
        "IT 0.40 0.95 25\n"
        "IT 0.50 0.998 10\n"
        + extra
    )


def write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


# --- AGF --------------------------------------------------------------------------------------


def test_agf_sellmeier_exact(tmp_path):
    p = write(tmp_path, "test.agf", agf_text())
    result = gi.convert_files([p])
    assert len(result.accepted) == 1
    g = result.accepted[0]
    assert g.method == "exakt"
    assert g.entry["B"] == BK7_B and g.entry["C"] == BK7_C
    assert g.entry["range_nm"] == [300.0, 2500.0]
    assert g.entry["reference"] == {"n_d": BK7_ND}
    assert abs(g.nd_error) < 1e-5


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "latin-1"])
def test_agf_encodings(tmp_path, encoding):
    p = write(tmp_path, "test.agf", agf_text(), encoding=encoding)
    glasses, warnings = gi.read_agf(p)
    assert [g.name for g in glasses] == ["TEST-BK7"] and warnings == []


def test_agf_transmission_prefers_most_frequent_thickness(tmp_path):
    p = write(tmp_path, "test.agf", agf_text())
    entry = gi.convert_files([p]).accepted[0].entry
    assert entry["transmission"] == {"thickness_mm": 10.0, "nm": [350.0, 400.0, 500.0],
                                     "tau": [0.5, 0.98, 0.998]}


def test_agf_schott_formula_exact_as_polynomial(tmp_path):
    a = schott_coefficients()
    extra = (f"NM TEST-SCHOTT 1 517642 {BK7_ND} 64.17 0 1\n"
             f"CD {' '.join(map(repr, a))}\nLD 0.365 1.1\n")
    p = write(tmp_path, "test.agf", agf_text(extra))
    g = next(x for x in gi.convert_files([p]).accepted if x.source.name == "TEST-SCHOTT")
    assert g.method == "exakt" and g.entry["model"] == "polynomial"
    assert g.entry["terms"] == [[c, float(e)] for c, e in zip(a, [0, 2, -2, -4, -6, -8])]
    lam = np.linspace(0.365, 1.1, 997)
    assert np.max(np.abs(gi.model_index(g.entry, lam) - schott_index(a, lam))) < 1e-15


def test_agf_schott_formula_legacy_is_fitted_within_tolerance(tmp_path):
    a = schott_coefficients()
    extra = (f"NM TEST-SCHOTT 1 517642 {BK7_ND} 64.17 0 1\n"
             f"CD {' '.join(map(repr, a))}\nLD 0.365 1.1\n")
    p = write(tmp_path, "test.agf", agf_text(extra))
    result = gi.convert_files([p], tolerance=5e-6, legacy=True)
    g = next(x for x in result.accepted if x.source.name == "TEST-SCHOTT")
    assert g.method == "Fit" and g.max_dn <= 5e-6
    assert g.entry["model"] == "sellmeier" and len(g.entry["B"]) == 3 and "A" not in g.entry
    lam = np.linspace(0.365, 1.1, 997)
    dn = gi.sellmeier_index(lam, g.entry["B"], g.entry["C"]) - schott_index(a, lam)
    assert np.max(np.abs(dn)) <= 5e-6
    # Pole außerhalb des Gültigkeitsbereichs (Bedingung des Katalog-Lesers)
    for c in g.entry["C"]:
        assert not (0.365**2 <= c <= 1.1**2)


@pytest.mark.parametrize("formula,cd", [
    (4, [0.02, 1.0, 0.08, 0.004, 0.12]),           # Sellmeier 2
    (9, [2.1, 0.2, 0.02, 1.0, 100.0]),             # Sellmeier 4
    (6, [1.0, 0.006, 0.2, 0.02, 0.05, 0.04, 1.0, 100.0]),  # Sellmeier 3 (vier Terme)
    (11, [1.0, 0.006, 0.2, 0.02, 0.05, 0.04, 0.01, 0.06, 1.0, 100.0]),  # Sellmeier 5
    (10, [2.27, -0.0094, 0.011, 0.00015, -1e-5, 8e-7, 1e-8, -1e-9]),    # Extended
    (12, [2.27, -0.0094, 0.011, 0.00015, -1e-5, 8e-7, 1e-5, -1e-6]),    # Extended 2
    (13, [2.27, -0.0094, 1e-5, 0.011, 0.00015, -1e-5, 8e-7, 1e-8, -1e-9]),  # Extended 3
])
def test_agf_formulas_exact_in_new_models(formula, cd):
    model = gi.agf_exact_model(formula, cd)
    assert model is not None
    lam = np.linspace(0.4, 1.0, 301)
    ref = gi.agf_index_function(formula, cd)(lam)
    assert np.max(np.abs(gi.model_index(model, lam) - ref)) < 1e-13


def test_agf_without_exact_model_is_fitted():
    for formula in (3, 5, 7, 8):
        assert gi.agf_exact_model(formula, [1.5, 0.01, 0, 0, 0, 0]) is None


def test_agf_fit_narrows_range_when_needed(tmp_path):
    # Schott-Formel über einen weiten Bereich als 3-Term-Fit (legacy): Drei Sellmeier-Terme
    # reichen dort nicht, der Bereich wird eingeschränkt und das gemeldet.
    a = schott_coefficients(0.31, 2.3)
    extra = f"NM WIDE 1 517642 {BK7_ND} 64.17 0 0\nCD {' '.join(map(repr, a))}\nLD 0.31 2.3\n"
    p = write(tmp_path, "test.agf", agf_text(extra))
    g = next(x for x in gi.convert_files([p], legacy=True).glasses if x.source.name == "WIDE")
    if g.entry is not None and g.entry["range_nm"] != [310.0, 2300.0]:
        assert any("eingeschränkt" in w for w in g.warnings)
    if g.entry is not None:
        assert g.max_dn <= gi.DEFAULT_TOLERANCE
    # ohne legacy exakt über den ganzen Bereich
    g = next(x for x in gi.convert_files([p]).glasses if x.source.name == "WIDE")
    assert g.method == "exakt" and g.entry["range_nm"] == [310.0, 2300.0]


def test_agf_obsolete_skipped_by_default(tmp_path):
    b, c = BK7_B, BK7_C
    extra = (f"NM OLD 2 517642 {BK7_ND} 64.17 0 2\n"
             f"CD {b[0]} {c[0]} {b[1]} {c[1]} {b[2]} {c[2]}\nLD 0.3 2.5\n")
    p = write(tmp_path, "test.agf", agf_text(extra))
    r = gi.convert_files([p])
    assert [g.source.name for g in r.accepted] == ["TEST-BK7"]
    assert any(g.reason == "Status obsolete" for g in r.rejected)
    r = gi.convert_files([p], include_obsolete=True)
    assert {g.source.name for g in r.accepted} == {"TEST-BK7", "OLD"}


def test_agf_wrong_formula_rejected_by_nd_check(tmp_path):
    # Herzberger-Koeffizienten, die nicht zum Katalogwert n_d passen: verworfen statt still falsch.
    extra = f"NM WRONG 3 517642 {BK7_ND} 64.17 0 0\nCD 1.5 0.01 0 0 0 0\nLD 0.4 0.8\n"
    p = write(tmp_path, "test.agf", agf_text(extra))
    g = next(x for x in gi.convert_files([p]).glasses if x.source.name == "WRONG")
    assert g.entry is None and "n_d weicht" in g.reason


def test_agf_unknown_formula_is_warning(tmp_path):
    extra = "NM ODD 99 517642 1.5 60 0 0\nCD 1 2 3\nLD 0.4 0.8\n"
    p = write(tmp_path, "test.agf", agf_text(extra))
    r = gi.convert_files([p])
    assert any("ODD" in w and "Formel 99" in w for w in r.warnings)
    assert [g.source.name for g in r.accepted] == ["TEST-BK7"]


@pytest.mark.parametrize("formula", sorted(gi.AGF_FORMULAS))
def test_agf_formulas_evaluate(formula):
    # Jede unterstützte Formel liefert endliche Brechzahlen > 1 für plausible Koeffizienten
    # (Formeln mit Sellmeier-Termen: BK7; übrige: konstante Brechzahl 1,5 als Spezialfall).
    b, c = BK7_B, BK7_C
    cd = {
        1: [2.25] + [0] * 5, 2: [b[0], c[0], b[1], c[1], b[2], c[2]], 3: [1.5, 0, 0, 0, 0, 0],
        4: [1.25, 0, 0, 0, 0], 5: [1.5, 0, 0], 6: [b[0], c[0], b[1], c[1], b[2], c[2], 0, 0],
        7: [2.25, 0, 0, 0], 8: [2.25, 0, 0, 0], 9: [2.25, 0, 0, 0, 0], 10: [2.25] + [0] * 7,
        11: [b[0], c[0], b[1], c[1], b[2], c[2], 0, 0, 0, 0], 12: [2.25] + [0] * 7,
        13: [2.25] + [0] * 8,
    }[formula]
    f = gi.agf_index_function(formula, cd)
    lam = np.array([0.45, 0.5876, 0.65])
    n = f(lam)
    expected = bk7(lam) if formula in (2, 6, 11) else np.full(3, 1.5)
    assert np.allclose(n, expected, rtol=0, atol=1e-12)


def test_agf_sellmeier3_with_three_terms_is_exact(tmp_path):
    b, c = BK7_B, BK7_C
    extra = (f"NM S3 6 517642 {BK7_ND} 64.17 0 0\n"
             f"CD {b[0]} {c[0]} {b[1]} {c[1]} {b[2]} {c[2]} 0 0\nLD 0.3 2.5\n")
    p = write(tmp_path, "test.agf", agf_text(extra))
    g = next(x for x in gi.convert_files([p]).accepted if x.source.name == "S3")
    assert g.method == "exakt" and g.entry["B"] == BK7_B


# --- Namen ------------------------------------------------------------------------------------


def test_prefix_duplicates_and_excluded_catalog(tmp_path):
    p1 = write(tmp_path, "a.agf", agf_text())
    p2 = write(tmp_path, "b.agf", agf_text())
    r = gi.convert_files([p1, p2])
    assert len(r.accepted) == 1 and "doppelt" in r.rejected[0].reason
    r = gi.convert_files([p1], prefix="X_")
    assert r.accepted[0].entry["name"] == "X_TEST-BK7"
    basis = tmp_path / "basis.json"
    basis.write_text(json.dumps({"format": gi.FORMAT, "glasses": [{"name": "TEST-BK7"}]}))
    r = gi.convert_files([p1], exclude_names_from=[basis])
    assert not r.accepted


# --- refractiveindex.info ---------------------------------------------------------------------


def rii_yaml(kind, coeffs, rng="0.3 2.5", specs=""):
    return (
        "REFERENCES: synthetisch\nDATA:\n"
        f"  - type: {kind}\n    wavelength_range: {rng}\n    coefficients: {coeffs}\n"
        + (f"SPECS:\n{specs}" if specs else "")
    )


def test_rii_formula1_exact(tmp_path):
    yaml = pytest.importorskip("yaml")  # noqa: F841
    b, c = BK7_B, BK7_C
    coeffs = f"0 {b[0]} {math.sqrt(c[0])} {b[1]} {math.sqrt(c[1])} {b[2]} {math.sqrt(c[2])}"
    p = write(tmp_path, "test-bk7.yml", rii_yaml("formula 1", coeffs, specs="    nd: 1.5168\n"))
    g = gi.convert_files([p]).accepted[0]
    assert g.method == "exakt" and g.entry["name"] == "TEST-BK7"
    assert np.allclose(g.entry["C"], BK7_C, rtol=1e-14)


def test_rii_formula2_with_constant(tmp_path):
    pytest.importorskip("yaml")
    b, c = BK7_B, BK7_C
    coeffs = f"0.01 {b[0]} {c[0]} {b[1]} {c[1]} {b[2]} {c[2]}"
    p = write(tmp_path, "k.yml", rii_yaml("formula 2", coeffs, rng="0.4 1.0"))
    g = gi.convert_files([p], name="K-CONST").accepted[0]
    assert g.method == "exakt" and g.entry["A"] == 0.01 and g.entry["name"] == "K-CONST"
    g = gi.convert_files([p], name="K-CONST", legacy=True).accepted[0]
    assert g.method == "Fit" and g.max_dn <= gi.DEFAULT_TOLERANCE


def test_rii_formula3_exact_as_polynomial(tmp_path):
    pytest.importorskip("yaml")
    a = schott_coefficients()
    coeffs = f"{a[0]} {a[1]} 2 {a[2]} -2 {a[3]} -4 {a[4]} -6 {a[5]} -8"
    p = write(tmp_path, "poly.yml", rii_yaml("formula 3", coeffs, rng="0.365 1.1"))
    g = gi.convert_files([p]).accepted[0]
    assert g.method == "exakt" and g.entry["model"] == "polynomial"
    lam = np.linspace(0.365, 1.1, 101)
    assert np.max(np.abs(gi.model_index(g.entry, lam) - schott_index(a, lam))) < 1e-14


def test_rii_tabulated_with_k_gives_transmission(tmp_path):
    pytest.importorskip("yaml")
    lam = np.linspace(0.35, 1.0, 60)
    n = bk7(lam)
    k = np.full_like(lam, 1e-9)
    rows = "\n".join(f"      {x:.6f} {y:.8f} {z:.3e}" for x, y, z in zip(lam, n, k))
    text = f"DATA:\n  - type: tabulated nk\n    data: |\n{rows}\n"
    p = write(tmp_path, "tab.yml", text)
    g = gi.convert_files([p], tolerance=1e-5).accepted[0]
    tau = g.entry["transmission"]["tau"]
    expected = math.exp(-4 * math.pi * 1e-9 * 10e3 / 0.35)
    assert g.entry["transmission"]["thickness_mm"] == 10.0
    assert abs(tau[0] - expected) < 1e-12


def test_rii_absolute_index_converted_to_air(tmp_path):
    pytest.importorskip("yaml")
    b, c = BK7_B, BK7_C
    coeffs = f"0 {b[0]} {c[0]} {b[1]} {c[1]} {b[2]} {c[2]}"
    specs = "    n_absolute: true\n    wavelength_vacuum: false\n"
    p = write(tmp_path, "abs.yml", rii_yaml("formula 2", coeffs, rng="0.4 1.0", specs=specs))
    g = gi.convert_files([p]).accepted[0]
    lam = np.array([0.5876])
    n_rel = gi.sellmeier_index(lam, g.entry["B"], g.entry["C"])
    assert abs(n_rel[0] - bk7(lam)[0] / gi.air_index(lam)[0]) <= gi.DEFAULT_TOLERANCE
    assert "Luft" in g.entry["comment"]


def test_rii_properties_nd_and_forced_vacuum(tmp_path):
    pytest.importorskip("yaml")
    b, c = BK7_B, BK7_C
    coeffs = f"0 {b[0]} {c[0]} {b[1]} {c[1]} {b[2]} {c[2]}"
    text = rii_yaml("formula 2", coeffs, rng="0.4 1.0") + "PROPERTIES:\n    nd: 1.5168\n"
    p = write(tmp_path, "props.yml", text)
    g = gi.convert_files([p]).accepted[0]
    assert g.method == "exakt" and g.entry["reference"] == {"n_d": 1.5168}
    g = gi.convert_files([p], vacuum=True).accepted[0]
    lam = np.array([0.5876])
    expected = bk7(lam * gi.air_index(lam))
    assert abs(gi.sellmeier_index(lam, g.entry["B"], g.entry["C"])[0] - expected[0]) <= 5e-6


def test_air_index_standard_value():
    # Edlén/Birch-Downs bei 15 °C, 101325 Pa: n − 1 ≈ 2,77·10⁻⁴ bei 589 nm
    assert abs(gi.air_index(np.array([0.589]))[0] - 1.0 - 2.77e-4) < 2e-6


# --- Katalog, Leser von Mikroptim, Kommandozeile ----------------------------------------------


def test_catalog_is_accepted_by_mikroptim(tmp_path):
    p = write(tmp_path, "test.agf", agf_text(
        f"NM TEST-SCHOTT 1 517642 {BK7_ND} 64.17 0 1\n"
        f"CD {' '.join(map(repr, schott_coefficients()))}\nLD 0.365 1.1\n"))
    result = gi.convert_files([p])
    cat = tmp_path / "katalog.json"
    gi.write_catalog(result, cat)
    data = json.loads((SYSTEMS / "einzellinse_bk7.json").read_text(encoding="utf-8"))
    reference = mikroptim.system_from_dict(data, base_dir=ROOT)
    data["glass_catalogs"] = [str(cat)]
    for s in data["surfaces"]:
        if s.get("material") == "N-BK7":
            s["material"] = "TEST-BK7"
    imported = mikroptim.system_from_dict(data, base_dir=tmp_path)
    # gleiche Koeffizienten wie N-BK7 in basis.json: Paraxialdaten bitgleich
    assert mikroptim.paraxial(imported).focal_length == mikroptim.paraxial(reference).focal_length
    for s in data["surfaces"]:
        if s.get("material") == "TEST-BK7":
            s["material"] = "TEST-SCHOTT"
    fitted = mikroptim.system_from_dict(data, base_dir=tmp_path)
    assert abs(mikroptim.paraxial(fitted).focal_length
               - mikroptim.paraxial(reference).focal_length) < 0.01


def test_cli_writes_catalog_and_report(tmp_path, capsys):
    p = write(tmp_path, "test.agf", agf_text())
    out = tmp_path / "private" / "test.json"
    rep = tmp_path / "bericht.json"
    assert gi.main([str(p), "-o", str(out), "--report", str(rep)]) == 0
    cat = json.loads(out.read_text(encoding="utf-8"))
    assert cat["format"] == "mikroptim.glass/1" and cat["glasses"][0]["name"] == "TEST-BK7"
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["accepted"] == 1 and report["glasses"][0]["method"] == "exakt"
    assert "übernommen: 1" in capsys.readouterr().out


def test_cli_no_glass_no_output(tmp_path):
    p = write(tmp_path, "leer.agf", "CC nichts\n")
    out = tmp_path / "x.json"
    assert gi.main([str(p), "-o", str(out)]) == 2
    assert not out.exists()


def test_cli_unknown_format(tmp_path):
    p = write(tmp_path, "x.txt", "")
    assert gi.main([str(p), "-o", str(tmp_path / "x.json")]) == 1
