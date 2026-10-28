"""Aberrationsbericht (M10-T1) aus Python: gleiche Werte wie die CLI, Identitäten."""

import json

import pytest

import mikroptim

from conftest import SYSTEMS


def test_aberrations_matches_cli_and_petzval():
    path = SYSTEMS / "testsystem_mehrlinser.json"
    rep = mikroptim.aberrations(path, field_angle_deg=1.0)
    assert rep["format"] == "mikroptim.aberrations/1"
    code, out, _ = mikroptim.run_cli(["aberrations", str(path), "--json", "--field-deg", "1"])
    assert code == 0 and json.loads(out)["totals"] == rep["totals"]
    system = mikroptim.load_system(path)
    H = rep["lagrange"]
    petzval = mikroptim.paraxial(system).petzval_sum
    assert rep["totals"]["S"][3] == pytest.approx(H * H * petzval, rel=1e-13)
    assert len(rep["surfaces"]) == len(mikroptim.system_to_dict(system)["surfaces"])
    w = rep["totals"]["waves"]
    assert w["W040"] == pytest.approx(rep["totals"]["S"][0] / 8 / (rep["reference_nm"] * 1e-6))


def test_aberrations_error_point_source(tmp_path):
    data = json.loads((SYSTEMS / "einzellinse_bk7.json").read_text(encoding="utf-8"))
    data["source"] = {"type": "point", "point": [0, 0, -100], "numerical_aperture": 0.05}
    p = tmp_path / "punkt.json"
    data["glass_catalogs"] = [str(SYSTEMS.parent / "glass" / "basis.json")]
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="kollimierte Quelle"):
        mikroptim.aberrations(p)
