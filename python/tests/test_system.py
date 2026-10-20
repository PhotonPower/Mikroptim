"""Systeme laden, Eigenschaften, JSON-Export und -Import, Paraxialdaten gegen die CLI."""

import json

import numpy as np
import pytest

import mikroptim
from conftest import SYSTEMS, cli_json, system_files


@pytest.mark.parametrize("path", system_files(), ids=lambda p: p.name)
def test_load_all_systems(path):
    s = mikroptim.load_system(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert s.name == data["name"]
    assert s.num_surfaces == len(data["surfaces"])
    assert s.curvatures.shape == (s.num_surfaces,)
    assert s.curvatures.dtype == np.float64
    assert s.indices.shape == (s.num_surfaces, len(data["spectrum"]["wavelengths_nm"]))
    for i, surface in enumerate(data["surfaces"]):
        assert s.thicknesses[i] == surface["thickness"]
        assert s.semi_diameters[i] == surface["semi_diameter"]
        assert s.surface_types[i] == surface["type"]
        assert s.curvatures[i] == surface.get("curvature", 0.0)
    assert list(s.wavelengths_nm) == data["spectrum"]["wavelengths_nm"]
    assert s.reference_nm == data["spectrum"]["reference_nm"]
    assert s.source_type == data["source"]["type"]
    assert s.file == str(path)
    assert s.num_ghost_plates == len(data.get("ghost_plates", []))


@pytest.mark.parametrize("path", system_files(), ids=lambda p: p.name)
def test_json_roundtrip(path):
    s = mikroptim.load_system(path)
    text = s.to_json()
    again = mikroptim.System.from_json(text, path.parent)
    assert again.to_json() == text
    assert again.file is None
    d = mikroptim.system_to_dict(s)
    assert d["format"] == "mikroptim.system/1"
    assert mikroptim.system_from_dict(d, path.parent).to_json() == text


def test_save_rewrites_glass_catalogs(tmp_path):
    s = mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json")
    out = tmp_path / "kopie.json"
    s.save(out)
    loaded = mikroptim.load_system(out)
    assert loaded.catalog_files == s.catalog_files
    assert np.array_equal(loaded.curvatures, s.curvatures)
    assert np.array_equal(loaded.indices, s.indices)


def test_from_dict_changes_system():
    s = mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json")
    d = mikroptim.system_to_dict(s)
    d["surfaces"][1]["curvature"] = 0.012
    changed = mikroptim.system_from_dict(d, SYSTEMS)
    assert changed.curvatures[1] == 0.012
    assert s.curvatures[1] == 0.00967  # das Original bleibt unverändert


def test_arrays_are_copies():
    s = mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json")
    c = s.curvatures
    c[1] = 1.0
    assert s.curvatures[1] == 0.00967


def test_variables():
    s = mikroptim.load_system(SYSTEMS / "einzellinse_bk7_biegung.json")
    v = s.variables()
    assert [e["name"] for e in v] == ["c[1]", "c[2]", "w"]
    assert v[0]["value"] == 0.00967
    assert v[2]["lower"] == 50.0 and v[2]["upper"] == 150.0
    assert mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json").variables() == []


@pytest.mark.parametrize("path", system_files(), ids=lambda p: p.name)
def test_paraxial_equals_cli(path):
    s = mikroptim.load_system(path)
    expected = cli_json(["trace", path, "--rings", "1", "--spokes", "1"])["paraxial"]
    assert mikroptim.paraxial(s).to_dict() == expected


def test_paraxial_properties():
    s = mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json")
    p = mikroptim.paraxial(s)
    assert p.wavelength_nm == s.reference_nm
    a, b, c, d = p.abcd.ravel()
    assert p.power == -c
    assert abs(a * d - b * c - 1.0) < 1e-12
    other = mikroptim.paraxial(s, 486.13)
    assert other.wavelength_nm == 486.13
    assert other.focal_length < p.focal_length  # N-BK7: kürzer im Blauen


def test_afocal_paraxial():
    p = mikroptim.paraxial(mikroptim.load_system(SYSTEMS / "planplatte_bk7.json"))
    assert p.afocal
    assert p.to_dict()["focal_length"] is None
    assert np.isinf(p.focal_length)
