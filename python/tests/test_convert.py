"""Raytatouille-Adapter (M9-T2) über ``mikroptim.run_cli(["convert", ...])``: Roundtrip
Mikroptim → Raytatouille → Mikroptim mit gleichen Paraxialdaten, nicht abbildbare Stellen."""

import json

import mikroptim

from conftest import SYSTEMS

BASIS = SYSTEMS.parent / "glass" / "basis.json"


def test_convert_roundtrip_keeps_paraxial_data(tmp_path):
    src = SYSTEMS / "einzellinse_bk7.json"
    rtt = tmp_path / "linse.rtt.json"
    code, out, err = mikroptim.run_cli(["convert", str(src), "-o", str(rtt)])
    assert code == 0, err
    data = json.loads(rtt.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.11.0"
    assert data["root"]["children"][1]["material"] == "BASIS:N-BK7"
    back = tmp_path / "linse.json"
    code, out, err = mikroptim.run_cli(["convert", str(rtt), "-o", str(back),
                                        "--glass-catalog", str(BASIS)])
    assert code == 0, err
    a = mikroptim.paraxial(mikroptim.load_system(src))
    b = mikroptim.paraxial(mikroptim.load_system(back))
    assert a.focal_length == b.focal_length
    assert a.back_focal_distance == b.back_focal_distance


def test_convert_reports_non_mappable(tmp_path):
    code, _, err = mikroptim.run_cli(["convert", str(SYSTEMS / "testsystem_mehrlinser_dezentriert.json"),
                                      "-o", str(tmp_path / "x.rtt.json")])
    assert code == 1
    assert "nicht abbildbar" in err and "surfaces[1].decenter" in err
    assert not (tmp_path / "x.rtt.json").exists()
