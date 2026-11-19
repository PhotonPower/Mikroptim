"""Monte-Carlo-Tolerierung (M8-T4) über `mikroptim.run_cli`: gleiche Ausgabe wie das Programm."""

import json

import mikroptim

from conftest import SYSTEMS


def test_tolerance_cli_in_module():
    path = SYSTEMS / "testsystem_toleranz.json"
    code, out, err = mikroptim.run_cli(["tolerance", str(path), "--json", "--samples", "20"])
    assert code == 0, err
    res = json.loads(out)
    assert res["format"] == "mikroptim.tolerance/1"
    assert res["draws"] == 26 and res["rms_waves"]["count"] == 20
    again = json.loads(mikroptim.run_cli(["tolerance", str(path), "--json", "--samples", "20"])[1])
    assert again["rms_waves"] == res["rms_waves"]  # reproduzierbar mit festem Startwert
    code, _, err = mikroptim.run_cli(["tolerance", str(SYSTEMS / "einzellinse_bk7.json")])
    assert code == 1 and "keine Toleranzen" in err
