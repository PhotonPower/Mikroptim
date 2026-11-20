"""Hilfen des Skripts tools/m6t6_zielspezifikation.py (M6-T6b), ohne Programmaufrufe."""

import importlib.util
import json
import types

import pytest

from conftest import ROOT


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "m6t6", ROOT / "tools" / "m6t6_zielspezifikation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _system(tmp_path, reference=None):
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    if reference:
        s["reference_sphere"] = reference
    path = tmp_path / "system.json"
    path.write_text(json.dumps(s), encoding="utf-8")
    return s, path


def test_psf_retries_with_numerical_aperture(tool, tmp_path, monkeypatch):
    """eval --psf scheitert bei vollem Feld mit „NA vorgeben“: das Skript wiederholt mit --na
    (NA = Pupillenradius / Brennweite) statt abzubrechen (Fehler aus M6-T7)."""
    s, path = _system(tmp_path, "image_infinite")
    calls = []

    def fake_run(args, stdout, stderr, cwd):
        calls.append(list(args))
        if ("--psf" in args or "--vector" in args) and "--na" not in args:
            stderr.write("Fehler: PSF: NA der Pupillenabbildung für Feld 2, Wellenlänge 0 nicht "
                         "bestimmbar (Feldstrahl versagt oder afokal); NA vorgeben\n")
            return types.SimpleNamespace(returncode=1)
        stdout.write("{}")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    runner = tool.Runner("mikroptim", str(tmp_path / "out"), synth_options=[])
    out = tool.evaluate(runner, "a3", str(path), grid=16)
    assert sorted(out) == ["psf", "pupille", "unendlich", "vektor"]
    na = s["surfaces"][0]["semi_diameter"] / s["target"]["focal_length"]
    assert tool.psf_numerical_aperture(s) == pytest.approx(na)
    retried = [c for c in calls if "--na" in c]
    assert len(retried) == 2
    assert all(float(c[c.index("--na") + 1]) == na for c in retried)
    second = next(c for c in calls if "--reference" in c and "exit-pupil" not in c)
    assert second[second.index("--reference") + 1] == "infinite"
    assert (tmp_path / "out" / "a3_psf_na.txt").exists()


def test_eval_failure_leaves_no_result_file(tool, tmp_path, monkeypatch):
    _, path = _system(tmp_path)

    def fake_run(args, stdout, stderr, cwd):
        stderr.write("anderer Fehler\n")
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    runner = tool.Runner("mikroptim", str(tmp_path / "out"), synth_options=[])
    with pytest.raises(RuntimeError):
        tool.evaluate(runner, "x", str(path), grid=16)
    # eine Fortsetzung darf das fehlgeschlagene Ergebnis nicht als vorhanden überspringen
    assert not (tmp_path / "out" / "x_pupille.json").exists()


def test_reference_sphere_default_and_f12_guard(tool):
    s = {}
    tool.set_reference_sphere(s, "image_infinite")
    assert s["reference_sphere"] == "image_infinite"
    tool.set_reference_sphere(s, "last_surface")
    assert s["reference_sphere"] == "last_surface"

    def args(ref, ohne=False, mit=False):
        return types.SimpleNamespace(referenzsphaere=ref, ohne_f12_schutz=ohne, mit_f12_schutz=mit)

    assert tool.f12_guard(args("last_surface"))
    assert not tool.f12_guard(args("last_surface", ohne=True))
    assert not tool.f12_guard(args("image_infinite"))
    assert not tool.f12_guard(args("image_exit_pupil"))
    assert tool.f12_guard(args("image_infinite", mit=True))
