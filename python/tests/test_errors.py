"""Fehlerfälle: C++-Ausnahmen als Python-Ausnahmen mit Meldung."""

import json

import pytest

import mikroptim
from conftest import ROOT, SYSTEMS

LINSE = SYSTEMS / "einzellinse_bk7.json"


def _variant(tmp_path, change):
    data = json.loads(LINSE.read_text(encoding="utf-8"))
    data["glass_catalogs"] = [str(ROOT / "data" / "glass" / "basis.json")]
    change(data)
    path = tmp_path / "system.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_catalog(tmp_path):
    path = tmp_path / "system.json"
    path.write_text(LINSE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(mikroptim.FormatError) as e:
        mikroptim.load_system(path)
    assert e.value.json_path == "glass_catalogs[0]"


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        mikroptim.load_system(SYSTEMS / "gibt_es_nicht.json")


def test_format_error_has_json_path(tmp_path):
    def unknown_glass(d):
        d["surfaces"][1]["material"] = "N-GIBTSNICHT"

    path = _variant(tmp_path, unknown_glass)
    with pytest.raises(mikroptim.FormatError) as e:
        mikroptim.load_system(path)
    assert e.value.json_path == "surfaces[1].material"
    assert "N-GIBTSNICHT" in e.value.message
    assert e.value.file is None or "system.json" in e.value.file
    assert "surfaces[1].material" in str(e.value)
    assert isinstance(e.value, ValueError)


def test_format_error_from_json():
    with pytest.raises(mikroptim.FormatError) as e:
        mikroptim.System.from_json('{"format": "mikroptim.system/1"}')
    assert e.value.json_path == "name"
    with pytest.raises(mikroptim.FormatError) as e:
        mikroptim.System.from_json("{kein json")
    assert e.value.json_path is None


def test_weights_mismatch(tmp_path):
    def mismatch(d):
        d["spectrum"]["weights"] = [1.0]

    with pytest.raises(mikroptim.FormatError) as e:
        mikroptim.load_system(_variant(tmp_path, mismatch))
    assert e.value.json_path.startswith("spectrum")


def test_glass_error_wavelength_out_of_range():
    s = mikroptim.load_system(LINSE)
    with pytest.raises(mikroptim.GlassError):
        mikroptim.paraxial(s, 5000.0)


def test_invalid_arguments():
    s = mikroptim.load_system(LINSE)
    with pytest.raises(ValueError, match="rings"):
        mikroptim.trace(s, rings=0)
    with pytest.raises(ValueError, match="spokes"):
        mikroptim.trace(s, spokes=0)
    with pytest.raises(ValueError, match="precision"):
        mikroptim.trace(s, precision="fp32")
    with pytest.raises(ValueError, match="df64"):
        mikroptim.trace(s, precision="df64")
    with pytest.raises(ValueError, match="reference"):
        mikroptim.evaluate(s, reference="irgendwo")
    with pytest.raises(ValueError):
        mikroptim.evaluate(s, grid=0)
    w = mikroptim.trace(s)
    with pytest.raises(IndexError):
        w.samples(3, 0)
    e = mikroptim.evaluate(s, grid=8)
    with pytest.raises(IndexError):
        e.psf(0, 5)
    with pytest.raises(ValueError, match="failure"):
        mikroptim.residuals_cpu(mikroptim.SystemBatch(s), failure="weich")
    with pytest.raises(ValueError):
        mikroptim.SystemBatch(s, 0)


def test_optimize_errors():
    s = mikroptim.load_system(LINSE)
    with pytest.raises(ValueError, match="keine Variablen"):
        mikroptim.optimize(s)
    b = mikroptim.load_system(SYSTEMS / "einzellinse_bk7_biegung.json")
    with pytest.raises(ValueError, match="jacobian"):
        mikroptim.optimize(b, jacobian="exakt")
    with pytest.raises(ValueError, match="fail_weight"):
        mikroptim.optimize(b, hard_failure=True, fail_weight=1.0)
    with pytest.raises(TypeError):
        mikroptim.optimize(b, callback=3)


def test_gpu_unavailable():
    if mikroptim.gpu_available():
        pytest.skip("GPU vorhanden")
    s = mikroptim.load_system(LINSE)
    for call in (
        lambda: mikroptim.trace(s, gpu=True),
        lambda: mikroptim.evaluate(s, gpu=True),
        lambda: mikroptim.optimize(mikroptim.load_system(SYSTEMS / "einzellinse_bk7_biegung.json"), gpu=True),
    ):
        with pytest.raises(mikroptim.GpuUnavailableError):
            call()
    assert issubclass(mikroptim.GpuUnavailableError, RuntimeError)


def test_invalid_device():
    if not mikroptim.gpu_available():
        pytest.skip("ohne GPU")
    with pytest.raises(ValueError, match="device"):
        mikroptim.trace(mikroptim.load_system(LINSE), gpu=True, device=99)


def test_cli_errors_in_module():
    code, out, err = mikroptim.run_cli(["trace", SYSTEMS / "gibt_es_nicht.json"])
    assert code == 1 and out == "" and "Fehler" in err
    code, _, err = mikroptim.run_cli(["trace", "--rings", "99", LINSE])
    assert code == 2
    code, _, err = mikroptim.run_cli(["unsinn"])
    assert code == 2 and "unbekannter Befehl" in err
