"""Wellenfront (trace) gegen `mikroptim trace --json` auf CPU und GPU."""

import numpy as np
import pytest

import mikroptim
from conftest import SYSTEMS, assert_close, cli_json, requires_gpu, system_files, tol


def _cli_arrays(expected, fields, wavelengths):
    def grid(key, dtype=float):
        values = [np.nan if e[key] is None else e[key] for e in expected["wavefront"]]
        return np.array(values, dtype=dtype).reshape(fields, wavelengths)

    return grid


def _traceable(path):
    s = mikroptim.load_system(path)
    try:
        mikroptim.trace(s, rings=1, spokes=1)
    except ValueError:
        return False
    return True


TRACEABLE = [p for p in system_files() if _traceable(p)]


@pytest.mark.parametrize("path", TRACEABLE, ids=lambda p: p.name)
@pytest.mark.parametrize("rings,spokes", [(10, 8), (6, 3)])
def test_wavefront_equals_cli(path, rings, spokes):
    s = mikroptim.load_system(path)
    w = mikroptim.trace(s, rings=rings, spokes=spokes)
    expected = cli_json(["trace", path, "--rings", str(rings), "--spokes", str(spokes)])
    assert w.backend == "cpu" and w.precision == "fp64" and w.device is None
    assert w.rms_waves.shape == (w.fields, w.wavelengths)
    grid = _cli_arrays(expected, w.fields, w.wavelengths)
    # Gleicher Code, verlustfreies JSON: bitgleich (NaN an denselben Stellen)
    np.testing.assert_array_equal(w.rms_waves, grid("rms_waves"))
    np.testing.assert_array_equal(w.rms_mm, grid("rms_mm"))
    np.testing.assert_array_equal(w.piston_mm, grid("piston_mm"))
    np.testing.assert_array_equal(w.failed, grid("failed_rays", np.int64))
    np.testing.assert_array_equal(w.rays, grid("rays", np.int64))
    assert int(w.rays.sum()) == expected["rays"]
    assert int(w.failed.sum()) == expected["failed_rays"]
    points = [e["image_point"] for e in expected["wavefront"][:: w.wavelengths]]
    np.testing.assert_array_equal(w.image_points, np.array(points))
    assert w.pupil_modes == [e["pupil_mode"] for e in expected["wavefront"][:: w.wavelengths]]
    np.testing.assert_array_equal(w.wavelengths_nm, expected["wavelengths_nm"])
    if expected["target_focal_length"] is None:
        assert np.isnan(w.target_focal_length)
    else:
        assert w.target_focal_length == expected["target_focal_length"]


def test_samples_reproduce_rms():
    s = mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json")
    w = mikroptim.trace(s)
    for f in range(w.fields):
        for l in range(w.wavelengths):
            d = w.samples(f, l)
            assert d["opd_mm"].shape == (w.rays[f, l],)
            assert np.all(d["status"] == 0)
            # RMS = √(Σ W OPD² / Σ W), Σ W = 1 ohne Versagen
            rms = np.sqrt(np.sum(d["weight"] * d["opd_mm"] ** 2) / np.sum(d["weight"]))
            assert abs(rms - w.rms_mm[f, l]) <= tol("kLengthFp64")
            assert np.allclose(d["opd_waves"], d["opd_mm"] / (w.reference_nm * 1e-6))


def test_results_are_readonly_views():
    w = mikroptim.trace(mikroptim.load_system(SYSTEMS / "einzellinse_bk7.json"))
    a = w.rms_waves
    assert not a.flags.writeable
    del w  # die Sicht hält das Ergebnis am Leben
    assert a.shape == (3, 3) and np.isfinite(a).all()


def test_afocal_without_target_raises_like_cli():
    path = SYSTEMS / "planplatte_bk7.json"
    expected = cli_json(["trace", path])
    assert expected["wavefront"] is None
    with pytest.raises(ValueError) as e:
        mikroptim.trace(mikroptim.load_system(path))
    assert str(e.value) == expected["wavefront_error"]


@requires_gpu
@pytest.mark.parametrize("path", TRACEABLE, ids=lambda p: p.name)
def test_gpu_wavefront(path):
    s = mikroptim.load_system(path)
    cpu = mikroptim.trace(s)
    gpu = mikroptim.trace(s, gpu=True)
    assert gpu.backend == "gpu" and gpu.device == 0
    assert np.isnan(gpu.piston_mm).all()
    np.testing.assert_array_equal(gpu.failed, cpu.failed)
    assert_close(gpu.rms_mm, cpu.rms_mm, tol("kCpuGpuFp64"), tol("kCpuGpuFp64Relative"))
    # gegen die CLI mit --gpu: gleicher Kernel, gleiche Eingaben
    expected = cli_json(["trace", path, "--gpu"])
    grid = _cli_arrays(expected, gpu.fields, gpu.wavelengths)
    np.testing.assert_array_equal(gpu.rms_mm, grid("rms_mm"))
    np.testing.assert_array_equal(gpu.rms_waves, grid("rms_waves"))
    for f in range(gpu.fields):
        for l in range(gpu.wavelengths):
            a, b = gpu.samples(f, l), cpu.samples(f, l)
            np.testing.assert_array_equal(a["status"], b["status"])
            np.testing.assert_array_equal(a["weight"], b["weight"])
            assert_close(a["opd_mm"], b["opd_mm"], tol("kCpuGpuFp64"), tol("kCpuGpuFp64Relative"))


@requires_gpu
def test_gpu_df64():
    path = SYSTEMS / "testsystem_mehrlinser.json"
    s = mikroptim.load_system(path)
    fp64 = mikroptim.trace(s, gpu=True)
    df64 = mikroptim.trace(s, gpu=True, precision="df64")
    assert df64.precision == "df64"
    assert_close(df64.rms_mm, fp64.rms_mm, tol("kDf64VsFp64"))
    expected = cli_json(["trace", path, "--gpu", "--precision", "df64"])
    grid = _cli_arrays(expected, df64.fields, df64.wavelengths)
    np.testing.assert_array_equal(df64.rms_mm, grid("rms_mm"))
