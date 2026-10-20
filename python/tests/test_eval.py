"""Auswertung (Modus EVAL) und PSF/MTF gegen `mikroptim eval --json`."""

import numpy as np
import pytest

import mikroptim
from conftest import SYSTEMS, assert_close, cli_json, requires_gpu, tol

LINSE = SYSTEMS / "einzellinse_bk7.json"
MEHRLINSER = SYSTEMS / "testsystem_mehrlinser.json"


def _nan(v):
    return np.nan if v is None else v


@pytest.mark.parametrize("path", [LINSE, MEHRLINSER], ids=lambda p: p.name)
def test_stats_equal_cli(path):
    e = mikroptim.evaluate(mikroptim.load_system(path), grid=32)
    expected = cli_json(["eval", path, "--grid", "32"])
    assert e.grid == expected["grid"] and e.grid_points == expected["grid_points"]
    assert e.failed_rays == expected["failed_rays"]
    lam = e.reference_nm * 1e-6
    stats = e.stats
    for t in expected["tuples"]:
        f, l = t["field"], t["wavelength"]
        assert stats["rms_opd"][f, l] == _nan(t["rms_opd_mm"])
        assert stats["pv_opd"][f, l] / lam == _nan(t["pv_opd_waves"])
        assert stats["rms_spot"][f, l] == _nan(t["rms_spot_mm"])
        assert stats["centroid_y"][f, l] == _nan(t["centroid_mm"][1])
        assert stats["focus_tangential"][f, l] == _nan(t["focus_tangential_mm"])
        assert stats["astigmatism"][f, l] == _nan(t["astigmatism_mm"])
        assert stats["image_height"][f, l] == _nan(t["image_height_mm"])
        assert stats["failed"][f, l] == t["failed_rays"]


def test_opd_map_and_spots():
    e = mikroptim.evaluate(mikroptim.load_system(LINSE), grid=16)
    m = e.opd_map(1, 1)
    assert m.shape == (16, 16)
    assert np.isnan(m[0, 0])  # Ecke außerhalb der Pupille
    points = e.pupil_points()
    assert points.shape == (e.grid_points, 3)
    assert abs(points[:, 2].sum() - 1.0) <= tol("kEvalWeightSum")
    # RMS der Karte (gleiche Gewichte in der vollen Pupille) = rms_opd
    valid = m[~np.isnan(m)]
    assert valid.size == e.grid_points
    rms = np.sqrt(np.mean(valid**2))
    assert abs(rms - e.stats["rms_opd"][1, 1]) <= tol("kLengthFp64")
    tilt = e.opd_map(1, 1, remove="tilt")
    assert np.nanstd(tilt) <= np.nanstd(m) + tol("kLengthFp64")
    spots = e.spots(1, 1)
    assert spots["x"].shape == (e.grid_points,)
    assert np.all(spots["status"] == 0)
    with pytest.raises(ValueError):
        e.opd_map(0, 0, remove="zernike")


def test_psf_equals_cli():
    e = mikroptim.evaluate(mikroptim.load_system(LINSE), grid=32)
    expected = cli_json(["eval", LINSE, "--grid", "32", "--psf"])
    for t in expected["tuples"]:
        p = e.psf(t["field"], t["wavelength"])
        q = t["psf"]
        assert p.fft_size == expected["psf"]["fft_size"]
        assert p.strehl == q["strehl"]
        assert p.strehl_fft == q["strehl_fft"]
        assert [p.dx, p.dy] == q["pixel_mm"]
        assert p.airy_radius == q["airy_radius_mm"]
        assert p.psf.shape == (p.fft_size, p.fft_size)
        assert p.mtf_sagittal.shape == (p.grid + 1,)
        assert p.mtf_sagittal[0] == pytest.approx(1.0)
        assert p.frequencies()[-1] == 1.0


@requires_gpu
def test_gpu_eval_and_psf():
    s = mikroptim.load_system(MEHRLINSER)
    cpu = mikroptim.evaluate(s, grid=32)
    gpu = mikroptim.evaluate(s, grid=32, gpu=True)
    assert gpu.backend == "gpu"
    for key in ("rms_opd", "pv_opd", "rms_spot"):
        assert_close(gpu.stats[key], cpu.stats[key], tol("kCpuGpuFp64"), tol("kCpuGpuFp64Relative"))
    np.testing.assert_array_equal(gpu.stats["failed"], cpu.stats["failed"])
    a = cpu.psf(0, 0)
    b = cpu.psf(0, 0, gpu=True)
    assert b.backend == "gpu"
    assert np.max(np.abs(a.psf - b.psf)) < tol("kPsfCpuGpu")
    assert abs(a.strehl - b.strehl) < tol("kPsfCpuGpu")
