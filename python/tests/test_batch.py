"""SystemBatch (NumPy-Sichten), Residuen und Merit auf der CPU und über die TraceEngine."""

import numpy as np
import pytest

import mikroptim
from conftest import SYSTEMS, assert_close, requires_gpu, tol

MEHRLINSER = SYSTEMS / "testsystem_mehrlinser.json"
BIEGUNG = SYSTEMS / "einzellinse_bk7_biegung.json"


def _perturbed_batch(path, variants, scale=1e-3, seed=1):
    s = mikroptim.load_system(path)
    b = mikroptim.SystemBatch(s, variants)
    rng = np.random.default_rng(seed)
    c = b.curvatures
    c[1:] *= 1.0 + scale * rng.standard_normal(c[1:].shape)
    return s, b


def test_views_share_memory():
    s = mikroptim.load_system(MEHRLINSER)
    b = mikroptim.SystemBatch(s, 4)
    assert b.variants == 4
    assert b.curvatures.shape == (4, b.surfaces)
    assert b.indices.shape == (4, b.surfaces, b.wavelengths)
    assert b.field_targets.shape == (4, b.fields)
    assert b.curvatures.flags.writeable
    np.testing.assert_array_equal(b.curvatures[2], s.curvatures)
    b.curvatures[2, 1] = 0.5
    assert b.curvatures[2, 1] == 0.5  # Sicht: geändert im Batch
    view = b.thicknesses
    del b  # die Sicht hält den Batch am Leben
    view[0, 0] += 1.0
    assert view.shape[0] == 4


def test_copy_and_select():
    s = mikroptim.load_system(MEHRLINSER)
    b = mikroptim.SystemBatch(s, 3)
    b.curvatures[1, 1] = 0.01
    c = b.copy()
    c.curvatures[1, 1] = 0.02
    assert b.curvatures[1, 1] == 0.01
    one = b.select(1)
    assert one.variants == 1 and one.curvatures[0, 1] == 0.01


def test_variables_roundtrip():
    s = mikroptim.load_system(BIEGUNG)
    b = mikroptim.SystemBatch(s, 2)
    assert b.variable_names() == ["c[1]", "c[2]", "w"]
    x = b.get_variables()
    assert x.shape == (2, 3)
    x[1] = [0.0167, -0.0027, 96.5]
    b.set_variables(x)
    np.testing.assert_array_equal(b.get_variables(), x)
    assert b.curvatures[1, 1] == 0.0167 and b.thicknesses[1, -1] == 96.5
    with pytest.raises(ValueError):
        b.set_variables(np.zeros((3, 3)))


def test_residuals_cpu_match_trace():
    # Harte Auswertung einer aus dem System gebauten Variante: bitgleich zu evaluate_wavefront
    s = mikroptim.load_system(MEHRLINSER)
    b = mikroptim.SystemBatch(s, 2)
    r = mikroptim.residuals_cpu(b, failure="hard")
    w = mikroptim.trace(s)
    assert r["rms_mm"].shape == (2, w.fields, w.wavelengths)
    np.testing.assert_array_equal(r["rms_mm"][0], w.rms_mm)
    np.testing.assert_array_equal(r["rms_mm"][1], w.rms_mm)
    np.testing.assert_array_equal(r["failed"][0], w.failed)
    assert r["residuals"].shape == (2, r["stride"])
    assert r["penalty_rows"] == 0
    soft = mikroptim.residuals_cpu(b)
    assert soft["penalty_rows"] == soft["rays_per_variant"]


def test_merit_is_sum_of_squares():
    _, b = _perturbed_batch(MEHRLINSER, 5)
    r = mikroptim.residuals_cpu(b)
    m = mikroptim.merit_cpu(b)
    expected = np.sum(r["residuals"] ** 2, axis=1)
    assert_close(m["merit"], expected, 0.0, tol("kMeritVsResidualSum"))
    assert m["merit"].shape == (5,)
    assert m["failed"].dtype == np.int64


def test_engine_requires_gpu():
    if mikroptim.gpu_available():
        pytest.skip("GPU vorhanden")
    with pytest.raises(mikroptim.GpuUnavailableError):
        mikroptim.TraceEngine()


@requires_gpu
def test_engine_residuals_and_merit_match_cpu():
    _, b = _perturbed_batch(MEHRLINSER, 64)
    engine = mikroptim.TraceEngine(0)
    assert engine.device == 0 and engine.device_name
    gpu = engine.residuals(b)
    cpu = mikroptim.residuals_cpu(b)
    np.testing.assert_array_equal(gpu["status"], cpu["status"])
    n = cpu["rays_per_variant"]
    assert_close(gpu["opd"], cpu["opd"], tol("kCpuGpuFp64"), tol("kCpuGpuFp64Relative"))
    assert_close(
        gpu["residuals"][:, :n], cpu["residuals"][:, :n],
        tol("kCpuGpuFp64") / (b.reference_nm * 1e-6), tol("kCpuGpuFp64Relative"),
    )
    # MERIT gegen die Quadratsumme der RESIDUALS-Ausgabe desselben Geräts (M2-T3)
    merit_gpu = engine.merit(b)
    np.testing.assert_array_equal(merit_gpu["failed"], mikroptim.merit_cpu(b)["failed"])
    assert_close(
        merit_gpu["merit"], np.sum(gpu["residuals"] ** 2, axis=1), 0.0, tol("kMeritVsResidualSum")
    )
    # deterministisch: zweiter Lauf bitgleich
    np.testing.assert_array_equal(engine.merit(b)["merit"], merit_gpu["merit"])
    df64 = engine.merit(b, precision="df64")
    assert df64["merit"].shape == (64,)
    no_opd = engine.residuals(b, opd=False, status=False)
    assert no_opd["opd"] is None and no_opd["status"] is None
