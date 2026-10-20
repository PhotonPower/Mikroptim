"""Optimierung gegen `mikroptim optimize --json` (gleicher Ablauf, gleiche Ergebnisse)."""

import numpy as np
import pytest

import mikroptim
from conftest import SYSTEMS, cli_json, requires_gpu, run_cli

BIEGUNG = SYSTEMS / "einzellinse_bk7_biegung.json"


def _nan(v):
    return np.nan if v is None else v


def _compare_with_cli(result, expected):
    r = expected["result"]
    assert result.status == r["status"]
    assert result.converged == r["converged"]
    assert result.iterations == r["iterations"]
    assert result.accepted == r["accepted"]
    assert result.jacobians == r["jacobians"]
    # Gleicher Code und verlustfreies JSON: bitgleich
    assert result.merit == r["merit"]
    assert result.start_merit == r["start_merit"]
    assert result.failed_rays == r["failed_rays"]
    assert result.start_failed_rays == r["start_failed_rays"]
    assert result.hard_failed_rays == r["hard_failed_rays"]
    assert result.penalty_rounds == r["penalty_rounds"]
    assert result.best_run == expected["best_run"]
    names = [v["name"] for v in expected["variables"]]
    assert result.variable_names == names
    np.testing.assert_array_equal(result.x0, [v["start"] for v in expected["variables"]])
    np.testing.assert_array_equal(result.x, [v["result"] for v in expected["variables"]])
    runs = result.runs
    assert len(runs) == len(expected["runs"])
    for mine, theirs in zip(runs, expected["runs"]):
        assert mine["status"] == theirs["status"]
        assert mine["merit"] == theirs["merit"]
        np.testing.assert_array_equal(mine["x"], theirs["x"])
        np.testing.assert_array_equal(mine["start"], theirs["start"])
    log = result.log
    its = expected["iterations"]
    np.testing.assert_array_equal(log["run"], [e["run"] for e in its])
    np.testing.assert_array_equal(log["iteration"], [e["iteration"] for e in its])
    np.testing.assert_array_equal(log["merit"], [_nan(e["merit"]) for e in its])
    np.testing.assert_array_equal(log["trial_merit"], [_nan(e["trial_merit"]) for e in its])
    np.testing.assert_array_equal(log["accepted"], [e["accepted"] for e in its])
    assert log["status"] == [e["status"] for e in its]


def test_bending_equals_cli(tmp_path):
    out = tmp_path / "cli_ergebnis.json"
    expected = cli_json(["optimize", BIEGUNG, "-o", out])
    result = mikroptim.optimize(mikroptim.load_system(BIEGUNG))
    _compare_with_cli(result, expected)
    assert result.backend == "cpu"
    # Verlauf: [0] am Start, danach ein Eintrag pro Zyklus (ein Lauf, harte Runde)
    assert result.history[0] == result.start_merit
    assert result.history[-1] == result.merit
    assert len(result.history) == result.iterations + 1
    np.testing.assert_array_equal(result.history[1:], result.log["merit"])
    # Ergebnissystem gleich der Ergebnisdatei der CLI
    cli_system = mikroptim.load_system(out)
    np.testing.assert_array_equal(result.system.curvatures, cli_system.curvatures)
    np.testing.assert_array_equal(result.system.thicknesses, cli_system.thicknesses)
    saved = tmp_path / "py_ergebnis.json"
    result.system.save(saved)
    assert saved.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
    # Ergebnis: Biegung minimaler sphärischer Aberration, deutlich kleinere Merit
    assert result.converged
    assert result.merit < 1e-3 * result.start_merit


def test_multistart_equals_cli():
    args = ["--runs", "3", "--seed", "7", "--perturbation", "0.02", "--max-iterations", "20"]
    expected = cli_json(["optimize", BIEGUNG, *args])
    result = mikroptim.optimize(
        mikroptim.load_system(BIEGUNG), runs=3, seed=7, perturbation=0.02, max_iterations=20
    )
    _compare_with_cli(result, expected)


def test_hard_failure_equals_cli():
    expected = cli_json(["optimize", BIEGUNG, "--hard-failure", "--rings", "6", "--spokes", "4"])
    result = mikroptim.optimize(
        mikroptim.load_system(BIEGUNG), hard_failure=True, rings=6, spokes=4
    )
    _compare_with_cli(result, expected)
    assert np.isnan(result.fail_weight)


def test_adjoint_equals_cli():
    expected = cli_json(["optimize", BIEGUNG, "--jacobian", "adjoint"])
    assert expected["settings"]["jacobian"] == "adjoint"
    result = mikroptim.optimize(mikroptim.load_system(BIEGUNG), jacobian="adjoint")
    assert result.jacobian == "adjoint"
    _compare_with_cli(result, expected)
    fd = mikroptim.optimize(mikroptim.load_system(BIEGUNG))
    assert fd.jacobian == "fd"
    # gleiches Optimum wie mit finiten Differenzen (bis auf die Abbruchgenauigkeit)
    assert result.converged and abs(result.merit - fd.merit) <= 1e-6 * fd.merit


def test_callback():
    seen = []
    result = mikroptim.optimize(
        mikroptim.load_system(BIEGUNG), max_iterations=5, callback=seen.append
    )
    assert len(seen) == len(result.log["merit"]) == 5
    assert [e["merit"] for e in seen] == list(result.log["merit"])
    assert seen[-1]["x"].shape == (3,)


def test_callback_exception_aborts():
    def stop(info):
        if info["iteration"] == 2:
            raise KeyboardInterrupt("Abbruch im Rückruf")

    with pytest.raises(KeyboardInterrupt, match="Abbruch im Rückruf"):
        mikroptim.optimize(mikroptim.load_system(BIEGUNG), callback=stop)


def test_zero_iterations_only_evaluates():
    result = mikroptim.optimize(mikroptim.load_system(BIEGUNG), max_iterations=0)
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, result.x0)
    assert result.merit == result.start_merit


@requires_gpu
@pytest.mark.parametrize("jacobian", ["fd", "adjoint"])
def test_gpu_equals_cli_gpu(jacobian):
    expected = cli_json(["optimize", BIEGUNG, "--gpu", "--jacobian", jacobian])
    result = mikroptim.optimize(mikroptim.load_system(BIEGUNG), gpu=True, jacobian=jacobian)
    assert result.backend == "gpu"
    _compare_with_cli(result, expected)


def test_run_cli_text_mode():
    code, out, err = run_cli(["optimize", BIEGUNG, "--quiet", "--max-iterations", "3"])
    assert code == 0, err
    assert "Ergebnis" in out
