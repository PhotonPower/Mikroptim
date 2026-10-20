"""Gemeinsame Hilfen der pytest-Suite der Python-Bindings (M9-T1).

- Toleranzen kommen aus tests/tolerances.hpp (AGENTS.md, Abschnitt 6.5), hier eingelesen.
- Vergleiche mit der CLI rufen das Programm aus der Umgebungsvariable MIKROPTIM_CLI auf (ctest
  setzt sie auf das Programm desselben Builds); ohne sie dieselben Unterbefehle im Modul
  (mikroptim.run_cli, gleicher Code wie src/cli/main.cpp).
- GPU-Tests werden ohne CUDA-Build oder ohne Gerät übersprungen.
"""

import json
import os
import pathlib
import re
import subprocess

import pytest

import mikroptim

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYSTEMS = ROOT / "data" / "systems"


def _read_tolerances():
    text = (ROOT / "tests" / "tolerances.hpp").read_text(encoding="utf-8")
    values = {}
    for name, value in re.findall(
        r"inline constexpr double (k\w+) = ([0-9.eE+\-]+);", text
    ):
        values[name] = float(value)
    return values


TOLERANCES = _read_tolerances()


def tol(name):
    """Toleranz `name` aus tests/tolerances.hpp."""
    return TOLERANCES[name]


def system_files():
    return sorted(SYSTEMS.glob("*.json"))


def run_cli(args):
    """CLI-Aufruf `mikroptim <args>`; Rückgabe (exit_code, stdout, stderr)."""
    args = [os.fspath(a) for a in args]
    exe = os.environ.get("MIKROPTIM_CLI")
    if exe:
        p = subprocess.run(
            [os.path.abspath(exe), *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return p.returncode, p.stdout, p.stderr
    return mikroptim.run_cli(args)


def cli_json(args):
    code, out, err = run_cli([*args, "--json"])
    assert code == 0, err
    return json.loads(out)


requires_gpu = pytest.mark.skipif(
    not mikroptim.gpu_available(), reason="ohne CUDA-Build oder ohne CUDA-Gerät"
)


def assert_close(actual, expected, absolute, relative=0.0):
    """|a − e| <= max(absolute, relative·|e|) elementweise, NaN an denselben Stellen."""
    import numpy as np

    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert a.shape == e.shape
    assert np.array_equal(np.isnan(a), np.isnan(e))
    ok = ~np.isnan(e)
    bound = np.maximum(absolute, relative * np.abs(e[ok]))
    diff = np.abs(a[ok] - e[ok])
    assert np.all(diff <= bound), f"größte Abweichung {diff.max() if diff.size else 0}"
