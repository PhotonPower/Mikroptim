"""Mikroptim: Raytracing, Auswertung und Optimierung von Mikroskopobjektiven.

Python-Bindings des C++-Kerns (M9-T1, nanobind). Überblick, Kopien und Sichten:
python/README.md.

Beispiel::

    import mikroptim as mko
    system = mko.load_system("data/systems/einzellinse_bk7.json")
    print(mko.paraxial(system).focal_length)
    wf = mko.trace(system, rings=10, spokes=8)
    print(wf.rms_waves)            # NumPy-Array (Felder, Wellenlängen)
"""

import json as _json
import os as _os

from . import _build_info


def _add_cuda_dll_directory():
    # Python >= 3.8 sucht unter Windows abhängige DLLs nicht mehr über PATH. Ein CUDA-Build
    # braucht cudart und cuFFT aus dem Toolkit, mit dem er gebaut wurde.
    if _os.name != "nt" or not _build_info.HAS_CUDA:
        return
    candidates = [_build_info.CUDA_BIN_DIR, _os.environ.get("MIKROPTIM_CUDA_BIN_DIR", "")]
    for directory in candidates:
        if directory and _os.path.isdir(directory):
            _os.add_dll_directory(directory)


_add_cuda_dll_directory()

from ._mikroptim import (  # noqa: E402  (nach dem DLL-Verzeichnis)
    CudaError,
    Evaluation,
    FormatError,
    GlassError,
    GpuUnavailableError,
    OptimizeResult,
    Paraxial,
    Psf,
    System,
    SystemBatch,
    TraceEngine,
    Wavefront,
    __version__,
    device_count,
    devices,
    evaluate,
    has_cuda,
    merit_cpu,
    optimize,
    paraxial,
    residuals_cpu,
    trace,
)
from ._mikroptim import load_system as _load_system  # noqa: E402

__all__ = [
    "aberrations",
    "CudaError",
    "Evaluation",
    "FormatError",
    "GlassError",
    "GpuUnavailableError",
    "OptimizeResult",
    "Paraxial",
    "Psf",
    "System",
    "SystemBatch",
    "TraceEngine",
    "Wavefront",
    "__version__",
    "device_count",
    "devices",
    "evaluate",
    "gpu_available",
    "has_cuda",
    "load_system",
    "merit_cpu",
    "optimize",
    "paraxial",
    "residuals_cpu",
    "run_cli",
    "system_from_dict",
    "system_to_dict",
    "trace",
]


def load_system(path):
    """Lädt eine Systemdatei ``mikroptim.system/1`` samt Glaskatalogen.

    Fehler: ``FileNotFoundError`` (Datei fehlt), ``FormatError`` (Inhalt, mit ``json_path``),
    ``GlassError`` (Glasdaten, Brechzahl außerhalb des Bereichs).
    """
    path = _os.fspath(path)
    if not _os.path.isfile(path):
        raise FileNotFoundError(f"Systemdatei nicht gefunden: {path}")
    return _load_system(path)


def system_to_dict(system):
    """Systembeschreibung als ``dict`` (über ``System.to_json``)."""
    return _json.loads(system.to_json())


def system_from_dict(data, base_dir="."):
    """System aus einem ``dict`` im Aufbau von ``mikroptim.system/1``.

    Relative Glaskataloge werden ab ``base_dir`` gesucht (und in dessen Elternverzeichnissen).
    """
    return System.from_json(_json.dumps(data), _os.fspath(base_dir))


def gpu_available():
    """``True``, wenn das Modul mit CUDA gebaut ist und ein CUDA-Gerät sichtbar ist."""
    return has_cuda and device_count() > 0


def aberrations(path, field_angle_deg=None, reference_nm=None, short_nm=None, long_nm=None):
    """Aberrationsbericht pro Fläche (M10-T1) als ``dict`` im Format ``mikroptim.aberrations/1``.

    Paraxiale Rand- und Hauptstrahlen, Seidel-Beiträge S_I … S_V, C_I, C_II und
    Sphärochromasie pro Dateifläche, dazu Summen und Wellenfrontkoeffizienten in Wellen
    (docs/numerik.md, Abschnitt 14). Fehler: ``RuntimeError`` mit der Meldung der CLI.
    """
    args = ["aberrations", _os.fspath(path), "--json"]
    for option, value in (("--field-deg", field_angle_deg), ("--reference-nm", reference_nm),
                          ("--short-nm", short_nm), ("--long-nm", long_nm)):
        if value is not None:
            args += [option, repr(float(value))]
    code, out, err = run_cli(args)
    if code != 0:
        raise RuntimeError(err.strip())
    return _json.loads(out)


def run_cli(args):
    """Führt einen Unterbefehl der CLI im Prozess aus, z. B. ``run_cli(["trace", "a.json"])``.

    Rückgabe: ``(exit_code, stdout, stderr)`` wie das Programm ``mikroptim`` (docs/cli.md).
    """
    from ._mikroptim import _run_cli

    return _run_cli([_os.fspath(a) for a in args])
