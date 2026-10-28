"""Aufruf der Mikroptim-Kommandozeile im Prozess (``mikroptim.run_cli``) mit JSON-Ausgabe."""

from __future__ import annotations

import json

from .. import run_cli


class ToolError(RuntimeError):
    """Ein Werkzeug (Unterbefehl der CLI) ist fehlgeschlagen; Meldung der CLI."""


def cli_json(args: list[str]) -> dict:
    """Führt ``mikroptim <args> --json`` aus und liefert die JSON-Ausgabe."""
    code, out, err = run_cli([str(a) for a in args] + ["--json"])
    if code != 0:
        raise ToolError(err.strip() or f"Exit-Code {code}")
    return json.loads(out)


def cli_text(args: list[str]) -> str:
    code, out, err = run_cli([str(a) for a in args])
    if code != 0:
        raise ToolError(err.strip() or f"Exit-Code {code}")
    return out
