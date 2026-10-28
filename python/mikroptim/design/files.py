"""Systemdateien des Kandidatenbaums: Laden, Schreiben, Pfade der Glaskataloge, Elemente."""

from __future__ import annotations

import json
import os
import pathlib


def read_system(path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_system(data: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_catalogs(data: dict, base_dir) -> dict:
    """Relative Einträge von ``glass_catalogs`` absolut machen (Suche ab ``base_dir``
    aufwärts, wie der Leser von Mikroptim), damit die Datei an anderer Stelle gültig bleibt."""
    out = dict(data)
    resolved = []
    for entry in data.get("glass_catalogs", []):
        p = pathlib.Path(entry)
        if p.is_absolute():
            resolved.append(str(p))
            continue
        found = None
        for d in [pathlib.Path(base_dir).resolve(), *pathlib.Path(base_dir).resolve().parents]:
            if (d / p).is_file():
                found = (d / p).resolve()
                break
        resolved.append(str(found) if found else entry)
    out["glass_catalogs"] = resolved
    return out


def is_air(material) -> bool:
    return material == "AIR"


def elements(data: dict) -> list[dict]:
    """Glaselemente: Stück zwischen Fläche s und s+1 mit Glas (nicht AIR) nach Fläche s.
    Gekittete Glieder bestehen aus mehreren aufeinanderfolgenden Elementen."""
    surfaces = data["surfaces"]
    out = []
    for s in range(len(surfaces) - 1):
        m = surfaces[s].get("material", "AIR")
        if not is_air(m):
            out.append({"element": len(out), "front": s, "back": s + 1,
                        "material": m if isinstance(m, str) else m.get("model", "?"),
                        "cemented_to_next": not is_air(surfaces[s + 1].get("material", "AIR"))})
    return out
