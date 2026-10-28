"""Antwortschema für die schema-beschränkte Ausgabe des Sprachmodells.

Teilmenge von ``data/design/schemas/design_action.schema.json`` mit den umgesetzten Aktionen
(``actions.IMPLEMENTED``); die Definitionen der einzelnen Aktionen sind dort identisch
(Test ``test_embedded_schema_matches_repository``). Eingebettet, weil das installierte Paket
die Dateien des Repositorys nicht mitbringt.
"""

from __future__ import annotations

_REASON = {"type": "string", "maxLength": 300}
_SURFACE = {"type": "integer", "minimum": 0, "maximum": 63}
_RANGE = {"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}],
          "minItems": 2, "maxItems": 2}
_GLASS = {"type": "object", "additionalProperties": False, "required": ["name"],
          "properties": {"name": {"type": "string"}}}

ACTION_DEFS = {
    "get_surface_table": {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"const": "get_surface_table"},
                       "quantities": {"type": "array", "items": {"enum": [
                           "paraxial", "seidel", "full_sa", "zonal_sa", "sphchrom", "extrema"]}},
                       "surfaces": _RANGE, "reason": _REASON}},
    "get_ghost_diagnosis": {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"const": "get_ghost_diagnosis"}, "reason": _REASON}},
    "evaluate": {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"const": "evaluate"}, "grid": {"enum": [64, 128, 256]},
                       "psf": {"type": "boolean"}, "reference": {"enum": ["exit_pupil", "vertex"]},
                       "reason": _REASON}},
    "optimize": {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"const": "optimize"},
                       "cycles": {"type": "integer", "minimum": 1, "maximum": 2000},
                       "variables": {"enum": ["current", "all_curvatures", "curvatures_and_gaps",
                                              "everything"]},
                       "jacobian": {"enum": ["fd", "adjoint"]}, "reason": _REASON}},
    "synth_segment": {
        "type": "object", "additionalProperties": False, "required": ["action", "target"],
        "properties": {"action": {"const": "synth_segment"},
                       "target": {"type": "object", "additionalProperties": False, "properties": {
                           "aperture": {"type": "number", "minimum": 0.01, "maximum": 0.99},
                           "field": {"type": "number", "minimum": 0, "maximum": 1},
                           "spectrum_stage": {"type": "integer", "minimum": 0}}},
                       "beam_width": {"type": "integer", "minimum": 1, "maximum": 8},
                       "alternatives": {"type": "integer", "minimum": 1, "maximum": 4},
                       "max_new_elements": {"type": "integer", "minimum": 0, "maximum": 4},
                       "reason": _REASON}},
    "snap_glasses": {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"const": "snap_glasses"},
                       "k": {"type": "integer", "minimum": 1, "maximum": 6},
                       "max_combinations": {"type": "integer", "minimum": 1, "maximum": 512},
                       "reason": _REASON}},
    "free_gaps": {
        "type": "object", "additionalProperties": False, "required": ["action"],
        "properties": {"action": {"const": "free_gaps"},
                       "surfaces": {"oneOf": [{"const": "all"},
                                              {"type": "array", "items": _SURFACE}]},
                       "include_glass_thickness": {"type": "boolean"},
                       "max_gap": {"type": "number", "minimum": 0.1, "maximum": 40},
                       "reason": _REASON}},
    "insert_null_meniscus": {
        "type": "object", "additionalProperties": False,
        "required": ["action", "after_surface", "glass"],
        "properties": {"action": {"const": "insert_null_meniscus"}, "after_surface": _SURFACE,
                       "position": {"type": "number", "minimum": 0}, "glass": _GLASS,
                       "curvature": {"oneOf": [{"type": "number"}, {
                           "type": "object", "additionalProperties": False,
                           "required": ["from_surface"], "properties": {"from_surface": _SURFACE}}]},
                       "saddle_branches": {"type": "boolean"},
                       "group": {"enum": ["front", "middle", "rear"]}, "reason": _REASON}},
    "split_lens": {
        "type": "object", "additionalProperties": False, "required": ["action", "element"],
        "properties": {"action": {"const": "split_lens"},
                       "element": {"type": "integer", "minimum": 0},
                       "fraction": {"type": "number", "minimum": 0.1, "maximum": 0.9},
                       "cement_curvature": {"oneOf": [{"enum": ["achromat", "mean"]},
                                                      {"type": "number"}]},
                       "free_part": {"enum": [0, 1]}, "target_glass": _GLASS, "reason": _REASON}},
    "abandon": {
        "type": "object", "additionalProperties": False, "required": ["action", "reason"],
        "properties": {"action": {"const": "abandon"}, "reason": _REASON}},
    "finish": {
        "type": "object", "additionalProperties": False, "required": ["action", "reason"],
        "properties": {"action": {"const": "finish"}, "reason": _REASON}},
}

#: Lese-Werkzeuge, die das Modell vor seiner Antwort aufrufen darf (Tool-Calls)
TOOL_ACTIONS = ("get_surface_table", "get_ghost_diagnosis", "evaluate")


def response_schema() -> dict:
    """JSON-Schema einer Antwort mit den umgesetzten Aktionen (ohne ``branch``)."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["format", "candidate", "assessment", "actions"],
        "properties": {
            "format": {"const": "mikroptim.design.action/1"},
            "candidate": {"type": "integer", "minimum": 0},
            "assessment": {"type": "string", "maxLength": 600},
            "actions": {"type": "array", "minItems": 1, "maxItems": 4,
                        "items": {"oneOf": [ACTION_DEFS[k] for k in ACTION_DEFS
                                            if k not in TOOL_ACTIONS]}},
        },
    }


def tool_specs() -> list[dict]:
    """Lese-Werkzeuge als Funktionen im Format der Chat-API (Ollama, OpenAI-kompatibel)."""
    desc = {
        "get_surface_table": "Aberrationsbericht pro Fläche (paraxiale Strahlen, Seidel-Beiträge, "
                             "Farbfehler, Sphärochromasie; Wellen nach Welford)",
        "get_ghost_diagnosis": "Diagnose der Geisterplatten (aktive Platten, Ordnungen, Gewinn)",
        "evaluate": "Auswertung auf dem dichten Pupillenraster (RMS, Strehl, optional PSF)",
    }
    out = []
    for name in TOOL_ACTIONS:
        params = {k: v for k, v in ACTION_DEFS[name]["properties"].items() if k != "action"}
        out.append({"type": "function", "function": {
            "name": name, "description": desc[name],
            "parameters": {"type": "object", "properties": params}}})
    return out
