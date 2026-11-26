#!/usr/bin/env python3
"""Bewertung der Entwurfssteuerung (M10-T7, docs/entwurfssteuerung.md, Abschnitt 8).

Benchmarks (Startsysteme frei erfunden):

    B1  NA 0,3 → 0,6 auf der Achse, monochromatisch (Start von M6-T3)
    B2  Zielspezifikation 50x/0,95, Aperturteil: NA 0,3 → 0,95 bei 10 % Feld, 546 nm
        (Start von M6-T6 mit Feldwinkeln × 0,1; Feld und Spektrum können die Entwurfswerkzeuge
        noch nicht stufenweise ändern, bei vollem Feld wird schon NA 0,3 verworfen)
    B3  40x/0,85 trocken, Aperturteil: NA 0,3 → 0,85 bei 10 % Feld, 546 nm
    B4  Patentsysteme aus dem privaten Verzeichnis ``data/systems/private/`` – nur lokal, nur
        mit Ollama (Entscheidung Maintainer 2026-11-19). Die Liste steht in
        ``data/systems/private/b4.json`` (Format: data/systems/README.md); fehlt sie, wird B4
        übersprungen. Sitzungen unter ``<work>/private/``, Ergebnisse unter ``<out>/private/``
        (beides nicht im Repository); ClaudeDecider ist für B4 gesperrt.

Varianten:

    V0  run_homotopy:            mikroptim synth --target-na NA
    V1  Strahlsuche:             mikroptim synth --beam 3 --alternatives 2
    V2  M6-T6-Skript:            nur B2; tools/m6t6_zielspezifikation.py (eigener Lauf, lang)
    V3  regelbasierte Strategie: mikroptim.design mit RuleDecider (ohne Modell); seit M10-T7b
        mit Stufen bei Stillstand (Spalte, aplanatische Schale, Wörterbuch, Null-Meniskus)
    V3AL wie V3, Homotopie-Abschnitte mit --augmented-lagrangian und --sine-condition (M10-T7b)
    LM  Sprachmodell:            mikroptim.design mit OllamaDecider oder ClaudeDecider
                                 (--decider ollama|claude, --model, Rückfall auf die Regel)
    V4  Mensch:                  mikroptim.design mit HumanDecider (Eingabe auf der Konsole)

Referenzsphäre (Entscheidung Maintainer 2026-11-20, M10-T7c): Alle Varianten rechnen
standardmäßig mit ``reference_sphere: image_infinite`` im Start (ADR-0017; ``--referenzsphaere``
wählt eine andere Art). Die Startsysteme unter ``data/systems/`` bleiben unverändert, das Feld
setzt das Skript in der Kopie im Arbeitsverzeichnis. Das Ergebnis-Kürzel trägt die Konvention:
``RS`` = ``image_infinite``, ``RSAP`` = ``image_exit_pupil``, ohne Zusatz = alte Konvention
``last_surface`` (Startdatei ohne Feld, bitgleich zu M10-T7/M10-T7b). Also:

    V3       → Ergebnis ``V3RS`` (Referenz der Regel), entsprechend V0RS, V1RS, V3ALRS, LMRS_<modell>
    V3-alt   → Ergebnis ``V3`` (alte Konvention wie M10-T7b), entsprechend V0-alt, LM-alt, …
    V0RS, V3RS  wie V0 bzw. V3 mit image_infinite unabhängig von --referenzsphaere (Namen aus
        M10-T7b)

Bewertung mit trace in der Konvention des Starts, RMS AP wie immer durch die Austrittspupille.

Alle Läufe rechnen auf der CPU im Prozess (``mikroptim.run_cli``), damit Trace und lokales
Sprachmodell nicht um die GPU konkurrieren. Ergebnisse pro Lauf als JSON unter --out
(``<B>_<V>[_<modell>].json``), Sitzungen unter ``--work``; ein vorhandenes Ergebnis wird
übersprungen. ``--tabelle`` druckt die Tabelle des Berichts aus den Ergebnisdateien.

Bewertet wird jedes Endsystem gleich: Geisterplatten-Koeffizienten auf 0 (reales System, wie die
Zielprüfung von synth), ``mikroptim trace`` (10 × 8), größter RMS über Feld und Wellenlänge,
versagte Strahlen, NA = Pupillenradius / Brennweite, Elementzahl. Ziel erreicht heißt
NA ≥ Soll-NA, RMS ≤ 1/14 Welle und keine versagten Strahlen.

Beispiele:

    python tools/m10t7_bewertung.py --varianten V0,V1,V3,V3AL
    python tools/m10t7_bewertung.py --varianten LM --decider ollama --model qwen3:14b --no-think
    python tools/m10t7_bewertung.py --varianten LM --decider ollama --model gpt-oss:20b --think
    python tools/m10t7_bewertung.py --varianten LM --decider claude --model claude-opus-5-5 --max-usd 20
    python tools/m10t7_bewertung.py --tabelle
    python tools/m10t7_bewertung.py --benchmarks B4 --varianten V0,LM --decider ollama --model qwen3:14b
    python tools/m10t7_bewertung.py --vergleich docs/ergebnisse/m10-t7b  # alt gegen neu
    python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7d --varianten LM --decider claude         --model claude-haiku-5-5 --wiederholung 1-3     # Kürzel LMRS_<modell>_w1 … _w3
    python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7d --zusammenfassung  # x/n, Median
    python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7b --varianten V0-alt,V3-alt,V3RS
    python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7e --varianten V3RS \
        --gestoerte-starts 5 --stoer-sigma 0.02         # Kürzel V3RS_s1 … _s5 (M10-T7e)

Gestörte Starts (M10-T7e, Entscheidung Maintainer 2026-11-20): ``--gestoerte-starts N`` rechnet
jede Variante zusätzlich auf N leicht gestörten Kopien des Starts (Kürzel ``<tag>_s<k>``). Gestört
werden nur Startwerte von Variablen der Datei: Radien r → r·(1 + σ ξ) (also c → c/(1 + σ ξ)) und
Dicken bzw. Arbeitsabstand t → t·(1 + σ ξ), ξ standardnormal, danach auf die Schranken der
Variable begrenzt (Dicken nicht unter die Lage einer Geisterplatte im Spalt); deterministisch aus
``--stoer-seed``, Benchmark und k. Die Zusammenfassung (``--zusammenfassung``) fasst ``_s<k>`` wie
Wiederholungen zusammen (Treffer x/N, Median, Spanne).

Referenz der Regel (M6-T11b, Entscheidung Maintainer 2026-11-22): der **Median über fünf gestörte
Starts** (``V3RS_s1`` … ``_s5``, σ = 2 %), nicht der Einzellauf (die Regel ist stark
pfadabhängig, Bericht Abschnitte 12.4 und 13.3). Ohne ``--gestoerte-starts`` rechnen die
Regel-Varianten (V3, V3AL) deshalb den ungestörten Einzellauf und ``RULE_REFERENCE_STARTS``
gestörte Starts; ``--regel-einzellauf`` rechnet nur den Einzellauf (wie bis M10-T7e),
``--gestoerte-starts N`` gilt wie bisher für alle Varianten (0: nur Einzelläufe). Tabelle und
Zusammenfassung führen die Referenz als eigene Zeile (``V3RS Referenz (Median 5 gestörte
Starts)``), den Einzellauf als ``V3RS Einzellauf``.

Standards von ``synth`` (M6-T14b, Entscheidung Maintainer 2026-11-25): Seit M6-T13b vergrößert
``synth`` die Schrittweite auch nach langsamen Schritten (``--grow-margin 0.8``); die Läufe der
Regel und von V0/V1 verschieben sich dadurch gegenüber den Referenzen aus M10-T7 bis M6-T11b.
``--synth-vor-m6t13b`` rechnet mit den Standards davor (``--no-grow-margin``,
``--no-insertion-improve-finish``, ``--no-color-variants`` an jedem ``synth`` von V0/V1 und von
``synth_segment``) und reproduziert so die Referenzen; die Ergebnisse tragen
``synth_defaults: "vor M6-T13b"``.

Ollama: ``--num-ctx`` (Standard 65536) setzt die Kontextlänge; größer als das Maximum des
Modells (``/api/show``) wird auf das Maximum begrenzt, 0 heißt Modellmaximum.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYSTEMS = ROOT / "data" / "systems"
#: privates Verzeichnis der Patentsysteme (B4, nie im Repository) und seine Benchmark-Liste
PRIVATE_SYSTEMS = SYSTEMS / "private"
PRIVATE_MANIFEST = PRIVATE_SYSTEMS / "b4.json"
RMS_TARGET = 1.0 / 14.0
#: V3AL: Augmented Lagrangian (höchstens K Läufe) und Gewicht der Sinusbedingung [Wellen²]
#: (w = 13,1 wie im Versuch M11-T4: kostet so viel wie die Koma bei 10 % Feld des 50x/0,95)
V3AL_SYNTH = {"augmented_lagrangian": 5, "sine_condition": 13.1}
#: relative Toleranz der NA (die Brennweite weicht über die Nebenbedingung leicht vom Soll ab)
NA_TOLERANCE = 1e-3
#: Referenzsphäre der Benchmarks (Entscheidung Maintainer 2026-11-20) und Kürzel im Ergebnisnamen
DEFAULT_REFERENCE_SPHERE = "image_infinite"
REFERENCE_SUFFIX = {"image_infinite": "RS", "image_exit_pupil": "RSAP", "last_surface": ""}
#: Referenz der Regel in Bericht und Tabelle: Median dieser Variante über RULE_REFERENCE_STARTS
#: gestörte Starts (M6-T11b, Entscheidung Maintainer 2026-11-22); der Einzellauf bleibt daneben
RULE_REFERENCE = "V3RS"
RULE_REFERENCE_STARTS = 5
#: Grundvarianten der Regel (ohne --gestoerte-starts: Einzellauf plus RULE_REFERENCE_STARTS
#: gestörte Starts)
RULE_VARIANTS = ("V3", "V3AL")
VARIANTS = ("V0", "V1", "V2", "V3", "V3AL", "LM", "V4")
#: Kontextlänge für Ollama (gpt-oss:20b brach in M10-T7b mit 32768 ab, E12)
DEFAULT_NUM_CTX = 65536

_TUBE = {"tube_focal_length": 164.5, "parfocal_max": 45.0}
BENCHMARKS = {
    "B1": {"start": SYSTEMS / "einzellinse_homotopie_start.json", "na": 0.6, "max_elements": 6,
           "spec": {"na": 0.6}, "steps": 12},
    "B2": {"start": SYSTEMS / "zielspezifikation_start.json", "na": 0.95, "max_elements": 15,
           "spec": {**_TUBE, "na": 0.95, "magnification": 50, "field_number": 2.5,
                    "target_color_class": "c", "target_field_class": [4, 5]}, "steps": 12,
           "field_scale": 0.1},
    "B3": {"start": SYSTEMS / "bewertung_b3_40x085_start.json", "na": 0.85, "max_elements": 15,
           "spec": {**_TUBE, "na": 0.85, "magnification": 40, "field_number": 2.5,
                    "target_color_class": "c", "target_field_class": [4, 5]}, "steps": 12,
           "field_scale": 0.1},
}


def private_benchmarks(manifest=None) -> dict:
    """Benchmarks B4 aus ``data/systems/private/b4.json`` (nur lokal, Entscheidung Maintainer
    2026-11-19): ``{"benchmarks": [{"name": "B4…", "start": "<datei>.json", "na": …,
    "max_elements": …, "spec": {…}, "steps": …, "field_scale": …}]}`` mit ``start`` relativ zum
    Verzeichnis der Liste. Leer, wenn die Liste fehlt."""
    manifest = pathlib.Path(manifest or PRIVATE_MANIFEST)
    if not manifest.is_file():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out = {}
    for e in data.get("benchmarks", []):
        name = str(e["name"])
        if not name.startswith("B4"):
            raise ValueError(f"{manifest}: Name {name} beginnt nicht mit B4")
        b = {"start": manifest.parent / e["start"], "na": float(e["na"]),
             "max_elements": int(e.get("max_elements", 15)),
             "spec": dict(e.get("spec") or {"na": float(e["na"])}),
             "steps": int(e.get("steps", 12)), "private": True}
        if "field_scale" in e:
            b["field_scale"] = float(e["field_scale"])
        out[name] = b
    return out


def expand_benchmarks(names, manifest=None) -> list[tuple[str, dict]]:
    """Benchmarks der Kommandozeile; ``B4`` steht für alle Einträge der privaten Liste (ohne
    Liste: übersprungen mit Hinweis), ``B4…`` für einen Eintrag."""
    private = None
    out = []
    for b in names:
        if b in BENCHMARKS:
            out.append((b, BENCHMARKS[b]))
            continue
        if not b.startswith("B4"):
            raise SystemExit(f"unbekannter Benchmark {b}")
        private = private_benchmarks(manifest) if private is None else private
        if not private:
            print(f"{b}: keine private Liste {manifest or PRIVATE_MANIFEST}, übersprungen")
            continue
        sel = private if b == "B4" else {k: v for k, v in private.items() if k == b}
        if not sel:
            raise SystemExit(f"Benchmark {b} fehlt in der privaten Liste")
        out += list(sel.items())
    return out


def _mko():
    import mikroptim  # noqa: PLC0415 - erst nach dem Parsen der Argumente

    return mikroptim


def cli_json(args):
    code, out, err = _mko().run_cli([str(a) for a in args] + ["--json"])
    if code != 0:
        raise RuntimeError(f"mikroptim {' '.join(map(str, args[:2]))}: {err.strip()}")
    return json.loads(out)


# --- einheitliche Bewertung eines Endsystems ---------------------------------------------------


def glass_elements(system):
    s = system["surfaces"]
    return sum(1 for i in range(len(s) - 1)
               if not (isinstance(s[i].get("material"), str) and s[i]["material"] == "AIR"))


def assess(path, target_na) -> dict:
    """Reales System (Geisterplatten-Koeffizienten 0) mit ``mikroptim trace`` bewerten."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for g in data.get("ghost_plates", []):
        g["coefficients"] = [0.0] * len(g.get("coefficients", []))
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "real.json"
        # Kataloge relativ zum Original auflösen
        base = pathlib.Path(path).resolve().parent
        cats = []
        for c in data.get("glass_catalogs", []):
            cp = pathlib.Path(c)
            if not cp.is_absolute():
                for b in [base, *base.parents]:
                    if (b / cp).is_file():
                        cp = (b / cp).resolve()
                        break
            cats.append(str(cp))
        data["glass_catalogs"] = cats
        p.write_text(json.dumps(data), encoding="utf-8")
        tr = cli_json(["trace", p])
        # Gegenprobe mit der Referenzsphäre durch die Austrittspupille (Falle F12 aus M6-T6:
        # Fokus im Scheitel einer Hilfsebene täuscht in trace/Merit kleine OPD vor)
        try:
            ev = cli_json(["eval", p, "--grid", 64])
            ev_all = [t["rms_opd_waves"] for t in ev["tuples"]]
            # ein Tupel ohne Wert (alle Strahlen versagt) macht die Gegenprobe ungültig (M10-T7e:
            # vorher TypeError, Lauf abgebrochen)
            ev_rms = None if any(x is None for x in ev_all) else max(ev_all)
        except (RuntimeError, ValueError, KeyError):
            ev_rms = None
    f = tr["paraxial"]["focal_length"]
    # NA wie synth: Pupillenradius / Soll-Brennweite (sonst paraxiale Brennweite)
    f_target = (data.get("target") or {}).get("focal_length") or f
    na = data["surfaces"][0]["semi_diameter"] / f_target if f_target else None
    rms = [w["rms_waves"] for w in tr.get("wavefront") or []]
    rms_max = None if not rms or any(r is None for r in rms) else max(rms)
    failed = tr.get("failed_rays", 0)
    goal = (na is not None and na >= target_na * (1 - NA_TOLERANCE) and rms_max is not None
            and rms_max <= RMS_TARGET and failed == 0)
    return {"na": na, "rms_max_waves": rms_max, "rms_eval_exit_pupil": ev_rms,
            "failed_rays": failed,
            "elements": glass_elements(data), "focal_length": f, "goal": goal}


def rank_key(a):
    """Bester Kandidat: Ziel erreicht, dann größte NA, dann kleinster RMS, dann wenig Elemente."""
    rms = a["rms_max_waves"] if a["rms_max_waves"] is not None else math.inf
    na = a["na"] or 0.0
    return (not a["goal"], -round(na, 3), a["failed_rays"] > 0, rms, a["elements"])


# --- Varianten ----------------------------------------------------------------------------------


def prepare_start(bench, work_root, name) -> dict:
    """Startsystem des Benchmarks im Arbeitsverzeichnis: Glaskataloge absolut, Feldwinkel mal
    ``field_scale`` (B2, B3), ``reference_sphere`` aus ``bench`` (ADR-0017; bei
    ``last_surface`` oder ohne Angabe bleibt die Datei ohne Feld, wie in M10-T7/M10-T7b)."""
    from mikroptim.design.files import read_system, resolve_catalogs, write_system

    data = resolve_catalogs(read_system(bench["start"]), pathlib.Path(bench["start"]).parent)
    k = bench.get("field_scale", 1.0)
    if k != 1.0:
        src = data["source"]
        src["field_angles_deg"] = [a * k for a in src.get("field_angles_deg", [0.0])]
        data["name"] = f"{data.get('name', '')} (Feldwinkel × {k:g})"
    rs = bench.get("reference_sphere")
    if rs and rs != "last_surface":  # ADR-0017; last_surface: Datei ohne Feld wie M10-T7b
        data["reference_sphere"] = rs
    path = pathlib.Path(work_root) / f"start_{name}.json"
    # atomar ersetzen: parallele Läufe desselben Benchmarks lasen sonst eine halb geschriebene
    # Datei (M10-T7b)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_system(data, tmp)
    os.replace(tmp, path)
    return {**bench, "start": path}


def _covered(item, n_surfaces) -> list[int]:
    """Dateiflächen einer Krümmungs- oder Dickenvariable (``surface``, Liste oder ``all``)."""
    if "surface" in item:
        return [int(item["surface"])]
    if item.get("surfaces") == "all":
        return list(range(n_surfaces if item.get("type") == "curvature" else n_surfaces - 1))
    return [int(x) for x in item.get("surfaces", [])]


def perturb_start(data, sigma, rng) -> dict:
    """Leicht gestörter Start (M10-T7e): Startwerte der Krümmungs- und Dickenvariablen sowie des
    Arbeitsabstands relativ um σ·ξ (ξ ~ N(0, 1) aus ``rng``) ändern – Radien r(1 + σξ), d. h.
    c/(1 + σξ), Dicken t(1 + σξ) – und auf die Schranken der Variable begrenzen; eine Dicke nicht
    unter die größte Lage einer Geisterplatte im Spalt. Nur Flächen, die eine Variable sind;
    ebene Flächen (c = 0) bleiben eben. Rückgabe: Liste der Änderungen (Fläche, Art, alt, neu)."""
    surf = data["surfaces"]
    n = len(surf)
    changes = []
    done = set()
    for it in (data.get("variables") or {}).get("items", []):
        kind = it.get("type")
        if kind == "working_distance":
            targets = [n - 1]
        elif kind in ("curvature", "thickness"):
            targets = _covered(it, n)
        else:
            continue
        lo = it.get("min", -math.inf)
        hi = it.get("max", math.inf)
        for j in targets:
            if (kind, j) in done or not 0 <= j < n:
                continue
            done.add((kind, j))
            xi = rng.gauss(0.0, 1.0)
            if kind == "curvature":
                if surf[j].get("type", "sphere") != "sphere":
                    continue
                old = float(surf[j].get("curvature", 0.0))
                if old == 0.0:
                    continue
                new = min(max(old / (1.0 + sigma * xi), lo), hi)
                surf[j]["curvature"] = new
            else:
                old = float(surf[j]["thickness"])
                floor = max([g["distance"] for g in data.get("ghost_plates", [])
                             if g["after_surface"] == j] or [0.0])
                new = min(max(old * (1.0 + sigma * xi), lo, floor), hi)
                surf[j]["thickness"] = new
            if new == old:  # an der Schranke
                continue
            changes.append({"surface": j, "type": "curvature" if kind == "curvature"
                            else "thickness", "old": old, "new": new})
    return changes


def prepare_perturbed(vb, work_root, name, k, sigma, seed) -> dict:
    """Gestörte Kopie ``start_<name>_s<k>.json`` des vorbereiteten Starts ``vb``; Zufallsfolge
    deterministisch aus Seed, Benchmark-Name und k."""
    import random

    from mikroptim.design.files import read_system, write_system

    data = read_system(vb["start"])
    rng = random.Random(f"{seed}:{name}:{k}")
    changes = perturb_start(data, sigma, rng)
    data["name"] = f"{data.get('name', '')} (gestört, sigma {sigma:g}, Start {k})"
    path = pathlib.Path(work_root) / f"start_{name}_s{k}.json"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_system(data, tmp)
    os.replace(tmp, path)
    return {**vb, "start": path, "perturbation": {"index": k, "sigma": sigma, "seed": seed,
                                                    "changes": changes}}


def run_synth(bench, work, beam=False, before_m6t13b=False) -> dict:
    out = work / "ergebnis.json"
    args = ["synth", bench["start"], "--target-na", bench["na"], "--max-elements",
            bench["max_elements"], "-o", out, "--quiet"]
    if beam:
        args += ["--beam", 3, "--alternatives", 2]
    if before_m6t13b:
        # M6-T14b: Standards von synth wie vor M6-T13b (wie synth_segment)
        from mikroptim.design import actions  # noqa: PLC0415
        args += list(actions.SYNTH_ARGS_BEFORE_M6T13B)
    t = time.perf_counter()
    rep = cli_json(args)
    sec = time.perf_counter() - t
    res = rep.get("result") or {}
    a = assess(out, bench["na"])
    return {"seconds": sec, "status": res.get("status"), "synth_goal": (res.get("goal") or {})
            .get("reached"), "lm_solves": res.get("solves"), "best": a,
            "system": str(out)}


def best_open_resume(target_na):
    """``resume`` für ``Session.run``: nach ``abandon`` weiter mit dem besten offenen Kandidaten
    in der Rangfolge der Bewertung (``rank_key``; Bewertungen pro Kandidat zwischengespeichert,
    die Systemdateien ändern sich nicht)."""
    cache = {}

    def resume(session, _abandoned):
        ranked = []
        for c in session.open_candidates():
            if c not in cache:
                try:
                    cache[c] = assess(session.system_path(c), target_na)
                except RuntimeError:
                    cache[c] = None
            if cache[c] is not None:
                ranked.append((rank_key(cache[c]), c))
        return min(ranked)[1] if ranked else None

    return resume


def run_session(bench, work, decider, fallback, steps) -> dict:
    from mikroptim import design

    sdir = work / "sitzung"
    if sdir.exists():
        shutil.rmtree(sdir)
    s = design.Session.create(bench["start"], sdir, spec=bench["spec"],
                              budget={"actions": steps})
    t = time.perf_counter()
    aborted = aborted_trace = None
    try:
        s.run(decider, 0, max_steps=steps, fallback=fallback,
              resume=best_open_resume(bench["na"]))
    except Exception as e:  # Budget, Konfiguration: Lauf abbrechen, Teilergebnis bewerten
        import traceback  # noqa: PLC0415

        aborted = f"{type(e).__name__}: {e}"
        aborted_trace = traceback.format_exc()[-4000:]
    sec = time.perf_counter() - t
    log = s.log()
    cands = {}
    for cid in s.candidates():
        try:
            cands[cid] = assess(s.system_path(cid), bench["na"])
        except RuntimeError as e:
            cands[cid] = {"error": str(e)}
    valid = {k: v for k, v in cands.items() if "error" not in v}
    best_id = min(valid, key=lambda k: rank_key(valid[k])) if valid else None
    actions = [r for e in log for r in e.get("results", [])]
    n_act = sum(len((e.get("response") or {}).get("actions") or []) for e in log)
    rejected = sum(1 for r in actions if r.get("status") != "ok")
    lm_solves = sum((rep.get("result") or {}).get("solves") or 0
                    for r in actions if r.get("action") == "synth_segment"
                    for rep in r.get("reports", []))
    lm_optimize = sum(1 for r in actions if r.get("action") == "optimize"
                      and r.get("status") == "ok")
    usage = usage_summary(log)
    return {"seconds": sec, "aborted": aborted, "aborted_trace": aborted_trace,
            "decisions": len(log), "actions": n_act,
            "rejected_actions": rejected,
            "rejected_share": (rejected / n_act) if n_act else None,
            "fallback_decisions": sum(1 for e in log if e["decider"].get("failed_decider")),
            "lm_solves_synth": lm_solves, "lm_optimize": lm_optimize, "candidates": len(cands),
            "best_candidate": best_id, "best": valid.get(best_id), "usage": usage,
            "session": str(sdir),
            "system": str(s.system_path(best_id)) if best_id is not None else None,
            "decisions_summary": [
                {"candidate": e["candidate"], "decider": e["decider"].get("kind"),
                 "actions": [a.get("action") for a in (e.get("response") or {}).get("actions")
                             or []],
                 "status": [r.get("status") for r in e.get("results", [])]} for e in log]}


def usage_summary(log) -> dict:
    """Tokens (Ollama: Prompt/Antwort, Denktext in Zeichen) bzw. Kosten (Claude) aus dem
    Protokoll, einschließlich gescheiterter Entscheider vor einem Rückfall."""
    out = {"prompt_tokens": 0, "output_tokens": 0, "thinking_chars": 0, "requests": 0,
           "cache_read_tokens": 0, "cost_usd": 0.0}
    for e in log:
        for d in (e["decider"], e["decider"].get("failed_decider")):
            if not d:
                continue
            tr = d.get("trace") or {}
            if d.get("kind") == "ollama":
                for u in tr.get("usage", []):
                    out["requests"] += 1
                    out["prompt_tokens"] += u.get("prompt_eval_count") or 0
                    out["output_tokens"] += u.get("eval_count") or 0
                    out["thinking_chars"] += u.get("thinking_chars") or 0
            elif d.get("kind") == "claude":
                for r in tr.get("requests", []):
                    u = r.get("usage") or {}
                    out["requests"] += 1
                    out["prompt_tokens"] += (u.get("input_tokens", 0)
                                             + u.get("cache_creation_input_tokens", 0))
                    out["cache_read_tokens"] += u.get("cache_read_input_tokens", 0)
                    out["output_tokens"] += u.get("output_tokens", 0)
                    out["cost_usd"] += r.get("cost_usd") or 0.0
                out["thinking_chars"] += sum(len(t) for t in tr.get("thinking", []))
    return out


# --- Ablauf -------------------------------------------------------------------------------------


def resolve_num_ctx(requested: int, model_max: int | None) -> int | None:
    """Kontextlänge für Ollama: ``requested`` begrenzt auf das Modellmaximum; 0 heißt
    Modellmaximum (``None``, wenn das Maximum unbekannt ist: dann gilt der Standard des
    Entscheiders)."""
    if requested <= 0:
        return model_max
    return min(requested, model_max) if model_max else requested


def make_llm(args, ledger):
    from mikroptim import design

    if args.decider == "ollama":
        d = design.OllamaDecider(args.model, host=args.host, think=args.think,
                                 keep_alive=args.keep_alive)
        model_max = d.model_context_length()
        n = resolve_num_ctx(args.num_ctx, model_max)
        if n is not None:
            d.options["num_ctx"] = n
        print(f"Ollama {args.model}: num_ctx {d.options.get('num_ctx')} (Modellmaximum "
              f"{model_max if model_max else 'unbekannt'})", flush=True)
        return d
    return design.ClaudeDecider(args.model or "claude-opus-5-5", effort=args.effort,
                                ledger=ledger, strict=args.strict)


def variant_config(variant, reference_sphere=DEFAULT_REFERENCE_SPHERE) -> tuple[str, str]:
    """Variante der Kommandozeile → (Grundvariante, Referenzsphäre): ``<V>-alt`` rechnet mit der
    alten Konvention ``last_surface``, ``V0RS``/``V3RS`` (Namen aus M10-T7b) immer mit
    ``image_infinite``, sonst gilt ``reference_sphere`` (``--referenzsphaere``)."""
    if variant.endswith("-alt"):
        base, rs = variant[:-4], "last_surface"
    elif variant in ("V0RS", "V3RS"):
        base, rs = variant[:2], "image_infinite"
    else:
        base, rs = variant, reference_sphere
    if base not in VARIANTS:
        raise SystemExit(f"unbekannte Variante {variant}")
    if rs not in REFERENCE_SUFFIX:
        raise SystemExit(f"unbekannte Referenzsphäre {rs}")
    return base, rs


def tag_of(variant, args):
    """Ergebnis-Kürzel: Grundvariante plus Kürzel der Referenzsphäre (``V3RS``, ``V3`` für die
    alte Konvention), bei LM zusätzlich das Modell (``LMRS_claude-opus-5-5``)."""
    base, rs = variant_config(variant, getattr(args, "referenzsphaere", DEFAULT_REFERENCE_SPHERE))
    tag = base + REFERENCE_SUFFIX[rs]
    if base != "LM":
        return tag
    model = args.model or "claude-opus-5-5"
    think = "" if args.decider != "ollama" or args.think is None else (
        "_think" if args.think else "_nothink")
    return tag + "_" + (model.replace(":", "-").replace("/", "-")) + think


def reassess(out) -> None:
    """Ergebnisdateien neu bewerten (gleiche Systeme, aktuelle Bewertungsregeln)."""
    for p in sorted(pathlib.Path(out).glob("B*_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        na = r["target_na"]
        if r.get("session") and pathlib.Path(r["session"]).is_absolute():
            from mikroptim import design

            s = design.Session(r["session"])
            cands = {c: assess(s.system_path(c), na) for c in s.candidates()}
            best = min(cands, key=lambda k: rank_key(cands[k]))
            r["best_candidate"], r["best"] = best, cands[best]
            r["system"] = str(s.system_path(best))
        elif r.get("system"):
            sys_path = pathlib.Path(r["system"])
            if not sys_path.is_absolute():
                sys_path = pathlib.Path(out) / sys_path
            r["best"] = assess(sys_path, na)
        p.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _portable_system(src, dst) -> None:
    """Systemdatei ohne absolute Pfade: Kataloge als ``data/glass/<name>`` (der Leser sucht
    relative Kataloge vom Ort der Datei aufwärts)."""
    data = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
    data["glass_catalogs"] = [f"data/glass/{pathlib.Path(c).name}"
                              for c in data.get("glass_catalogs", [])]
    pathlib.Path(dst).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")


def archive(out) -> None:
    """Für das Repository: pro Lauf Protokoll (``log.jsonl``, ``session.json``) und bestes
    System nach ``<out>/<B>_<V>/`` kopieren, absolute Pfade durch relative ersetzen. Mit dem
    Protokoll lässt sich jeder Lauf ohne Modell wiederholen (``python -m mikroptim.design
    replay``)."""
    out = pathlib.Path(out)
    for p in sorted(out.glob("B*_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        d = out / p.stem
        d.mkdir(exist_ok=True)
        rel = d.resolve().relative_to(ROOT).as_posix()
        if r.get("session") and pathlib.Path(r["session"]).exists():
            sdir = pathlib.Path(r["session"])
            shutil.copy(sdir / "log.jsonl", d / "log.jsonl")
            meta = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
            _portable_system(pathlib.Path(meta["start"]), d / "start.json")
            meta["start"] = f"{rel}/start.json"  # relativ zum Wurzelverzeichnis
            (d / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)
                                            + "\n", encoding="utf-8")
            r["session"] = f"{p.stem}/"
        if r.get("system") and pathlib.Path(r["system"]).exists():
            _portable_system(pathlib.Path(r["system"]), d / "bestes_system.json")
            r["system"] = f"{p.stem}/bestes_system.json"
        if not r.get("aborted"):
            r.pop("aborted_trace", None)
        p.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmarks", default="B1,B2,B3")
    ap.add_argument("--varianten", default="V0,V1,V3",
                    help="Varianten (V0, V1, V3, V3AL, LM, V4; <V>-alt = alte Konvention "
                    "last_surface; V0RS/V3RS = image_infinite)")
    ap.add_argument("--referenzsphaere", choices=list(REFERENCE_SUFFIX),
                    default=DEFAULT_REFERENCE_SPHERE,
                    help="reference_sphere der Starts (Standard image_infinite, Entscheidung "
                    "Maintainer 2026-11-20; last_surface = Startdatei ohne Feld wie M10-T7b)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "ergebnisse" / "m10-t7c"))
    ap.add_argument("--work", default=None, help="Arbeitsverzeichnis der Läufe (Sitzungen, "
                    "Systeme); Standard: <out>/../<out>-arbeit (nicht im Repository)")
    ap.add_argument("--steps", type=int, default=None, help="Entscheidungen pro Sitzung")
    ap.add_argument("--decider", choices=["ollama", "claude"], default="ollama")
    ap.add_argument("--model")
    ap.add_argument("--host", default="http://localhost:11434")
    th = ap.add_mutually_exclusive_group()
    th.add_argument("--think", dest="think", action="store_true", default=None)
    th.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--keep-alive", default="2m")
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                    help="Ollama: Kontextlänge (begrenzt auf das Modellmaximum aus /api/show; "
                    f"0 = Modellmaximum; Standard {DEFAULT_NUM_CTX})")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-usd", type=float, default=20.0)
    st = ap.add_mutually_exclusive_group()
    st.add_argument("--strict", dest="strict", action="store_true", default=False,
                    help="Claude: Werkzeuge mit strict (lehnt die API derzeit ab, M10-T7c)")
    st.add_argument("--no-strict", dest="strict", action="store_false",
                    help="Claude: Werkzeuge ohne strict (Standard)")
    ap.add_argument("--cost-file", default=None,
                    help="Kostensumme der Claude-Läufe (Standard <work>/claude_cost.json)")
    ap.add_argument("--tabelle", action="store_true")
    ap.add_argument("--vergleich", metavar="ALT",
                    help="Tabelle alt (Ergebnisverzeichnis ALT) gegen neu (--out)")
    ap.add_argument("--archiv", action="store_true",
                    help="Protokolle und beste Systeme ins Ergebnisverzeichnis kopieren")
    ap.add_argument("--wiederholung", default=None, metavar="N[,M…|N-M]",
                    help="Wiederholungen mit eigenem Kürzel <tag>_w<N> (M10-T7d), z. B. 1-3; "
                    "Läufe mit Modell sind nicht deterministisch, Ergebnisse werden nicht "
                    "überschrieben")
    ap.add_argument("--gestoerte-starts", type=int, default=None, metavar="N",
                    help="jede Variante auf N leicht gestörten Starts statt des Einzellaufs "
                    "(Kürzel <tag>_s<k>, M10-T7e; 0: nur Einzelläufe); gestört werden "
                    "Startwerte der Krümmungs- und Dickenvariablen im Rahmen ihrer Schranken. "
                    "Ohne die Option rechnen die Regel-Varianten Einzellauf und "
                    f"{RULE_REFERENCE_STARTS} gestörte Starts (Referenz der Regel, M6-T11b)")
    ap.add_argument("--nur-starts", default=None, metavar="LISTE",
                    help="nur diese Starts rechnen (0 = Einzellauf, k = gestörter Start k; "
                    "z. B. 0,1 oder 2-3), um Läufe auf mehrere Prozesse zu verteilen")
    ap.add_argument("--regel-einzellauf", action="store_true",
                    help="Regel-Varianten nur als Einzellauf ohne gestörte Starts (wie bis "
                    "M10-T7e)")
    ap.add_argument("--stoer-sigma", type=float, default=0.02,
                    help="relative Störung σ der Radien und Dicken (Standard 0,02)")
    ap.add_argument("--stoer-seed", type=int, default=0,
                    help="Seed der Störungen (deterministisch mit Benchmark und k)")
    ap.add_argument("--zusammenfassung", action="store_true",
                    help="Tabelle über Wiederholungen: Ziel-Treffer x/n, beste und mediane NA "
                    "und RMS, Elemente, Kosten pro Lauf")
    ap.add_argument("--synth-vor-m6t13b", action="store_true",
                    help="M6-T14b: synth (V0/V1 und synth_segment) mit den Standards vor M6-T13b "
                    "(--no-grow-margin u. a.), um die Referenzen aus M10-T7 bis M6-T11b zu "
                    "reproduzieren")
    ap.add_argument("--neu-bewerten", action="store_true",
                    help="vorhandene Ergebnisse mit den aktuellen Regeln neu bewerten")
    return ap


def _numbers(spec, option, lowest=1) -> list:
    """Nummernliste aus ``"1,2"``, ``"1-3"`` oder ``"4"`` (ab ``lowest``)."""
    out = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        try:
            a, b = int(lo), int(hi or lo)
        except ValueError:
            raise SystemExit(f"{option}: {spec!r} ist keine Nummernliste") from None
        if a < lowest or b < a:
            raise SystemExit(f"{option}: ungültiger Bereich {part!r}")
        out.extend(n for n in range(a, b + 1) if n not in out)
    return out


def repetitions(spec) -> list:
    """``--wiederholung``: ``None`` → ein Lauf ohne Zusatz, sonst Nummern aus ``"1,2"``,
    ``"1-3"`` oder ``"4"``."""
    if spec is None:
        return [None]
    out = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        try:
            a, b = int(lo), int(hi or lo)
        except ValueError:
            raise SystemExit(f"--wiederholung: {spec!r} ist keine Nummernliste") from None
        if a < 1 or b < a:
            raise SystemExit(f"--wiederholung: ungültiger Bereich {part!r}")
        out.extend(n for n in range(a, b + 1) if n not in out)
    return out


def rep_tag(tag, n, k=None) -> str:
    """Kürzel einer Wiederholung ``<tag>_w<N>`` und eines gestörten Starts ``_s<k>``; ohne
    Nummern das Kürzel selbst."""
    tag = tag if n is None else f"{tag}_w{n}"
    return tag if k is None else f"{tag}_s{k}"


def starts_of(base, args) -> list:
    """Starts einer Grundvariante: ``None`` = ungestörter Einzellauf, k = gestörter Start k.
    ``--gestoerte-starts N`` gilt für alle Varianten (N gestörte Starts, 0: Einzellauf); ohne die
    Option rechnen die Regel-Varianten Einzellauf und ``RULE_REFERENCE_STARTS`` gestörte Starts
    (Referenz der Regel, M6-T11b), mit ``--regel-einzellauf`` nur den Einzellauf."""
    n = args.gestoerte_starts
    if n is not None:
        out = list(range(1, n + 1)) if n else [None]
    elif base in RULE_VARIANTS and not getattr(args, "regel_einzellauf", False):
        out = [None] + list(range(1, RULE_REFERENCE_STARTS + 1))
    else:
        out = [None]
    only = getattr(args, "nur_starts", None)
    if only is not None:
        keep = {None if k == 0 else k for k in _numbers(only, "--nur-starts", lowest=0)}
        out = [k for k in out if k in keep]
    return out


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    reps = repetitions(args.wiederholung)
    if (args.gestoerte_starts is not None and args.gestoerte_starts < 0) or \
            not args.stoer_sigma >= 0:
        raise SystemExit("--gestoerte-starts und --stoer-sigma müssen ≥ 0 sein")
    out = pathlib.Path(args.out)
    work_root = pathlib.Path(args.work) if args.work else out.parent / f"{out.name}-arbeit"
    if args.neu_bewerten:
        reassess(out)
    if args.archiv:
        archive(out)
    if (args.neu_bewerten or args.archiv) and not (args.tabelle or args.vergleich
                                                   or args.zusammenfassung):
        return 0
    if args.vergleich:
        print(comparison(args.vergleich, out))
        return 0
    if args.tabelle or args.zusammenfassung:
        if args.tabelle:
            print(table(out))
        if args.zusammenfassung:
            print(("\n" if args.tabelle else "") + summary(out))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    from mikroptim import design
    from mikroptim.design import actions as design_actions

    # M6-T14b: synth mit den Standards vor M6-T13b (gilt für alle Sitzungen dieses Prozesses)
    design_actions.SYNTH_DEFAULTS_BEFORE_M6T13B = bool(args.synth_vor_m6t13b)

    variants = args.varianten.split(",")
    for v in variants:  # Tippfehler vor dem ersten Lauf melden
        variant_config(v, args.referenzsphaere)
    ledger = None
    if any(variant_config(v)[0] == "LM" for v in variants) and args.decider == "claude":
        ledger = design.CostLedger(args.max_usd, args.cost_file or work_root / "claude_cost.json")
    llm = None
    try:
        for b, bench in expand_benchmarks(args.benchmarks.split(",")):
            # private Benchmarks (B4): Arbeits- und Ergebnisverzeichnis unter private/, das die
            # Datenschutzprüfung des ClaudeDecider erkennt und .gitignore ausschließt
            b_work = work_root / "private" if bench.get("private") else work_root
            b_out = out / "private" if bench.get("private") else out
            b_work.mkdir(parents=True, exist_ok=True)
            b_out.mkdir(parents=True, exist_ok=True)
            for v, n, k in ((v, n, k) for v in variants for n in reps
                            for k in starts_of(variant_config(v, args.referenzsphaere)[0], args)):
                base, rs = variant_config(v, args.referenzsphaere)
                tag = rep_tag(tag_of(v, args), n, k)
                if bench.get("private") and base == "LM" and args.decider == "claude":
                    print(f"{b} {tag}: private Daten nur lokal (Ollama), übersprungen")
                    continue
                res_path = b_out / f"{b}_{tag}.json"
                if res_path.exists():
                    print(f"{b} {tag}: vorhanden, übersprungen")
                    continue
                work = b_work / b / tag
                work.mkdir(parents=True, exist_ok=True)
                # Startkopie pro Konvention (Namen wie M10-T7b: start_<B>.json alte Konvention,
                # start_<B>_rs.json image_infinite)
                start_name = b + ("_" + REFERENCE_SUFFIX[rs].lower() if REFERENCE_SUFFIX[rs] else "")
                vb = prepare_start({**bench, "reference_sphere": rs}, b_work, start_name)
                if k is not None:
                    vb = prepare_perturbed(vb, b_work, start_name, k, args.stoer_sigma,
                                           args.stoer_seed)
                steps = args.steps or vb["steps"]
                print(f"{b} {tag}: läuft …", flush=True)
                if base == "V0":
                    res = run_synth(vb, work, before_m6t13b=args.synth_vor_m6t13b)
                elif base == "V1":
                    res = run_synth(vb, work, beam=True, before_m6t13b=args.synth_vor_m6t13b)
                elif base == "V3":
                    res = run_session(vb, work, design.RuleDecider(), None, steps)
                elif base == "V3AL":
                    res = run_session(vb, work,
                                      design.RuleDecider(synth_options=V3AL_SYNTH), None, steps)
                elif base == "LM":
                    if args.decider == "ollama" and not args.model:
                        raise SystemExit("--model fehlt")
                    llm = llm or make_llm(args, ledger)
                    res = run_session(vb, work, llm, design.RuleDecider(), steps)
                    res["decider"] = {k: v2 for k, v2 in llm.describe().items()
                                      if k not in ("trace", "usage")}
                    if ledger is not None:
                        res["ledger_spent_usd"] = ledger.spent_usd
                elif base == "V4":  # Mensch auf der Kommandozeile, gleiche Werkzeuge und Prüfungen
                    res = run_session(vb, work, design.HumanDecider(), None, steps)
                else:
                    raise SystemExit(f"Variante {v}: im Skript nicht umgesetzt (V2: "
                                     "tools/m6t6_zielspezifikation.py)")
                res["reference_sphere"] = rs
                if args.synth_vor_m6t13b:
                    res["synth_defaults"] = "vor M6-T13b"  # M6-T14b
                res.update({"benchmark": b, "variant": v, "tag": tag, "target_na": bench["na"]})
                if n is not None:
                    res["repetition"] = n
                if k is not None:
                    res["perturbation"] = vb["perturbation"]
                aborted = res.get("aborted") or ""
                if any(k in aborted for k in ("Unavailable", "PrivacyError", "HTTPError", "URLError")):
                    # Konfigurations- oder Serverfehler (SDK, Zugangsdaten, private Daten, Ollama): kein
                    # Ergebnis, Lauf wiederholen
                    print(f"Abbruch ohne Ergebnis: {aborted}")
                    return 3
                res_path.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
                best = res.get("best") or {}
                print(f"{b} {tag}: NA {fmt(best.get('na'), 3)}, RMS "
                      f"{fmt(best.get('rms_max_waves'))}, {best.get('elements')} Elemente, Ziel "
                      f"{'ja' if best.get('goal') else 'nein'}, {res['seconds']:.0f} s", flush=True)
                if "BudgetExceeded" in aborted:  # Teilergebnis bleibt, weitere Läufe nicht
                    print(f"Abbruch: {aborted}")
                    return 3
    finally:
        if llm is not None and args.decider == "ollama":
            try:
                llm.unload()
            except OSError:
                pass
    return 0


# --- Tabelle ------------------------------------------------------------------------------------


def fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}".replace(".", ",")
    return str(x)


def lm_count(r):
    """LM-Läufe: Optimierungen des Homotopie-Controllers (synth) plus einzelne optimize-Aktionen."""
    if "lm_solves" in r:
        return r["lm_solves"]
    if "lm_optimize" in r:
        return r["lm_solves_synth"] + r["lm_optimize"]
    return None


def table(out) -> str:
    rows = []
    for p in sorted(pathlib.Path(out).glob("B*_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        b = r.get("best") or {}
        u = r.get("usage") or {}
        tokens = (f"{u['prompt_tokens']} / {u['output_tokens']}" if u.get("requests") else "–")
        cost = f"{u['cost_usd']:.2f} $" if u.get("cost_usd") else "–"
        rej = r.get("rejected_share")
        ds = r.get("decisions_summary") or []
        bad = sum(1 for d in ds if any(x != "ok" for x in d["status"]))
        rows.append(
            f"| {r['benchmark']} | {r['tag']} | {'ja' if b.get('goal') else 'nein'} | "
            f"{fmt(b.get('na'), 3)} | {fmt(b.get('rms_max_waves'))} | {fmt(b.get('rms_eval_exit_pupil'))} | "
            f"{fmt(b.get('elements'))} | "
            f"{fmt(lm_count(r))} | "
            f"{r['seconds']:.0f} | {fmt(r.get('decisions'))} | "
            f"{'–' if rej is None else f'{100 * rej:.0f} %'} | "
            f"{f'{bad}/{len(ds)}' if ds else '–'} | "
            f"{fmt(r.get('fallback_decisions'))} | {tokens} | {cost} |")
    note = (f"Referenz der Regel: {RULE_REFERENCE} (V3 mit `reference_sphere: image_infinite`, "
            "ADR-0017; Kürzel RS = image_infinite, RSAP = image_exit_pupil, ohne Zusatz = "
            f"last_surface), als Median über {RULE_REFERENCE_STARTS} gestörte Starts "
            f"({RULE_REFERENCE}_s1 … _s{RULE_REFERENCE_STARTS}, M6-T11b); {RULE_REFERENCE} allein "
            "ist der Einzellauf\n\n")
    ref = rule_reference_table(out)
    if ref:
        note += ref + "\n\n"
    head = note + ("| Bench. | Variante | Ziel | NA | RMS max [Wellen] | RMS AP³ | Elemente | LM-Läufe¹ | Zeit [s] | "
            "Entsch. | abgelehnt² | Entsch. mit Fehler | Rückfälle | Tokens Prompt / Ausgabe | Kosten |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def rule_reference(out) -> dict:
    """Referenz der Regel pro Benchmark (M6-T11b): Median von NA, RMS und Elementzahl über die
    gestörten Starts ``<RULE_REFERENCE>_s<k>``, Ziel-Treffer, Zahl der Starts und der Einzellauf
    ``<RULE_REFERENCE>`` daneben (``None``, wenn er fehlt). Benchmarks ohne gestörte Starts der
    Regel fehlen."""
    runs, single = {}, {}
    pat = re.compile(rf"^{re.escape(RULE_REFERENCE)}_s\d+$")
    for p in sorted(pathlib.Path(out).glob("B*_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if pat.match(r.get("tag", "")):
            runs.setdefault(r["benchmark"], []).append(r.get("best") or {})
        elif r.get("tag") == RULE_REFERENCE:
            single[r["benchmark"]] = r.get("best") or {}
    ref = {}
    for b, best in sorted(runs.items()):
        na = [x.get("na") for x in best if x.get("na") is not None]
        ref[b] = {"starts": len(best), "goal": sum(1 for x in best if x.get("goal")),
                  "na": _median(na), "rms": _median([x.get("rms_max_waves") for x in best]),
                  "elements": _median([x.get("elements") for x in best]),
                  "na_range": (min(na), max(na)) if na else None,
                  "single": single.get(b)}
    return ref


def rule_reference_table(out) -> str:
    """Tabelle der Referenz der Regel (``rule_reference``); leer ohne gestörte Starts."""
    ref = rule_reference(out)
    if not ref:
        return ""
    rows = []
    for b, x in ref.items():
        s = x["single"] or {}
        rng = "–" if x["na_range"] is None else \
            f"{fmt(x['na_range'][0], 3)}–{fmt(x['na_range'][1], 3)}"
        el = x["elements"]
        el_s = "–" if el is None else (str(int(el)) if el == int(el) else fmt(el, 1))
        rows.append(f"| {b} | {x['goal']}/{x['starts']} | {fmt(x['na'], 3)} ({rng}) | "
                    f"{fmt(x['rms'])} | {el_s} | "
                    f"{('ja' if s.get('goal') else 'nein') if s else '–'} | "
                    f"{fmt(s.get('na'), 3)} | {fmt(s.get('rms_max_waves'))} |")
    head = (f"| Bench. | Referenz {RULE_REFERENCE}: Ziel | NA Median (Spanne) | RMS Median | "
            "Elemente Median | Einzellauf: Ziel | NA | RMS |\n"
            "|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def summary_label(group, groups_of_bench) -> str:
    """Beschriftung einer Gruppe der Zusammenfassung (M6-T11b): die gestörten Starts der Regel
    sind die Referenz, der ungestörte Lauf daneben der Einzellauf. ``{n}`` = Zahl der Läufe."""
    if group == f"{RULE_REFERENCE}_s*":
        return f"{RULE_REFERENCE} Referenz (Median {{n}} gestörte Starts)"
    if group == RULE_REFERENCE and f"{RULE_REFERENCE}_s*" in groups_of_bench:
        return f"{RULE_REFERENCE} Einzellauf"
    return group


def summary_group(tag) -> str:
    """Gruppe der Zusammenfassung: Kürzel ohne ``_w<N>``; gestörte Starts (``_s<k>``) bilden eine
    eigene Gruppe ``<tag>_s*`` neben dem ungestörten Lauf (M10-T7e)."""
    m = re.match(r"^(.*?)(_w\d+)?(_s\d+)?$", tag)
    return m.group(1) + ("_s*" if m.group(3) else "")


def summary(out) -> str:
    """Wiederholungen zusammenfassen (M10-T7d): pro Benchmark und Kürzel ohne ``_w<N>``
    Ziel-Treffer x/n, beste und mediane NA, bester und medianer RMS (max. über Felder, Wellen),
    Elemente (Median, Spanne), Kosten pro Lauf (Mittel, Spanne)."""
    groups = {}
    for p in sorted(pathlib.Path(out).glob("B*_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        base = summary_group(r["tag"])
        groups.setdefault((r["benchmark"], base), []).append(r)
    rows = []
    benches = {}
    for b, g in groups:
        benches.setdefault(b, set()).add(g)

    def order(item):  # Referenz der Regel zuerst, dann ihr Einzellauf, sonst alphabetisch
        (b, g), _ = item
        rank = 0 if g == f"{RULE_REFERENCE}_s*" else 1 if g == RULE_REFERENCE else 2
        return (b, rank, g)

    for (b, tag), rs in sorted(groups.items(), key=order):
        label = summary_label(tag, benches[b]).format(n=len(rs))
        best = [r.get("best") or {} for r in rs]
        hits = sum(1 for x in best if x.get("goal"))
        na = [x.get("na") for x in best]
        rms = [x.get("rms_max_waves") for x in best]
        el = [x.get("elements") for x in best if x.get("elements") is not None]
        cost = [(r.get("usage") or {}).get("cost_usd") for r in rs]
        cost = [c for c in cost if c]
        ranked = [x for x in best if x.get("rms_max_waves") is not None or x.get("na")]
        top = min(ranked, key=lambda x: rank_key({"goal": False, "na": 0.0, "failed_rays": 0,
                                                  "rms_max_waves": None, "elements": 0, **x})
                  ) if ranked else {}
        cost_s = ("–" if not cost else f"{fmt(sum(cost) / len(cost), 2)} ({fmt(min(cost), 2)}–"
                  f"{fmt(max(cost), 2)})")
        med = _median(el)
        med_s = "–" if med is None else (str(int(med)) if med == int(med) else fmt(med, 1))
        el_s = "–" if not el else f"{med_s} ({min(el)}–{max(el)})"
        rows.append(f"| {b} | {label} | {hits}/{len(rs)} | {fmt(top.get('na'), 3)} | "
                    f"{fmt(_median(na), 3)} | {fmt(top.get('rms_max_waves'))} | "
                    f"{fmt(_median(rms))} | {el_s} | {cost_s} |")
    head = ("| Bench. | Variante | Ziel | NA bester Lauf | NA Median | RMS bester Lauf | "
            "RMS Median | Elemente Median (Spanne) | Kosten/Lauf [$] Mittel (Spanne) |\n"
            "|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


def comparison(old, new) -> str:
    """Alt gegen neu pro Benchmark und Variante: Ziel, NA, RMS, Elemente, Zeit."""
    def load(d):
        out = {}
        for p in sorted(pathlib.Path(d).glob("B*_*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            out[(r["benchmark"], r["tag"])] = r
        return out

    a, b = load(old), load(new)

    def cell(r):
        if r is None:
            return "– | – | – | – | –"
        x = r.get("best") or {}
        return (f"{'**ja**' if x.get('goal') else 'nein'} | {fmt(x.get('na'), 3)} | "
                f"{fmt(x.get('rms_max_waves'))} | {fmt(x.get('elements'))} | {r['seconds']:.0f}")

    rows = [f"| {k[0]} | {k[1]} | {cell(a.get(k))} | {cell(b.get(k))} |"
            for k in sorted(set(a) | set(b))]
    head = ("| Bench. | Variante | Ziel alt | NA alt | RMS alt | Elem. alt | Zeit alt [s] | "
            "Ziel neu | NA neu | RMS neu | Elem. neu | Zeit neu [s] |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
