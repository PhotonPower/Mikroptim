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
    python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7b --varianten V0-alt,V3-alt,V3RS

Ollama: ``--num-ctx`` (Standard 65536) setzt die Kontextlänge; größer als das Maximum des
Modells (``/api/show``) wird auf das Maximum begrenzt, 0 heißt Modellmaximum.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
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
#: Referenz der Regel in Bericht und Tabelle
RULE_REFERENCE = "V3RS"
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
            ev_rms = max(t["rms_opd_waves"] for t in ev["tuples"])
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


def run_synth(bench, work, beam=False) -> dict:
    out = work / "ergebnis.json"
    args = ["synth", bench["start"], "--target-na", bench["na"], "--max-elements",
            bench["max_elements"], "-o", out, "--quiet"]
    if beam:
        args += ["--beam", 3, "--alternatives", 2]
    t = time.perf_counter()
    rep = cli_json(args)
    sec = time.perf_counter() - t
    res = rep.get("result") or {}
    a = assess(out, bench["na"])
    return {"seconds": sec, "status": res.get("status"), "synth_goal": (res.get("goal") or {})
            .get("reached"), "lm_solves": res.get("solves"), "best": a,
            "system": str(out)}


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
        s.run(decider, 0, max_steps=steps, fallback=fallback)
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
                                ledger=ledger, strict=not args.no_strict)


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
    ap.add_argument("--no-strict", action="store_true",
                    help="Claude: Werkzeuge ohne strict (falls die API die Schemas ablehnt)")
    ap.add_argument("--cost-file", default=None,
                    help="Kostensumme der Claude-Läufe (Standard <work>/claude_cost.json)")
    ap.add_argument("--tabelle", action="store_true")
    ap.add_argument("--vergleich", metavar="ALT",
                    help="Tabelle alt (Ergebnisverzeichnis ALT) gegen neu (--out)")
    ap.add_argument("--archiv", action="store_true",
                    help="Protokolle und beste Systeme ins Ergebnisverzeichnis kopieren")
    ap.add_argument("--neu-bewerten", action="store_true",
                    help="vorhandene Ergebnisse mit den aktuellen Regeln neu bewerten")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    out = pathlib.Path(args.out)
    work_root = pathlib.Path(args.work) if args.work else out.parent / f"{out.name}-arbeit"
    if args.neu_bewerten:
        reassess(out)
    if args.archiv:
        archive(out)
    if (args.neu_bewerten or args.archiv) and not (args.tabelle or args.vergleich):
        return 0
    if args.vergleich:
        print(comparison(args.vergleich, out))
        return 0
    if args.tabelle:
        print(table(out))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    from mikroptim import design

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
            for v in variants:
                base, rs = variant_config(v, args.referenzsphaere)
                tag = tag_of(v, args)
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
                steps = args.steps or vb["steps"]
                print(f"{b} {tag}: läuft …", flush=True)
                if base == "V0":
                    res = run_synth(vb, work)
                elif base == "V1":
                    res = run_synth(vb, work, beam=True)
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
                res.update({"benchmark": b, "variant": v, "tag": tag, "target_na": bench["na"]})
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
            "last_surface)\n\n")
    head = note + ("| Bench. | Variante | Ziel | NA | RMS max [Wellen] | RMS AP³ | Elemente | LM-Läufe¹ | Zeit [s] | "
            "Entsch. | abgelehnt² | Entsch. mit Fehler | Rückfälle | Tokens Prompt / Ausgabe | Kosten |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
