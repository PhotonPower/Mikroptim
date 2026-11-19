#!/usr/bin/env python3
"""Bewertung der Entwurfssteuerung (M10-T7, docs/entwurfssteuerung.md, Abschnitt 8).

Benchmarks (Startsysteme frei erfunden):

    B1  NA 0,3 → 0,6 auf der Achse, monochromatisch (Start von M6-T3)
    B2  Zielspezifikation 50x/0,95, Aperturteil: NA 0,3 → 0,95 bei 10 % Feld, 546 nm
        (Start von M6-T6 mit Feldwinkeln × 0,1; Feld und Spektrum können die Entwurfswerkzeuge
        noch nicht stufenweise ändern, bei vollem Feld wird schon NA 0,3 verworfen)
    B3  40x/0,85 trocken, Aperturteil: NA 0,3 → 0,85 bei 10 % Feld, 546 nm
    B4  Patentsysteme aus dem privaten Verzeichnis – nur lokal, hier nicht umgesetzt

Varianten:

    V0  run_homotopy:            mikroptim synth --target-na NA
    V1  Strahlsuche:             mikroptim synth --beam 3 --alternatives 2
    V2  M6-T6-Skript:            nur B2; tools/m6t6_zielspezifikation.py (eigener Lauf, lang)
    V3  regelbasierte Strategie: mikroptim.design mit RuleDecider (ohne Modell)
    LM  Sprachmodell:            mikroptim.design mit OllamaDecider oder ClaudeDecider
                                 (--decider ollama|claude, --model, Rückfall auf die Regel)
    V4  Mensch:                  mikroptim.design mit HumanDecider (Eingabe auf der Konsole)

Alle Läufe rechnen auf der CPU im Prozess (``mikroptim.run_cli``), damit Trace und lokales
Sprachmodell nicht um die GPU konkurrieren. Ergebnisse pro Lauf als JSON unter --out
(``<B>_<V>[_<modell>].json``), Sitzungen unter ``--work``; ein vorhandenes Ergebnis wird
übersprungen. ``--tabelle`` druckt die Tabelle des Berichts aus den Ergebnisdateien.

Bewertet wird jedes Endsystem gleich: Geisterplatten-Koeffizienten auf 0 (reales System, wie die
Zielprüfung von synth), ``mikroptim trace`` (10 × 8), größter RMS über Feld und Wellenlänge,
versagte Strahlen, NA = Pupillenradius / Brennweite, Elementzahl. Ziel erreicht heißt
NA ≥ Soll-NA, RMS ≤ 1/14 Welle und keine versagten Strahlen.

Beispiele:

    python tools/m10t7_bewertung.py --varianten V0,V1,V3
    python tools/m10t7_bewertung.py --varianten LM --decider ollama --model qwen3:14b --no-think
    python tools/m10t7_bewertung.py --varianten LM --decider ollama --model gpt-oss:20b --think
    python tools/m10t7_bewertung.py --varianten LM --decider claude --model claude-opus-5-5 --max-usd 20
    python tools/m10t7_bewertung.py --tabelle
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
RMS_TARGET = 1.0 / 14.0
#: relative Toleranz der NA (die Brennweite weicht über die Nebenbedingung leicht vom Soll ab)
NA_TOLERANCE = 1e-3

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
    ``field_scale`` (B2, B3)."""
    from mikroptim.design.files import read_system, resolve_catalogs, write_system

    data = resolve_catalogs(read_system(bench["start"]), pathlib.Path(bench["start"]).parent)
    k = bench.get("field_scale", 1.0)
    if k != 1.0:
        src = data["source"]
        src["field_angles_deg"] = [a * k for a in src.get("field_angles_deg", [0.0])]
        data["name"] = f"{data.get('name', '')} (Feldwinkel × {k:g})"
    path = pathlib.Path(work_root) / f"start_{name}.json"
    write_system(data, path)
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


def make_llm(args, ledger):
    from mikroptim import design

    if args.decider == "ollama":
        return design.OllamaDecider(args.model, host=args.host, think=args.think,
                                    keep_alive=args.keep_alive)
    return design.ClaudeDecider(args.model or "claude-opus-5-5", effort=args.effort,
                                ledger=ledger, strict=not args.no_strict)


def tag_of(variant, args):
    if variant != "LM":
        return variant
    model = args.model or "claude-opus-5-5"
    think = "" if args.decider != "ollama" or args.think is None else (
        "_think" if args.think else "_nothink")
    return "LM_" + (model.replace(":", "-").replace("/", "-")) + think


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmarks", default="B1,B2,B3")
    ap.add_argument("--varianten", default="V0,V1,V3")
    ap.add_argument("--out", default=str(ROOT / "docs" / "ergebnisse" / "m10-t7"))
    ap.add_argument("--work", default=None, help="Arbeitsverzeichnis der Läufe (Sitzungen, "
                    "Systeme); Standard: <out>/../m10-t7-arbeit (nicht im Repository)")
    ap.add_argument("--steps", type=int, default=None, help="Entscheidungen pro Sitzung")
    ap.add_argument("--decider", choices=["ollama", "claude"], default="ollama")
    ap.add_argument("--model")
    ap.add_argument("--host", default="http://localhost:11434")
    th = ap.add_mutually_exclusive_group()
    th.add_argument("--think", dest="think", action="store_true", default=None)
    th.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--keep-alive", default="2m")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--no-strict", action="store_true",
                    help="Claude: Werkzeuge ohne strict (falls die API die Schemas ablehnt)")
    ap.add_argument("--cost-file", default=None,
                    help="Kostensumme der Claude-Läufe (Standard <work>/claude_cost.json)")
    ap.add_argument("--tabelle", action="store_true")
    ap.add_argument("--archiv", action="store_true",
                    help="Protokolle und beste Systeme ins Ergebnisverzeichnis kopieren")
    ap.add_argument("--neu-bewerten", action="store_true",
                    help="vorhandene Ergebnisse mit den aktuellen Regeln neu bewerten")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    work_root = pathlib.Path(args.work) if args.work else out.parent / "m10-t7-arbeit"
    if args.neu_bewerten:
        reassess(out)
    if args.archiv:
        archive(out)
    if (args.neu_bewerten or args.archiv) and not args.tabelle:
        return 0
    if args.tabelle:
        print(table(out))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    from mikroptim import design

    ledger = None
    if "LM" in args.varianten.split(",") and args.decider == "claude":
        ledger = design.CostLedger(args.max_usd, args.cost_file or work_root / "claude_cost.json")
    llm = None
    try:
        for b in args.benchmarks.split(","):
            work_root.mkdir(parents=True, exist_ok=True)
            bench = prepare_start(BENCHMARKS[b], work_root, b)
            for v in args.varianten.split(","):
                tag = tag_of(v, args)
                res_path = out / f"{b}_{tag}.json"
                if res_path.exists():
                    print(f"{b} {tag}: vorhanden, übersprungen")
                    continue
                work = work_root / b / tag
                work.mkdir(parents=True, exist_ok=True)
                steps = args.steps or bench["steps"]
                print(f"{b} {tag}: läuft …", flush=True)
                if v == "V0":
                    res = run_synth(bench, work)
                elif v == "V1":
                    res = run_synth(bench, work, beam=True)
                elif v == "V3":
                    res = run_session(bench, work, design.RuleDecider(), None, steps)
                elif v == "LM":
                    if args.decider == "ollama" and not args.model:
                        raise SystemExit("--model fehlt")
                    llm = llm or make_llm(args, ledger)
                    res = run_session(bench, work, llm, design.RuleDecider(), steps)
                    res["decider"] = {k: v2 for k, v2 in llm.describe().items()
                                      if k not in ("trace", "usage")}
                    if ledger is not None:
                        res["ledger_spent_usd"] = ledger.spent_usd
                elif v == "V4":  # Mensch auf der Kommandozeile, gleiche Werkzeuge und Prüfungen
                    res = run_session(bench, work, design.HumanDecider(), None, steps)
                else:
                    raise SystemExit(f"Variante {v}: im Skript nicht umgesetzt (V2: "
                                     "tools/m6t6_zielspezifikation.py)")
                res.update({"benchmark": b, "variant": v, "tag": tag, "target_na": bench["na"]})
                aborted = res.get("aborted") or ""
                if any(k in aborted for k in ("Unavailable", "PrivacyError")):
                    # Konfigurationsfehler (SDK, Zugangsdaten, private Daten): kein Ergebnis
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
    head = ("| Bench. | Variante | Ziel | NA | RMS max [Wellen] | RMS AP³ | Elemente | LM-Läufe¹ | Zeit [s] | "
            "Entsch. | abgelehnt² | Entsch. mit Fehler | Rückfälle | Tokens Prompt / Ausgabe | Kosten |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
