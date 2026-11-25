"""Hilfen des Skripts tools/m6t6_zielspezifikation.py (M6-T6b), ohne Programmaufrufe."""

import importlib.util
import json
import os
import types

import pytest

from conftest import ROOT

#: synth-Optionen ohne --aktuell/--farbregel: wie im dokumentierten Lauf (ohne Stagnationsabbruch,
#: seit M6-T13b auch ohne Vergrößern mit Abstand zur Schwelle)
LEGACY = ["--stall", "0", "--grow-margin", "0"]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "m6t6", ROOT / "tools" / "m6t6_zielspezifikation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _system(tmp_path, reference=None):
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    if reference:
        s["reference_sphere"] = reference
    path = tmp_path / "system.json"
    path.write_text(json.dumps(s), encoding="utf-8")
    return s, path


def test_psf_retries_with_numerical_aperture(tool, tmp_path, monkeypatch):
    """eval --psf scheitert bei vollem Feld mit „NA vorgeben“: das Skript wiederholt mit --na
    (NA = Pupillenradius / Brennweite) statt abzubrechen (Fehler aus M6-T7)."""
    s, path = _system(tmp_path, "image_infinite")
    calls = []

    def fake_run(args, stdout, stderr, cwd):
        calls.append(list(args))
        if ("--psf" in args or "--vector" in args) and "--na" not in args:
            stderr.write("Fehler: PSF: NA der Pupillenabbildung für Feld 2, Wellenlänge 0 nicht "
                         "bestimmbar (Feldstrahl versagt oder afokal); NA vorgeben\n")
            return types.SimpleNamespace(returncode=1)
        stdout.write("{}")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    runner = tool.Runner("mikroptim", str(tmp_path / "out"), synth_options=[])
    out = tool.evaluate(runner, "a3", str(path), grid=16)
    assert sorted(out) == ["psf", "pupille", "unendlich", "vektor"]
    na = s["surfaces"][0]["semi_diameter"] / s["target"]["focal_length"]
    assert tool.psf_numerical_aperture(s) == pytest.approx(na)
    retried = [c for c in calls if "--na" in c]
    assert len(retried) == 2
    assert all(float(c[c.index("--na") + 1]) == na for c in retried)
    second = next(c for c in calls if "--reference" in c and "exit-pupil" not in c)
    assert second[second.index("--reference") + 1] == "infinite"
    assert (tmp_path / "out" / "a3_psf_na.txt").exists()


def test_eval_failure_leaves_no_result_file(tool, tmp_path, monkeypatch):
    _, path = _system(tmp_path)

    def fake_run(args, stdout, stderr, cwd):
        stderr.write("anderer Fehler\n")
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    runner = tool.Runner("mikroptim", str(tmp_path / "out"), synth_options=[])
    with pytest.raises(RuntimeError):
        tool.evaluate(runner, "x", str(path), grid=16)
    # eine Fortsetzung darf das fehlgeschlagene Ergebnis nicht als vorhanden überspringen
    assert not (tmp_path / "out" / "x_pupille.json").exists()


def test_reference_sphere_default_and_f12_guard(tool):
    s = {}
    tool.set_reference_sphere(s, "image_infinite")
    assert s["reference_sphere"] == "image_infinite"
    tool.set_reference_sphere(s, "last_surface")
    assert s["reference_sphere"] == "last_surface"

    def args(ref, ohne=False, mit=False):
        return types.SimpleNamespace(referenzsphaere=ref, ohne_f12_schutz=ohne, mit_f12_schutz=mit)

    assert tool.f12_guard(args("last_surface"))
    assert not tool.f12_guard(args("last_surface", ohne=True))
    assert not tool.f12_guard(args("image_infinite"))
    assert not tool.f12_guard(args("image_exit_pupil"))
    assert tool.f12_guard(args("image_infinite", mit=True))


def test_synth_options_field_rule(tool):
    """M6-T8: --feldregel reicht synth --field-rule durch; ohne bleibt die Liste wie bisher."""
    def args(**kw):
        base = dict(aktuell=False, farbregel=False, farb_achromat=False, sinusbedingung=None,
                    feldregel=False, synth_arg=[])
        base.update(kw)
        return types.SimpleNamespace(**base)

    assert tool.synth_options(args()) == LEGACY
    assert tool.synth_options(args(feldregel=True)) == LEGACY + ["--field-rule"]
    assert tool.synth_options(args(feldregel=True, farbregel=True)) == [
        "--color-rule", "--field-rule"]


def test_synth_options_prune(tool):
    """M6-T9: --rueckbau reicht synth --prune und die Einstellungen durch, --elementbudget nur die
    Einstellungen; limit_elements setzt das Budget statt --max-elements anzuhängen."""
    def args(**kw):
        base = dict(aktuell=False, farbregel=False, farb_achromat=False, sinusbedingung=None,
                    feldregel=False, synth_arg=[])
        base.update(kw)
        return types.SimpleNamespace(**base)

    assert tool.synth_options(args(rueckbau=False, elementbudget=None)) == LEGACY
    assert tool.synth_options(args(rueckbau=True)) == LEGACY + ["--prune"]
    assert tool.synth_options(args(rueckbau=True, rueckbau_faktor=0.5, rueckbau_versuche=6,
                                   rueckbau_stufen=4)) == LEGACY + [
        "--prune", "--prune-factor", "0.5", "--prune-attempts", "6",
        "--prune-stages", "4"]
    assert tool.synth_options(args(elementbudget=9, rueckbau_versuche=2)) == LEGACY + [
        "--prune-attempts", "2"]
    assert tool.synth_options(args(nur_rueckbau=True)) == LEGACY + ["--prune"]
    assert tool.limit_elements(["--rings", "16"], 7) == ["--rings", "16", "--max-elements", "7"]
    assert tool.limit_elements(["--element-budget", "9"], 7) == ["--element-budget", "7"]
    assert tool.limit_elements(["--element-budget", "5"], 7) == ["--element-budget", "5"]


def test_remove_dummy_planes_merged_gap_once(tool):
    """M6-T9: Haben beide Spalte um eine Hilfsebene Dickenvariablen (in verschiedenen Einträgen),
    steht der vereinigte Spalt danach nur einmal in den Variablen."""
    s = {"surfaces": [{"type": "stop", "thickness": 0.5, "material": "AIR"},
                      {"type": "plane", "thickness": 0.1, "material": "AIR"},
                      {"type": "sphere", "curvature": 0.1, "thickness": 1.0, "material": "N-BK7"},
                      {"type": "sphere", "curvature": -0.1, "thickness": 0.2, "material": "AIR"},
                      {"type": "sphere", "curvature": 0.1, "thickness": 5.0, "material": "AIR"}],
         "variables": {"items": [{"type": "thickness", "surface": 0},
                                 {"type": "thickness", "surfaces": [1, 3]},
                                 {"type": "thickness", "surface": 1}]},
         "constraints": {"items": []}}
    tool.remove_dummy_planes(s)
    assert len(s["surfaces"]) == 4
    assert s["surfaces"][0]["thickness"] == 0.6
    assert s["variables"]["items"] == [{"type": "thickness", "surface": 0},
                                       {"type": "thickness", "surfaces": [2]}]


def test_system_state_and_stage_done(tool):
    """M6-T10: Punkt eines realen Systems (NA aus Pupille und Brennweite, Feldanteil aus dem
    größten Feldwinkel, Spektrum) und Ziel der Stufen."""
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    st = tool.system_state(s)
    assert st["aperture"] == pytest.approx(0.987 / 3.29)
    assert st["field"] == pytest.approx(1.0, rel=1e-5)
    assert st["spectrum"] == "mono"
    assert st["elements"] == 1
    tool.set_field(s, [a * 0.25 for a in tool.FIELD_ANGLES], tool.FIELD_WEIGHTS)
    tool.set_spectrum(s, *tool.SPECTRA["voll"])
    st = tool.system_state(s)
    assert st["field"] == pytest.approx(0.25)
    assert st["spectrum"] == "voll"
    assert not tool.stage_done("feld", st)
    assert tool.stage_done("farbe", st)
    assert not tool.stage_done("apertur", st)
    assert tool.stage_done("apertur", dict(st, aperture=0.95))
    assert tool.stage_done("feld", dict(st, field=1.0))


def test_return_to_ghosts_command(tool, tmp_path, monkeypatch):
    """M6-T10: Rückkehr zu Geisterplatten über mikroptim insert --ghost-return mit den Gläsern
    des Startsystems; der JSON-Bericht wird auch vor Warnungen gefunden."""
    calls = []

    def fake_run(args, stdout, stderr, cwd):
        calls.append(list(args))
        stdout.write('{"details": {"positions": []}, "relative_change": 0.0}\n'
                     "mikroptim insert: Warnung: field: Spalt nach Fläche 3 schon belegt\n")
        out = args[args.index("-o") + 1]
        with open(out, "w", encoding="utf-8") as f:
            f.write("{}")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    runner = tool.Runner("mikroptim", str(tmp_path / "out"), synth_options=[])
    path, report = tool.return_to_ghosts(runner, "c0_feld_start", "real.json",
                                         positions="aperture,field,front")
    assert path.endswith("c0_feld_start.json")
    assert report["relative_change"] == 0.0
    a = calls[0]
    assert a[1:3] == ["insert", "real.json"]
    assert "--ghost-return" in a
    assert a[a.index("--ghost-positions") + 1] == "aperture,field,front"
    assert a[a.index("--ghost-order") + 1] == "10"
    glasses = [a[i + 1] for i, x in enumerate(a) if x == "--ghost-glass"]
    assert glasses == ["SYN-36", "CAF2", "SYN-30"]
    # Fortsetzung: vorhandene Datei wird nicht neu erzeugt
    tool.return_to_ghosts(runner, "c0_feld_start", "real.json")
    assert len(calls) == 1


def test_cycle_run_stops_at_goal_or_stall(tool, tmp_path, monkeypatch):
    """M6-T10: Der zyklische Ablauf überspringt Stufen am Ziel, beginnt jede Stufe am Ergebnis
    der vorigen (nach dem Rückbau) und endet, wenn ein Durchlauf nichts mehr öffnet."""
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    tool.set_field(s, [a * 0.5 for a in tool.FIELD_ANGLES], tool.FIELD_WEIGHTS)
    start = tmp_path / "start.json"
    start.write_text(json.dumps(s), encoding="utf-8")
    seen = []

    def fake_stage(r, args, k, kind, cur, common, relax, rings, log=print):
        seen.append((k, kind, os.path.basename(cur)))
        x = json.loads(open(cur, encoding="utf-8").read())
        if kind == "feld":  # öffnet nur bis 80 %
            f = min(0.8, tool.system_state(x)["field"] + 0.3)
            tool.set_field(x, [a * f for a in tool.FIELD_ANGLES], tool.FIELD_WEIGHTS)
        elif kind == "farbe":
            tool.set_spectrum(x, *tool.SPECTRA["voll"])
        out = os.path.join(r.out, f"c{k}_{kind}_z2r_s0.json")
        tool.save(x, out)
        return {"kind": kind, "tag": f"c{k}_{kind}", "z2r": out,
                "end_state": tool.system_state(x)}

    monkeypatch.setattr(tool, "cycle_stage", fake_stage)
    args = types.SimpleNamespace(
        exe="mikroptim", out=str(tmp_path / "zyklus"), gpu=False, aktuell=False, farbregel=False,
        farb_achromat=False, sinusbedingung=None, feldregel=True, synth_arg=[],
        elementbudget=12, rueckbau=False, start=str(start), referenzsphaere="image_infinite",
        zyklus="feld,farbe", zyklus_wiederholen=3, geister_lagen="aperture,field")
    assert tool.cycle_run(args) == 0
    # Durchlauf 1: feld (0,5 → 0,8), farbe; Durchlauf 2: feld (bleibt 0,8), farbe am Ziel;
    # kein Fortschritt → Ende
    assert [(k, kind) for k, kind, _ in seen] == [(0, "feld"), (1, "farbe"), (2, "feld")]
    assert seen[1][2] == "c0_feld_z2r_s0.json"
    summary = json.loads((tmp_path / "zyklus" / "m6t10_zyklus.json").read_text(encoding="utf-8"))
    assert summary["final_state"]["spectrum"] == "voll"
    assert summary["final_state"]["field"] == pytest.approx(0.8)
    assert len(summary["stages"]) == 3


def test_synth_options_color_stage(tool):
    """M6-T11: Optionen der Farbstufe werden durchgereicht; ohne sie bleibt die Liste wie bisher."""
    def args(**kw):
        base = dict(aktuell=False, farbregel=True, farb_achromat=False, sinusbedingung=None,
                    feldregel=False, synth_arg=[])
        base.update(kw)
        return types.SimpleNamespace(**base)

    assert tool.synth_options(args()) == ["--color-rule"]
    assert tool.synth_options(args(farb_spektrum_rang=True, farb_startradien=True,
                                   farb_chromatisch=0.5, farb_modul="doublet",
                                   farb_modul_immer=True, farb_anomal=True)) == [
        "--color-rule", "--split-spectrum", "--split-member-start", "--color-chromatic", "0.5",
        "--color-module", "doublet", "--color-module-always", "--color-module-anomalous"]
    # ohne Modul keine Modul-Unteroptionen
    assert tool.synth_options(args(farb_modul_immer=True, farb_anomal=True)) == ["--color-rule"]
    # M6-T11b: Rang und Startradien sind Standard von synth, abschaltbar
    assert tool.synth_options(args(ohne_farb_spektrum_rang=True, ohne_farb_startradien=True)) == [
        "--color-rule", "--no-split-spectrum", "--no-split-member-start"]


def test_synth_options_ghost_dictionary(tool):
    """M6-T12: erweitertes Glas-Wörterbuch wird an jeden synth-Aufruf durchgereicht; ohne Option
    bleibt die Liste wie bisher; --woerterbuch-extra ohne Zahl nimmt die Standardgröße."""
    def args(**kw):
        base = dict(aktuell=False, farbregel=False, farb_achromat=False, sinusbedingung=None,
                    feldregel=False, synth_arg=[])
        base.update(kw)
        return types.SimpleNamespace(**base)

    assert tool.synth_options(args()) == LEGACY
    assert tool.synth_options(args(woerterbuch_extra=2, woerterbuch_glas=["SYN-21"])) == LEGACY + [
        "--ghost-dictionary-extra", "2", "--ghost-dictionary-glass", "SYN-21"]
    assert tool.synth_options(args(woerterbuch_extra=0)) == LEGACY
    assert isinstance(tool.WOERTERBUCH_STANDARD, int) and 2 <= tool.WOERTERBUCH_STANDARD <= 4


def test_prune_point(tool):
    """M6-T11: Rückbau bei Q × Parameter der Stufe: Apertur (Pupillenradius), Feld, Farbe
    (Spektrumbreite um λ_ref)."""
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    f = s.get("target", {}).get("focal_length", tool.FOCAL)
    a = json.loads(json.dumps(s))
    assert tool.prune_point(a, "apertur", 0.85, 0.6, 0.5) == pytest.approx((0.51, 0.5))
    assert a["surfaces"][0]["semi_diameter"] == pytest.approx(f * 0.51)
    b = json.loads(json.dumps(s))
    assert tool.prune_point(b, "feld", 0.8, 0.6, 0.5) == pytest.approx((0.6, 0.4))
    assert b == s  # Feld über --start-field, die Datei bleibt
    c = json.loads(json.dumps(s))
    tool.set_spectrum(c, *tool.SPECTRA["voll"])
    assert tool.prune_point(c, "farbe", 0.5, 0.6, 0.5) == (0.6, 0.5)
    ref = c["spectrum"]["reference_nm"]
    for w, w0 in zip(c["spectrum"]["wavelengths_nm"], tool.SPECTRA["voll"][0]):
        assert w == pytest.approx(ref + 0.5 * (w0 - ref))


def test_reopen_after_prune(tool, tmp_path, monkeypatch):
    """M6-T11: nach dem Rückbau bei Q × NA zurück zum erreichten Punkt ohne Einfügen; erreicht
    synth ihn nicht, bleibt der Stand vor dem Rückbau (None)."""
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    r = tool.Runner("mikroptim", str(tmp_path))
    pruned = tmp_path / "c0_apertur_rueckbau.json"
    tool.save(s, str(pruned))
    calls = []
    reach = {"na": 0.6}

    def fake_synth(name, start, options):
        calls.append((name, options))
        tool.save({"result": {"status": "goal_reached",
                              "point": {"aperture": reach["na"], "field": 1.0}}},
                  r.path(name + "_log.json"))
        out = r.path(name + ".json")
        tool.save(s, out)
        return out

    monkeypatch.setattr(r, "synth", fake_synth)
    out, info = tool.reopen_after_prune(r, "c0_apertur", str(pruned), "apertur", 0.6, 1.0, {},
                                        ["--rings", "16"], [])
    assert out == r.path("c0_apertur_wieder.json") and info["reached"]
    name, opts = calls[0]
    assert name == "c0_apertur_wieder"
    assert opts[:2] == ["--target-na", "0.6"]
    assert opts[-2:] == ["--max-elements", str(tool.glass_elements(s))]
    reach["na"] = 0.55
    out, info = tool.reopen_after_prune(r, "c1_apertur", str(pruned), "apertur", 0.6, 1.0, {},
                                        [], [])
    assert out is None and not info["reached"]
    out, info = tool.reopen_after_prune(r, "c2_farbe", str(pruned), "farbe", 0.6, 1.0, {}, [], [])
    assert out == str(pruned) and info == {"reached": True, "via": "z2"}


def _fake_main_path(tool, tmp_path, monkeypatch, argv):
    """Hauptlauf ohne Programmaufrufe: synth_loop, polish und evaluate aufgezeichnet; liefert die
    Liste (Stufe, Startdatei, Optionen, Fortschritt) und die Zusatzoptionen der synth-Aufrufe."""
    s = json.loads((ROOT / "data" / "systems" / "zielspezifikation_start.json")
                   .read_text(encoding="utf-8"))
    tmp_path.mkdir(parents=True, exist_ok=True)
    start = tmp_path / "start.json"
    start.write_text(json.dumps(s), encoding="utf-8")
    points = {"p1": (0.5, 0.1), "p1f": (0.5, 0.1), "p2": (0.5, 0.4), "p3": (0.6, 0.4),
              "p4": (0.6, 0.4), "p5": (0.6, 0.4)}
    calls, runner = [], {}

    def fake_loop(r, tag, start_file, synth_options, relax, progress, done, **kw):
        runner["r"] = r
        opts = synth_options(None) if callable(synth_options) else synth_options
        x = tool.load(start_file)
        na, f = points[tag]
        pt = {"aperture": na, "field": f, "spectrum_width": 1.0}
        calls.append({"tag": tag, "start": os.path.basename(start_file),
                      "spectrum": list(x["spectrum"]["wavelengths_nm"]), "opts": opts,
                      "progress": progress(pt), "kw": kw})
        out = r.path(tag + "_final.json")
        tool.save(x, out)
        tool.save({"result": {"point": pt}}, r.path(tag + "_r0_log.json"))
        return out, [{"synth": tag + "_r0", "point": pt}]

    polished = []

    def fake_polish(r, name, src, relax, spectrum="erweitert"):
        polished.append((name, spectrum))
        return src

    monkeypatch.setattr(tool, "synth_loop", fake_loop)
    monkeypatch.setattr(tool, "polish", fake_polish)
    monkeypatch.setattr(tool, "evaluate", lambda r, name, src, grid=256: {})
    monkeypatch.setattr(tool, "has_glass_variables", lambda s: False)
    monkeypatch.setattr(tool.sys, "argv", ["m6t6", "--exe", "mikroptim", "--out",
                                           str(tmp_path / "out"), "--start", str(start)] + argv)
    assert tool.main() == 0
    return calls, runner["r"].synth_options, polished


def test_color_mode_default_is_color_early(tool, tmp_path, monkeypatch):
    """M6-T13: Standard „Farbe früh“ (--farbe apertur): stetige Farbstufe p1f am Ende der
    Aperturstufe bei 10 % Feld direkt zu 436–656 nm, danach Feld- und Aperturstufe am
    polychromatischen System; Farbwerkzeuge an; Z2 am Spektrum der Feldstufe."""
    calls, opts, polished = _fake_main_path(tool, tmp_path, monkeypatch, [])
    assert [c["tag"] for c in calls] == ["p1", "p1f", "p2", "p3"]
    assert calls[0]["spectrum"] == [546.07]
    p1f = calls[1]
    assert p1f["start"] == "p1f_in.json" and p1f["spectrum"] == tool.SPECTRA["voll"][0]
    assert "--spectrum-continuous" in p1f["opts"]
    i = p1f["opts"].index("--start-field")
    assert p1f["opts"][i + 1] == "0.1" and p1f["opts"][i + 3] == "0.1"
    assert p1f["progress"] == 1.0  # Fortschritt = Spektrumbreite
    assert p1f["kw"]["stage_budget"] == 4
    assert calls[2]["start"] == "p1f_final.json"  # Feldstufe ab dem polychromatischen System
    assert calls[2]["spectrum"] == tool.SPECTRA["voll"][0]
    assert calls[3]["start"] == "p2_final.json"
    for c in calls[2:]:
        assert "--spectrum-continuous" not in c["opts"]
    for o in ["--color-rule", "--color-chromatic", "--color-module", "--field-rule",
              "--insertion-improve"]:
        assert o in opts
    assert "--stall" not in opts
    assert ("z2", None) in polished
    summary = json.loads((tmp_path / "out" / "m6t6_lauf.json").read_text(encoding="utf-8"))
    assert summary["color_mode"] == "apertur"


def test_color_mode_late_and_legacy_unchanged(tool, tmp_path, monkeypatch):
    """M6-T13: --farbe spaet läuft wie der Standard bis M6-T12 (monochromatische Stufen 1–3,
    Farbstufen p4/p5 danach, synth --stall 0, Z2 monochromatisch); --farbe-frueh allein wie
    bisher; --farbe anfang mit dem vollen Spektrum ab dem Start."""
    calls, opts, polished = _fake_main_path(tool, tmp_path / "a", monkeypatch, ["--farbe", "spaet"])
    assert [c["tag"] for c in calls] == ["p1", "p2", "p3", "p4", "p5"]
    assert opts == LEGACY
    assert ("z2", "mono") in polished
    assert calls[3]["progress"] == 0.0 and calls[3]["kw"]["stage_budget"] == 1
    assert "--spectrum-continuous" not in calls[3]["opts"]
    summary = json.loads((tmp_path / "a" / "out" / "m6t6_lauf.json").read_text(encoding="utf-8"))
    assert "color_mode" not in summary

    calls, opts, _ = _fake_main_path(tool, tmp_path / "b", monkeypatch, ["--farbe-frueh"])
    assert [c["tag"] for c in calls] == ["p1", "p2", "p4", "p5", "p3"]
    assert opts == LEGACY

    calls, opts, polished = _fake_main_path(tool, tmp_path / "c", monkeypatch,
                                            ["--farbe", "anfang"])
    assert [c["tag"] for c in calls] == ["p1", "p2", "p3"]
    assert calls[0]["spectrum"] == tool.SPECTRA["voll"][0]
    assert "--color-rule" in opts and ("z2", None) in polished


def test_apply_color_mode(tool):
    """M6-T13: Ablauf und Werkzeuge; angegebene Optionen bleiben, Widersprüche sind Fehler."""
    def args(**kw):
        base = dict(farbe=None, farbe_frueh=False, farbe_von_anfang=None, gemeinsam=False,
                    farbregel=False, farb_stetig=False, farb_ziel=None, farb_modul=None,
                    farb_chromatisch=None, feldregel=False, einfuegen_verbessern=None,
                    farb_budget=None)
        base.update(kw)
        return types.SimpleNamespace(**base)

    a = args()
    assert tool.apply_color_mode(a) == tool.FARBE_STANDARD == "apertur"
    for k, v in tool.COLOR_TOOLS.items():
        assert getattr(a, k) == v
    a = args(farb_chromatisch=0.3, farb_budget=2, einfuegen_verbessern=6)
    tool.apply_color_mode(a)
    assert (a.farb_chromatisch, a.farb_budget, a.einfuegen_verbessern) == (0.3, 2, 6)
    a = args(farbe="spaet")
    assert tool.apply_color_mode(a) == "spaet" and not a.farbregel and a.farb_budget is None
    a = args(farbe_von_anfang="zwei")
    assert tool.apply_color_mode(a) == "spaet" and not a.farbregel
    a = args(farbe="frueh")
    assert tool.apply_color_mode(a) == "frueh" and a.farbe_frueh and not a.farbregel
    with pytest.raises(SystemExit):
        tool.apply_color_mode(args(farbe="apertur", farbe_frueh=True))
    with pytest.raises(SystemExit):
        tool.apply_color_mode(args(farbe="apertur", gemeinsam=True))
    assert tool.farb_budget(types.SimpleNamespace()) == 1


def test_synth_options_color_variants(tool):
    """M6-T13: --farb-varianten reicht synth --color-variants durch."""
    a = types.SimpleNamespace(aktuell=False, farbregel=True, farb_achromat=False,
                              sinusbedingung=None, feldregel=False, synth_arg=[],
                              farb_varianten=True)
    assert tool.synth_options(a) == ["--color-rule", "--color-variants"]
    a.wachsen_marge = 0.5
    assert tool.synth_options(a) == ["--color-rule", "--color-variants", "--grow-margin", "0.5"]
    a.einfuegen_abschluss = True
    assert tool.synth_options(a) == ["--color-rule", "--color-variants",
                                     "--insertion-improve-finish", "--grow-margin", "0.5"]


def test_synth_options_m6t13b_switches(tool):
    """M6-T13b: Varianten, Einfügen im Abschluss und Wachsen sind Standard von synth; die Schalter
    --ohne-farb-varianten, --ohne-einfuegen-abschluss und --wachsen-marge 0 reichen die Abschalter
    durch (vor den positiven Optionen, auch wenn COLOR_TOOLS diese setzt); ohne --aktuell bzw.
    --farbregel rechnet synth wie vor M6-T13b (--grow-margin 0, mit --einfuegen-verbessern
    --no-insertion-improve-finish)."""
    def args(**kw):
        base = dict(aktuell=False, farbregel=True, farb_achromat=False, sinusbedingung=None,
                    feldregel=False, synth_arg=[])
        base.update(kw)
        return types.SimpleNamespace(**base)

    assert tool.synth_options(args(farb_varianten=True, ohne_farb_varianten=True,
                                   einfuegen_abschluss=True, ohne_einfuegen_abschluss=True,
                                   wachsen_marge=0.0)) == [
        "--color-rule", "--no-color-variants", "--no-insertion-improve-finish",
        "--grow-margin", "0.0"]
    # ohne --aktuell/--farbregel: alte Standards
    legacy = args(farbregel=False, einfuegen_verbessern=4)
    assert tool.synth_options(legacy) == LEGACY + [
        "--insertion-improve", "4", "--no-insertion-improve-finish"]
    assert tool.synth_options(args(farbregel=False, wachsen_marge=0.8)) == [
        "--stall", "0", "--grow-margin", "0.8"]
    assert tool.synth_options(args(farbregel=False, einfuegen_verbessern=2,
                                   einfuegen_abschluss=True)) == LEGACY + [
        "--insertion-improve-finish", "--insertion-improve", "2"]
    # COLOR_TOOLS überschreibt eine ausdrückliche 0 nicht
    a = types.SimpleNamespace(farbe="apertur", farbe_frueh=False, farbe_von_anfang=None,
                              gemeinsam=False, wachsen_marge=0.0)
    tool.apply_color_mode(a)
    assert a.wachsen_marge == 0.0 and a.farb_varianten is True


def test_reference_point_and_distortion(tool):
    """M6-T14 (ADR-0019): --bildpunkt hauptstrahl setzt reference_point, --verzeichnung D[,W]
    ersetzt bzw. hängt den Eintrag distortion an; ohne Schalter bleibt die Datei unverändert."""
    s = {"reference_sphere": "image_infinite",
         "constraints": {"items": [{"type": "focal_length", "weight": 1.0},
                                   {"type": "distortion", "max_distortion": 0.5}]}}
    before = json.loads(json.dumps(s))
    tool.set_reference_point(s, types.SimpleNamespace(bildpunkt=None, verzeichnung=None))
    assert s == before
    tool.set_reference_point(s, types.SimpleNamespace(bildpunkt="hauptstrahl",
                                                      verzeichnung="0.01"))
    assert s["reference_point"] == "chief_ray"
    assert s["constraints"]["items"] == [
        {"type": "focal_length", "weight": 1.0},
        {"type": "distortion", "max_distortion": 0.01, "weight": tool.DISTORTION_WEIGHT}]
    tool.set_reference_point(s, types.SimpleNamespace(bildpunkt="schwerpunkt", verzeichnung=None))
    assert s["reference_point"] == "centroid"
    tool.set_reference_point(s, types.SimpleNamespace(bildpunkt="ideal", verzeichnung="0,2e3"))
    assert "reference_point" not in s
    assert s["constraints"]["items"][-1] == {"type": "distortion", "max_distortion": 0.0,
                                             "weight": 2000.0}
    t = {}
    tool.set_reference_point(t, types.SimpleNamespace(bildpunkt=None, verzeichnung="0.005,10"))
    assert t == {"constraints": {"items": [{"type": "distortion", "max_distortion": 0.005,
                                            "weight": 10.0}]}}
    with pytest.raises(ValueError):
        tool.parse_distortion("-0.01")


def test_synth_options_lateral_rays(tool):
    """M6-T14: --farb-querfarbe-strahlen reicht synth --color-lateral-rays durch (mit Modul)."""
    a = types.SimpleNamespace(aktuell=False, farbregel=True, farb_achromat=False,
                              sinusbedingung=None, feldregel=False, synth_arg=[],
                              farb_modul="doublet", farb_querfarbe_strahlen=True)
    assert tool.synth_options(a) == ["--color-rule", "--color-module", "doublet",
                                     "--color-lateral-rays"]
    a.farb_modul = None
    assert tool.synth_options(a) == ["--color-rule"]
