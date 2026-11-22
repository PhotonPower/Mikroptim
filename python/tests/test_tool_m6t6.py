"""Hilfen des Skripts tools/m6t6_zielspezifikation.py (M6-T6b), ohne Programmaufrufe."""

import importlib.util
import json
import os
import types

import pytest

from conftest import ROOT


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

    assert tool.synth_options(args()) == ["--stall", "0"]
    assert tool.synth_options(args(feldregel=True)) == ["--stall", "0", "--field-rule"]
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

    assert tool.synth_options(args(rueckbau=False, elementbudget=None)) == ["--stall", "0"]
    assert tool.synth_options(args(rueckbau=True)) == ["--stall", "0", "--prune"]
    assert tool.synth_options(args(rueckbau=True, rueckbau_faktor=0.5, rueckbau_versuche=6,
                                   rueckbau_stufen=4)) == [
        "--stall", "0", "--prune", "--prune-factor", "0.5", "--prune-attempts", "6",
        "--prune-stages", "4"]
    assert tool.synth_options(args(elementbudget=9, rueckbau_versuche=2)) == [
        "--stall", "0", "--prune-attempts", "2"]
    assert tool.synth_options(args(nur_rueckbau=True)) == ["--stall", "0", "--prune"]
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
