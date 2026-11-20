"""Optionen des Skripts tools/m10t7_bewertung.py (M10-T7c, Entscheidung Maintainer 2026-11-20):
Referenzsphäre der Benchmarks und Kontextlänge für Ollama, ohne Rechnung und ohne Netz."""

import importlib.util
import json

import pytest

from conftest import ROOT, SYSTEMS


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("m10t7", ROOT / "tools" / "m10t7_bewertung.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_reference_sphere_is_image_infinite(tool):
    args = tool.parser().parse_args([])
    assert args.referenzsphaere == "image_infinite"
    assert args.out.endswith("m10-t7c")
    assert tool.RULE_REFERENCE == "V3RS"
    assert [tool.tag_of(v, args) for v in ("V0", "V1", "V3", "V3AL", "V4")] == [
        "V0RS", "V1RS", "V3RS", "V3ALRS", "V4RS"]
    for v in ("V0", "V3", "V3AL"):
        assert tool.variant_config(v, args.referenzsphaere)[1] == "image_infinite"
    # Namen aus M10-T7b bleiben gültig und meinen dasselbe
    assert tool.variant_config("V3RS", args.referenzsphaere) == ("V3", "image_infinite")
    assert tool.tag_of("V3RS", args) == "V3RS"
    lm = tool.parser().parse_args(["--decider", "claude", "--model", "claude-sonnet-5-5"])
    assert tool.tag_of("LM", lm) == "LMRS_claude-sonnet-5-5"
    with pytest.raises(SystemExit):
        tool.variant_config("V9")


def test_alt_variants_reproduce_old_convention(tool):
    args = tool.parser().parse_args(["--decider", "ollama", "--model", "qwen3:14b",
                                     "--no-think"])
    # Kürzel ohne Zusatz = Ergebnisnamen von M10-T7/M10-T7b
    assert [tool.tag_of(v, args) for v in ("V0-alt", "V1-alt", "V3-alt", "V3AL-alt")] == [
        "V0", "V1", "V3", "V3AL"]
    assert tool.tag_of("LM-alt", args) == "LM_qwen3-14b_nothink"
    assert tool.tag_of("LM", args) == "LMRS_qwen3-14b_nothink"
    old = tool.parser().parse_args(["--referenzsphaere", "last_surface"])
    assert tool.tag_of("V3", old) == "V3"
    assert tool.tag_of("V3RS", old) == "V3RS"  # ausdrücklicher Name gewinnt
    ap = tool.parser().parse_args(["--referenzsphaere", "image_exit_pupil"])
    assert tool.tag_of("V3", ap) == "V3RSAP"


@pytest.mark.parametrize("bench", ["B1", "B2"])
def test_prepare_start_sets_field_only_for_new_convention(tool, tmp_path, bench):
    """image_infinite steht in der Startkopie; last_surface lässt die Datei ohne Feld (bitgleich
    zu M10-T7b); die Systemdatei unter data/systems bleibt unverändert."""
    b = tool.BENCHMARKS[bench]
    original = b["start"].read_bytes()
    assert "reference_sphere" not in json.loads(original)
    new = tool.prepare_start({**b, "reference_sphere": "image_infinite"}, tmp_path, "neu")
    alt = tool.prepare_start({**b, "reference_sphere": "last_surface"}, tmp_path, "alt")
    plain = tool.prepare_start(b, tmp_path, "ohne")  # Aufruf wie in M10-T7b
    d_new = json.loads(new["start"].read_text(encoding="utf-8"))
    d_alt = json.loads(alt["start"].read_text(encoding="utf-8"))
    assert d_new["reference_sphere"] == "image_infinite"
    assert "reference_sphere" not in d_alt
    assert alt["start"].read_bytes() == plain["start"].read_bytes()
    assert {k: v for k, v in d_new.items() if k != "reference_sphere"} == d_alt
    assert b["start"].read_bytes() == original
    assert b["start"].parent == SYSTEMS


def test_main_runs_each_variant_in_its_convention(tool, tmp_path, monkeypatch):
    """Der Ablauf gibt jeder Variante die Startkopie ihrer Konvention (Rechnung ersetzt)."""
    seen = {}

    def fake_synth(bench, work, beam=False):
        d = json.loads(bench["start"].read_text(encoding="utf-8"))
        seen[work.name] = (bench["start"].name, d.get("reference_sphere"))
        return {"seconds": 0.0, "best": {}}

    monkeypatch.setattr(tool, "run_synth", fake_synth)
    out, work = tmp_path / "out", tmp_path / "arbeit"
    assert tool.main(["--benchmarks", "B1", "--varianten", "V0,V0-alt", "--out", str(out), "--work", str(work)]) == 0
    assert seen == {"V0RS": ("start_B1_rs.json", "image_infinite"),
                    "V0": ("start_B1.json", None)}
    r_new = json.loads((out / "B1_V0RS.json").read_text(encoding="utf-8"))
    r_old = json.loads((out / "B1_V0.json").read_text(encoding="utf-8"))
    assert r_new["reference_sphere"] == "image_infinite" and r_new["tag"] == "V0RS"
    assert r_old["reference_sphere"] == "last_surface" and r_old["tag"] == "V0"
    assert "Referenz der Regel: V3RS" in tool.table(out)
    with pytest.raises(SystemExit):  # Tippfehler vor dem ersten Lauf
        tool.main(["--benchmarks", "B1", "--varianten", "V0,V7", "--out", str(out),
                   "--work", str(work)])


def test_resolve_num_ctx(tool):
    assert tool.DEFAULT_NUM_CTX == 65536
    assert tool.resolve_num_ctx(65536, 131072) == 65536  # gpt-oss:20b
    assert tool.resolve_num_ctx(65536, 40960) == 40960  # qwen3:14b
    assert tool.resolve_num_ctx(65536, None) == 65536
    assert tool.resolve_num_ctx(0, 40960) == 40960
    assert tool.resolve_num_ctx(0, None) is None


@pytest.mark.parametrize("model_max,expected", [(131072, 50000), (40960, 40960), (None, 50000)])
def test_num_ctx_reaches_ollama_request(tool, monkeypatch, model_max, expected):
    """``--num-ctx`` kommt (begrenzt auf das Modellmaximum aus /api/show) in den Optionen der
    Chat-Anfrage an; Attrappe statt Server."""
    from mikroptim import design

    bodies = []

    def fake_post(self, path, body):
        bodies.append((path, body))
        if path == "/api/show":
            if model_max is None:
                raise OSError("kein Server")
            return {"model_info": {"general.architecture": "testarch",
                                   "testarch.context_length": model_max,
                                   "testarch.rope.scaling.original_context_length": 4096}}
        return {"message": {"role": "assistant", "content": "{}"}}

    monkeypatch.setattr(design.OllamaDecider, "_post", fake_post)
    args = tool.parser().parse_args(["--varianten", "LM", "--model", "test:1b",
                                     "--num-ctx", "50000"])
    d = tool.make_llm(args, None)
    assert d.model_context_length() == model_max
    d._chat([{"role": "user", "content": "x"}])
    path, body = bodies[-1]
    assert path == "/api/chat"
    assert body["options"]["num_ctx"] == expected
    assert body["options"]["temperature"] == 0.0
    assert d.options["num_ctx"] == expected


def test_strict_off_by_default(tool):
    """M10-T7c: Die API lehnt die strict-Schemas ab; Standard ohne strict, --strict wählt es."""
    assert tool.parser().parse_args([]).strict is False
    assert tool.parser().parse_args(["--strict"]).strict is True
    assert tool.parser().parse_args(["--no-strict"]).strict is False


def test_resume_after_abandon_picks_best_open_candidate(tool, tmp_path, monkeypatch):
    """Nach abandon geht die Sitzung mit dem besten offenen Kandidaten in der Rangfolge der
    Bewertung weiter (rank_key); jede Bewertung nur einmal."""
    from mikroptim import design

    s = design.Session.create(SYSTEMS / "testsystem_mehrlinser.json", tmp_path / "s")
    for c, surf in ((0, 3), (0, 5), (1, 5)):
        s.apply(c, {"format": "mikroptim.design.action/1", "candidate": c, "assessment": "",
                    "actions": [{"action": "free_gaps", "surfaces": [surf]}]}, decider={})
    s.meta["candidates"]["3"]["status"] = "abandoned"
    na = {0: 0.3, 1: 0.5, 2: 0.5, 3: 0.9}
    rms = {0: 0.01, 1: 0.2, 2: 0.1, 3: 0.0}
    calls = []

    def fake_assess(path, target_na):
        c = int(path.stem)
        calls.append(c)
        return {"na": na[c], "rms_max_waves": rms[c], "failed_rays": 0, "elements": 4,
                "goal": False}

    monkeypatch.setattr(tool, "assess", fake_assess)
    resume = tool.best_open_resume(0.6)
    assert resume(s, 3) == 2  # größte NA, dann kleinster RMS; 3 ist verworfen
    s.meta["candidates"]["2"]["status"] = "abandoned"
    assert resume(s, 2) == 1
    assert sorted(calls) == [0, 1, 2]


def test_repetitions_get_own_tags_and_summary(tool, tmp_path, monkeypatch):
    """M10-T7d: --wiederholung rechnet jede Wiederholung unter <tag>_w<N>, ohne Vorhandenes zu
    überschreiben; --zusammenfassung zählt Ziel-Treffer x/n und bildet Mediane."""
    assert tool.repetitions(None) == [None]
    assert tool.repetitions("1-3") == [1, 2, 3]
    assert tool.repetitions("2,4-5,2") == [2, 4, 5]
    for bad in ("0", "3-1", "x"):
        with pytest.raises(SystemExit):
            tool.repetitions(bad)
    calls = []

    def fake_synth(bench, work, beam=False):
        calls.append(work.name)
        k = len(calls)
        return {"seconds": 1.0, "best": {"goal": k == 2, "na": 0.3 * k, "rms_max_waves": 0.1 / k,
                                         "failed_rays": 0, "elements": k}}

    monkeypatch.setattr(tool, "run_synth", fake_synth)
    out, work = tmp_path / "out", tmp_path / "arbeit"
    argv = ["--benchmarks", "B1", "--varianten", "V0", "--out", str(out), "--work", str(work)]
    assert tool.main(argv + ["--wiederholung", "1-2"]) == 0
    assert tool.main(argv + ["--wiederholung", "1-3"]) == 0  # 1 und 2 vorhanden
    assert calls == ["V0RS_w1", "V0RS_w2", "V0RS_w3"]
    r = json.loads((out / "B1_V0RS_w3.json").read_text(encoding="utf-8"))
    assert r["tag"] == "V0RS_w3" and r["repetition"] == 3
    # Ziel nur in w2 (NA 0,6); bester Lauf nach rank_key ist w2, Mediane über alle drei
    assert ("| B1 | V0RS | 1/3 | 0,600 | 0,600 | 0,0500 | 0,0500 | 2 (1–3) | – |"
            in tool.summary(out))


def test_assess_tolerates_eval_tuple_without_value(tool, monkeypatch):
    """M10-T7e: liefert die Gegenprobe (eval) für ein Tupel keinen RMS (alle Strahlen versagt),
    ist RMS AP None statt eines TypeError, der den Lauf abbrach."""
    def fake(args):
        if args[0] == "trace":
            return {"paraxial": {"focal_length": 10.0}, "wavefront": [{"rms_waves": 0.01}],
                    "failed_rays": 0}
        return {"tuples": [{"rms_opd_waves": 0.02}, {"rms_opd_waves": None}]}

    monkeypatch.setattr(tool, "cli_json", fake)
    a = tool.assess(SYSTEMS / "einzellinse_homotopie_start.json", 0.3)
    assert a["rms_eval_exit_pupil"] is None and a["rms_max_waves"] == 0.01


def test_perturbed_starts_within_bounds_and_summary(tool, tmp_path, monkeypatch):
    """M10-T7e: --gestoerte-starts N rechnet jede Variante auf N leicht gestörten Starts
    (Kürzel <tag>_s<k>); gestört werden nur Startwerte von Krümmungs- und Dickenvariablen und
    der Arbeitsabstand, innerhalb ihrer Schranken und nicht unter die Lage einer Geisterplatte;
    deterministisch aus Seed, Benchmark und k. Die Zusammenfassung fasst _s<k> als eigene
    Gruppe zusammen."""
    b = tool.BENCHMARKS["B2"]
    vb = tool.prepare_start({**b, "reference_sphere": "image_infinite"}, tmp_path, "B2_rs")
    base = json.loads(vb["start"].read_text(encoding="utf-8"))
    p1 = tool.prepare_perturbed(vb, tmp_path, "B2_rs", 1, 0.05, 0)
    first = p1["start"].read_bytes()
    again = tool.prepare_perturbed(vb, tmp_path, "B2_rs", 1, 0.05, 0)
    assert again["start"].read_bytes() == first  # deterministisch
    p2 = tool.prepare_perturbed(vb, tmp_path, "B2_rs", 2, 0.05, 0)
    assert p2["start"].read_bytes() != first
    d = json.loads(first)
    changes = p1["perturbation"]["changes"]
    assert changes and p1["perturbation"]["sigma"] == 0.05
    n = len(base["surfaces"])

    def inside(j, key, value, items):
        for it in items:
            kind = it["type"]
            if kind == "working_distance":
                covered, k = [n - 1], "thickness"
            elif kind in ("curvature", "thickness"):
                covered, k = tool._covered(it, n), kind
            else:
                continue
            if k == key and j in covered:
                assert it.get("min", -1e9) <= value <= it.get("max", 1e9), (j, key, value)
        if key == "thickness":
            for g in d.get("ghost_plates", []):
                if g["after_surface"] == j:
                    assert value >= g["distance"]

    for c in changes:
        j = c["surface"]
        assert c["old"] == base["surfaces"][j][c["type"]] != c["new"]
        assert d["surfaces"][j][c["type"]] == c["new"]
        inside(j, c["type"], c["new"], base["variables"]["items"])
    # nur Startwerte: Variablen, Gläser und Platten unverändert
    assert d["variables"] == base["variables"] and d["ghost_plates"] == base["ghost_plates"]
    assert [s.get("material") for s in d["surfaces"]] == [s.get("material")
                                                          for s in base["surfaces"]]
    changed = {(c["surface"], c["type"]) for c in changes}
    for j, (s0, s1) in enumerate(zip(base["surfaces"], d["surfaces"])):
        for key in ("curvature", "thickness"):
            if (j, key) not in changed:
                assert s0.get(key) == s1.get(key)
    # große Störung: Schranken greifen
    big = json.loads(json.dumps(base))
    import random
    for c in tool.perturb_start(big, 3.0, random.Random(7)):
        inside(c["surface"], c["type"], c["new"], base["variables"]["items"])
    # Ablauf und Zusammenfassung (Rechnung ersetzt)
    calls = []

    def fake_synth(bench, work, beam=False):
        calls.append((work.name, bench["start"].name))
        k = len(calls)
        return {"seconds": 1.0, "best": {"goal": k != 2, "na": 0.3 * k, "rms_max_waves": 0.01,
                                         "failed_rays": 0, "elements": 4}}

    monkeypatch.setattr(tool, "run_synth", fake_synth)
    out, work = tmp_path / "out", tmp_path / "arbeit"
    argv = ["--benchmarks", "B1", "--varianten", "V0", "--out", str(out), "--work", str(work)]
    assert tool.main(argv) == 0
    assert tool.main(argv + ["--gestoerte-starts", "3", "--stoer-sigma", "0.01"]) == 0
    assert calls == [("V0RS", "start_B1_rs.json"), ("V0RS_s1", "start_B1_rs_s1.json"),
                     ("V0RS_s2", "start_B1_rs_s2.json"), ("V0RS_s3", "start_B1_rs_s3.json")]
    r = json.loads((out / "B1_V0RS_s2.json").read_text(encoding="utf-8"))
    assert r["tag"] == "V0RS_s2" and r["perturbation"]["index"] == 2
    assert r["perturbation"]["sigma"] == 0.01 and r["perturbation"]["seed"] == 0
    s = tool.summary(out)
    assert "| B1 | V0RS | 1/1 |" in s and "| B1 | V0RS_s* | 2/3 |" in s
