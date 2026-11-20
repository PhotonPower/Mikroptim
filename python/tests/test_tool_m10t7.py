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
