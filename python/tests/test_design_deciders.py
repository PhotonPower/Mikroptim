"""Entscheider (M10-T6b): Ollama über einen nachgebildeten Server, Regel, Bestätigung, Rückfall."""

import http.server
import io
import json
import threading

import pytest

from mikroptim import design
from mikroptim.design import schema

from conftest import ROOT, SYSTEMS

START = SYSTEMS / "testsystem_mehrlinser.json"


class FakeOllama:
    """Nachgebildeter Ollama-Server: liefert vorgegebene Antworten auf /api/chat der Reihe nach
    und merkt sich die Anfragen."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"models": [{"name": "test:1b", "digest": "abc123"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                if self.path == "/api/generate":  # Entladen (keep_alive 0)
                    self._send({"model": body["model"], "done": True})
                    return
                self._send({"model": body["model"], "done": True, "message": fake.replies.pop(0)})

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def answer(*actions, assessment="Test"):
    return {"role": "assistant", "content": json.dumps(
        {"format": "mikroptim.design.action/1", "candidate": 99, "assessment": assessment,
         "actions": list(actions)})}


@pytest.fixture
def session(tmp_path):
    return design.Session.create(START, tmp_path / "s", spec={"tube_focal_length": 164.5})


def test_ollama_tool_call_then_answer(session):
    fake = FakeOllama([
        {"role": "assistant", "content": "", "thinking": "Ich schaue mir die Flächen an.",
         "tool_calls": [{"function": {"name": "get_surface_table",
                                      "arguments": {"surfaces": [1, 2]}}}]},
        {"role": "assistant", "content": "fertig mit Werkzeugen"},
        answer({"action": "free_gaps", "surfaces": [3]}),
    ])
    try:
        d = design.OllamaDecider("test:1b", host=fake.url, think=True)
        entry = session.step(d, 0)
    finally:
        fake.close()
    assert entry["results"][0]["status"] == "ok" and entry["response"]["candidate"] == 0
    dec = entry["decider"]
    assert dec["kind"] == "ollama" and dec["model"] == "test:1b" and dec["digest"] == "abc123"
    assert dec["trace"]["tool_calls"] == [{"name": "get_surface_table",
                                            "arguments": {"surfaces": [1, 2]}}]
    assert dec["trace"]["thinking"] == ["Ich schaue mir die Flächen an."]
    first, second, last = fake.requests
    assert first["tools"] and first["think"] is True and first["options"]["temperature"] == 0.0
    assert any(m["role"] == "tool" for m in second["messages"])
    assert last["format"] == schema.response_schema() and "tools" not in last


def test_ollama_invalid_answer_retried_then_fallback(session):
    fake = FakeOllama([
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "kein JSON"},
        answer({"action": "free_gaps", "surfaces": [3]}),
    ])
    try:
        entry = session.step(design.OllamaDecider("test:1b", host=fake.url, max_retries=1), 0)
    finally:
        fake.close()
    assert entry["decider"]["trace"]["attempts"] == 2
    assert entry["decider"]["trace"]["rejected"]
    # immer ungültig: Rückfall auf die Regel
    fake = FakeOllama([{"role": "assistant", "content": ""}] +
                      [{"role": "assistant", "content": "nein"}] * 2)
    try:
        d = design.OllamaDecider("test:1b", host=fake.url, max_retries=1)
        with pytest.raises(design.DecisionError):
            session.step(d, 0)
        fake.replies = [{"role": "assistant", "content": ""}] + \
            [{"role": "assistant", "content": "nein"}] * 2
        entry = session.step(d, 0, fallback=design.RuleDecider())
    finally:
        fake.close()
    assert entry["decider"]["kind"] == "regel"


def test_rule_decider_run(session):
    last = session.run(design.RuleDecider(cycles=5), 0, max_steps=3)
    kinds = [c["action"] for c in session.candidates().values()]
    assert kinds[:3] == [None, "free_gaps", "optimize"]
    assert last in session.candidates()
    assert all(e["decider"]["kind"] == "regel" for e in session.log())


def test_confirm_decider(session):
    proposer = design.RuleDecider()
    accept = design.ConfirmDecider(proposer, stdin=io.StringIO("\n"), stdout=io.StringIO())
    entry = session.step(accept, 0)
    assert entry["decider"]["choice"] == "angenommen"
    assert entry["results"][0]["action"] == "free_gaps"
    reject = design.ConfirmDecider(proposer, stdin=io.StringIO("x\n"), stdout=io.StringIO())
    entry = session.step(reject, 1)
    assert entry["results"][0]["action"] == "abandon"
    assert session.candidates()[1]["status"] == "abandoned"


def test_replay_of_model_run_without_ollama(session, tmp_path):
    fake = FakeOllama([{"role": "assistant", "content": ""},
                       answer({"action": "free_gaps", "surfaces": [3]})])
    try:
        session.step(design.OllamaDecider("test:1b", host=fake.url), 0)
    finally:
        fake.close()
    again = design.replay(session.log(), START, tmp_path / "w", spec=session.meta["spec"])
    assert again.system_path(1).read_text() == session.system_path(1).read_text()


def test_embedded_schema_matches_repository():
    repo = json.loads((ROOT / "data" / "design" / "schemas" / "design_action.schema.json")
                      .read_text(encoding="utf-8"))["$defs"]
    for name, d in schema.ACTION_DEFS.items():
        assert d["required"] == repo[name]["required"], name
        assert set(d["properties"]) <= set(repo[name]["properties"]), name
    assert {t["function"]["name"] for t in schema.tool_specs()} == set(schema.TOOL_ACTIONS)


def test_rule_decider_opens_aperture(tmp_path):
    """M10-T7: Liegt die aktuelle NA unter der Soll-NA, öffnet die Regel die Apertur mit
    synth_segment; nach einem Abschnitt ohne Ziel folgen Spalte freigeben und optimieren."""
    s = design.Session.create(SYSTEMS / "einzellinse_homotopie_start.json", tmp_path / "a",
                              spec={"na": 0.6})
    st = s.state(0)
    assert st["homotopy_point"]["aperture"] == pytest.approx(0.3, rel=1e-3)
    assert st["spec"]["na"] == 0.6
    r = design.RuleDecider().decide(st, s)
    assert r["actions"] == [{"action": "synth_segment", "target": {"aperture": 0.6},
                             "max_new_elements": 4, "reason": "Soll-NA der Spezifikation"}]
    # Kandidat nach einem synth_segment, der die Soll-NA nicht erreicht hat
    s.meta["candidates"]["0"]["action"] = "synth_segment"
    r = design.RuleDecider().decide(st, s)
    assert [a["action"] for a in r["actions"]] == ["free_gaps", "optimize"]
    # höchstens max_synth Abschnitte: danach die übrigen Regeln
    s.meta["candidates"]["0"]["action"] = None
    r = design.RuleDecider(max_synth=0).decide(st, s)
    assert r["actions"][0]["action"] == "free_gaps"
    # ein gescheiterter Abschnitt wird nicht wiederholt
    failed = {**st, "last_errors": ["synth_segment: mikroptim synth: Fehler"]}
    r = design.RuleDecider().decide(failed, s)
    assert r["actions"][0]["action"] == "free_gaps"


def test_split_lens_relaxes_thickness_bound(tmp_path):
    """Befund M10-T7: Die Hälften einer Linse mit Dickenvariable (Schranke 0,4 mm) dürfen
    dünner sein als die Schranke; vorher lehnte insert das geteilte System ab."""
    data = json.loads((SYSTEMS / "zielspezifikation_start.json").read_text(encoding="utf-8"))
    data["surfaces"][2]["thickness"] = 0.6  # Hälften 0,3 mm < 0,4 mm
    from mikroptim.design.files import resolve_catalogs

    start = tmp_path / "start.json"
    start.write_text(json.dumps(resolve_catalogs(data, SYSTEMS)), encoding="utf-8")
    s = design.Session.create(start, tmp_path / "s")
    entry = s.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                        "actions": [{"action": "split_lens", "element": 0}]})
    r = entry["results"][0]
    assert r["status"] == "ok", r
    assert r["reports"][0]["relaxed_thickness_min"] == pytest.approx(0.3)
    # dieselbe Variable in der Listenform (so legt sie Operator B für beide Hälften an)
    for it in data["variables"]["items"]:
        if it.get("type") == "thickness" and it.get("surface") == 2:
            del it["surface"]
            it["surfaces"] = [2]
    start.write_text(json.dumps(resolve_catalogs(data, SYSTEMS)), encoding="utf-8")
    s2 = design.Session.create(start, tmp_path / "s2")
    entry = s2.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                         "actions": [{"action": "split_lens", "element": 0}]})
    assert entry["results"][0]["status"] == "ok", entry["results"][0]


def test_free_gaps_skips_gap_thinner_than_plate(tmp_path):
    """Befund M10-T7: Spalt dünner als die Lage seiner Geisterplatte bekommt keine Variable
    (der Startwert läge unter der Schranke, synth lehnte das System ab)."""
    data = json.loads((SYSTEMS / "einzellinse_homotopie_start.json").read_text(encoding="utf-8"))
    data["surfaces"][2]["thickness"] = 0.3  # Platte nach Fläche 2 liegt bei 0,5 mm
    start = tmp_path / "start.json"
    start.write_text(json.dumps(data), encoding="utf-8")
    s = design.Session.create(start, tmp_path / "s")
    entry = s.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                        "actions": [{"action": "free_gaps"}]})
    assert entry["results"][0]["status"] == "ok"
    new = json.loads(s.system_path(entry["results"][0]["new_candidates"][0])
                     .read_text(encoding="utf-8"))
    gaps = [it["surface"] for it in new["variables"]["items"] if it["type"] == "thickness"]
    assert 2 not in gaps and 0 in gaps


def test_wrong_parameter_type_is_structured_error(session):
    """M10-T7 (Befund mit gpt-oss:20b): Parameter falschen Typs brechen die Sitzung nicht ab."""
    entry = session.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0,
                              "assessment": "", "actions": [
                                  {"action": "split_lens", "element": "achromat"}]})
    r = entry["results"][0]
    assert r["status"] == "error" and r["code"] == "invalid_parameter"


def test_tool_call_with_wrong_type_returns_error(session):
    """Befund mit gpt-oss:20b: glass_preselect mit element "achromat" bricht nicht ab, das
    Modell bekommt die Fehlermeldung als Werkzeugergebnis."""
    fake = FakeOllama([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "glass_preselect", "arguments": {"element": "achromat"}}}]},
        {"role": "assistant", "content": ""},
        answer({"action": "free_gaps", "surfaces": [3]}),
    ])
    try:
        entry = session.step(design.OllamaDecider("test:1b", host=fake.url), 0)
    finally:
        fake.close()
    assert entry["results"][0]["status"] == "ok"
    tool_msg = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "error" in json.loads(tool_msg["content"])


def test_ollama_usage_and_unload(session):
    """Tokenangaben von Ollama landen in der Spur; unload schickt keep_alive 0."""
    fake = FakeOllama([{"role": "assistant", "content": ""},
                       answer({"action": "free_gaps", "surfaces": [3]})])
    try:
        d = design.OllamaDecider("test:1b", host=fake.url, keep_alive="2m")
        entry = session.step(d, 0)
        d.unload()
    finally:
        fake.close()
    assert len(entry["decider"]["trace"]["usage"]) == 2
    assert fake.requests[0]["keep_alive"] == "2m"
    assert fake.requests[-1] == {"model": "test:1b", "keep_alive": 0}
