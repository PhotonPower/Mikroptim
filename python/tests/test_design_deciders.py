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
