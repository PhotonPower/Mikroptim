"""Claude-Entscheider (M10-T6c) mit einer Attrappe des SDK-Clients: ohne Netz, ohne Schlüssel
und ohne installiertes ``anthropic``; die Tests mit echten Fehlerklassen des SDK laufen nur, wenn
das Extra installiert ist."""

import json
import types

import pytest

from mikroptim import design
from mikroptim.design import claude, schema

from conftest import SYSTEMS

START = SYSTEMS / "testsystem_mehrlinser.json"


def block(**kw):
    return types.SimpleNamespace(**kw)


def tool_use(name, inp, uid="tu_1"):
    return block(type="tool_use", id=uid, name=name, input=inp)


def message(*content, stop="tool_use", model="claude-opus-5-5", usage=None, rid="req_1"):
    u = usage or {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 3000,
                  "cache_creation_input_tokens": 0}
    return types.SimpleNamespace(content=list(content), stop_reason=stop, model=model,
                                 usage=types.SimpleNamespace(**u), _request_id=rid,
                                 stop_details=None)


class FakeClient:
    """Liefert vorgegebene Antworten auf ``beta.messages.create`` und merkt sich die Anfragen
    (tiefe Kopie der Nachrichtenliste, weil der Entscheider sie danach weiter anhängt)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def submit(*actions, assessment="Test", uid="tu_s"):
    return tool_use(claude.SUBMIT_TOOL, {"assessment": assessment, "actions": list(actions)}, uid)


@pytest.fixture
def session(tmp_path):
    return design.Session.create(START, tmp_path / "s", spec={"tube_focal_length": 164.5})


def test_tool_call_then_submit(session):
    reply1 = message(block(type="thinking", thinking="Erst die Flächen ansehen.", signature="sig"),
                     tool_use("get_surface_table", {"surfaces": [1, 2]}))
    fake = FakeClient([reply1, message(submit({"action": "free_gaps", "surfaces": [3]}),
                                       rid="req_2")])
    ledger = design.CostLedger(20.0)
    d = design.ClaudeDecider(client=fake, ledger=ledger)
    entry = session.step(d, 0)
    assert entry["results"][0]["status"] == "ok" and entry["response"]["candidate"] == 0
    dec = entry["decider"]
    assert dec["kind"] == "claude" and dec["model"] == "claude-opus-5-5"
    assert dec["effort"] == "medium" and dec["fallbacks"] is True
    tr = dec["trace"]
    assert tr["tool_calls"] == [{"name": "get_surface_table", "arguments": {"surfaces": [1, 2]}}]
    assert tr["thinking"] == ["Erst die Flächen ansehen."]
    assert [r["request_id"] for r in tr["requests"]] == ["req_1", "req_2"]
    assert all("signature" not in b for raw in tr["raw"] for b in raw)
    # Kosten: 2 × (1000·4 + 200·20 + 3000·0,2)/1e6
    assert dec["cost_usd"] == pytest.approx(2 * 8.6e-3)
    assert ledger.spent_usd == pytest.approx(2 * 8.6e-3) and ledger.requests == 2
    first, second = fake.requests
    assert first["model"] == "claude-opus-5-5" and first["tool_choice"] == {"type": "auto"}
    assert first["thinking"]["type"] == "adaptive"
    assert first["output_config"] == {"effort": "medium"}
    assert first["betas"] == [claude.FALLBACK_BETA] and first["fallbacks"] == "default"
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in first["tools"]] == sorted(t["name"] for t in first["tools"])
    # Standard ohne strict (die API lehnt die strict-Schemas ab, Befund M10-T7c)
    assert not any("strict" in t for t in first["tools"]) and dec["strict"] is False
    # Verlauf nur angehängt: assistant-Antwort unverändert, danach tool_result
    assert second["messages"][:1] == first["messages"]
    assert second["messages"][1] == {"role": "assistant", "content": reply1.content}
    assert second["messages"][2]["content"][0]["type"] == "tool_result"
    assert second["messages"][2]["content"][0]["tool_use_id"] == "tu_1"
    # gleicher Präfix für das Caching
    assert first["system"] == second["system"] and first["tools"] == second["tools"]


def test_haiku_without_fallbacks_and_prices(session):
    fake = FakeClient([message(submit({"action": "free_gaps"}), model="claude-haiku-5-5")])
    d = design.ClaudeDecider("claude-haiku-5-5", client=fake, effort="low")
    session.step(d, 0)
    req = fake.requests[0]
    assert "fallbacks" not in req and "betas" not in req
    assert req["output_config"] == {"effort": "low"}
    u = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 0,
         "cache_creation_input_tokens": 0}
    assert claude.cost_usd("claude-opus-5-5", u) == pytest.approx(24.0)
    assert claude.cost_usd("claude-sonnet-5-5", u) == pytest.approx(12.0)
    assert claude.cost_usd("claude-haiku-5-5", {**u, "input_tokens": 50_000}) == \
        pytest.approx(0.005 + 0.5)
    # unbekanntes Modell (z. B. Rückfall) konservativ
    assert claude.cost_usd("claude-opus-4-8", u) == pytest.approx(30.0)


def test_invalid_submission_retried_then_fallback(session):
    fake = FakeClient([
        message(submit({"action": "get_surface_table"})),        # Lese-Werkzeug als Aktion
        message(submit({"action": "free_gaps", "foo": 1})),       # unbekannter Parameter
        message(submit({"action": "free_gaps", "surfaces": [3]})),
    ])
    entry = session.step(design.ClaudeDecider(client=fake, max_retries=2), 0)
    tr = entry["decider"]["trace"]
    assert tr["attempts"] == 3 and len(tr["rejected"]) == 2
    assert fake.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    # immer ungültig bzw. ohne Werkzeugaufruf: Rückfall auf die Regel, Spur bleibt im Protokoll
    text = message(block(type="text", text="Ich würde free_gaps nehmen."), stop="end_turn")
    fake = FakeClient([text] * 2)
    d = design.ClaudeDecider(client=fake, max_retries=1)
    with pytest.raises(design.DecisionError):
        session.step(d, 0)
    fake.replies = [text] * 2
    entry = session.step(d, 0, fallback=design.RuleDecider())
    assert entry["decider"]["kind"] == "regel"
    failed = entry["decider"]["failed_decider"]
    assert failed["kind"] == "claude" and len(failed["trace"]["requests"]) == 2
    assert failed["cost_usd"] > 0


def test_refusal_and_max_tokens(session):
    refusal = message(stop="refusal")
    refusal.stop_details = types.SimpleNamespace(type="refusal", category="cyber",
                                                 explanation="x")
    with pytest.raises(design.DecisionError, match="abgelehnt"):
        session.step(design.ClaudeDecider(client=FakeClient([refusal])), 0)
    fake = FakeClient([message(tool_use(claude.SUBMIT_TOOL, {"assessment": "x"}), stop="max_tokens"),
                       message(submit({"action": "free_gaps"}))])
    entry = session.step(design.ClaudeDecider(client=fake), 0)
    assert entry["results"][0]["status"] == "ok"
    assert entry["decider"]["trace"]["rejected"] == ["max_tokens: Antwort abgeschnitten"]
    last = fake.requests[1]["messages"][-1]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["is_error"] is True


def test_budget_aborts_before_request(session, tmp_path):
    path = tmp_path / "kosten.json"
    ledger = design.CostLedger(0.15, path)
    fake = FakeClient([message(submit({"action": "free_gaps"}))])
    # Reserve einer Anfrage (volle max_tokens zu Opus-Preisen) übersteigt 0,15 $ (Prompt mit
    # Kurzfassung der Wissensbasis, M10-T5)
    with pytest.raises(design.BudgetExceeded):
        session.step(design.ClaudeDecider(client=fake, ledger=ledger), 0)
    assert fake.requests == [] and ledger.spent_usd == 0.0
    # mit kleinem max_tokens passt es; Summe wird in der Datei fortgeschrieben
    session.step(design.ClaudeDecider(client=fake, ledger=ledger, max_tokens=500), 0)
    again = design.CostLedger(0.15, path)
    assert again.spent_usd == pytest.approx(8.6e-3) and again.requests == 1
    assert "claude-opus-5-5" in json.loads(path.read_text(encoding="utf-8"))["by_model"]
    # BudgetExceeded ist kein DecisionError: kein stiller Rückfall auf die Regel
    with pytest.raises(design.BudgetExceeded):
        session.step(design.ClaudeDecider(client=FakeClient([]), ledger=again), 1,
                     fallback=design.RuleDecider())


def test_private_data_not_sent(session, tmp_path):
    data = json.loads(session.system_path(0).read_text(encoding="utf-8"))
    data["glass_catalogs"] = [str(tmp_path / "data" / "glass" / "private" / "x.json")]
    session.system_path(0).write_text(json.dumps(data), encoding="utf-8")
    fake = FakeClient([])
    with pytest.raises(design.PrivacyError):
        design.ClaudeDecider(client=fake).decide({"candidate": 0}, session)
    assert fake.requests == []


def test_private_start_system_not_sent(tmp_path):
    """B4 (Entscheidung Maintainer 2026-11-19): ein Startsystem aus data/systems/private/ geht
    nicht an die Claude-API, auch wenn seine Glaskataloge öffentlich sind."""
    private = tmp_path / "data" / "systems" / "private"
    private.mkdir(parents=True)
    data = json.loads(START.read_text(encoding="utf-8"))
    root = SYSTEMS.parents[1]
    data["glass_catalogs"] = [str((root / c).resolve()) for c in data["glass_catalogs"]]
    start = private / "patent.json"
    start.write_text(json.dumps(data), encoding="utf-8")
    s = design.Session.create(start, tmp_path / "s", spec={"tube_focal_length": 164.5})
    fake = FakeClient([])
    with pytest.raises(design.PrivacyError):
        design.ClaudeDecider(client=fake).decide({"candidate": 0}, s)
    assert fake.requests == []


def test_b4_benchmarks_from_private_manifest(tmp_path, capsys):
    """tools/m10t7_bewertung.py lädt B4 aus der privaten Liste und überspringt ohne Liste."""
    import importlib.util

    from conftest import ROOT

    spec = importlib.util.spec_from_file_location("m10t7", ROOT / "tools" / "m10t7_bewertung.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.PRIVATE_SYSTEMS == ROOT / "data" / "systems" / "private"
    missing = tmp_path / "fehlt" / "b4.json"
    assert tool.private_benchmarks(missing) == {}
    assert [b for b, _ in tool.expand_benchmarks(["B1", "B4"], missing)] == ["B1"]
    assert "übersprungen" in capsys.readouterr().out
    manifest = tmp_path / "private" / "b4.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"benchmarks": [
        {"name": "B4a", "start": "a.json", "na": 0.95, "field_scale": 0.1},
        {"name": "B4b", "start": "b.json", "na": 0.9, "max_elements": 9, "steps": 3}]}),
        encoding="utf-8")
    b4 = tool.private_benchmarks(manifest)
    assert list(b4) == ["B4a", "B4b"]
    assert b4["B4a"]["start"] == manifest.parent / "a.json"
    assert b4["B4a"]["private"] and b4["B4a"]["field_scale"] == 0.1
    assert b4["B4a"]["spec"] == {"na": 0.95} and b4["B4a"]["max_elements"] == 15
    assert b4["B4b"]["max_elements"] == 9 and b4["B4b"]["steps"] == 3
    assert [b for b, _ in tool.expand_benchmarks(["B4b"], manifest)] == ["B4b"]
    with pytest.raises(SystemExit):
        tool.expand_benchmarks(["B4c"], manifest)


def test_replay_of_claude_run_without_api(session, tmp_path):
    fake = FakeClient([message(submit({"action": "free_gaps", "surfaces": [3]}))])
    session.step(design.ClaudeDecider(client=fake), 0)
    again = design.replay(session.log(), START, tmp_path / "w", spec=session.meta["spec"])
    assert again.system_path(1).read_text() == session.system_path(1).read_text()
    assert again.log()[0]["decider"]["original"]["kind"] == "claude"


def _optional_parameters(s):
    """Zahl der optionalen Parameter wie im Limit der API für strict-Werkzeuge."""
    n = 0
    if isinstance(s, dict):
        if "properties" in s:
            n += sum(1 for k in s["properties"] if k not in set(s.get("required", [])))
        n += sum(_optional_parameters(v) for v in s.values())
    elif isinstance(s, list):
        n += sum(_optional_parameters(v) for v in s)
    return n


def test_strict_is_opt_in_and_answers_still_validated(session):
    """M10-T7c: Die API lehnte strict ab (185 optionale Parameter, Grenze 24; das
    Entscheidungswerkzeug allein liegt darüber); Standard ist strict aus, strict=True setzt es
    auf jedes Werkzeug.
    Ohne strict prüft der Entscheider die Antwort weiter gegen das Aktionsschema."""
    assert not any("strict" in t for t in claude.tool_definitions())
    assert all(t["strict"] for t in claude.tool_definitions(strict=True))
    submit_tool = next(t for t in claude.tool_definitions() if t["name"] == claude.SUBMIT_TOOL)
    assert _optional_parameters(submit_tool["input_schema"]) > 24
    fake = FakeClient([message(submit({"action": "synth_segment"})),          # target fehlt
                       message(submit({"action": "optimize", "variables": "alles"})),
                       message(submit({"action": "free_gaps", "surfaces": [3]}))])
    d = design.ClaudeDecider(client=fake)
    entry = session.step(d, 0)
    assert entry["results"][0]["status"] == "ok" and len(fake.requests) == 3
    rejected = entry["decider"]["trace"]["rejected"]
    assert len(rejected) == 2 and "target fehlt" in rejected[0] and "'alles'" in rejected[1]


def test_strict_tool_schemas():
    tools = claude.tool_definitions(strict=True)
    names = [t["name"] for t in tools]
    assert names == sorted(set(schema.TOOL_ACTIONS) | {claude.SUBMIT_TOOL})

    def walk(s):
        if isinstance(s, dict):
            assert not {"oneOf", "prefixItems", "minimum", "maximum", "maxLength",
                        "minItems", "maxItems"} & set(s), s
            if s.get("type") == "object":
                assert s["additionalProperties"] is False and "required" in s
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)

    for t in tools:
        walk(t["input_schema"])
    submit_tool = next(t for t in tools if t["name"] == claude.SUBMIT_TOOL)
    items = submit_tool["input_schema"]["properties"]["actions"]["items"]["anyOf"]
    assert {i["properties"]["action"]["const"] for i in items} == set(claude.decision_actions())
    # Tool-Liste deterministisch (Prompt-Caching)
    assert json.dumps(tools, sort_keys=True) == json.dumps(claude.tool_definitions(strict=True),
                                                           sort_keys=True)


def test_cli_claude_without_sdk_or_key(session, monkeypatch, capsys):
    """Ohne SDK bzw. ohne Zugangsdaten: klarer Abbruch (Exit 3) statt Rückfall, kein Netz."""
    from mikroptim.design.__main__ import main

    monkeypatch.setattr(claude, "_import_sdk", _no_sdk)
    code = main(["run", str(session.dir), "--decider", "claude", "--steps", "1",
                 "--fallback", "rule"])
    err = capsys.readouterr().err
    assert code == 3 and "mikroptim[claude]" in err
    assert session.log() == []


def _no_sdk(required):
    if required:
        raise claude.ClaudeUnavailableError("Das Python-SDK 'anthropic' fehlt: "
                                            "pip install mikroptim[claude]")
    return None


def test_sdk_errors_mapped():
    anthropic = pytest.importorskip("anthropic")
    import httpx2  # Abhängigkeit des SDK

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    d = design.ClaudeDecider(client=FakeClient([anthropic.APIConnectionError(request=req)]))
    trace = {"requests": [], "usage": dict.fromkeys(claude.usage_dict({}), 0), "cost_usd": 0.0}
    with pytest.raises(design.DecisionError, match="Verbindung"):
        d._request([{"role": "user", "content": "x"}], trace)
    resp = httpx2.Response(401, request=req, json={"error": {"message": "invalid x-api-key"}})
    err = anthropic.AuthenticationError("invalid x-api-key", response=resp, body=None)
    d = design.ClaudeDecider(client=FakeClient([err]))
    with pytest.raises(design.ClaudeUnavailableError):
        d._request([{"role": "user", "content": "x"}], trace)
    # ohne Zugangsdaten meldet das SDK erst beim Aufruf einen TypeError
    d = design.ClaudeDecider(client=FakeClient([TypeError("Could not resolve authentication")]))
    with pytest.raises(design.ClaudeUnavailableError, match="authentication"):
        d._request([{"role": "user", "content": "x"}], trace)


def test_sdk_accepts_request_parameters():
    """Die Parameter der Anfrage gibt es in der Signatur des installierten SDK."""
    anthropic = pytest.importorskip("anthropic")
    import inspect

    params = inspect.signature(anthropic.Anthropic(api_key="x").beta.messages.create).parameters
    for name in ("model", "max_tokens", "system", "tools", "tool_choice", "messages", "thinking",
                 "output_config", "betas", "fallbacks", "cache_control"):
        assert name in params, name
