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
                reply = fake.replies.pop(0)
                if "_status" in reply:  # Serverfehler nachbilden
                    self.send_response(reply["_status"])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self._send({"model": body["model"], "done": True, "message": reply})

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


def test_ollama_retries_server_error(session):
    """M10-T7b: HTTP 500 von Ollama (z. B. Modell lädt nicht, weil die geteilte GPU voll ist)
    wird wiederholt statt die Sitzung abzubrechen; 4xx nicht."""
    fake = FakeOllama([{"_status": 500}, {"role": "assistant", "content": ""},
                       answer({"action": "free_gaps", "surfaces": [3]})])
    try:
        d = design.OllamaDecider("test:1b", host=fake.url)
        d.server_retry_wait = 0.0
        entry = session.step(d, 0)
    finally:
        fake.close()
    assert entry["results"][0]["status"] == "ok"
    fake = FakeOllama([{"_status": 400}])
    try:
        d = design.OllamaDecider("test:1b", host=fake.url)
        d.server_retry_wait = 0.0
        with pytest.raises(Exception, match="400"):
            session.step(d, 0)
    finally:
        fake.close()


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


def _mark_used(session, step):
    """Protokolleintrag, als hätte die Regel die Stufe ``step`` erfolgreich benutzt."""
    entry = {"format": "mikroptim.design.log/1", "candidate": 0, "decider": {"kind": "regel"},
             "response": {"actions": [{"action": "free_gaps", "reason": f"Stillstand: {step}"}]},
             "results": [{"action": "free_gaps", "status": "ok"}]}
    with open(session.dir / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def test_rule_decider_stall_escalation(tmp_path):
    """M10-T7b: Bleibt ein Abschnitt unter der Soll-NA stehen, wählt die Regel nacheinander je
    einmal: Spalte freigeben, aplanatische Schale im Arbeitsabstand (hochbrechendes Katalogglas),
    Geisterplatten-Wörterbuch, Null-Meniskus mit Dicken-Homotopie; danach die übrigen Regeln.
    Die Antworten sind schema-gültig und lassen sich ausführen."""
    jsonschema = pytest.importorskip("jsonschema")
    repo = json.loads((ROOT / "data" / "design" / "schemas" / "design_action.schema.json")
                      .read_text(encoding="utf-8"))
    check = jsonschema.Draft202012Validator(repo).validate
    s = design.Session.create(SYSTEMS / "einzellinse_homotopie_start.json", tmp_path / "a",
                              spec={"na": 0.6})
    s.meta["candidates"]["0"]["action"] = "synth_segment"
    st = s.state(0)
    rule = design.RuleDecider(cycles=5)

    def step():
        r = rule.decide(st, s)
        check(r)
        return r

    r = step()
    assert [a["action"] for a in r["actions"]] == ["free_gaps", "optimize"]
    assert r["actions"][0]["reason"] == "Stillstand: free_gaps"
    _mark_used(s, "free_gaps")
    r = step()
    shell = r["actions"][0]
    assert [a["action"] for a in r["actions"]] == ["insert_module", "free_gaps", "optimize"]
    assert shell["module"] == "front.shell_ca" and shell["after_surface"] == 3
    assert shell["glasses"] == [{"name": "N-BK7"}]  # höchste Brechzahl in basis.json
    # Arbeitsabstand 5 mm reicht: Dicke (5 − 0,05)/n′ − 1,05 · 0,2, begrenzt auf 2 mm
    assert shell["position"] == 0.05 and shell["thickness"] == 2.0
    # die Schale lässt sich einfügen und nachoptimieren; danach folgt der nächste Abschnitt
    entry = s.apply(0, r, decider=rule.describe())
    assert [x["status"] for x in entry["results"]] == ["ok"] * 3, entry["results"]
    c = entry["results"][-1]["new_candidates"][0]
    assert s.state(c)["elements"] == 2
    assert rule.decide(s.state(c), s)["actions"][0]["action"] == "synth_segment"
    r = step()
    assert [a["action"] for a in r["actions"]] == ["place_ghost_dictionary", "synth_segment"]
    assert r["actions"][0]["after_surface"] == 2 and r["actions"][0]["distance"] == 1.5
    assert r["actions"][0]["glasses"] == ["N-BK7"]
    _mark_used(s, "ghost_dictionary")
    r = step()
    men = r["actions"][0]
    assert men["module"] == "generic.null_meniscus" and men["after_surface"] == 2
    assert men["thickness"] == 1.0
    _mark_used(s, "null_meniscus")
    # alle Stufen benutzt: übrige Regeln (Synthese nicht wiederholt)
    assert step()["actions"][0]["action"] != "synth_segment"
    # ... außer der letzte Abschnitt hat die NA merklich geöffnet: weiter mit Spalten
    for na in (0.45, 0.52):
        entry = {"format": "mikroptim.design.log/1", "candidate": 0,
                 "decider": {"kind": "regel"}, "response": {"actions": [
                     {"action": "synth_segment", "target": {"aperture": 0.6}}]},
                 "results": [{"action": "synth_segment", "status": "ok", "reports": [
                     {"result": {"status": "step_too_small", "point": {"aperture": na}}}]}]}
        with open(s.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    r = step()
    assert [a["action"] for a in r["actions"]] == ["free_gaps", "optimize"]
    assert r["actions"][0]["reason"] == "Stillstand: repeat"
    # nur ausgewählte Stufen; Optionen des Abschnitts
    rule = design.RuleDecider(escalation=["null_meniscus"],
                              synth_options={"augmented_lagrangian": 3, "sine_condition": 2.0})
    s.meta["candidates"]["0"]["action"] = None
    r = rule.decide(st, s)
    check(r)
    assert r["actions"][0]["augmented_lagrangian"] == 3
    assert r["actions"][0]["sine_condition"] == 2.0
    with pytest.raises(ValueError):
        design.RuleDecider(escalation=["unbekannt"])


def test_rule_decider_skips_shell_without_room(tmp_path):
    """M10-T7b: Ist der Arbeitsabstand für die Schale zu kurz (Schranke 3 mm, Abstand 3,5 mm:
    die Schale ließe weniger als 1,05 · 3 mm), überspringt die Regel die Stufe."""
    data = json.loads((SYSTEMS / "einzellinse_homotopie_start.json").read_text(encoding="utf-8"))
    data["surfaces"][-1]["thickness"] = 3.5
    for it in data["variables"]["items"]:
        if it["type"] == "working_distance":
            it["min"] = 3.0
    from mikroptim.design.files import resolve_catalogs

    start = tmp_path / "start.json"
    start.write_text(json.dumps(resolve_catalogs(data, SYSTEMS)), encoding="utf-8")
    s = design.Session.create(start, tmp_path / "s", spec={"na": 0.6})
    s.meta["candidates"]["0"]["action"] = "synth_segment"
    rule = design.RuleDecider(escalation=["aplanatic_shell", "null_meniscus"])
    r = rule.decide(s.state(0), s)
    assert r["actions"][0]["module"] == "generic.null_meniscus"


def test_set_bounds_action(tmp_path):
    """M10-T7b: set_bounds für wd und t[i]; Einträge mit mehreren Flächen werden geteilt, der
    Startwert wird auf neue Schranken geklemmt."""
    s = design.Session.create(START, tmp_path / "s")
    def act(cand, *actions):
        return s.apply(cand, {"format": "mikroptim.design.action/1", "candidate": cand,
                              "assessment": "", "actions": list(actions)})["results"]
    [r] = act(0, {"action": "free_gaps", "surfaces": [3, 5]})
    c = r["new_candidates"][0]
    [r] = act(c, {"action": "set_bounds", "variable": "t[3]", "min": 1.5, "max": 4.0})
    assert r["status"] == "ok", r
    rep = r["reports"][0]
    assert rep["value_before"] == 1.0 and rep["value"] == 1.5
    data = json.loads(s.system_path(r["new_candidates"][0]).read_text(encoding="utf-8"))
    t3 = [it for it in data["variables"]["items"] if it.get("surface") == 3]
    assert len(t3) == 1 and t3[0]["min"] == 1.5 and t3[0]["max"] == 4.0
    assert data["surfaces"][3]["thickness"] == 1.5
    [bad] = act(c, {"action": "set_bounds", "variable": "wd", "min": 1.0})
    assert bad["status"] == "error" and bad["code"] == "precondition"  # keine Variable wd
    [bad] = act(c, {"action": "set_bounds", "variable": "t[1]", "min": 1.0})
    assert bad["code"] == "precondition"
    [bad] = act(c, {"action": "set_bounds", "variable": "c[1]", "min": 1.0})
    assert bad["code"] == "not_implemented"


def test_synth_segment_options(tmp_path):
    """M10-T7b: synth_segment reicht augmented_lagrangian und sine_condition an synth weiter."""
    s = design.Session.create(SYSTEMS / "einzellinse_homotopie_start.json", tmp_path / "a")
    entry = s.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                        "actions": [{"action": "synth_segment", "target": {"aperture": 0.31},
                                     "augmented_lagrangian": 2, "sine_condition": 0.5}]})
    r = entry["results"][0]
    assert r["status"] == "ok", r
    data = json.loads(s.system_path(1).read_text(encoding="utf-8"))
    sine = [c for c in data["constraints"]["items"] if c["type"] == "sine_condition"]
    assert sine and sine[0]["weight"] == 0.5


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


def test_free_gaps_skips_gap_thinner_than_plate(tmp_path, monkeypatch):
    """Befund M10-T7: Spalt dünner als die Lage seiner Geisterplatte bekommt keine Variable
    (der Startwert läge unter der Schranke, synth lehnte das System ab). Der konstruierte Start
    ist selbst nicht ladbar (Platte hinter dem Spalt); die Prüfung neuer Kandidaten (M10-T7b)
    ist hier deshalb abgeschaltet."""
    from mikroptim.design import session as session_module

    monkeypatch.setattr(session_module, "_check_results", lambda outs: None)
    data = json.loads((SYSTEMS / "einzellinse_homotopie_start.json").read_text(encoding="utf-8"))
    data["surfaces"][2]["thickness"] = 0.3  # Platte nach Fläche 2 liegt bei 0,5 mm
    from mikroptim.design.files import resolve_catalogs

    start = tmp_path / "start.json"
    start.write_text(json.dumps(resolve_catalogs(data, SYSTEMS)), encoding="utf-8")
    s = design.Session.create(start, tmp_path / "s")
    entry = s.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                        "actions": [{"action": "free_gaps"}]})
    assert entry["results"][0]["status"] == "ok"
    # seit M10-T7e werden Plattenspalte wieder freigegeben (Schranke = Plattenlage), aber nicht
    # Spalt 2, der dünner ist als die Lage seiner Platte
    new = json.loads(s.system_path(entry["results"][0]["new_candidates"][0])
                     .read_text(encoding="utf-8"))
    gaps = [it["surface"] for it in new["variables"]["items"] if it["type"] == "thickness"]
    assert 2 not in gaps and 0 in gaps
    # mit include_ghost_plate_gaps = false bleiben sie fest wie in M10-T7d: nichts Neues
    entry = s.apply(0, {"format": "mikroptim.design.action/1", "candidate": 0, "assessment": "",
                        "actions": [{"action": "free_gaps", "include_ghost_plate_gaps": False}]})
    assert entry["results"][0]["status"] == "ok"
    assert not entry["results"][0].get("new_candidates")


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
