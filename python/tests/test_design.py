"""Entwurfswerkzeuge (M10-T6a): Zustand, Aktionen, Kandidatenbaum, Protokoll, Wiedereinspielen."""

import io
import json

import pytest

from mikroptim import design
from mikroptim.design.files import catalog_names

from conftest import ROOT, SYSTEMS

START = SYSTEMS / "testsystem_mehrlinser.json"
SCHEMAS = ROOT / "data" / "design" / "schemas"


def response(*actions, candidate=0):
    return {"format": "mikroptim.design.action/1", "candidate": candidate,
            "assessment": "Test", "actions": list(actions)}


def validate(instance, schema_name):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


@pytest.fixture
def session(tmp_path):
    return design.Session.create(START, tmp_path / "s", spec={"tube_focal_length": 164.5})


def test_state_content_and_schema(session):
    st = session.state(0)
    validate(st, "design_state.schema.json")
    assert st["format"] == "mikroptim.design.state/1" and st["elements"] == 4
    assert st["spec"]["magnification"] == pytest.approx(164.5 / st["paraxial"]["f"])
    assert st["spec"]["zone"] in (1, 2, 3, 4, 5, 6)
    groups = st["groups"]
    assert "middle" in groups
    # Gruppenbilanzen summieren sich zur Summe des Aberrationsberichts
    import mikroptim
    total = mikroptim.aberrations(session.system_path(0))["totals"]["waves"]["W040"]
    assert sum(g["W040"] for g in groups.values()) == pytest.approx(total, rel=1e-12)
    assert "Gruppenbilanzen" in design.format_state(st)


def test_state_reference_sphere(session, tmp_path):
    """Der Zustand nennt die Konvention der Referenzsphäre (ADR-0017, Entscheidung Maintainer
    2026-11-19): ohne Feld die alte Konvention last_surface, sonst das Feld der Datei."""
    st = session.state(0)
    assert st["reference_sphere"] == {"kind": "last_surface", "explicit": False}
    assert "Referenzsphäre: last_surface (ohne Feld" in design.format_state(st)
    data = json.loads(session.system_path(0).read_text(encoding="utf-8"))
    data["reference_sphere"] = "image_infinite"
    path = tmp_path / "unendlich.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    st = design.build_state(path, with_merit=False, with_sensitivity=False)
    validate(st, "design_state.schema.json")
    assert st["reference_sphere"] == {"kind": "image_infinite", "explicit": True}
    assert "Referenzsphäre: image_infinite\n" in design.format_state(st) + "\n"


def test_read_actions_do_not_create_candidates(session):
    r = response({"action": "get_surface_table", "surfaces": [1, 2]})
    validate(r, "design_action.schema.json")
    entry = session.apply(0, r, decider={"kind": "test"})
    assert entry["results"][0]["status"] == "ok"
    assert [s["surface"] for s in entry["results"][0]["result"]["surfaces"]] == [1, 2]
    assert list(session.candidates()) == [0]


def test_insert_with_saddle_branches_and_optimize(session):
    r = response({"action": "insert_null_meniscus", "after_surface": 3,
                  "glass": {"name": "N-BK7"}, "saddle_branches": True})
    validate(r, "design_action.schema.json")
    entry = session.apply(0, r, decider={"kind": "test"})
    assert entry["results"][0]["new_candidates"] == [1, 2]
    st = session.state(2)
    assert st["elements"] == 5 and st["parent"] == 0
    r2 = response({"action": "free_gaps"}, {"action": "optimize", "cycles": 5}, candidate=2)
    validate(r2, "design_action.schema.json")
    entry = session.apply(2, r2, decider={"kind": "test"})
    assert [x["status"] for x in entry["results"]] == ["ok", "ok"]
    assert entry["results"][1]["new_candidates"] == [4]
    assert session.candidates()[4]["parent"] == 3
    data = json.loads(session.system_path(3).read_text(encoding="utf-8"))
    assert any(v["type"] == "thickness" for v in data["variables"]["items"])


def test_split_lens_and_errors(session):
    entry = session.apply(0, response({"action": "split_lens", "element": 0}), decider={})
    assert entry["results"][0]["status"] == "ok"
    assert session.state(1)["elements"] == 5
    bad = session.apply(0, response({"action": "split_lens", "element": 9}), decider={})
    assert bad["results"][0]["code"] == "precondition"
    todo = session.apply(0, response({"action": "freeze"}), decider={})
    assert todo["results"][0]["code"] == "not_implemented"
    fmt = session.apply(0, {"actions": []}, decider={})
    assert fmt["results"][0]["code"] == "format"
    # Fehler stoppen die folgenden Aktionen der Antwort
    stop = session.apply(0, response({"action": "split_lens", "element": 9},
                                     {"action": "free_gaps"}), decider={})
    assert len(stop["results"]) == 1


def test_branch_abandon_finish(session):
    r = response({"action": "branch", "alternatives": [
        [{"action": "free_gaps", "surfaces": [3]}],
        [{"action": "free_gaps", "surfaces": [5]}]]})
    validate(r, "design_action.schema.json")
    entry = session.apply(0, r, decider={})
    assert entry["results"][0]["new_candidates"] == [1, 2]
    session.apply(1, response({"action": "abandon", "reason": "Test"}, candidate=1), decider={})
    session.apply(2, response({"action": "finish", "reason": "Test"}, candidate=2), decider={})
    c = session.candidates()
    assert c[1]["status"] == "abandoned" and c[2]["status"] == "finished"
    assert session.open_candidates() == [0]


class ScriptDecider(design.Decider):
    """Antwortet nacheinander mit den vorgegebenen Aktionslisten und merkt sich die Zustände."""

    kind = "skript"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.states = []

    def decide(self, state, session):
        self.states.append(state)
        return response(*self.answers.pop(0), candidate=state["candidate"])


def test_run_continues_after_abandon(session):
    """M10-T7c: abandon verwirft nur den Kandidaten; run macht mit dem nächsten offenen
    Vorfahren weiter (Sonnet 5.5 beendete sonst die Sitzung mit Restbudget). Auch ein abandon
    nach einer verändernden Aktion derselben Antwort betrifft den neuen Kandidaten."""
    d = ScriptDecider([{"action": "free_gaps", "surfaces": [3]}],              # 0 → 1
                      [{"action": "free_gaps", "surfaces": [5]},               # 1 → 2
                       {"action": "abandon", "reason": "Test"}],              # 2 verworfen
                      [{"action": "abandon", "reason": "Test"}],              # 1 verworfen
                      [{"action": "finish", "reason": "Test"}])               # 0 fertig
    last = session.run(d, 0, max_steps=10)
    assert [e["candidate"] for e in session.log()] == [0, 1, 1, 0]
    c = session.candidates()
    assert c[2]["status"] == "abandoned" and c[1]["status"] == "abandoned"
    assert c[0]["status"] == "finished" and last == 0
    assert "abandon: Kandidat 2 verworfen, weiter mit Kandidat 1" in d.states[2]["last_errors"]
    assert d.answers == []


def test_run_resume_hook_and_end(session):
    """Eigene Wahl nach abandon (Bewertungsskript: bester offener Kandidat); ``None`` beendet
    den Lauf, ebenso ein abandon ohne offenen Kandidaten."""
    picked = []

    def resume(s, gone):
        picked.append(gone)
        return None

    d = ScriptDecider([{"action": "free_gaps", "surfaces": [3]}],
                      [{"action": "abandon", "reason": "Test"}],
                      [{"action": "finish", "reason": "nie erreicht"}])
    assert session.run(d, 0, max_steps=10, resume=resume) == 1
    assert picked == [1] and len(session.log()) == 2 and len(d.answers) == 1
    d = ScriptDecider([{"action": "abandon", "reason": "Test"}])
    assert session.run(d, 0, max_steps=10) == 0
    assert session.open_candidates() == [] and len(session.log()) == 3


def test_history_names_synth_outcome(session):
    """M10-T7c: Der Verlauf im Zustand nennt Ausgang und NA jedes synth_segment; start_rejected
    sah vorher wie ein Erfolg aus."""
    entries = [
        {"seq": 1, "candidate": 0, "results": [{"action": "synth_segment", "status": "ok",
         "new_candidates": [1], "reports": [{"result": {
             "status": "start_rejected", "message": "Startpunkt nicht annehmbar (F_B > F_max)",
             "start_point": {"aperture": 0.3}}}]}]},
        {"seq": 2, "candidate": 1, "results": [{"action": "synth_segment", "status": "ok",
         "new_candidates": [2], "reports": [{"result": {
             "status": "step_too_small", "point": {"aperture": 0.4475}}}]},
                                               {"action": "free_gaps", "status": "ok",
                                                "new_candidates": [3]}]}]
    with open(session.dir / "log.jsonl", "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    assert session.history() == [
        "#0: synth_segment → [1] (start_rejected, Apertur nicht geöffnet: Startpunkt nicht "
        "annehmbar (F_B > F_max))",
        "#1: synth_segment → [2] (step_too_small, NA 0.4475)",
        "#1: free_gaps → [3]"]


def test_optimize_variable_selection(session):
    """optimize variables (vorher not_implemented außer current): jede Auswahl erweitert die
    Variablen der Datei; Glas bleibt fest, die letzte Dicke (Arbeitsabstand) auch."""
    expect = {"all_curvatures": ([1, 2, 3, 4, 6, 7], []),
              "curvatures_and_gaps": ([1, 2, 3, 4, 6, 7], [0, 3, 5]),
              "everything": ([1, 2, 3, 4, 6, 7], [0, 1, 2, 3, 4, 5, 6])}
    for sel, (curv, thick) in expect.items():
        r = response({"action": "optimize", "variables": sel, "cycles": 2})
        validate(r, "design_action.schema.json")
        entry = session.apply(0, r, decider={})
        res = entry["results"][0]
        assert res["status"] == "ok", res
        assert res["reports"][0]["variables"] == {"selection": sel, "curvatures": curv,
                                                  "thicknesses": thick}
        data = json.loads(session.system_path(res["new_candidates"][0]).read_text(
            encoding="utf-8"))
        items = data["variables"]["items"]
        assert [i["surfaces"] for i in items if i["type"] == "curvature"] == [curv]
        assert sorted(i["surface"] for i in items if i["type"] == "thickness") == thick
        assert not any(i["type"] == "glass" for i in items)
    # schon variable Flächen werden nicht doppelt freigegeben
    entry = session.apply(3, response({"action": "optimize", "variables": "everything",
                                       "cycles": 2}, candidate=3), decider={})
    assert entry["results"][0]["status"] == "ok"
    assert "variables" not in entry["results"][0]["reports"][0]
    assert not list((session.dir / "tmp").glob("*.json"))  # keine Zwischendateien


def test_optimize_widening_and_ghost_plate_gaps(tmp_path):
    """E14 (M10-T7c): In M10-T7d blieben Spalte mit Geisterplatten bei curvatures_and_gaps/
    everything und free_gaps fest. Seit M10-T7e gibt free_gaps sie wieder frei (untere Schranke
    = Plattenlage; Operator A senkt im zu kurzen Spalt die Schranke der Teilstücke);
    include_ghost_plate_gaps = false hält sie fest wie in M10-T7d."""
    s = design.Session.create(SYSTEMS / "einzellinse_homotopie_start.json", tmp_path / "s")
    data = json.loads(s.system_path(0).read_text(encoding="utf-8"))
    plate_gaps = {g["after_surface"] for g in data["ghost_plates"]}
    assert plate_gaps
    entry = s.apply(0, response({"action": "optimize", "variables": "everything", "cycles": 2}),
                    decider={})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    freed = res["reports"][0]["variables"]["thicknesses"]
    assert plate_gaps <= set(freed)
    new = json.loads(s.system_path(res["new_candidates"][0]).read_text(encoding="utf-8"))
    for it in new["variables"]["items"]:
        if it["type"] == "thickness" and it.get("surface") in plate_gaps:
            lage = max(g["distance"] for g in data["ghost_plates"]
                       if g["after_surface"] == it["surface"])
            assert it["min"] == pytest.approx(max(lage, 0.1))
    r = response({"action": "free_gaps", "surfaces": sorted(plate_gaps),
                  "include_ghost_plate_gaps": False})
    validate(r, "design_action.schema.json")
    entry = s.apply(0, r, decider={})
    assert entry["results"][0]["status"] == "ok"
    assert not entry["results"][0].get("new_candidates")
    entry = s.apply(0, response({"action": "free_gaps", "surfaces": sorted(plate_gaps)}),
                    decider={})
    assert entry["results"][0]["new_candidates"]


def _b1_start(tmp_path):
    """Startsystem von B1 wie im Bewertungsskript (Referenzsphäre image_infinite)."""
    from mikroptim.design.files import read_system, resolve_catalogs, write_system
    src = SYSTEMS / "einzellinse_homotopie_start.json"
    data = resolve_catalogs(read_system(src), SYSTEMS)
    data["reference_sphere"] = "image_infinite"
    path = tmp_path / "start_B1_rs.json"
    write_system(data, path)
    return path


def test_free_gaps_releases_plate_gaps_without_e14(tmp_path, monkeypatch):
    """E14 (M10-T7c) ohne Modell auf B1: free_gaps + optimize vor dem ersten Abschnitt. Bis
    M10-T7c schob der Optimierer den freien Spalt nach der Blende auf die Plattenlage (2 → 1 mm),
    Operator A fand keinen Platz für zwei Teilstücke und synth lehnte den Start ab
    (start_rejected, NA blieb 0,30); M10-T7d hielt die Plattenspalte deshalb fest. Seit M10-T7e
    werden sie wieder freigegeben, Operator A senkt im zu kurzen Spalt die Schranke der
    Teilstücke (synth --shrink-short-gaps, Warnung), und der Abschnitt erreicht das Ziel."""
    from mikroptim.design import actions

    s = design.Session.create(_b1_start(tmp_path), tmp_path / "s", spec={"na": 0.6})
    data = json.loads(s.system_path(0).read_text(encoding="utf-8"))
    plate_gaps = {g["after_surface"] for g in data["ghost_plates"]}
    before = [x["thickness"] for x in data["surfaces"]]
    entry = s.apply(0, response({"action": "free_gaps"}, {"action": "optimize", "cycles": 20}),
                    decider={})
    assert [r["status"] for r in entry["results"]] == ["ok", "ok"]
    c = entry["results"][-1]["new_candidates"][0]
    opt = json.loads(s.system_path(c).read_text(encoding="utf-8"))
    freed = {i["surface"] for i in opt["variables"]["items"]
             if i["type"] == "thickness" and "surface" in i}
    assert 0 in freed and 0 in plate_gaps  # Plattenspalt nach der Blende ist frei
    plate0 = max(g["distance"] for g in data["ghost_plates"] if g["after_surface"] == 0)
    assert [i["min"] for i in opt["variables"]["items"]
            if i["type"] == "thickness" and i.get("surface") == 0] == [plate0]
    assert opt["surfaces"][0]["thickness"] != before[0]
    # Gegenprobe ohne gesenkte Schranke: die Synthese schiebt den Spalt auf die Plattenlage, und
    # Operator A findet dort keinen Platz (E14) – das Ziel wird nicht erreicht
    monkeypatch.setattr(actions, "SHRINK_SHORT_GAPS", False)
    entry = s.apply(c, response({"action": "synth_segment", "target": {"aperture": 0.6},
                                 "start_retry_cycles": 0}, candidate=c), decider={})
    res = entry["results"][0]
    assert res["status"] == "error" or res["reports"][0]["result"]["status"] != "goal_reached"
    monkeypatch.setattr(actions, "SHRINK_SHORT_GAPS", True)
    entry = s.apply(c, response({"action": "synth_segment", "target": {"aperture": 0.6}},
                                candidate=c), decider={})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    rep = res["reports"][0]
    result = rep["result"]
    assert result["status"] == "goal_reached", result.get("message")
    assert result["point"]["aperture"] == pytest.approx(0.6)
    # Operator A im kurzen Plattenspalt: Warnung im Bericht und im Verlauf
    assert any("Schranke der Teilstücke" in w for w in rep.get("warnings", [])), rep.get("warnings")
    assert any("Warnung" in line for line in s.history())


def test_synth_segment_retries_after_start_rejected_e19(session, monkeypatch):
    """E19 (M10-T7d, behoben in M10-T7e): lehnt synth den Start knapp über F_max ab und meldet
    keine Geisterplatte Bedarf, läuft erst optimize (start_retry_cycles, Standard 20) und dann
    ein zweiter Versuch; Bericht start_retry und Verlauf zeigen das. Ohne Wirkung bleibt das
    erste Ergebnis; mit Bedarf (oder start_retry_cycles 0) kein zweiter Versuch. Die
    Kommandozeile ist hier ersetzt (Ablauf, nicht Optik)."""
    import shutil

    from mikroptim.design import actions

    calls = []
    rejected = {"status": "start_rejected", "steps": [],
                "message": "Startpunkt nicht annehmbar (F_B = 0.0057 > F_max = 0.0051); keine "
                           "Geisterplatte aktiv (kein Bedarf)"}
    plan = {}

    def fake(args):
        cmd, src = args[0], args[1]
        out = args[args.index("-o") + 1]
        shutil.copy(src, out)
        calls.append((cmd, list(args)))
        if cmd == "optimize":
            return {"result": {"merit": 1.0, "status": "converged"}}
        n = sum(1 for c, _ in calls if c == "synth")
        return {"result": plan[n]}

    monkeypatch.setattr(actions, "cli_json", fake)
    a = {"action": "synth_segment", "target": {"aperture": 0.5}}
    validate(response(a), "design_action.schema.json")
    # zweiter Versuch erreicht das Ziel
    plan.update({1: rejected, 2: {"status": "goal_reached", "point": {"aperture": 0.5},
                                  "steps": [{"operation": "Operator A ...; Warnung: Test"}]}})
    entry = session.apply(0, response(a), decider={})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    rep = res["reports"][0]
    assert [c for c, _ in calls] == ["synth", "optimize", "synth"]
    assert "--shrink-short-gaps" in calls[0][1]
    assert calls[1][1][calls[1][1].index("--max-iterations") + 1] == 20
    assert calls[2][1][1] == calls[1][1][3]  # zweiter synth auf dem optimierten System
    assert rep["result"]["status"] == "goal_reached"
    assert rep["start_retry"]["outcome"] == "goal_reached" and rep["start_retry"]["cycles"] == 20
    assert rep["warnings"] == ["Test"]
    assert "nach start_rejected optimize (20 Zyklen) und zweiter Versuch" in session.history()[-1]
    assert not list((session.dir / "tmp").glob("*.json"))
    # zweiter Versuch ebenfalls abgelehnt: erstes Ergebnis bleibt, im Verlauf sichtbar
    calls.clear()
    plan.update({1: rejected, 2: rejected})
    entry = session.apply(0, response({**a, "start_retry_cycles": 5}), decider={})
    rep = entry["results"][0]["reports"][0]
    assert [c for c, _ in calls] == ["synth", "optimize", "synth"]
    assert rep["result"]["status"] == "start_rejected"
    assert rep["start_retry"]["outcome"] == "start_rejected"
    assert "ebenfalls abgelehnt" in session.history()[-1]
    # abgeschaltet bzw. Bedarf gemeldet: kein zweiter Versuch
    for extra, result in (({"start_retry_cycles": 0}, rejected),
                          ({}, {**rejected, "message": "Startpunkt nicht annehmbar; Einfügen "
                                                       "fehlgeschlagen"})):
        calls.clear()
        plan.update({1: result})
        entry = session.apply(0, response({**a, **extra}), decider={})
        assert [c for c, _ in calls] == ["synth"]
        assert "start_retry" not in entry["results"][0]["reports"][0]


def test_human_decider_and_replay(session, tmp_path):
    answer = json.dumps(response({"action": "free_gaps", "surfaces": [3]}))
    human = design.HumanDecider(stdin=io.StringIO(answer + "\n"), stdout=io.StringIO())
    entry = session.step(human, 0)
    assert entry["decider"] == {"kind": "mensch"} and entry["state_sha256"]
    session.apply(1, response({"action": "optimize", "cycles": 3}, candidate=1), decider={})
    log = session.log()
    assert [e["seq"] for e in log] == [1, 2]
    again = design.replay(log, START, tmp_path / "wieder", spec=session.meta["spec"])
    for cid in session.candidates():
        a = json.loads(session.system_path(cid).read_text(encoding="utf-8"))
        b = json.loads(again.system_path(cid).read_text(encoding="utf-8"))
        assert a == b


def test_command_line(tmp_path, capsys):
    from mikroptim.design.__main__ import main
    d = tmp_path / "k"
    assert main(["new", str(START), str(d)]) == 0
    assert "Kandidat 0" in capsys.readouterr().out
    f = tmp_path / "a.json"
    f.write_text(json.dumps(response({"action": "free_gaps", "surfaces": [3]})), encoding="utf-8")
    assert main(["act", str(d), str(f), "-c", "0"]) == 0
    assert main(["tree", str(d)]) == 0
    assert "free_gaps" in capsys.readouterr().out


def test_glass_preselect_action(session):
    r = response({"action": "glass_preselect", "order": 1, "element": 2, "top": 3})
    validate(r, "design_action.schema.json")
    entry = session.apply(0, r, decider={"kind": "test"})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    out = res["result"]
    assert out["order"] == 1 and 1 <= len(out["results"]) <= 3
    assert all(x["glasses"][0] == "N-BK7" for x in out["results"])  # Element 2: N-BK7 fest
    assert list(session.candidates()) == [0]
    bad = session.apply(0, response({"action": "glass_preselect", "element": 1}), decider={})
    assert bad["results"][0]["code"] == "precondition"  # Buchdahl-Modell, kein Katalogglas



def test_split_lens_achromat_with_preselection(tmp_path):
    s = design.Session.create(SYSTEMS / "einzellinse_achromat_start.json", tmp_path / "a")
    r = response({"action": "split_lens", "element": 0, "cement_curvature": "achromat",
                  "free_part": 1})
    validate(r, "design_action.schema.json")
    entry = s.apply(0, r, decider={})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    rep = res["reports"][0]
    assert rep["partner"] and rep["secondary_spectrum"] >= 0
    data = json.loads(s.system_path(1).read_text(encoding="utf-8"))
    assert any(v["type"] == "glass" for v in data["variables"]["items"])
    # mit Zielglas: das freigegebene Glas startet dort (Buchdahl-Fit des Partners)
    r2 = response({"action": "split_lens", "element": 0, "cement_curvature": "achromat",
                   "free_part": 1, "target_glass": {"name": rep["partner"]}})
    e2 = s.apply(0, r2, decider={})
    assert e2["results"][0]["status"] == "ok", e2["results"][0]
    d2 = json.loads(s.system_path(2).read_text(encoding="utf-8"))
    assert d2["surfaces"][2]["material"] != data["surfaces"][2]["material"]


def test_sensitivity_action_and_state(session):
    """M10-T3: Lese-Aktion sensitivity (Filter nach Art und Element) und Kurzfassung im Zustand."""
    from mikroptim.design import files

    r = response({"action": "sensitivity", "kinds": ["decenter", "tilt"], "elements": [0]})
    validate(r, "design_action.schema.json")
    entry = session.apply(0, r, decider={"kind": "test"})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    out = res["result"]
    assert out["set"] == "default"  # Testsystem ohne Toleranzsatz
    assert out["tolerances"], out
    assert {t["type"] for t in out["tolerances"]} <= {"element_decenter", "element_tilt"}
    front = files.elements(files.read_system(session.system_path(0)))[0]["front"]
    for t in out["tolerances"]:
        first, last = t["element"]
        assert first <= front < last
        assert t["delta_rms_waves"] is not None
    deltas = [t["worst_delta_q_waves2"] for t in out["tolerances"]]
    assert deltas == sorted(deltas, reverse=True)
    assert list(session.candidates()) == [0]
    bad = session.apply(0, response({"action": "sensitivity", "elements": [99]}), decider={})
    assert bad["results"][0]["code"] == "precondition"

    st = session.state(0)
    validate(st, "design_state.schema.json")
    sens = st["sensitivity"]
    assert sens["set"] == "default" and len(sens["decenter_worst"]) >= 1
    assert all(len(e) == 2 and len(e[0]) == 2 for e in sens["decenter_worst"])
    assert "manufacturability" in st["mechanics"]
    assert "Empfindlichkeit" in design.format_state(st)


def test_state_glass_check_and_snap_precondition(session):
    """Befund E7 (M10-T7): Vorprüfung „einrastbar“ und Katalognamen im Zustand; snap_glasses
    ohne Glasvariablen wird vor dem Aufruf von snap abgelehnt."""
    st = session.state(0)
    validate(st, "design_state.schema.json")
    gc = st["glass_check"]
    assert gc["snappable"] is False and gc["glass_variables"] == 0
    # Elemente 1 und 3 des Mehrlinsers tragen ein Buchdahl-Modell
    assert gc["model_glass_elements"] == [1, 3]
    assert "N-BK7" in gc["catalog_names"] and "CAF2" in gc["catalog_names"]
    assert gc["catalog_glass_count"] == len(gc["catalog_names"])
    assert "einrastbar nein" in design.format_state(st)
    entry = session.apply(0, response({"action": "snap_glasses"}), decider={"kind": "test"})
    res = entry["results"][0]
    assert res["status"] == "error" and res["code"] == "precondition"
    assert "glass_check" in res["message"]
    # mit Glasvariable (split_lens mit free_part 1): einrastbar
    entry = session.apply(0, response({"action": "split_lens", "element": 0, "free_part": 1}),
                          decider={"kind": "test"})
    gc1 = session.state(entry["results"][0]["new_candidates"][0])["glass_check"]
    assert gc1["snappable"] is True and gc1["glass_variables"] == 1


def test_unknown_glass_name_is_rejected(session):
    """Befund E7 (M10-T7): erfundene Glasnamen (z. B. ``buchdahl(...)`` aus der Geisterdiagnose,
    ``F2``) sind ein Fehler ``unknown_glass`` vor dem Aufruf der Kommandozeile."""
    for action in ({"action": "insert_null_meniscus", "after_surface": 3,
                    "glass": {"name": "buchdahl(n0=1.8, nu=-0.1/0.04)"}},
                   {"action": "split_lens", "element": 0, "free_part": 1,
                    "target_glass": {"name": "F2"}},
                   {"action": "insert_module", "module": "generic.null_meniscus",
                    "after_surface": 3, "glasses": [{"name": "N-BK7"}, {"name": "SF99"}]}):
        res = session.apply(0, response(action), decider={"kind": "test"})["results"][0]
        assert res["status"] == "error" and res["code"] == "unknown_glass", res
        assert "catalog_names" in res["message"]
    ok = session.apply(0, response({"action": "insert_null_meniscus", "after_surface": 3,
                                    "glass": {"name": "N-BK7"}}), decider={"kind": "test"})
    assert ok["results"][0]["status"] == "ok"


def test_catalog_names_json_and_agf(tmp_path):
    agf = tmp_path / "test.agf"
    agf.write_bytes("CC Testkatalog\nNM GLAS-A 2 0 1.5 60 0 0 0\nNM GLAS-B 2 0 1.6 40\n"
                    .encode("utf-16"))
    data = {"glass_catalogs": [str(ROOT / "data" / "glass" / "basis.json"), str(agf),
                               str(tmp_path / "fehlt.json")]}
    assert catalog_names(data) == ["N-BK7", "CAF2", "F_SILICA", "GLAS-A", "GLAS-B"]


def test_state_monochromatic_has_no_target_color_class(tmp_path):
    """Befund E8 (M10-T7): Ein monochromatisches System nennt keine Ziel-Farbklasse, auch wenn
    die Spezifikation eine enthält; mit mehreren Wellenlängen bleibt sie."""
    spec = {"tube_focal_length": 164.5, "target_color_class": "c"}
    mono = design.Session.create(SYSTEMS / "einzellinse_homotopie_start.json", tmp_path / "m",
                                 spec=spec)
    st = mono.state(0)
    validate(st, "design_state.schema.json")
    assert st["spec"]["monochromatic"] is True and st["spec"]["target_color_class"] is None
    assert "keine Ziel-Farbklasse" in design.format_state(st)
    poly = design.Session.create(START, tmp_path / "p", spec=spec)
    st = poly.state(0)
    assert st["spec"]["monochromatic"] is False and st["spec"]["target_color_class"] == "c"


def test_unevaluable_result_is_rejected(session, tmp_path):
    """Befund M10-T7b: Ein neues System, dessen Zustand sich nicht bilden lässt (hier Brechzahl
    ≤ 0 eines Glasmodells), ist ein Fehler der Aktion ``invalid_result``; es entsteht kein
    Kandidat und die Sitzung läuft weiter."""
    from mikroptim.design import session as session_module

    data = json.loads(session.system_path(0).read_text(encoding="utf-8"))
    for s in data["surfaces"]:
        if isinstance(s.get("material"), dict):
            s["material"] = {**s["material"], "n0": -0.5}
    bad = tmp_path / "kaputt.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(design.ActionError) as e:
        session_module._check_results([(bad, {})])
    assert e.value.code == "invalid_result" and not bad.exists()
    session_module._check_results([(session.system_path(0), {})])  # gültig: kein Fehler
