"""Entwurfswerkzeuge (M10-T6a): Zustand, Aktionen, Kandidatenbaum, Protokoll, Wiedereinspielen."""

import io
import json

import pytest

from mikroptim import design

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
    todo = session.apply(0, response({"action": "remove_element", "element": 0}), decider={})
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

