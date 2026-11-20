"""Strukturelle Aktionen der Entwurfssteuerung (M10-T4): place_ghost_dictionary (Merit
unverändert), insert_module (aplanatische Vorlage ändert die Gruppensumme nicht, Null-Vorlagen),
remove_element nach insert_null_meniscus = Ausgangssystem, Rückbau-Homotopie, replace_glass,
replace_group, start_from_skeleton."""

import json

import pytest

import mikroptim
from mikroptim import design

from conftest import ROOT, SYSTEMS, tol

START = SYSTEMS / "testsystem_mehrlinser.json"
SCHEMAS = ROOT / "data" / "design" / "schemas"


def response(*actions, candidate=0):
    return {"format": "mikroptim.design.action/1", "candidate": candidate,
            "assessment": "Test", "actions": list(actions)}


def validate(instance):
    """Gegen das Aktionsschema, falls jsonschema installiert ist (sonst nur die Ausführung)."""
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads((SCHEMAS / "design_action.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


def apply_ok(session, candidate, *actions):
    r = response(*actions, candidate=candidate)
    validate(r)
    entry = session.apply(candidate, r, decider={"kind": "test"})
    for res in entry["results"]:
        assert res["status"] == "ok", res
    return entry["results"]


def load(session, cid):
    return json.loads(session.system_path(cid).read_text(encoding="utf-8"))


@pytest.fixture
def session(tmp_path):
    return design.Session.create(START, tmp_path / "s", spec={"tube_focal_length": 164.5})


def with_variables(session, cid=0):
    """Kandidat mit Variablen (Luftspalte 3 und 5 frei)."""
    [res] = apply_ok(session, cid, {"action": "free_gaps", "surfaces": [3, 5]})
    return res["new_candidates"][0]


def test_place_ghost_dictionary_merit_unchanged(session):
    c = with_variables(session)
    [res] = apply_ok(session, c, {"action": "place_ghost_dictionary", "after_surface": 3,
                                  "distance": 0.5, "glasses": ["N-BK7", "CAF2"], "max_order": 6,
                                  "weight": 10.0})
    rep = res["reports"][0]
    assert rep["relative_change"] <= tol("kInsertionMeritRelative")
    assert rep["new_plates"] == [0, 1]
    data = load(session, res["new_candidates"][0])
    assert [g["glass"] for g in data["ghost_plates"]] == ["N-BK7", "CAF2"]
    assert all(g["coefficients"] == [0.0] * 6 for g in data["ghost_plates"])
    # zweites Wörterbuch an anderer Lage: Gläser von den vorhandenen Platten, deren Strafe
    [res2] = apply_ok(session, res["new_candidates"][0],
                      {"action": "place_ghost_dictionary", "after_surface": 0, "distance": 1.0})
    rep2 = res2["reports"][0]
    assert rep2["glasses"] == ["N-BK7", "CAF2"] and rep2["new_plates"] == [0, 1]
    assert rep2["relative_change"] <= tol("kInsertionMeritRelative")
    # ohne Strafe und ohne weight: strukturierter Fehler
    entry = session.apply(0, response({"action": "place_ghost_dictionary", "after_surface": 3,
                                       "distance": 0.5, "glasses": ["N-BK7"]}), decider={})
    assert entry["results"][0]["code"] == "tool_failed"


def test_remove_after_null_meniscus_restores_system(session):
    """Abnahme: remove_element nach insert_null_meniscus ergibt das Ausgangssystem."""
    [ins] = apply_ok(session, 0, {"action": "insert_null_meniscus", "after_surface": 3,
                                  "position": 0.4, "glass": {"name": "N-BK7"}})
    c1 = ins["new_candidates"][0]
    elems = design.files.elements(load(session, c1))
    k = next(e["element"] for e in elems if e["front"] == 4)
    [rem] = apply_ok(session, c1, {"action": "remove_element", "element": k})
    rep = rem["reports"][0]
    assert rep["null_element"] and rep["homotopy"] == [] and rep["removed_surfaces"] == 2
    before, after = load(session, 0), load(session, rem["new_candidates"][0])
    assert len(after["surfaces"]) == len(before["surfaces"])
    for s0, s1 in zip(before["surfaces"], after["surfaces"]):
        assert s1["thickness"] == pytest.approx(s0["thickness"],
                                                rel=tol("kRemovalRoundtripRelative"))
        assert {k: v for k, v in s1.items() if k != "thickness"} == \
            {k: v for k, v in s0.items() if k != "thickness"}
    assert rep["relative_change"] <= tol("kInsertionMeritRelative")
    t0 = mikroptim.trace(mikroptim.load_system(session.system_path(0))).rms_waves
    t1 = mikroptim.trace(mikroptim.load_system(session.system_path(rem["new_candidates"][0])))
    assert t1.rms_waves == pytest.approx(t0, rel=1e-10)


def test_aplanatic_module_keeps_group_sum(session):
    """Abnahme: aplanatische Vorlage (front.shell_ca) vor dem Objekt ändert die Gruppensumme der
    sphärischen Aberration aller Ordnungen um < 1e-3."""
    def front_sum(path):
        rep = mikroptim.aberrations(path)
        return sum(s["waves"]["W040_full"] for s in rep["surfaces"] if s["surface"] >= 6)

    before = front_sum(session.system_path(0))
    [res] = apply_ok(session, 0, {"action": "insert_module", "module": "front.shell_ca",
                                  "after_surface": 7, "position": 1.0, "thickness": 1.5,
                                  "glasses": [{"name": "N-BK7"}]})
    rep = res["reports"][0]
    assert rep["template"] == "aplanatic_shell" and rep["surfaces"] == [8, 9]
    after = front_sum(session.system_path(res["new_candidates"][0]))
    assert abs(after - before) < tol("kAplanaticGroupSpherical") * abs(before)
    assert rep["image_intercept"] == pytest.approx(rep["intercept"] / 1.5187, rel=2e-3)
    # Feld group (Entscheidung Maintainer 2026-11-19): Gruppe des Moduls an den neuen Flächen,
    # Gruppensumme im Aberrationsbericht
    data = load(session, res["new_candidates"][0])
    assert [s.get("group") for s in data["surfaces"]] == [None] * 8 + ["front", "front"]
    groups = mikroptim.aberrations(session.system_path(res["new_candidates"][0]))["groups"]
    assert [(g["group"], g["surfaces"]) for g in groups] == [("front", [8, 9])]


def test_group_field_overrides_automatic_groups(session):
    """Feld group: insert_module mit Parameter group setzt es an den neuen Flächen, der Zustand
    ordnet die Elemente danach statt nach der Randstrahlhöhe zu (Entscheidung Maintainer
    2026-11-19)."""
    st0 = session.state(0)
    auto = {g["element"]: g["group"] for g in st0["glasses"]}
    [res] = apply_ok(session, 0, {"action": "insert_module", "module": "generic.null_meniscus",
                                  "after_surface": 5, "position": 1.0, "group": "rear",
                                  "glasses": [{"name": "N-BK7"}]})
    assert res["reports"][0]["group"] == "rear"
    c = res["new_candidates"][0]
    data = load(session, c)
    assert [s.get("group") for s in data["surfaces"][6:8]] == ["rear", "rear"]
    st = session.state(c)
    new = next(g for g in st["glasses"] if g["surface"] == 6)
    assert new["group"] == "rear"
    # ausdrückliche Gruppe an einem bestehenden Element überschreibt die Automatik (hier ordnet
    # die Automatik alle Elemente der Mittelgruppe zu)
    first = st0["glasses"][0]
    assert auto[first["element"]] == "middle"
    surface = first["surface"]
    data["surfaces"][surface]["group"] = "front"
    session.system_path(c).write_text(json.dumps(data), encoding="utf-8")
    st2 = design.build_state(session.system_path(c), with_merit=False,
                             with_sensitivity=False)
    assert next(g for g in st2["glasses"] if g["surface"] == surface)["group"] == "front"
    lo, hi = st2["groups"]["front"]["surfaces"]
    assert lo <= surface <= hi


def test_thickness_homotopy_keeps_gap_bound(session):
    """M10-T7b: Wächst ein Modul dicker, als der Luftspalt dahinter über seiner Schranke Platz
    hat, bleibt der Spalt auf der Schranke und die folgenden Scheitel rücken (vorher lehnte
    optimize das System ab)."""
    c = with_variables(session)  # Spalt 5: 2 mm, untere Schranke 0,1 mm
    [res] = apply_ok(session, c, {"action": "insert_module", "module": "generic.null_meniscus",
                                  "after_surface": 5, "position": 1.0, "thickness": 0.95,
                                  "glasses": [{"name": "N-BK7"}], "stages": 1, "cycles": 0})
    data = load(session, res["new_candidates"][0])
    assert data["surfaces"][6]["thickness"] == 0.95
    assert data["surfaces"][7]["thickness"] == 0.1


@pytest.mark.parametrize("mod,glasses,added", [
    ("generic.null_meniscus", [{"name": "N-BK7"}], 2),
    ("middle.merte_doublet", [{"name": "N-BK7"}, {"name": "CAF2"}], 3),
    ("rear.gauss_pair", [{"name": "N-BK7"}], 4),
])
def test_null_templates_merit_unchanged(session, mod, glasses, added):
    c = with_variables(session)
    [res] = apply_ok(session, c, {"action": "insert_module", "module": mod, "after_surface": 5,
                                  "glasses": glasses, "group": "middle"})
    rep = res["reports"][0]
    data = load(session, res["new_candidates"][0])
    assert len(data["surfaces"]) == len(load(session, c)["surfaces"]) + added
    assert abs(rep["merit_after"] - rep["merit_before"]) <= \
        tol("kInsertionMeritRelative") * rep["merit_before"]
    assert rep["group"] == "middle"


def test_insert_module_default_glass(session):
    """M10-T7e (44 Antworten der Modelle in M10-T7d ohne glasses): Fehlt glasses, nimmt
    insert_module das erste Katalogglas der Glasregel des Moduls (front.shell_ca:
    low_n_low_dispersion), ohne Regel das hochbrechendste Katalogglas; Warnung im Bericht und im
    Verlauf."""
    from mikroptim.design.deciders import high_index_glass
    from mikroptim.design.glass_query import resolve_glass

    r = response({"action": "insert_module", "module": "front.shell_ca", "after_surface": 7,
                  "position": 1.0, "thickness": 1.5})
    validate(r)
    entry = session.apply(0, r, decider={"kind": "test"})
    res = entry["results"][0]
    assert res["status"] == "ok", res
    rep = res["reports"][0]
    assert rep["glass"] == resolve_glass(session.system_path(0), {"rule": "low_n_low_dispersion"})
    assert any("glasses fehlte" in w and "low_n_low_dispersion" in w for w in rep["warnings"])
    assert "Warnung: glasses fehlte" in session.history()[-1]
    # generic.null_meniscus hat keine Glasregel: hochbrechendstes Katalogglas
    c = with_variables(session)
    [res] = apply_ok(session, c, {"action": "insert_module", "module": "generic.null_meniscus",
                                  "after_surface": 5, "position": 0.5})
    rep = res["reports"][0]
    assert rep["glasses"] == [high_index_glass(session.system_path(0))]
    assert "keine Glasregel" in rep["warnings"][0]
    assert rep["merit_after"] == pytest.approx(rep["merit_before"], rel=1e-9)


def test_insert_module_raises_thickness_to_bound(session, tmp_path):
    """M10-T7e (81 Antworten der Modelle in M10-T7d): Liegt der Arbeitsabstand nach dem
    Nachfokussieren der Schale unter der Schranke seiner Variable, lehnte mikroptim insert das
    Ergebnis ab. Jetzt hebt insert_module ihn auf die Schranke (Warnung) und lässt die übrigen
    Flächen, wie sie ohne Schranke eingefügt würden."""
    from mikroptim.design.files import read_system, resolve_catalogs, write_system

    shell = {"action": "insert_module", "module": "front.shell_ca", "after_surface": 7,
             "position": 1.0, "thickness": 1.5, "glasses": [{"name": "N-BK7"}]}
    [free] = apply_ok(session, 0, shell)
    wd_free = free["reports"][0]["working_distance"]
    free_sys = load(session, free["new_candidates"][0])
    wd0 = read_system(session.system_path(0))["surfaces"][-1]["thickness"]
    assert 0 < wd_free < wd0
    bound = 0.5 * (wd_free + wd0)
    data = resolve_catalogs(read_system(START), SYSTEMS)
    data["variables"] = {"items": [{"type": "working_distance", "min": bound, "max": 50.0}]}
    start = tmp_path / "wd.json"
    write_system(data, start)
    s = design.Session.create(start, tmp_path / "wd")
    [res] = apply_ok(s, 0, shell)
    rep = res["reports"][0]
    new = load(s, res["new_candidates"][0])
    assert new["surfaces"][-1]["thickness"] == bound
    assert new["variables"]["items"][0]["min"] == bound
    # übrige Flächen wie ohne Schranke
    assert [x.get("curvature") for x in new["surfaces"]] == [x.get("curvature") for x in
                                                          free_sys["surfaces"]]
    assert [x["thickness"] for x in new["surfaces"][:-1]] == [x["thickness"] for x in
                                                               free_sys["surfaces"][:-1]]
    assert len(rep["warnings"]) == 1 and "Arbeitsabstand" in rep["warnings"][0]
    assert "Warnung: Arbeitsabstand" in s.history()[-1]
    # ein anderer Fehler der Kommandozeile bleibt ein Fehler
    bad = s.apply(0, response({**shell, "thickness": 500.0}), decider={})
    assert bad["results"][0]["status"] == "error"


def test_insert_module_thickness_homotopy(session):
    c = with_variables(session)
    [res] = apply_ok(session, c, {"action": "insert_module", "module": "generic.null_meniscus",
                                  "after_surface": 5, "position": 0.5,
                                  "glasses": [{"name": "N-BK7"}], "thickness": 0.4,
                                  "stages": 2, "cycles": 0})
    rep = res["reports"][0]
    assert [h["thickness"] for h in rep["homotopy"]] == [0.2, 0.4]
    assert all(h["optimize"] is None for h in rep["homotopy"])
    data = load(session, res["new_candidates"][0])
    assert data["surfaces"][6]["thickness"] == 0.4
    # Lage der folgenden Scheitel bleibt: Spalt 5 + Glas + Rest = 2,0 mm
    assert sum(data["surfaces"][i]["thickness"] for i in (5, 6, 7)) == pytest.approx(2.0)


def test_remove_element_homotopy(session):
    c = with_variables(session)
    n0 = len(load(session, c)["surfaces"])
    [res] = apply_ok(session, c, {"action": "remove_element", "element": 2, "stages": 2,
                                  "cycles": 3})
    rep = res["reports"][0]
    assert not rep["null_element"] and len(rep["homotopy"]) == 2
    assert rep["relative_change"] <= tol("kInsertionMeritRelative")
    data = load(session, res["new_candidates"][0])
    assert len(data["surfaces"]) == n0 - 2
    assert len(design.files.elements(data)) == 3
    # Kittglied: Element 1 gehört zum Block ab Element 0, beide werden entfernt
    [res2] = apply_ok(session, c, {"action": "remove_element", "element": 1, "stages": 1,
                                   "cycles": 0})
    assert res2["reports"][0]["removed_elements"] == [0, 1]
    assert len(load(session, res2["new_candidates"][0])["surfaces"]) == n0 - 3


def test_remove_element_reports_raised_bounds(tmp_path):
    """Rückbau über die obere Schranke einer Dickenvariable hinaus: Schranke angehoben, Warnung
    im Bericht der Aktion und damit im Sitzungsprotokoll (Entscheidung Maintainer 2026-11-19)."""
    data = json.loads(START.read_text(encoding="utf-8"))
    data["glass_catalogs"] = [str(ROOT / c) for c in data["glass_catalogs"]]
    data["variables"] = {"items": [{"type": "thickness", "surfaces": [3, 5], "min": 0.1,
                                    "max": 3.0}]}
    start = tmp_path / "start.json"
    start.write_text(json.dumps(data), encoding="utf-8")
    s = design.Session.create(start, tmp_path / "s", spec={"tube_focal_length": 164.5})
    [res] = apply_ok(s, 0, {"action": "remove_element", "element": 2, "stages": 1, "cycles": 0})
    warnings = res["reports"][0]["warnings"]
    assert warnings == ["obere Schranke der Dicke nach Fläche 5 von 3 auf 5 mm angehoben",
                        "obere Schranke der Dicke nach Fläche 3 von 5 auf 6 mm angehoben"]
    out = load(s, res["new_candidates"][0])
    assert out["variables"]["items"][0]["max"] == 6.0
    logged = [r for e in s.log() for r in e["results"]]
    assert logged[-1]["reports"][0]["warnings"] == warnings


def test_replace_glass(session):
    [res] = apply_ok(session, 0, {"action": "replace_glass", "element": 2,
                                  "glass": {"name": "CAF2"}})
    data = load(session, res["new_candidates"][0])
    assert data["surfaces"][4]["material"] == "CAF2"
    assert res["reports"][0]["optimize"] is None
    # Filter: Partner aus der Glasvorauswahl, mehrere Kandidaten
    c = with_variables(session)
    pre = design.actions.read_action({"action": "glass_preselect", "order": 1, "element": 2,
                                      "top": 50}, session.system_path(c))["results"]
    partners = list(dict.fromkeys(r["glasses"][1] for r in pre))
    [res2] = apply_ok(session, c, {"action": "replace_glass", "element": 2,
                                   "glass": {"filter": {"partner_of_element": 2,
                                                        "max_results": 3}},
                                   "cycles": 2})
    assert len(res2["new_candidates"]) == min(3, len(partners)) >= 1
    assert {r["glass"] for r in res2["reports"]} == set(partners[:3])
    merits = [r["optimize"]["merit"] for r in res2["reports"]]
    assert merits == sorted(merits)
    # Paarregel ohne Bezugsglas (Glasabfrage, M10-T5)
    bad = session.apply(0, response({"action": "replace_glass", "element": 2,
                                     "glass": {"filter": {"rule": "merte_pair"}}}), decider={})
    assert bad["results"][0]["code"] == "missing_parameter"
    unknown = session.apply(0, response({"action": "replace_glass", "element": 2,
                                         "glass": {"name": "GIBTESNICHT"}}), decider={})
    assert unknown["results"][0]["code"] == "unknown_glass"


def skeleton(session, distribution="PPN"):
    [res] = apply_ok(session, 0, {
        "action": "start_from_skeleton", "power_distribution": distribution,
        "front": ["generic.lens"], "middle": ["middle.merte_doublet"],
        "rear": ["generic.lens"], "start_aperture": 0.2, "glasses": [{"name": "N-BK7"}]})
    return res


def test_replace_group(session):
    from mikroptim.design import operators
    # Gerüst PNP: die Linse an der Blende trägt die größte Randstrahlhöhe (Mittelgruppe),
    # Kittglied und Frontlinse liegen darunter (Frontgruppe)
    c = skeleton(session, "PNP")["new_candidates"][0]
    groups = operators.group_elements(session.system_path(c))
    assert groups == {"front": [1, 2, 3], "middle": [0], "rear": []}, groups
    [res] = apply_ok(session, c, {"action": "replace_group", "group": "middle",
                                  "modules": ["generic.null_meniscus"], "stages": 1,
                                  "cycles": 0})
    rep = res["reports"][0]
    assert rep["removed"] and len(rep["inserted"]) == 1
    removed = sum(r["removed_surfaces"] - 1 for r in rep["removed"])  # ganze Glasblöcke
    assert removed == len(groups["middle"])
    data = load(session, res["new_candidates"][0])
    assert len(design.files.elements(data)) == \
        len(design.files.elements(load(session, c))) - removed + 1
    # Frontgruppe bis zum Objekt durch eine aplanatische Schale ersetzen (im Arbeitsabstand)
    [res2] = apply_ok(session, c, {"action": "replace_group", "group": "front",
                                   "modules": ["front.shell_ca"], "stages": 1, "cycles": 0})
    data2 = load(session, res2["new_candidates"][0])
    assert len(design.files.elements(data2)) == 2  # Linse an der Blende und Schale
    assert res2["reports"][0]["inserted"][0]["template"] == "aplanatic_shell"
    # Null-Vorlagen passen nicht in den Arbeitsabstand: strukturierter Fehler
    bad = session.apply(c, response({"action": "replace_group", "group": "front",
                                     "modules": ["generic.null_meniscus"], "stages": 1,
                                     "cycles": 0}, candidate=c), decider={})
    assert bad["results"][0]["code"] == "precondition"


def test_start_from_skeleton_new_root(session):
    res = skeleton(session)
    cid = res["new_candidates"][0]
    assert session.candidates()[cid]["parent"] is None
    rep = res["reports"][0]
    data = load(session, cid)
    target = data["target"]["focal_length"]
    assert rep["focal_length"] == pytest.approx(target, rel=1e-9)
    assert len(design.files.elements(data)) == 4  # Linse, Kittglied (2), Linse
    assert data["surfaces"][0]["semi_diameter"] == pytest.approx(0.2 * target)
    # neues System mit der empfohlenen Referenzsphäre (Entscheidung Maintainer 2026-11-19)
    assert data["reference_sphere"] == "image_infinite"
    st = session.state(cid)
    assert st["parent"] is None and st["elements"] == 4
    assert st["reference_sphere"] == {"kind": "image_infinite", "explicit": True}
    # LM kommt vom Gerüst aus voran
    [opt] = apply_ok(session, cid, {"action": "optimize", "cycles": 5})
    assert opt["new_candidates"]
    bad = session.apply(0, response({"action": "start_from_skeleton",
                                     "power_distribution": "P(PN)0", "front": ["generic.lens"],
                                     "middle": ["generic.lens"], "rear": ["generic.lens"]}),
                        decider={})
    assert bad["results"][0]["code"] == "invalid_parameter"  # (PN) braucht zwei Elemente


def test_embedded_schema_lists_operators():
    from mikroptim.design import schema
    for name in ("place_ghost_dictionary", "insert_module", "remove_element", "replace_glass",
                 "replace_group", "start_from_skeleton"):
        assert name in schema.ACTION_DEFS
        assert name in design.actions.MODIFY_ACTIONS
