"""Wissensbasis und Glasabfrage (M10-T5): Schema und Quellen jedes Eintrags, glass_query gegen
Bruteforce auf den Testkatalogen (Kennzahlen unabhängig aus den Katalogformeln, Regeln aus der
Tabelle in docs/entwurfssteuerung.md, Abschnitt 5.3, ausgeschrieben), module_lookup, Glasfilter
in Aktionen und die neuen Vorlagen strong_doublet und petzval_group."""

import json
import math
import random

import numpy as np
import pytest

import mikroptim
from mikroptim import design
from mikroptim.design import knowledge, schema
from mikroptim.design.glass_query import glass_query
from mikroptim.design.operators import incidence_angle_deg

from conftest import ROOT, SYSTEMS, tol

KNOWLEDGE = ROOT / "data" / "design" / "knowledge"
SCHEMAS = ROOT / "data" / "design" / "schemas"
CATALOGS = [ROOT / "data" / "glass" / "synthetisch.json", ROOT / "data" / "glass" / "basis.json"]
START = SYSTEMS / "testsystem_mehrlinser.json"
LINES = {"d": 587.5618, "F": 486.1327, "C": 656.2725, "g": 435.8343}


def response(*actions, candidate=0):
    return {"format": "mikroptim.design.action/1", "candidate": candidate,
            "assessment": "Test", "actions": list(actions)}


def validate(instance, name="design_action.schema.json"):
    jsonschema = pytest.importorskip("jsonschema")
    schema_doc = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema_doc).validate(instance)


# --- Bruteforce: Kennzahlen direkt aus den Katalogformeln ------------------------------------


def catalog_index(g, nm):
    lam2 = (nm / 1000.0) ** 2
    if g["model"] == "sellmeier":
        n2 = 1.0 + g.get("A", 0.0) + sum(b * lam2 / (lam2 - c) for b, c in zip(g["B"], g["C"]))
    else:
        n2 = sum(a * (nm / 1000.0) ** p for a, p in g["terms"])
    return math.sqrt(n2)


def brute_glasses(catalogs):
    out = []
    for path in catalogs:
        for g in json.loads(path.read_text(encoding="utf-8"))["glasses"]:
            lo, hi = g["range_nm"]
            if lo > LINES["g"] or hi < LINES["C"]:
                continue
            n = {k: catalog_index(g, v) for k, v in LINES.items()}
            vd = (n["d"] - 1.0) / (n["F"] - n["C"])
            pgf = (n["g"] - n["F"]) / (n["F"] - n["C"])
            out.append({"name": g["name"], "nd": n["d"], "vd": vd, "pgf": pgf,
                        "dpgf": pgf - (0.6438 - 0.001682 * vd),
                        "transmission": g.get("transmission")})
    return out


def inside(x, lo, hi):
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def af_forbidden(g):
    zone = g["vd"] <= 35 or (g["vd"] <= 50 and g["nd"] <= 1.62)
    exception = 25 <= g["vd"] <= 41 and 1.65 <= g["nd"] <= 1.80
    return zone and not exception


#: Regeln der Tabelle 5.3, unabhängig von glass_rules.json ausgeschrieben (Schwellen der
#: Mikroptim-Annahmen wie dort dokumentiert)
SINGLE_RULES = {
    "selectable_pgf": lambda g: 0.535 <= g["pgf"] <= 0.58,
    "front_medium_n_medium_vd": lambda g: 1.60 <= g["nd"] <= 1.90 and 40 <= g["vd"] <= 60,
    "low_n_low_dispersion": lambda g: g["nd"] <= 1.60 and g["vd"] >= 66,
    "af_forbidden": af_forbidden,
    "not_af_forbidden": lambda g: not af_forbidden(g),
    "dense_flint_for_lca": lambda g: g["nd"] >= 1.65 and 25 <= g["vd"] <= 40,
    "anomalous_pgf": lambda g: abs(g["dpgf"]) >= 0.01,
}
PAIR_RULES = {
    "merte_pair": lambda g, r: (0.05 <= abs(g["nd"] - r["nd"]) <= 0.15
                                and abs(g["vd"] - r["vd"]) >= 15
                                and abs(g["pgf"] - r["pgf"]) <= 0.01),
    "buried_surface": lambda g, r: (abs(g["nd"] - r["nd"]) <= 0.01
                                    and abs(g["vd"] - r["vd"]) >= 10),
}


@pytest.fixture(scope="module")
def catalog_system(tmp_path_factory):
    """Testsystem mit beiden Testkatalogen (synthetisch, basis)."""
    data = json.loads(START.read_text(encoding="utf-8"))
    data["glass_catalogs"] = [str(p) for p in CATALOGS]
    path = tmp_path_factory.mktemp("kat") / "system.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def brute():
    return brute_glasses(CATALOGS)


def names(res):
    return [g["name"] for g in res["glasses"]]


# --- Wissensbasis -----------------------------------------------------------------------------


def test_knowledge_files_match_schema():
    for f in knowledge.FILES:
        validate(json.loads((KNOWLEDGE / f).read_text(encoding="utf-8")),
                 "design_knowledge.schema.json")


def test_every_entry_has_a_source():
    """Abnahme: jede Regel, jedes Modul und jede Verteilung mit nicht leerer Quelle; Einträge
    ohne Literaturbeleg sind als Mikroptim-Annahme gekennzeichnet."""
    docs = {f: json.loads((KNOWLEDGE / f).read_text(encoding="utf-8")) for f in knowledge.FILES}
    entries = (docs["modules.json"]["modules"] + docs["glass_rules.json"]["rules"]
               + docs["power_distributions.json"]["distributions"])
    assert len(entries) >= 30
    for e in entries:
        assert isinstance(e.get("source"), str) and len(e["source"].strip()) >= 3, e["id"]
        assert e["source"].strip() not in ("-", "?"), e["id"]
        if "assumption" in e:
            assert e["assumption"].startswith("Mikroptim-Annahme"), e["id"]
    assert docs["power_distributions.json"]["sign_weights_source"].strip()
    # Kurzzeichen der Quellen sind aufgelöst
    for f, doc in docs.items():
        refs = doc.get("references", {})
        key = {"modules.json": "modules", "glass_rules.json": "rules",
               "power_distributions.json": "distributions"}[f]
        for e in doc[key]:
            if "Zhang 2019" in e["source"]:
                assert "Zhang 2019" in refs, (f, e["id"])


def test_loaded_knowledge_matches_repository():
    kb = knowledge.load()
    for f in knowledge.FILES:
        here = (KNOWLEDGE / f).read_bytes()
        assert (knowledge.knowledge_dir() / f).read_bytes() == here, f
    assert len(kb["digest"]) == 64
    assert set(kb["rules"]) == set(SINGLE_RULES) | set(PAIR_RULES)
    # Aktionsschema des Repositorys und eingebettetes Schema kennen genau diese Regeln,
    # Verteilungen und einfügbaren Module
    repo = json.loads((SCHEMAS / "design_action.schema.json").read_text(encoding="utf-8"))
    assert set(repo["$defs"]["glass_filter"]["properties"]["rule"]["enum"]) == set(kb["rules"])
    assert set(schema.RULE_IDS) == set(kb["rules"])
    dist = repo["$defs"]["start_from_skeleton"]["properties"]["power_distribution"]["enum"]
    assert set(dist) == set(kb["distributions"]) == set(schema.DISTRIBUTION_IDS)
    assert "double_gauss" in dist
    assert set(schema.MODULE_IDS) == {k for k, m in kb["modules"].items()
                                      if m["template"]["kind"] in knowledge.INSERT_TEMPLATES}
    assert "sine_condition" in \
        repo["$defs"]["set_constraint"]["properties"]["type"]["enum"]


def test_inconsistent_knowledge_is_rejected(tmp_path):
    for f in knowledge.FILES:
        (tmp_path / f).write_bytes((KNOWLEDGE / f).read_bytes())
    rules = json.loads((tmp_path / "glass_rules.json").read_text(encoding="utf-8"))
    rules["rules"][0]["source"] = " "
    (tmp_path / "glass_rules.json").write_text(json.dumps(rules), encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="ohne Quelle"):
        knowledge._load(str(tmp_path))
    rules["rules"][0]["source"] = "x y"
    rules["rules"][0]["condition"] = {"rule": rules["rules"][0]["id"]}
    (tmp_path / "glass_rules.json").write_text(json.dumps(rules), encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="zyklisch"):
        knowledge._load(str(tmp_path))


# --- glass_query gegen Bruteforce -------------------------------------------------------------


def test_glass_properties_match_catalog_formulas(catalog_system, brute):
    res = glass_query(catalog_system, {}, limit=None)
    assert res["count"] == len(brute) == res["catalog_glasses"] == 39
    for got, exp in zip(res["glasses"], brute):
        assert got["name"] == exp["name"]
        for k, kk in (("nd", "nd"), ("vd", "vd"), ("pgf", "pgf")):
            assert got[k] == pytest.approx(exp[kk], rel=tol("kGlassPropertiesFormula"))
        assert got["dPgF"] == pytest.approx(exp["dpgf"], abs=tol("kGlassPropertiesFormula"))


@pytest.mark.parametrize("rule", sorted(SINGLE_RULES))
def test_single_rules_match_brute_force(catalog_system, brute, rule):
    """Abnahme: Filter gegen Bruteforce auf den Testkatalogen."""
    res = glass_query(catalog_system, {"rule": rule}, limit=None)
    expected = [g["name"] for g in brute if SINGLE_RULES[rule](g)]
    assert names(res) == expected
    assert res["count"] == len(expected)
    for g in res["glasses"]:
        assert rule in g["rules"]
    assert res["rules"][rule]["source"]


def test_rules_are_not_all_empty_on_test_catalogs(brute):
    hits = {r: sum(1 for g in brute if f(g)) for r, f in SINGLE_RULES.items()}
    assert sum(1 for v in hits.values() if v) >= 5, hits


@pytest.mark.parametrize("rule", sorted(PAIR_RULES))
def test_pair_rules_match_brute_force(catalog_system, brute, rule):
    total = 0
    for ref in brute:
        res = glass_query(catalog_system, {"rule": rule, "partner": ref["name"]}, limit=None)
        expected = [g["name"] for g in brute
                    if g["name"] != ref["name"] and PAIR_RULES[rule](g, ref)]
        assert names(res) == expected, ref["name"]
        assert res["reference"]["name"] == ref["name"]
        total += len(expected)
    assert total > 0


def test_random_filters_match_brute_force(catalog_system, brute):
    rng = random.Random(20261119)

    def rand_range(lo, hi):
        a, b = sorted(rng.uniform(lo, hi) for _ in range(2))
        k = rng.randrange(3)
        return [None if k == 0 else a, None if k == 1 else b]

    nonempty = 0
    for _ in range(40):
        flt, conds = {}, []
        for key, field, lo, hi in (("nd", "nd", 1.40, 1.85), ("vd", "vd", 20, 95),
                                   ("pgf", "pgf", 0.50, 0.63), ("dPgF", "dpgf", -0.01, 0.05)):
            if rng.random() < 0.5:
                r = rand_range(lo, hi)
                flt[key] = r
                conds.append(lambda g, f=field, r=r: inside(g[f], *r))
        rules = rng.sample(sorted(SINGLE_RULES), rng.randrange(3))
        if rules:
            flt["rules"] = rules
        exclude = rng.sample([g["name"] for g in brute], 2)
        flt["exclude"] = exclude
        expected = [g["name"] for g in brute if g["name"] not in exclude
                    and all(c(g) for c in conds) and all(SINGLE_RULES[r](g) for r in rules)]
        res = glass_query(catalog_system, flt, limit=None)
        assert names(res) == expected, flt
        nonempty += bool(expected)
    assert nonempty >= 10


def test_partner_of_element_uses_preselection_order(catalog_system, brute):
    # Element 2 ist N-BK7 (Fläche 4)
    pre = mikroptim.run_cli(["glasses", "--system", str(catalog_system), "--order", "1",
                             "--fixed", "N-BK7", "--top", "50", "--json"])
    ranked = list(dict.fromkeys(r["glasses"][1] for r in json.loads(pre[1])["results"]))
    by = {g["name"]: g for g in brute}
    ref = by["N-BK7"]
    res = glass_query(catalog_system, {"partner_of_element": 2}, limit=None)
    assert names(res) == [n for n in ranked if n != "N-BK7"]
    assert res["glasses"][0]["secondary_spectrum"] is not None
    res = glass_query(catalog_system, {"partner_of_element": 2, "rule": "merte_pair",
                                       "vd": [None, 60]}, limit=None)
    expected = [n for n in ranked if n != "N-BK7" and PAIR_RULES["merte_pair"](by[n], ref)
                and by[n]["vd"] <= 60]
    assert names(res) == expected


def test_transmission_filter_matches_brute_force(tmp_path, brute):
    """Reintransmission aus dem Katalog (frei erfundene Kurven an einer Kopie des synthetischen
    Katalogs), auf eine andere Dicke umgerechnet."""
    cat = json.loads(CATALOGS[0].read_text(encoding="utf-8"))
    for i, g in enumerate(cat["glasses"]):
        if i % 3 == 2:
            continue  # ohne Daten
        edge = 0.80 + 0.19 * ((i * 7) % 10) / 9.0
        g["transmission"] = {"thickness_mm": 10, "nm": [350, 400, 500, 700],
                             "tau": [edge * 0.5, edge, 0.998, 0.999]}
    cpath = tmp_path / "kat_tau.json"
    cpath.write_text(json.dumps(cat), encoding="utf-8")
    data = json.loads(START.read_text(encoding="utf-8"))
    data["glass_catalogs"] = [str(cpath), str(CATALOGS[1])]
    spath = tmp_path / "system.json"
    spath.write_text(json.dumps(data), encoding="utf-8")
    glasses = brute_glasses([cpath, CATALOGS[1]])
    for nm, tmin, d in ((380.0, 0.6, None), (400.0, 0.9, 25.0), (450.0, 0.95, 5.0)):
        flt = {"transmission": {"nm": nm, "min": tmin}}
        if d is not None:
            flt["transmission"]["thickness_mm"] = d
        res = glass_query(spath, flt, limit=None)
        expected, no_data = [], []
        for g in glasses:
            t = g["transmission"]
            if t is None:
                no_data.append(g["name"])
                continue
            tau = float(np.interp(nm, t["nm"], t["tau"]))
            if d is not None:
                tau = tau ** (d / t["thickness_mm"])
            if tau >= tmin:
                expected.append(g["name"])
        assert names(res) == expected
        assert res["no_transmission_data"] == no_data
        assert 0 < len(expected) < len(glasses)


def test_glass_query_errors(catalog_system):
    from mikroptim.design.glass_query import QueryError
    with pytest.raises(QueryError) as e:
        glass_query(catalog_system, {"rule": "gibt_es_nicht"})
    assert e.value.code == "invalid_parameter"
    with pytest.raises(QueryError) as e:
        glass_query(catalog_system, {"rule": "merte_pair"})
    assert e.value.code == "missing_parameter"
    with pytest.raises(QueryError) as e:
        glass_query(catalog_system, {"rule": "merte_pair", "partner": "UNBEKANNT"})
    assert e.value.code == "precondition"
    with pytest.raises(QueryError) as e:
        glass_query(catalog_system, {"nd": [1.7, 1.5]})
    assert e.value.code == "invalid_parameter"
    with pytest.raises(QueryError) as e:
        glass_query(catalog_system, {"farbe": "blau"})
    assert e.value.code == "invalid_parameter"


# --- module_lookup ----------------------------------------------------------------------------


def test_module_lookup_matches_brute_force():
    raw = json.loads((KNOWLEDGE / "modules.json").read_text(encoding="utf-8"))["modules"]
    for group in (None, "front", "middle", "rear"):
        for corrects in ([], ["SA"], ["FC", "AST"], ["LCA"], ["SPHCRM", "ZSA"]):
            got = knowledge.module_lookup(group=group, corrects=corrects)
            exp = [m for m in raw if (group is None or m["group"] in (None, group))
                   and (not corrects or any(m["effect"].get(c, 0) != 0 for c in corrects))]
            exp.sort(key=lambda m: (-sum(abs(m["effect"].get(c, 0)) for c in corrects), m["id"]))
            assert [m["id"] for m in got] == [m["id"] for m in exp], (group, corrects)
    dry = {m["id"] for m in knowledge.module_lookup(na=0.95, immersion=False)}
    assert "front.thick_meniscus_dry" in dry
    assert "front.thick_meniscus_dry" not in {m["id"] for m in knowledge.module_lookup(na=0.5)}
    assert all(m["insertable"] for m in knowledge.module_lookup(insertable_only=True))
    with pytest.raises(knowledge.KnowledgeError):
        knowledge.module_lookup(corrects=["XYZ"])


# --- Aktionen ---------------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path, catalog_system):
    return design.Session.create(catalog_system, tmp_path / "s", spec={"tube_focal_length": 164.5})


def apply_ok(session, candidate, *actions):
    r = response(*actions, candidate=candidate)
    validate(r)
    entry = session.apply(candidate, r, decider={"kind": "test"})
    for res in entry["results"]:
        assert res["status"] == "ok", res
    return entry["results"]


def test_read_actions_glass_query_and_module_lookup(session, brute):
    [res] = apply_ok(session, 0, {"action": "glass_query",
                                  "filter": {"rule": "front_medium_n_medium_vd"}, "limit": 3})
    out = res["result"]
    exp = [g["name"] for g in brute if SINGLE_RULES["front_medium_n_medium_vd"](g)]
    assert out["count"] == len(exp) and names(out) == exp[:3]
    assert not res.get("new_candidates")
    [res] = apply_ok(session, 0, {"action": "module_lookup", "group": "rear",
                                  "corrects": ["FC"]})
    ids = [m["id"] for m in res["result"]["modules"]]
    assert "rear.gauss_pair" in ids and "front.shell_ca" not in ids
    bad = session.apply(0, response({"action": "glass_query", "filter": {"rule": "merte_pair"}}),
                        decider={})
    assert bad["results"][0]["code"] == "missing_parameter"


def test_glass_filter_in_actions(session, brute):
    # insert_null_meniscus mit Filter: erstes Glas der Abfrage
    flt = {"rule": "low_n_low_dispersion", "nd": [1.45, None]}
    first = [g["name"] for g in brute if SINGLE_RULES["low_n_low_dispersion"](g)
             and g["nd"] >= 1.45][0]
    [res] = apply_ok(session, 0, {"action": "insert_null_meniscus", "after_surface": 5,
                                  "glass": {"filter": flt}})
    assert res["reports"][0]["glass"] == first
    # replace_glass mit Regelfilter: Kandidaten ohne das bisherige Glas, in Abfragereihenfolge
    [res] = apply_ok(session, 0, {"action": "replace_glass", "element": 2,
                                  "glass": {"filter": {"rule": "selectable_pgf",
                                                       "max_results": 2}}})
    exp = [g["name"] for g in brute if SINGLE_RULES["selectable_pgf"](g) and g["name"] != "N-BK7"]
    assert {r["glass"] for r in res["reports"]} == set(exp[:2])
    # leerer Filter: Fehler, nicht aufgeweicht
    bad = session.apply(0, response({"action": "replace_glass", "element": 2,
                                     "glass": {"filter": {"nd": [3.0, 4.0]}}}), decider={})
    assert bad["results"][0]["code"] == "precondition"


def test_incidence_angle_oracles():
    # achsparalleler Strahl: sin I = y c
    assert incidence_angle_deg(2.0, 0.0, 0.25) == pytest.approx(30.0, rel=1e-12)
    # Strahl durch den Krümmungsmittelpunkt: I = 0
    r = 10.0
    assert incidence_angle_deg(1.0, -1.0 / r, 1.0 / r) == pytest.approx(0.0, abs=1e-6)
    # Ebene: I = Neigungswinkel
    assert incidence_angle_deg(1.0, math.tan(math.radians(20)), 0.0) == pytest.approx(20.0)


def test_strong_doublet_template(session):
    # Im Spalt nach Fläche 5 erreicht der Randstrahl bei Halbkugelgrad 0,9 höchstens ≈ 39°
    [res] = apply_ok(session, 0, {"action": "insert_module", "module": "middle.strong_doublet",
                                  "after_surface": 5, "glasses": [{"name": "SYN-04"}],
                                  "incidence_deg": 36})
    rep = res["reports"][0]
    assert rep["template"] == "strong_doublet"
    assert rep["incidence_deg"] == pytest.approx(36.0, abs=1e-6)
    assert abs(rep["merit_after"] - rep["merit_before"]) <= \
        tol("kInsertionMeritRelative") * rep["merit_before"]
    # Partner nach merte_pair mit dem größten |Δν_d|: SYN-09 und SYN-11 kommen in Frage
    q = glass_query(session.system_path(0), {"rule": "merte_pair", "partner": "SYN-04"},
                    limit=None)
    assert names(q) == ["SYN-09", "SYN-11"]
    best = max(q["glasses"], key=lambda g: abs(g["vd"] - q["reference"]["vd"]))["name"]
    assert rep["glasses"] == ["SYN-04", best]
    data = json.loads(session.system_path(res["new_candidates"][0]).read_text(encoding="utf-8"))
    c = [data["surfaces"][k]["curvature"] for k in (6, 7, 8)]
    assert c[0] == c[1] == c[2] == rep["cement_curvature"]
    assert abs(c[0]) * 6.0 <= 0.9 + 1e-12
    # 50° ist hier nicht erreichbar: strukturierter Fehler
    bad = session.apply(0, response({"action": "insert_module", "module": "middle.strong_doublet",
                                     "after_surface": 5, "glasses": [{"name": "SYN-04"}],
                                     "incidence_deg": 50}), decider={})
    assert bad["results"][0]["code"] == "precondition"


def test_petzval_group_template(session):
    f0 = json.loads(mikroptim.run_cli(["trace", str(session.system_path(0)), "--json"])[1])
    [res] = apply_ok(session, 0, {"action": "insert_module", "module": "rear.petzval_group",
                                  "after_surface": 5, "glasses": [{"name": "SYN-14"}],
                                  "power": 0.3, "cycles": 0})
    rep = res["reports"][0]
    phi1, phi2 = rep["powers"]
    d = rep["positions"][1] - rep["positions"][0]
    assert phi1 == pytest.approx(0.3 / abs(f0["paraxial"]["focal_length"]))
    assert phi1 + phi2 - d * phi1 * phi2 == pytest.approx(0.0, abs=1e-15)  # afokal (dünn)
    data = json.loads(session.system_path(res["new_candidates"][0]).read_text(encoding="utf-8"))
    ab = mikroptim.aberrations(session.system_path(res["new_candidates"][0]))["surfaces"]
    t = rep["thickness"]
    for front, phi in ((6, phi1), (8, phi2)):
        n = ab[front]["n_after"]
        c1, c2 = data["surfaces"][front]["curvature"], data["surfaces"][front + 1]["curvature"]
        thick = (n - 1) * (c1 - c2) + t * (n - 1) ** 2 * c1 * c2 / n  # dicke Linse
        assert thick == pytest.approx(phi, rel=1e-12)
        assert data["surfaces"][front]["thickness"] == t
    assert rep["failed_rays"] == 0
    assert len(data["surfaces"]) == 8 + 4


def test_skeleton_double_gauss_with_new_modules(session):
    [res] = apply_ok(session, 0, {
        "action": "start_from_skeleton", "power_distribution": "double_gauss",
        "front": ["generic.lens", "generic.lens"], "middle": ["middle.strong_doublet"],
        "rear": ["rear.petzval_group"], "start_aperture": 0.2,
        "glasses": [{"name": "N-BK7"}, {"name": "SYN-14"}]})
    rep = res["reports"][0]
    assert rep["power_distribution"] == "double_gauss" and rep["elements"] == 5
    target = json.loads(session.system_path(res["new_candidates"][0])
                        .read_text(encoding="utf-8"))["target"]["focal_length"]
    assert rep["focal_length"] == pytest.approx(target, rel=1e-9)
    # fünf dicke Glieder erreichen die Brennweite erst mit halbierten Dicken und Abständen
    assert rep["length_scale"] < 1.0


def test_prompt_and_tools_include_knowledge():
    from mikroptim.design import claude, prompt
    assert "front.shell_ca" in prompt.SYSTEM_PROMPT
    assert "module_lookup" in prompt.SYSTEM_PROMPT and "glass_query" in prompt.SYSTEM_PROMPT
    assert {"glass_query", "module_lookup"} <= {t["function"]["name"]
                                                for t in schema.tool_specs()}
    tools = {t["name"]: t for t in claude.tool_definitions()}
    assert {"glass_query", "module_lookup"} <= set(tools)
    filt = tools["glass_query"]["input_schema"]["properties"]["filter"]
    assert filt["additionalProperties"] is False
