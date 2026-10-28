"""Farb- und Feldklassen nach Zhang (M10-T2): reine Funktionen mit konstruierten Daten und der
Zustand eines Testsystems."""

import pytest

from mikroptim import design
from mikroptim.design import classes as cl

from conftest import SYSTEMS

LINES = ("i", "h", "g", "F", "e", "d", "C", "A'", "s", "t")


def shifts(**override):
    base = {k: 0.0 for k in LINES}
    base.update(override)
    return base


def test_depth_of_focus_matches_zhang_appendix_iv():
    # NA 0,8, d-Linie: 0,918 µm trocken, 1,392 µm Öl (n = 1,515)
    assert cl.depth_of_focus(0.8) * 1e3 == pytest.approx(0.918, abs=1e-3)
    assert cl.depth_of_focus(0.8, 1.515) * 1e3 == pytest.approx(1.392, abs=2e-3)


@pytest.mark.parametrize("data,expected", [
    (shifts(), "g"),                                   # alles in einem Punkt
    (shifts(i=3.0), "e"),                              # i fällt heraus: IR-Superapo
    (shifts(t=3.0), "f"),                              # t fällt heraus: UV-Superapo
    (shifts(i=3.0, t=3.0), "d"),                       # g … A′
    (shifts(i=3.0, t=3.0, s=3.0, **{"A'": 3.0}), "c"),  # g … C
    (shifts(i=3.0, t=3.0, **{"A'": 3.0}, g=2.0, h=3.0), "b"),  # g … C in 2,5 DoF
])
def test_color_class_order(data, expected):
    assert cl.color_class(data, dof=1.0)["color"] == expected


def test_color_class_fluorite_achromat_none():
    far = {k: 10.0 for k in ("i", "h", "A'", "s", "t")}
    # g … C in 2,5 DoF, aber nicht in 1 DoF → b
    assert cl.color_class(shifts(**far, g=2.0, F=0.0, d=0.5, C=0.5), 1.0)["color"] == "b"
    # F–C ≤ 2 DoF, sekundäres Spektrum ≤ 1,5 DoF, g weit weg → a
    r = cl.color_class(shifts(**far, g=10.0, F=0.0, d=-0.4, C=1.8), 1.0)
    assert r["color"] == "a" and "b" not in r["satisfied"]
    assert cl.color_class(shifts(**far, g=10.0, F=0.0, d=0.0, C=5.0), 1.0)["color"] == "none"
    # fehlende Linie (Glas deckt sie nicht ab): Klassen mit dieser Linie unbestimmbar
    r = cl.color_class(shifts(i=float("nan")), 1.0)
    assert r["color"] == "e" and "g" not in r["satisfied"]


@pytest.mark.parametrize("kw,expected", [
    (dict(field_shift=3.0), 1),
    (dict(na=0.3), 2),
    (dict(edge_rms=0.2), 3),
    (dict(edge_failed=5), 4),
    (dict(), 5),
    (dict(etendue=1.2, edge_failed=2), 6),
    (dict(etendue=1.2), 7),
])
def test_field_class(kw, expected):
    args = dict(field_shift=0.5, dof=1.0, na=0.9, etendue=0.2, edge_rms=0.05, edge_failed=0)
    args.update(kw)
    assert cl.field_class(**args)["field"] == expected


def test_state_contains_classes(tmp_path):
    s = design.Session.create(SYSTEMS / "testsystem_mehrlinser.json", tmp_path / "k")
    c = s.state(0)["classes"]
    assert c["color"] in ("a", "b", "c", "d", "e", "f", "g", "none")
    assert 1 <= c["field"] <= 7
    assert c["dof_mm"] == pytest.approx(
        cl.depth_of_focus(s.state(0)["spec"]["na"]), rel=1e-12)
