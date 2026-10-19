// Strahlsuche (M6-T4) mit dem optischen Modell: Bausteine (Varianten aus der Diagnose, Abstand,
// gemeinsame LM-Aufrufe pro Topologie, gestörte Starts, FP32-Vorfilter), Kurzläufe (B = 1 gleich
// dem einzweigigen Controller, Gruppierung ohne Einfluss, Determinismus mit Startwert) und der
// längere Lauf NA 0,3 → 0,6 (docs/optimierung.md, Abschnitt 8).

#include "beam_test_tools.hpp"
#include "homotopy_test_systems.hpp"

#include "mikroptim/io/system_json.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace mko::test {
namespace {

// Startsystem mit aktiven Geisterplatten: hinter der Linse beide Gläser (Platten 2 und 3),
// davor das hochbrechende (Platte 1).
SystemDescription active_start(const OpticalHomotopyModel& m) {
    SystemDescription s = m.start_system();
    s.ghost_plates[3].coefficients[1] = 2e-3;
    s.ghost_plates[2].coefficients[1] = 1e-3;
    s.ghost_plates[1].coefficients[1] = 1e-3;
    return s;
}

// Varianten des Einfügens: die Empfehlung zuerst, dann die übrige Lage, dann das andere Glas der
// Lage mit zwei aktiven Platten; jede Variante lässt sich anwenden.
TEST(OpticalBeam, InsertionOptionsFromDiagnosis) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o;
    o.saddle_delta = 0.0;
    OpticalHomotopyModel m(start, o);
    const SystemDescription s = active_start(m);
    const HomotopyPoint p = m.start_point();
    const HomotopyNeed need = m.diagnose(s, p);
    ASSERT_EQ(need.recommendation, GhostOperator::A) << need.text;
    EXPECT_EQ(m.insertion_options(s, p, need, 1).size(), 1U);
    const std::vector<HomotopyNeed> opts = m.insertion_options(s, p, need, 5);
    ASSERT_EQ(opts.size(), 3U);
    const int rec = need.diagnosis.recommended_position;
    EXPECT_EQ(opts[0].diagnosis.recommended_position, rec);
    EXPECT_EQ(opts[0].text, need.text);
    EXPECT_NE(opts[1].diagnosis.recommended_position, rec);
    for (std::size_t i = 0; i < need.diagnosis.positions.size(); ++i) {
        EXPECT_EQ(opts[1].diagnosis.positions[i].active_plates,
                  need.diagnosis.positions[i].active_plates);
    }
    // Glasvariante: die Lage hinter der Linse (Platten 2 und 3), nur die schwächere Platte 2
    const int behind = opts[2].diagnosis.recommended_position;
    const GhostPositionReport& pr = opts[2].diagnosis.positions[static_cast<std::size_t>(behind)];
    EXPECT_EQ(pr.after_surface, 2);
    EXPECT_EQ(pr.active_plates, std::vector<int>{2});
    std::vector<int> surfaces;
    std::vector<int> plates;
    for (const HomotopyNeed& n : opts) {
        EXPECT_EQ(n.recommendation, GhostOperator::A);
        const HomotopyInsertion ins = m.insert(s, p, n);
        ASSERT_EQ(ins.branches.size(), 1U);
        surfaces.push_back(ins.first_surface);
        plates.push_back(ins.plate);
    }
    std::vector<int> sorted = plates;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(plates[2], 2);
    EXPECT_EQ(surfaces[2], 3); // Meniskus hinter Fläche 2
}

// Gemeinsamer LM-Aufruf: zwei Startpunkte am Sattelpunkt (gleiche Topologie) bilden eine Gruppe,
// das Startsystem eine eigene; die Ergebnisse sind bitgleich zu einzelnen Aufrufen.
TEST(OpticalBeam, SolveAllGroupsByTopologyBitwise) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o;
    o.saddle_delta = 0.01;
    o.lm.max_iterations = 4;
    OpticalHomotopyModel m(start, o);
    const SystemDescription s = active_start(m);
    const HomotopyPoint p = m.start_point();
    const HomotopyInsertion ins = m.insert(s, p, m.diagnose(s, p));
    ASSERT_EQ(ins.branches.size(), 2U);
    SystemDescription a = ins.branches[0];
    SystemDescription b = s;
    SystemDescription c = ins.branches[1];
    m.set_stage(a, ins, 0.1);
    m.set_stage(c, ins, 0.1);
    SystemDescription a1 = a;
    SystemDescription b1 = b;
    SystemDescription c1 = c;
    std::vector<SystemDescription*> sys{&a, &b, &c};
    std::vector<int> groups;
    const std::vector<HomotopySolve> all = m.solve_all(sys, p, &groups);
    EXPECT_EQ(groups, (std::vector<int>{0, 1, 0}));
    const HomotopySolve sa = m.solve(a1, p);
    const HomotopySolve sb = m.solve(b1, p);
    const HomotopySolve sc = m.solve(c1, p);
    EXPECT_TRUE(a == a1);
    EXPECT_TRUE(b == b1);
    EXPECT_TRUE(c == c1);
    const HomotopySolve* single[3] = {&sa, &sb, &sc};
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(all[i].merit, single[i]->merit) << i;
        EXPECT_EQ(all[i].start_merit, single[i]->start_merit) << i;
        EXPECT_EQ(all[i].iterations, single[i]->iterations) << i;
        EXPECT_EQ(all[i].rms, single[i]->rms) << i;
        EXPECT_EQ(all[i].wavefront, single[i]->wavefront) << i;
        EXPECT_EQ(all[i].ghost_penalty, single[i]->ghost_penalty) << i;
        EXPECT_EQ(all[i].status, single[i]->status) << i;
    }
    // andere Stufe (Dicke außerhalb der Variablen): eigene Gruppe
    SystemDescription d = ins.branches[1];
    m.set_stage(d, ins, 0.2);
    SystemDescription a2 = ins.branches[0];
    m.set_stage(a2, ins, 0.1);
    std::vector<SystemDescription*> sys2{&a2, &d};
    (void)m.solve_all(sys2, p, &groups);
    EXPECT_EQ(groups, (std::vector<int>{0, 1}));
    // ohne Bündelung: jede eine eigene Gruppe
    o.group_runs = false;
    OpticalHomotopyModel m2(start, o);
    SystemDescription a3 = ins.branches[0];
    SystemDescription c3 = ins.branches[1];
    std::vector<SystemDescription*> sys3{&a3, &c3};
    (void)m2.solve_all(sys3, p, &groups);
    EXPECT_EQ(groups, (std::vector<int>{0, 1}));
}

// Abstand: 0 für denselben Zustand, endlich für die Zweige am Sattelpunkt, ∞ für eine andere
// Topologie.
TEST(OpticalBeam, Distance) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o;
    o.saddle_delta = 0.01;
    OpticalHomotopyModel m(start, o);
    const SystemDescription s = active_start(m);
    const HomotopyPoint p = m.start_point();
    const HomotopyInsertion ins = m.insert(s, p, m.diagnose(s, p));
    const SystemDescription& a = ins.branches[0];
    const SystemDescription& b = ins.branches[1];
    EXPECT_EQ(m.distance(a, a, p), 0.0);
    const double d = m.distance(a, b, p);
    EXPECT_TRUE(std::isfinite(d));
    EXPECT_GT(d, 0.0);
    // größte relative Abweichung: die Meniskuskrümmungen (c₀ ± 0,01)
    const double c0 = 0.5 * (a.surfaces[3].curvature + a.surfaces[4].curvature);
    EXPECT_LE(d, 0.02 / std::abs(c0) * (1.0 + 1e-12));
    EXPECT_EQ(m.distance(a, s, p), std::numeric_limits<double>::infinity());
}

// Gestörte Starts: nur die Krümmungen der neuen Flächen, innerhalb ±s, reproduzierbar mit dem
// Startwert, anders mit einem anderen.
TEST(OpticalBeam, PerturbedStarts) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o;
    o.saddle_delta = 0.0;
    o.start_perturbation = 0.02;
    OpticalHomotopyModel m(start, o);
    const SystemDescription s = active_start(m);
    const HomotopyPoint p = m.start_point();
    const HomotopyInsertion ins = m.insert(s, p, m.diagnose(s, p));
    const SystemDescription& b = ins.branches[0];
    const std::vector<SystemDescription> x = m.perturbed_starts(b, ins, p, 3, 99);
    const std::vector<SystemDescription> y = m.perturbed_starts(b, ins, p, 3, 99);
    const std::vector<SystemDescription> z = m.perturbed_starts(b, ins, p, 3, 100);
    ASSERT_EQ(x.size(), 3U);
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_TRUE(x[j] == y[j]);
        EXPECT_FALSE(x[j] == z[j]);
        SystemDescription back = x[j];
        for (int f = ins.first_surface; f < ins.first_surface + ins.new_surfaces; ++f) {
            const auto i = static_cast<std::size_t>(f);
            EXPECT_LE(std::abs(x[j].surfaces[i].curvature - b.surfaces[i].curvature), 0.02);
            EXPECT_NE(x[j].surfaces[i].curvature, b.surfaces[i].curvature);
            back.surfaces[i].curvature = b.surfaces[i].curvature;
        }
        EXPECT_TRUE(back == b); // sonst nichts geändert
    }
    EXPECT_TRUE(m.perturbed_starts(b, ins, p, 0, 99).empty());
}

// FP32-Vorfilter: Die Startpunkte am Sattelpunkt bleiben, ein Start mit einer stark brechenden
// Linse statt des Meniskus (Merit um Größenordnungen schlechter) fällt heraus.
TEST(OpticalBeam, PrefilterRejectsHopelessStart) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o;
    o.saddle_delta = 0.002;
    OpticalHomotopyModel m(start, o);
    const SystemDescription s = active_start(m);
    const HomotopyPoint p = m.start_point();
    const HomotopyInsertion ins = m.insert(s, p, m.diagnose(s, p));
    SystemDescription bad = ins.branches[0];
    bad.surfaces[3].curvature = 0.8;
    bad.surfaces[4].curvature = -0.8;
    std::vector<const SystemDescription*> starts{&ins.branches[0], &ins.branches[1], &bad};
    const std::vector<bool> keep = m.prefilter(starts, p);
    EXPECT_EQ(keep, (std::vector<bool>{true, true, false}));
}

// Kurzlauf (CI) mit B = 2: alle Phasen, ein Einfügen mit zwei Zweigen im selben LM-Aufruf,
// Stammbaum, Abbruch bei der höchsten Elementzahl.
TEST(BeamSmoke, ShortRunCpu) {
    BeamSmokeSettings st;
    st.width = 2;
    const BeamResult r = beam_smoke_run(st);
    write_beam_step_header(std::cout);
    for (const BeamStep& s : r.steps) {
        write_beam_step_line(s, std::cout);
    }
    write_beam_summary(r, std::cout);
    write_beam_tree(r, std::cout);
    EXPECT_EQ(r.status, HomotopyStatus::MaxElements) << r.message;
    EXPECT_EQ(r.point.aperture, kHomotopySmokeAperture);
    EXPECT_EQ(r.insertions, 1);
    EXPECT_EQ(r.elements, 2);
    ASSERT_GE(r.nodes.size(), 3U);
    EXPECT_EQ(r.nodes[0].status, BeamNodeStatus::Expanded);
    EXPECT_EQ(r.nodes[0].children, (std::vector<int>{1, 2}));
    // Stufen der inneren Homotopie: beide Zweige im selben Aufruf
    int stage_steps = 0;
    for (const BeamStep& s : r.steps) {
        if (s.step.kind != HomotopyStepKind::InsertionStage) {
            continue;
        }
        ++stage_steps;
        for (const BeamStep& t : r.steps) {
            if (t.step.kind == HomotopyStepKind::InsertionStage && t.step.stage == s.step.stage) {
                EXPECT_EQ(t.group, s.group);
            }
        }
    }
    EXPECT_EQ(stage_steps, 4);
    EXPECT_LT(r.groups, r.solves);
    for (const BeamStep& s : r.steps) {
        if (s.step.solved) {
            EXPECT_TRUE(std::isfinite(s.step.solve.merit));
        }
    }
}

// B = 1: dieselben Entscheidungen und dasselbe Endsystem (bitgleich) wie run_homotopy.
TEST(BeamSmoke, SingleBeamMatchesHomotopyCpu) {
    BeamSmokeSettings st;
    st.width = 1;
    const BeamResult b = beam_smoke_run(st);
    const HomotopyResult h = homotopy_smoke_run();
    expect_beam_same_as_homotopy(h, b);
}

// Gemeinsame LM-Aufrufe ändern kein Ergebnis: Protokoll (ohne Gruppen und Laufzeiten),
// Stammbaum und Endsystem bitgleich zu Einzelaufrufen.
TEST(BeamSmoke, GroupingDoesNotChangeResultsCpu) {
    BeamSmokeSettings st;
    st.width = 2;
    const BeamResult grouped = beam_smoke_run(st);
    st.group = false;
    const BeamResult single = beam_smoke_run(st);
    EXPECT_EQ(beam_json_without_groups(grouped), beam_json_without_groups(single));
    EXPECT_TRUE(grouped.system == single.system);
    EXPECT_LT(grouped.groups, single.groups);
    EXPECT_EQ(single.groups, single.solves);
}

// Fester Startwert: zwei Läufe mit gestörten Starts und Vorfilter bitgleich (Protokoll ohne
// Laufzeiten, Stammbaum, Endsystem).
TEST(BeamSmoke, DeterministicWithSeedCpu) {
    BeamSmokeSettings st;
    st.width = 2;
    st.starts = 1;
    st.prefilter = true;
    st.seed = 2024;
    const BeamResult a = beam_smoke_run(st);
    const BeamResult b = beam_smoke_run(st);
    EXPECT_EQ(beam_json_without_seconds(a), beam_json_without_seconds(b));
    EXPECT_TRUE(a.system == b.system);
    // je Zweig ein gestörter Start: vier Kinder des Einfügens
    ASSERT_GE(a.nodes.size(), 5U);
    EXPECT_EQ(a.nodes[0].children.size(), 4U);
}

// Längerer Lauf (lokal mit --gtest_also_run_disabled_tests, Release ~1–2 min): NA 0,3 → 0,6 mit
// B = 2 und zwei Varianten pro Einfügen; die Strahlsuche ist mindestens so gut wie der
// einzweigige Controller (RMS des realen Systems am Ziel).
TEST(BeamSearchOptimization, DISABLED_ApertureToNa06BeamCpu) {
    const HomotopyRun single = homotopy_acceptance_run({}, false);
    const BeamResult r = beam_acceptance_run(2, 2, {}, true);
    write_beam_summary(r, std::cout);
    write_beam_tree(r, std::cout);
    std::cout << "einzweigig: " << to_string(single.result.status) << ", RMS "
              << single.result.goal.max_rms << " Wellen, " << single.result.elements
              << " Elemente, " << single.result.solves << " Optimierungen, "
              << single.result.seconds << " s\n";
    std::cout << "Strahlsuche: " << to_string(r.status) << ", RMS " << r.goal.max_rms << " Wellen, "
              << r.elements << " Elemente, " << r.solves << " Optimierungen in " << r.groups
              << " Aufrufen, " << r.seconds << " s\n";
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.point.aperture, kHomotopyTargetAperture);
    EXPECT_TRUE(r.goal.reached);
    EXPECT_LE(r.goal.max_rms, 1.0 / 14.0);
    // mindestens so gut: weniger Elemente, oder gleich viele und RMS nicht größer
    EXPECT_LE(r.elements, single.result.elements);
    if (r.elements == single.result.elements) {
        EXPECT_LE(r.goal.max_rms, single.result.goal.max_rms);
    }
}

} // namespace
} // namespace mko::test
