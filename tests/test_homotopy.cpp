// Homotopie-Controller (M6-T3): Steuerlogik mit einem synthetischen Modell (Schrittweite,
// Rücknahme, Einfügen bei Δ < Δ_min und Bedarf, Fortsetzungen, Abbruch, Protokoll) und die
// Bausteine des optischen Modells ohne lange Optimierung (docs/optimierung.md, Abschnitt 7).

#include "homotopy_mock.hpp"
#include "insertion_test_systems.hpp"
#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/optim/design_operators.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/synth/homotopy.hpp"
#include "mikroptim/synth/optical_homotopy.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

// Synthetisches Modell: tests/homotopy_mock.hpp (MockModel).

std::vector<const HomotopyStep*> steps_of(const HomotopyResult& r, HomotopyStepKind kind) {
    std::vector<const HomotopyStep*> out;
    for (const HomotopyStep& s : r.steps) {
        if (s.kind == kind) {
            out.push_back(&s);
        }
    }
    return out;
}

// Schnelle Schritte vergrößern Δ (Faktor 2), langsame behalten es; der letzte Schritt wird auf
// das Ziel gekürzt.
TEST(HomotopyControl, StepGrowsAfterFastConvergence) {
    MockModel m({});
    const HomotopyResult r =
        run_homotopy(m, MockModel::start(0.3), mock_point(0.3), mock_options(0.6));
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    const auto a = steps_of(r, HomotopyStepKind::Aperture);
    ASSERT_EQ(a.size(), 5U);
    const std::vector<double> deltas{0.02, 0.04, 0.08, 0.08, 0.08};
    const std::vector<HomotopyAction> actions{HomotopyAction::Grow, HomotopyAction::Grow,
                                              HomotopyAction::Keep, HomotopyAction::Keep,
                                              HomotopyAction::Keep};
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_DOUBLE_EQ(a[i]->delta, deltas[i]) << i;
        EXPECT_EQ(a[i]->action, actions[i]) << i;
        EXPECT_TRUE(a[i]->accepted);
    }
    EXPECT_EQ(a.back()->point.aperture, 0.6);
    EXPECT_EQ(r.point.aperture, 0.6);
    EXPECT_EQ(r.elements, 1);
    EXPECT_EQ(r.insertions, 0);
}

// Lehnt das Modell ein Einfügen ab (std::invalid_argument), endet der Lauf geordnet mit dem
// Status ohne Bedarf und dem Grund, statt abzustürzen (M6-T6: p4_r3, p3_r0).
TEST(HomotopyControl, RefusedInsertionStopsCleanly) {
    MockSpec spec;
    spec.capacity0 = 0.32;
    spec.insert_throws = true;
    MockModel m(spec);
    const HomotopyResult r =
        run_homotopy(m, MockModel::start(0.3), mock_point(0.3), mock_options(0.6));
    EXPECT_EQ(r.status, HomotopyStatus::StepTooSmall);
    EXPECT_NE(r.message.find("nicht möglich: Linse zu dünn"), std::string::npos) << r.message;
}

// Stetige Spektrumbreite (M11-T5f): eigene Phase mit Schrittweitensteuerung von 0 bis 1.
TEST(HomotopyControl, SpectrumWidthPhase) {
    MockModel m{MockSpec{}};
    HomotopyOptions o = mock_options(0.3);
    o.target_spectrum_width = 1.0;
    HomotopyPoint start = mock_point(0.3);
    start.spectrum_width = 0.0;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), start, o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.point.spectrum_width, 1.0);
    const auto w = steps_of(r, HomotopyStepKind::SpectrumWidth);
    ASSERT_FALSE(w.empty());
    EXPECT_DOUBLE_EQ(w.front()->point.spectrum_width, o.spectrum_step.initial);
    for (std::size_t i = 1; i < w.size(); ++i) {
        EXPECT_GT(w[i]->point.spectrum_width, w[i - 1]->point.spectrum_width);
    }
    HomotopyOptions bad = mock_options(0.3);
    bad.target_spectrum_width = 1.5;
    EXPECT_THROW((void)homotopy_target(start, bad), std::invalid_argument);
}

// Prädiktor (M11-T3): ab dem zweiten Aperturschritt Start aus der Sekante, Verhältnis der
// Schrittweiten (0,04/0,02, 0,08/0,04, 0,08/0,08, 0,08/0,08); ohne Option kein Aufruf.
TEST(HomotopyControl, PredictorUsesSecantRatios) {
    MockSpec spec;
    spec.predict = true;
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.6);
    o.predictor = true;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    ASSERT_EQ(m.predict_ratios.size(), 4U);
    const std::vector<double> ratios{2.0, 2.0, 1.0, 1.0};
    for (std::size_t i = 0; i < ratios.size(); ++i) {
        EXPECT_NEAR(m.predict_ratios[i], ratios[i], 1e-12) << i;
    }
    const auto a = steps_of(r, HomotopyStepKind::Aperture);
    ASSERT_EQ(a.size(), 5U);
    EXPECT_TRUE(a[0]->operation.empty());
    for (std::size_t i = 1; i < a.size(); ++i) {
        EXPECT_EQ(a[i]->operation, "Prädiktor (Sekante)") << i;
    }
    MockModel off(spec);
    (void)run_homotopy(off, MockModel::start(0.3), mock_point(0.3), mock_options(0.6));
    EXPECT_TRUE(off.predict_ratios.empty());
}

// Abgelehnter Schritt: Δ halbiert, der nächste Versuch startet am letzten guten Zustand (nicht
// am abgelehnten Ergebnis).
TEST(HomotopyControl, RejectedStepHalvesAndRestartsFromGoodState) {
    MockSpec spec;
    spec.max_jump = 0.03; // Schritte über 0,03 scheitern
    MockModel m(spec);
    const HomotopyResult r =
        run_homotopy(m, MockModel::start(0.3), mock_point(0.3), mock_options(0.4));
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    const auto a = steps_of(r, HomotopyStepKind::Aperture);
    ASSERT_GE(a.size(), 3U);
    EXPECT_EQ(a[0]->action, HomotopyAction::Grow); // 0,30 → 0,32
    EXPECT_FALSE(a[1]->accepted);                  // 0,32 → 0,36 zu weit
    EXPECT_EQ(a[1]->action, HomotopyAction::Shrink);
    EXPECT_DOUBLE_EQ(a[1]->next_delta, 0.02);
    EXPECT_TRUE(a[2]->accepted);
    EXPECT_NEAR(a[2]->point.aperture, 0.34, 1e-15);
    // Aufruf des dritten Schritts: Startwert = guter Zustand 0,32, nicht der Versuch 0,36
    std::vector<MockCall> aperture_calls;
    for (const MockCall& c : m.calls) {
        if (c.point.penalty_scale == 1.0 && c.point.aperture > 0.3) {
            aperture_calls.push_back(c);
        }
    }
    ASSERT_GE(aperture_calls.size(), 3U);
    EXPECT_NEAR(aperture_calls[1].point.aperture, 0.36, 1e-15);
    EXPECT_NEAR(aperture_calls[2].from, 0.32, 1e-15);
    EXPECT_EQ(r.point.aperture, 0.4);
}

// Δ < Δ_min mit Bedarf: Operator, innere Homotopie in zwei Zweigen (der bessere wird
// übernommen), Penalty-Fortsetzung, dann weiter mit Δ₀.
TEST(HomotopyControl, InsertionWhenStepBelowMinimumWithNeed) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.15; // 1 Element: 0,4; 2: 0,55; 3: 0,7
    spec.one_sided = true;
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.6);
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.insertions, 2);
    EXPECT_EQ(r.elements, 3);
    EXPECT_EQ(r.start_elements, 1);
    EXPECT_EQ(m.finishes, 2);
    int inserts = 0;
    for (std::size_t i = 0; i < r.steps.size(); ++i) {
        const HomotopyStep& s = r.steps[i];
        if (s.action != HomotopyAction::Insert) {
            continue;
        }
        ++inserts;
        EXPECT_FALSE(s.accepted);
        EXPECT_LT(s.next_delta, o.aperture_step.min);
        EXPECT_EQ(s.recommendation, GhostOperator::A);
        // danach: Operator, 2 Zweige × 2 Stufen, 3 Penalty-Stufen
        ASSERT_LT(i + 8, r.steps.size());
        EXPECT_EQ(r.steps[i + 1].kind, HomotopyStepKind::Insertion);
        for (int k = 0; k < 4; ++k) {
            const HomotopyStep& st = r.steps[i + 2 + static_cast<std::size_t>(k)];
            EXPECT_EQ(st.kind, HomotopyStepKind::InsertionStage);
            EXPECT_EQ(st.branch, k / 2);
            EXPECT_EQ(st.stage, k % 2 == 0 ? 0.25 : 0.5);
            // innere Homotopie am letzten guten Punkt
            EXPECT_EQ(st.point.aperture, r.steps[i + 1].point.aperture);
        }
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(r.steps[i + 6 + static_cast<std::size_t>(k)].kind, HomotopyStepKind::Penalty);
        }
        EXPECT_NE(r.steps[i + 6].operation.find("Zweig 1 von 2"), std::string::npos);
        // erster Aperturschritt danach mit Δ₀
        const HomotopyStep& next = r.steps[i + 9];
        EXPECT_EQ(next.kind, HomotopyStepKind::Aperture);
        EXPECT_DOUBLE_EQ(next.delta, o.aperture_step.initial);
    }
    EXPECT_EQ(inserts, 2);
    // Zweig 1 (besser) wurde übernommen
    EXPECT_EQ(r.system.surfaces.back().curvature, 1.0);
    EXPECT_EQ(r.point.aperture, 0.6);
}

// Δ < Δ_min ohne Bedarf: Abbruch mit Bericht am letzten guten Punkt.
TEST(HomotopyControl, StopsWithoutNeed) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.need = false;
    MockModel m(spec);
    const HomotopyResult r =
        run_homotopy(m, MockModel::start(0.3), mock_point(0.3), mock_options(0.6));
    EXPECT_EQ(r.status, HomotopyStatus::StepTooSmall);
    EXPECT_NE(r.message.find("kein Bedarf"), std::string::npos) << r.message;
    EXPECT_LE(r.point.aperture, 0.4);
    EXPECT_GT(r.point.aperture, 0.4 - 0.004);
    EXPECT_EQ(r.steps.back().action, HomotopyAction::Stop);
    EXPECT_EQ(r.insertions, 0);
}

// Bedarf, aber höchste Elementzahl erreicht.
TEST(HomotopyControl, StopsAtMaxElements) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.1;
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.6);
    o.max_elements = 2;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::MaxElements) << r.message;
    EXPECT_EQ(r.elements, 2);
    EXPECT_EQ(r.insertions, 1);
    EXPECT_LE(r.point.aperture, 0.5);
}

// Penalty-Fortsetzung am Start, ε-Fortsetzung am Ende, μ-Fortsetzung bis zum Ziel.
TEST(HomotopyControl, PenaltySmoothingAndGhostWeightContinuation) {
    MockSpec spec;
    spec.one_sided = true;
    spec.goal_ghost_weight = 100.0; // Ziel erst mit μ × 100
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.32);
    o.smoothing_stages = 2;
    o.smoothing_factor = 0.1;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    ASSERT_GE(m.calls.size(), 3U);
    EXPECT_DOUBLE_EQ(m.calls[0].point.penalty_scale, 0.01);
    EXPECT_DOUBLE_EQ(m.calls[1].point.penalty_scale, 0.1);
    EXPECT_DOUBLE_EQ(m.calls[2].point.penalty_scale, 1.0);
    std::vector<HomotopyStepKind> kinds;
    for (const HomotopyStep& s : r.steps) {
        kinds.push_back(s.kind);
    }
    const std::vector<HomotopyStepKind> expected{
        HomotopyStepKind::Penalty,     HomotopyStepKind::Penalty,     HomotopyStepKind::Penalty,
        HomotopyStepKind::Aperture,    HomotopyStepKind::Smoothing,   HomotopyStepKind::Smoothing,
        HomotopyStepKind::Goal,        HomotopyStepKind::GhostWeight, HomotopyStepKind::Goal,
        HomotopyStepKind::GhostWeight, HomotopyStepKind::Goal};
    EXPECT_EQ(kinds, expected);
    EXPECT_NEAR(r.point.ghost_smoothing, 0.01, 1e-17);
    EXPECT_DOUBLE_EQ(r.point.ghost_weight, 100.0);
    EXPECT_DOUBLE_EQ(r.point.penalty_scale, 1.0);
    EXPECT_FALSE(r.steps[6].goal.reached);
    EXPECT_TRUE(r.steps.back().goal.reached);
    EXPECT_TRUE(r.goal.reached);
}

// Augmented Lagrangian (M11-T2): statt der Penalty-Fortsetzung Läufe mit festem Faktor, bis
// max √w|Δθ| ≤ Toleranz; Fortschreibung nach jeder Optimierung (auch den stetigen Schritten),
// Protokoll mit shift_change. Ohne die Option keine Fortschreibung.
TEST(HomotopyControl, AugmentedLagrangianReplacesPenaltyContinuation) {
    MockSpec spec;
    spec.one_sided = true;
    spec.shift_changes = {0.5, 1e-3, 1e-8, 0.25};
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.32);
    o.augmented_lagrangian.rounds = 5;
    o.augmented_lagrangian.tolerance = 1e-6;
    o.al_penalty_scale = 0.1;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    const std::vector<const HomotopyStep*> penalty = steps_of(r, HomotopyStepKind::Penalty);
    ASSERT_EQ(penalty.size(), 3U); // dritter Lauf: 1e-8 ≤ 1e-6
    EXPECT_DOUBLE_EQ(penalty[0]->solve.shift_change, 0.5);
    EXPECT_DOUBLE_EQ(penalty[2]->solve.shift_change, 1e-8);
    EXPECT_NE(penalty[0]->operation.find("Augmented Lagrangian, Lauf 1"), std::string::npos);
    for (const MockCall& c : m.calls) {
        EXPECT_DOUBLE_EQ(c.point.penalty_scale, 0.1); // festes Gewicht im ganzen Lauf
    }
    // eine Fortschreibung pro Optimierung
    EXPECT_EQ(m.shift_points.size(), m.calls.size());
    const std::vector<const HomotopyStep*> aperture = steps_of(r, HomotopyStepKind::Aperture);
    ASSERT_FALSE(aperture.empty());
    EXPECT_DOUBLE_EQ(aperture[0]->solve.shift_change, 0.25);
    const nlohmann::json j = nlohmann::json::parse(homotopy_result_json(r));
    EXPECT_DOUBLE_EQ(j["steps"][0]["lm"]["shift_change"].get<double>(), 0.5);

    // Standard: keine Fortschreibung, kein Feld im Protokoll
    MockModel plain(spec);
    const HomotopyResult rp =
        run_homotopy(plain, MockModel::start(0.3), mock_point(0.3), mock_options(0.32));
    EXPECT_TRUE(plain.shift_points.empty());
    EXPECT_TRUE(std::isnan(rp.steps[0].solve.shift_change));
    const nlohmann::json jp = nlohmann::json::parse(homotopy_result_json(rp));
    EXPECT_FALSE(jp["steps"][0]["lm"].contains("shift_change"));

    HomotopyOptions bad = mock_options(0.32);
    bad.augmented_lagrangian.rounds = -1;
    EXPECT_THROW(check_homotopy_options(bad), std::invalid_argument);
    bad = mock_options(0.32);
    bad.al_penalty_scale = 0.0;
    EXPECT_THROW(check_homotopy_options(bad), std::invalid_argument);
}

// β-Fortsetzung der Strahl-Extrema (M11-T1b): Start mit β, am Ende β · 10, β · 100, dann hart,
// vor der ε-Fortsetzung; ohne extrema_beta keine solchen Schritte.
TEST(HomotopyControl, ExtremaSmoothingContinuation) {
    MockModel m{MockSpec{}};
    HomotopyOptions o = mock_options(0.32);
    o.extrema_beta = 300.0;
    o.extrema_stages = 2;
    o.extrema_factor = 10.0;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_DOUBLE_EQ(m.calls.front().point.extrema_beta, 300.0);
    const std::vector<const HomotopyStep*> e = steps_of(r, HomotopyStepKind::ExtremaSmoothing);
    ASSERT_EQ(e.size(), 3U);
    EXPECT_DOUBLE_EQ(e[0]->point.extrema_beta, 3000.0);
    EXPECT_DOUBLE_EQ(e[1]->point.extrema_beta, 30000.0);
    EXPECT_EQ(e[2]->point.extrema_beta, 0.0);
    EXPECT_EQ(r.point.extrema_beta, 0.0);
    const auto first_smoothing =
        std::find_if(r.steps.begin(), r.steps.end(),
                     [](const HomotopyStep& s) { return s.kind == HomotopyStepKind::Smoothing; });
    const auto last_extrema =
        std::find_if(r.steps.rbegin(), r.steps.rend(), [](const HomotopyStep& s) {
            return s.kind == HomotopyStepKind::ExtremaSmoothing;
        });
    EXPECT_LT(&*last_extrema, &*first_smoothing);
    // ohne β: keine Schritte, Punkt hart
    MockModel m2{MockSpec{}};
    const HomotopyResult r2 =
        run_homotopy(m2, MockModel::start(0.3), mock_point(0.3), mock_options(0.32));
    EXPECT_TRUE(steps_of(r2, HomotopyStepKind::ExtremaSmoothing).empty());
    HomotopyOptions bad = mock_options(0.32);
    bad.extrema_factor = 1.0;
    EXPECT_THROW(check_homotopy_options(bad), std::invalid_argument);
}

// Ziel am Endpunkt verfehlt: μ-Fortsetzung, danach Einfügen am Zustand vor dem Abschluss und
// erneuter Abschluss; ohne weitere Elemente Abbruch.
TEST(HomotopyControl, GoalMissedInsertsAtEndThenStops) {
    MockSpec spec;
    spec.goal_never = true;
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.32);
    o.max_elements = 2;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::MaxElements) << r.message;
    EXPECT_EQ(r.insertions, 1);
    // nach dem Einfügen beginnt der Abschluss wieder mit μ × 1 und ε × 1
    const auto ins = steps_of(r, HomotopyStepKind::Insertion);
    ASSERT_EQ(ins.size(), 1U);
    EXPECT_EQ(ins[0]->point.ghost_weight, 1.0);
    EXPECT_EQ(ins[0]->point.ghost_smoothing, 1.0);
    EXPECT_EQ(steps_of(r, HomotopyStepKind::Goal).size(), 6U);
}

// Startpunkt nicht annehmbar: Einfügen; ohne Bedarf Abbruch.
TEST(HomotopyControl, StartRejected) {
    MockSpec spec;
    spec.start_merit = 5.0;
    spec.need = false;
    MockModel m(spec);
    const HomotopyResult r =
        run_homotopy(m, MockModel::start(0.3), mock_point(0.3), mock_options(0.6));
    EXPECT_EQ(r.status, HomotopyStatus::StartRejected);
    EXPECT_EQ(r.solves, 1);

    MockSpec spec2;
    spec2.start_merit = 5.0;
    spec2.capacity0 = 0.2; // Start über der Kapazität: Bedarf
    spec2.capacity_step = 0.5;
    MockModel m2(spec2);
    const HomotopyResult r2 =
        run_homotopy(m2, MockModel::start(0.3), mock_point(0.3), mock_options(0.6));
    EXPECT_EQ(r2.status, HomotopyStatus::GoalReached) << r2.message;
    EXPECT_EQ(r2.insertions, 1);
    EXPECT_EQ(r2.steps[1].kind, HomotopyStepKind::Insertion);
}

// Feldanteil (stetig) und Spektrumstufen (diskret) nach der Apertur.
TEST(HomotopyControl, FieldAndSpectrumPhases) {
    MockModel m({});
    HomotopyOptions o = mock_options(0.32);
    o.target_field = 1.0;
    o.target_spectrum = 2;
    HomotopyPoint start = mock_point(0.3);
    start.field = 0.0;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), start, o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    const auto f = steps_of(r, HomotopyStepKind::Field);
    ASSERT_EQ(f.size(), 4U); // 0,125 → 0,375 → 0,875 → 1 (Δ wächst bis 0,5)
    EXPECT_DOUBLE_EQ(f[0]->point.field, 0.125);
    EXPECT_DOUBLE_EQ(f[1]->point.field, 0.375);
    EXPECT_DOUBLE_EQ(f[2]->point.field, 0.875);
    EXPECT_DOUBLE_EQ(f[3]->point.field, 1.0);
    EXPECT_DOUBLE_EQ(f[3]->delta, 0.5);
    const auto sp = steps_of(r, HomotopyStepKind::Spectrum);
    ASSERT_EQ(sp.size(), 2U);
    EXPECT_EQ(sp[0]->point.spectrum, 1);
    EXPECT_EQ(sp[1]->point.spectrum, 2);
    EXPECT_EQ(r.point.field, 1.0);
    EXPECT_EQ(r.point.spectrum, 2);
}

// Gemeinsame Apertur-/Feld-Homotopie (M6-T8): eine Phase über t, Punkte auf der Diagonale
// NA(t) = 0,3 + 0,16 t, φ(t) = 0,1 + 0,9 t; Schrittweite in t wie bei der Apertur (Δ₀ = 1/4,
// Aperturschritt 0,04 schnell, dann Δ = 1/2 behalten), keine Apertur- und Feldschritte.
TEST(HomotopyControl, JointApertureFieldDiagonal) {
    MockModel m({});
    HomotopyOptions o = mock_options(0.46);
    o.target_field = 1.0;
    o.joint_aperture_field = true;
    o.joint_step = {0.25, 1.0 / 64.0, 0.5, 2.0, 0.5};
    HomotopyPoint start = mock_point(0.3);
    start.field = 0.1;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), start, o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_TRUE(steps_of(r, HomotopyStepKind::Aperture).empty());
    EXPECT_TRUE(steps_of(r, HomotopyStepKind::Field).empty());
    const auto j = steps_of(r, HomotopyStepKind::ApertureField);
    ASSERT_EQ(j.size(), 3U); // t = 0,25 → 0,75 → 1
    const std::vector<double> t{0.25, 0.75, 1.0};
    const std::vector<double> deltas{0.25, 0.5, 0.5};
    for (std::size_t i = 0; i < j.size(); ++i) {
        EXPECT_DOUBLE_EQ(j[i]->point.aperture, 0.3 + 0.16 * t[i]) << i;
        EXPECT_DOUBLE_EQ(j[i]->point.field, 0.1 + 0.9 * t[i]) << i;
        EXPECT_DOUBLE_EQ(j[i]->delta, deltas[i]) << i;
        EXPECT_TRUE(j[i]->accepted);
    }
    EXPECT_EQ(j[0]->action, HomotopyAction::Grow); // Aperturschritt 0,04: schnell
    EXPECT_EQ(j[1]->action, HomotopyAction::Keep); // 0,08: langsam
    // Endpunkt genau das Ziel (nicht nur bis auf Rundung)
    EXPECT_EQ(r.point.aperture, 0.46);
    EXPECT_EQ(r.point.field, 1.0);
    EXPECT_EQ(std::string(to_string(HomotopyStepKind::ApertureField)), "aperture_field");
    const auto doc = nlohmann::json::parse(homotopy_result_json(r));
    EXPECT_EQ(doc["steps"][static_cast<std::size_t>(j[0]->index)]["kind"], "aperture_field");
}

// Exponent p < 1: das Feld läuft voraus, φ(t) = φ₀ + t^p (φ₁ − φ₀).
TEST(HomotopyControl, JointApertureFieldExponent) {
    MockModel m({});
    HomotopyOptions o = mock_options(0.5);
    o.target_field = 1.0;
    o.joint_aperture_field = true;
    o.joint_field_exponent = 0.5;
    o.joint_step = {0.25, 1.0 / 64.0, 0.25, 2.0, 0.5};
    HomotopyPoint start = mock_point(0.3);
    start.field = 0.0;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), start, o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    const auto j = steps_of(r, HomotopyStepKind::ApertureField);
    ASSERT_EQ(j.size(), 4U);
    for (std::size_t i = 0; i < j.size(); ++i) {
        const double t = 0.25 * static_cast<double>(i + 1);
        EXPECT_DOUBLE_EQ(j[i]->point.aperture, 0.3 + 0.2 * t) << i;
        EXPECT_DOUBLE_EQ(j[i]->point.field, std::sqrt(t)) << i;
    }
    EXPECT_GT(j[0]->point.field, 0.25); // voraus gegenüber der Diagonale
}

// Ablehnung, Halbieren in t und Einfügen am letzten guten Punkt der Bahn; danach weiter bis zum
// Ziel. Kapazität eines Elements NA 0,4 (t = 0,5), jedes weitere +0,2.
TEST(HomotopyControl, JointApertureFieldInsertsOnPath) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.2;
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.5);
    o.target_field = 1.0;
    o.joint_aperture_field = true;
    o.joint_step = {0.375, 0.1, 0.5, 2.0, 0.5};
    HomotopyPoint start = mock_point(0.3);
    start.field = 0.1;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), start, o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.insertions, 1);
    const auto j = steps_of(r, HomotopyStepKind::ApertureField);
    // t = 0,375 angenommen (NA 0,375); 0,75 → abgelehnt, Δ 0,1875; 0,5625 abgelehnt,
    // Δ 0,09375 < 0,1 → Einfügen bei t = 0,375; danach Δ₀: t = 0,75, 1
    ASSERT_EQ(j.size(), 5U);
    EXPECT_TRUE(j[0]->accepted);
    EXPECT_FALSE(j[1]->accepted);
    EXPECT_EQ(j[1]->action, HomotopyAction::Shrink);
    EXPECT_FALSE(j[2]->accepted);
    EXPECT_EQ(j[2]->action, HomotopyAction::Insert);
    const auto ins = steps_of(r, HomotopyStepKind::Insertion);
    ASSERT_EQ(ins.size(), 1U);
    EXPECT_DOUBLE_EQ(ins[0]->point.aperture, 0.375);
    EXPECT_DOUBLE_EQ(ins[0]->point.field, 0.1 + 0.9 * 0.375);
    EXPECT_TRUE(j[3]->accepted);
    EXPECT_DOUBLE_EQ(j[3]->point.aperture, 0.3 + 0.2 * 0.75);
    EXPECT_DOUBLE_EQ(j[3]->delta, 0.375);
    EXPECT_EQ(r.point.aperture, 0.5);
    EXPECT_EQ(r.point.field, 1.0);
}

// Ohne Option bzw. wenn sich nur ein Parameter ändert: dieselben Schritte wie bisher (Apertur,
// dann Feld), bitgleiche Protokolle.
TEST(HomotopyControl, JointApertureFieldOffOrSingleParameterUnchanged) {
    const auto run = [](bool joint, double target_field) {
        MockSpec spec;
        spec.capacity0 = 0.42;
        spec.capacity_step = 0.2;
        MockModel m(spec);
        HomotopyOptions o = mock_options(0.5);
        o.target_field = target_field;
        o.joint_aperture_field = joint;
        HomotopyPoint start = mock_point(0.3);
        start.field = 0.1;
        HomotopyResult r = run_homotopy(m, MockModel::start(0.3), start, o);
        for (HomotopyStep& s : r.steps) {
            s.seconds = 0.0;
        }
        r.seconds = 0.0;
        return homotopy_result_json(r);
    };
    const std::string off = run(false, 1.0);
    EXPECT_EQ(nlohmann::json::parse(off)["steps"].dump().find("aperture_field"), std::string::npos);
    EXPECT_NE(run(true, 1.0), off);
    // nur die Apertur ändert sich: mit und ohne Option gleich
    EXPECT_EQ(run(true, 0.1), run(false, 0.1));
}

// Einfügen nur mit Verbesserung (M6-T8): Variante 0 verschlechtert F_B (× 4) und wird verworfen,
// Variante 1 übernommen; ohne Option wird Variante 0 übernommen; verbessert keine, Abbruch.
TEST(HomotopyControl, InsertionMustImprove) {
    const auto run = [](bool improve, int bad) {
        MockSpec spec;
        spec.capacity0 = 0.4;
        spec.capacity_step = 0.2;
        spec.alternatives = 3;
        spec.bad_options = bad;
        MockModel m(spec);
        HomotopyOptions o = mock_options(0.5);
        o.insertion_must_improve = improve;
        o.insertion_attempts = 3;
        return run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    };
    const HomotopyResult off = run(false, 1);
    EXPECT_GE(off.insertions, 1);
    EXPECT_EQ(
        off.steps[static_cast<std::size_t>(steps_of(off, HomotopyStepKind::Insertion)[0]->index) +
                  1]
            .kind,
        HomotopyStepKind::InsertionStage); // Variante 0 übernommen, kein Vermerk
    for (const HomotopyStep& s : off.steps) {
        EXPECT_EQ(s.operation.find("verworfen"), std::string::npos);
    }

    const HomotopyResult r = run(true, 1);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.insertions, 2); // Variante 0 versucht und verworfen, Variante 1 übernommen
    const auto ins = steps_of(r, HomotopyStepKind::Insertion);
    ASSERT_EQ(ins.size(), 3U); // Variante 0, Vermerk „verworfen“, Variante 1
    EXPECT_NE(ins[1]->operation.find("Einfügen verworfen"), std::string::npos);
    EXPECT_FALSE(ins[1]->accepted);
    EXPECT_EQ(r.elements, 2);
    EXPECT_EQ(r.system.surfaces.back().semi_diameter, 1.0); // Variante 1

    const HomotopyResult none = run(true, 3);
    EXPECT_EQ(none.status, HomotopyStatus::StepTooSmall);
    EXPECT_NE(none.message.find("kein Einfügen verbessert F_B (3 Varianten"), std::string::npos)
        << none.message;
    EXPECT_EQ(none.elements, 1); // Zustand vor dem Einfügen
    EXPECT_EQ(steps_of(none, HomotopyStepKind::Insertion).size(), 6U);

    MockModel m({});
    HomotopyOptions bad = mock_options(0.5);
    bad.insertion_attempts = 0;
    EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), mock_point(0.3), bad),
                 std::invalid_argument);
}

// Rückbau (M6-T9): Startsystem aus der Basis (Element 0) und `groups.size()` weiteren Elementen.
SystemDescription mock_system(double aperture, const std::vector<const char*>& groups) {
    SystemDescription d = MockModel::start(aperture);
    for (const char* g : groups) {
        Surface s;
        if (g != nullptr) {
            s.group = g;
        }
        d.surfaces.push_back(s);
    }
    return d;
}

// Rückbau vor dem Abschluss: Kandidaten in der Reihenfolge des Modells (Element 2 zuerst); das
// notwendige Element 2 wird nach der Rückbau-Homotopie verworfen (Zustand zurück), das
// überflüssige Element 1 entfernt; danach ist nichts mehr entfernbar.
TEST(HomotopyControl, PruneRemovesSuperfluousKeepsNecessary) {
    MockSpec spec;
    spec.capacity0 = 0.31;
    spec.capacity_step = 0.1; // zwei beitragende Elemente: Kapazität 0,41
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.4);
    o.prune = true;
    const HomotopyResult r =
        run_homotopy(m, mock_system(0.3, {"useless", nullptr}), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.start_elements, 3);
    EXPECT_EQ(r.elements, 2);
    EXPECT_EQ(r.insertions, 0);
    EXPECT_EQ(r.removals, 1);
    EXPECT_EQ(r.removal_attempts, 3); // Element 2, Element 1 (entfernt), dann Element 1 (vorher 2)
    ASSERT_EQ(r.system.surfaces.size(), 2U);
    EXPECT_FALSE(r.system.surfaces[1].group.has_value()); // das notwendige Element blieb
    EXPECT_EQ(m.collapse_fractions,
              (std::vector<double>{1.0 / 3.0, 0.5, 1.0, 1.0 / 3.0, 0.5, 1.0, 1.0 / 3.0, 0.5, 1.0}));
    const auto stages = steps_of(r, HomotopyStepKind::RemovalStage);
    const auto notes = steps_of(r, HomotopyStepKind::Removal);
    ASSERT_EQ(stages.size(), 9U);
    ASSERT_EQ(notes.size(), 3U);
    EXPECT_FALSE(notes[0]->accepted);
    EXPECT_TRUE(notes[1]->accepted);
    EXPECT_FALSE(notes[2]->accepted);
    EXPECT_NE(notes[0]->operation.find("Rückbau verworfen (vor dem Abschluss): Mock-Element 2"),
              std::string::npos)
        << notes[0]->operation;
    EXPECT_NE(notes[1]->operation.find("Mock-Element 1 entfernt"), std::string::npos)
        << notes[1]->operation;
    EXPECT_EQ(notes[0]->elements, 3);
    EXPECT_EQ(notes[1]->elements, 2);
    for (const HomotopyStep* s : stages) {
        EXPECT_FALSE(s->accepted);
        EXPECT_EQ(s->point, stages.front()->point);
    }
    // nach dem Rückbau der Abschluss am entfernten System
    EXPECT_EQ(r.steps.back().kind, HomotopyStepKind::Goal);
    EXPECT_EQ(r.steps.back().elements, 2);
    // Zusammenfassung und JSON nennen den Rückbau
    std::ostringstream text;
    write_homotopy_summary(r, text);
    EXPECT_NE(text.str().find("1 Rückbauten in 3 Versuchen"), std::string::npos) << text.str();
    const nlohmann::json j = nlohmann::json::parse(homotopy_result_json(r));
    EXPECT_EQ(j["removals"], 1);
    EXPECT_EQ(j["removal_attempts"], 3);
    EXPECT_EQ(j["steps"][static_cast<std::size_t>(notes[1]->index)]["kind"], "removal");

    // höchstens ein Versuch pro Runde: nur Element 2, verworfen
    HomotopyOptions one = o;
    one.prune_attempts = 1;
    MockModel m1(spec);
    const HomotopyResult r1 =
        run_homotopy(m1, mock_system(0.3, {"useless", nullptr}), mock_point(0.3), one);
    EXPECT_EQ(r1.elements, 3);
    EXPECT_EQ(r1.removal_attempts, 1);
    // Mindestzahl: mit 3 Elementen nichts zu entfernen
    HomotopyOptions min3 = o;
    min3.prune_min_elements = 3;
    MockModel m3(spec);
    const HomotopyResult r3 =
        run_homotopy(m3, mock_system(0.3, {"useless", nullptr}), mock_point(0.3), min3);
    EXPECT_EQ(r3.removal_attempts, 0);
    EXPECT_EQ(r3.elements, 3);
    // Schwelle: F_B = F_max/2 nach dem Entfernen; Faktor 0,4 verwirft alles, Toleranz 0 nicht
    HomotopyOptions strict = o;
    strict.prune_merit_factor = 0.4;
    MockModel ms(spec);
    EXPECT_EQ(
        run_homotopy(ms, mock_system(0.3, {"useless", nullptr}), mock_point(0.3), strict).removals,
        0);
    HomotopyOptions tol = o;
    tol.prune_tolerance = 0.0;
    MockModel mt(spec);
    EXPECT_EQ(
        run_homotopy(mt, mock_system(0.3, {"useless", nullptr}), mock_point(0.3), tol).removals, 1);
}

// Rückbau nach einer Phase nur, wenn in ihr eingefügt wurde; vor dem Abschluss nur, wenn seit
// dem letzten Rückbau eingefügt wurde.
TEST(HomotopyControl, PruneAfterPhaseWithInsertions) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.2;
    MockModel m(spec);
    HomotopyOptions o = mock_options(0.5);
    o.prune = true;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.insertions, 1);
    EXPECT_EQ(r.elements, 2);         // das eingefügte Element ist notwendig
    EXPECT_EQ(r.removal_attempts, 1); // einmal nach der Aperturphase, nicht noch einmal am Ende
    const auto notes = steps_of(r, HomotopyStepKind::Removal);
    ASSERT_EQ(notes.size(), 1U);
    EXPECT_NE(notes[0]->operation.find("(nach der Phase)"), std::string::npos);
    EXPECT_EQ(notes[0]->point.aperture, 0.5);

    // Abbruch in der Phase (max_elements): Rückbau am letzten guten Zustand, Status bleibt
    MockSpec stop = spec;
    stop.capacity_step = 0.1;
    MockModel ms(stop);
    HomotopyOptions os = mock_options(0.6);
    os.prune = true;
    os.max_elements = 2;
    const HomotopyResult rs = run_homotopy(ms, MockModel::start(0.3), mock_point(0.3), os);
    EXPECT_EQ(rs.status, HomotopyStatus::MaxElements) << rs.message;
    EXPECT_EQ(rs.removal_attempts, 1);
    EXPECT_EQ(rs.elements, 2);
    const auto stop_notes = steps_of(rs, HomotopyStepKind::Removal);
    ASSERT_EQ(stop_notes.size(), 1U);
    EXPECT_NE(stop_notes[0]->operation.find("(am Ende)"), std::string::npos);
    EXPECT_EQ(rs.steps.back().kind, HomotopyStepKind::Removal);
}

// Elementbudget (M6-T9): Verlangt ein Einfügen mehr als max_elements, zuerst Rückbau; das
// überflüssige Element geht, das neue kommt. Ohne Option bzw. ohne entfernbares Element Abbruch.
TEST(HomotopyControl, ElementBudgetPrunesBeforeInserting) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.2;
    HomotopyOptions o = mock_options(0.5);
    o.max_elements = 2;
    MockModel off(spec);
    const HomotopyResult r0 = run_homotopy(off, mock_system(0.3, {"useless"}), mock_point(0.3), o);
    EXPECT_EQ(r0.status, HomotopyStatus::MaxElements) << r0.message;
    EXPECT_EQ(r0.removal_attempts, 0);

    o.prune_at_limit = true;
    MockModel m(spec);
    const HomotopyResult r = run_homotopy(m, mock_system(0.3, {"useless"}), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.removals, 1);
    EXPECT_EQ(r.insertions, 1);
    EXPECT_EQ(r.elements, 2);
    ASSERT_EQ(r.system.surfaces.size(), 2U);
    EXPECT_FALSE(r.system.surfaces[1].group.has_value()); // das eingefügte Element
    const auto notes = steps_of(r, HomotopyStepKind::Removal);
    ASSERT_EQ(notes.size(), 1U);
    EXPECT_NE(notes[0]->operation.find("(Elementbudget 2)"), std::string::npos);
    // die Phase versucht nach dem Rückbau neu (Δ₀), erst danach das Einfügen
    const auto it = std::find_if(r.steps.begin(), r.steps.end(), [](const HomotopyStep& s) {
        return s.kind == HomotopyStepKind::Insertion;
    });
    ASSERT_NE(it, r.steps.end());
    EXPECT_GT(it->index, notes[0]->index);

    // nur notwendige Elemente: ein Rückbau-Versuch an diesem Punkt, dann Abbruch
    MockSpec necessary = spec;
    necessary.capacity0 = 0.2; // beide Elemente tragen bei: Kapazität 0,4
    MockModel mn(necessary);
    const HomotopyResult rn = run_homotopy(mn, mock_system(0.3, {nullptr}), mock_point(0.3), o);
    EXPECT_EQ(rn.status, HomotopyStatus::MaxElements) << rn.message;
    EXPECT_EQ(rn.removal_attempts, 1);
    EXPECT_EQ(rn.removals, 0);
    EXPECT_EQ(rn.elements, 2);
}

// Ohne Rückbau: kein Aufruf des Modells, Protokoll wie vorher (Optionen gesetzt, aber aus);
// ungültige Optionen.
TEST(HomotopyControl, PruneOffUnchangedAndInvalidOptions) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.2;
    MockModel a(spec);
    MockModel b(spec);
    HomotopyOptions o = mock_options(0.5);
    HomotopyOptions p = o;
    p.prune_stages = 5;
    p.prune_attempts = 7;
    p.prune_merit_factor = 0.5;
    p.prune_tolerance = 0.1;
    const HomotopyResult ra = run_homotopy(a, mock_system(0.3, {"useless"}), mock_point(0.3), o);
    const HomotopyResult rb = run_homotopy(b, mock_system(0.3, {"useless"}), mock_point(0.3), p);
    EXPECT_TRUE(b.collapse_fractions.empty());
    ASSERT_EQ(ra.steps.size(), rb.steps.size());
    for (std::size_t i = 0; i < ra.steps.size(); ++i) {
        HomotopyStep x = ra.steps[i];
        HomotopyStep y = rb.steps[i];
        x.seconds = y.seconds = 0.0;
        EXPECT_EQ(homotopy_step_json(x), homotopy_step_json(y)) << i;
    }
    EXPECT_EQ(ra.system, rb.system);
    EXPECT_EQ(homotopy_result_json(ra).find("removal"), std::string::npos);

    MockModel m({});
    for (int k = 0; k < 5; ++k) {
        HomotopyOptions bad = mock_options(0.5);
        bad.prune = true;
        if (k == 0) {
            bad.prune_stages = 0;
        } else if (k == 1) {
            bad.prune_attempts = 0;
        } else if (k == 2) {
            bad.prune_merit_factor = 1.5;
        } else if (k == 3) {
            bad.prune_merit_factor = 0.0;
        } else {
            bad.prune_min_elements = -1;
        }
        EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), mock_point(0.3), bad),
                     std::invalid_argument)
            << k;
    }
}

TEST(HomotopyControl, JointApertureFieldInvalidOptions) {
    MockModel m({});
    HomotopyOptions o = mock_options(0.5);
    o.target_field = 1.0;
    o.joint_aperture_field = true;
    o.joint_field_exponent = 0.0;
    EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o), std::invalid_argument);
    o.joint_field_exponent = 1.0;
    o.joint_step.min = 1.0; // Δ_min > Δ₀
    EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o), std::invalid_argument);
}

TEST(HomotopyControl, MaxSteps) {
    MockModel m({});
    HomotopyOptions o = mock_options(0.6);
    o.max_steps = 3;
    const HomotopyResult r = run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::MaxSteps);
    EXPECT_EQ(r.solves, 3);
}

TEST(HomotopyControl, InvalidOptionsThrow) {
    MockModel m({});
    HomotopyOptions o = mock_options(0.6);
    o.aperture_step.min = 0.5; // Δ_min > Δ₀
    EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o), std::invalid_argument);
    o = mock_options(0.6);
    o.penalty_factor = 1.0;
    EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), mock_point(0.3), o), std::invalid_argument);
    o = mock_options(0.6);
    o.target_spectrum = 0;
    HomotopyPoint p = mock_point(0.3);
    p.spectrum = 1;
    EXPECT_THROW(run_homotopy(m, MockModel::start(0.3), p, o), std::invalid_argument);
    EXPECT_DOUBLE_EQ(homotopy_merit_threshold(HomotopyOptions{}), 1.0 / 196.0);
}

// Protokoll als Text und JSON.
TEST(HomotopyControl, ProtocolTextAndJson) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.15;
    MockModel m(spec);
    const HomotopyResult r =
        run_homotopy(m, MockModel::start(0.3), mock_point(0.3), mock_options(0.5));
    const nlohmann::json j = nlohmann::json::parse(homotopy_result_json(r));
    EXPECT_EQ(j["status"], "goal_reached");
    EXPECT_EQ(j["insertions"], 1);
    EXPECT_EQ(j["elements"], 2);
    ASSERT_EQ(j["steps"].size(), r.steps.size());
    for (std::size_t i = 0; i < r.steps.size(); ++i) {
        const nlohmann::json& s = j["steps"][i];
        EXPECT_EQ(s["index"], static_cast<int>(i));
        EXPECT_EQ(s["kind"], to_string(r.steps[i].kind));
        EXPECT_EQ(s["action"], to_string(r.steps[i].action));
        EXPECT_EQ(s["elements"], r.steps[i].elements);
        EXPECT_EQ(s["lm"].is_null(), !r.steps[i].solved);
        EXPECT_EQ(s["point"]["aperture"], r.steps[i].point.aperture);
    }
    EXPECT_TRUE(j["goal"]["reached"].get<bool>());
    std::ostringstream text;
    write_homotopy_step_header(text);
    for (const HomotopyStep& s : r.steps) {
        write_homotopy_step_line(s, text);
    }
    write_homotopy_summary(r, text);
    const std::string t = text.str();
    EXPECT_NE(t.find("insert"), std::string::npos);
    EXPECT_NE(t.find("Mock A"), std::string::npos);
    EXPECT_NE(t.find("Status: goal_reached"), std::string::npos);
    EXPECT_NE(homotopy_step_json(r.steps.front()).find("\"kind\""), std::string::npos);
}

// ---------------------------------------------------------------------------------------------
// Optisches Modell: Punkt realisieren, Einfügen, Ziel (ohne lange Optimierung)

LoadedSystem homotopy_start() {
    return load_residual_system("einzellinse_homotopie_start.json");
}

TEST(OpticalHomotopy, RealizeAppliesPointAndMargins) {
    const LoadedSystem start = load_residual_system("zweilinser_mechanik.json");
    OpticalHomotopyOptions o;
    o.spectra = {start.system.spectrum, start.system.spectrum};
    o.spectra[1].wavelengths_nm = {486.13, 587.56, 656.27};
    o.spectra[1].weights = {1.0, 1.0, 1.0};
    OpticalHomotopyModel m(start, o);
    const HomotopyPoint p0 = m.start_point();
    const double f = *start.system.target->focal_length;
    EXPECT_DOUBLE_EQ(p0.aperture, start.system.surfaces[0].semi_diameter / f);
    HomotopyPoint p = p0;
    p.aperture = 2.0 * p0.aperture;
    p.field = 0.5;
    p.spectrum = 1;
    p.penalty_scale = 0.01;
    const SystemDescription d = m.realize(m.start_system(), p);
    EXPECT_DOUBLE_EQ(d.surfaces[0].semi_diameter, f * p.aperture);
    for (std::size_t i = 0; i < d.source.field_angles_deg.size(); ++i) {
        EXPECT_DOUBLE_EQ(d.source.field_angles_deg[i],
                         0.5 * start.system.source.field_angles_deg[i]);
    }
    EXPECT_EQ(d.spectrum, o.spectra[1]);
    // Einseitige Bedingungen: Gewicht × 0,01 und Vorhalt; Brennweite (Gleichung) unverändert
    const auto& before = start.system.constraints->items;
    const auto& after = d.constraints->items;
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (!is_one_sided(before[i])) {
            EXPECT_EQ(before[i], after[i]);
            continue;
        }
        EXPECT_DOUBLE_EQ(*after[i].weight, 0.01 * before[i].weight_or_default());
        switch (before[i].type) {
        case ConstraintType::MinEdgeThickness:
        case ConstraintType::MinCenterThickness:
            EXPECT_DOUBLE_EQ(*after[i].thickness, *before[i].thickness + o.mechanical_margin);
            break;
        case ConstraintType::MaxCenterThickness:
            EXPECT_DOUBLE_EQ(*after[i].thickness, *before[i].thickness - o.mechanical_margin);
            break;
        case ConstraintType::MaxSemiDiameter:
            EXPECT_DOUBLE_EQ(*after[i].clearance,
                             before[i].clearance_or_default() + o.mechanical_margin);
            break;
        case ConstraintType::MaxIncidenceAngle:
            EXPECT_DOUBLE_EQ(*after[i].angle, *before[i].angle - o.angle_margin);
            break;
        default:
            break;
        }
    }
    // ohne Vorhalt nur die Gewichte
    const SystemDescription n = m.realize(m.start_system(), p, false);
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(n.constraints->items[i].thickness, before[i].thickness);
        EXPECT_EQ(n.constraints->items[i].angle, before[i].angle);
    }
    EXPECT_THROW(m.realize(m.start_system(), HomotopyPoint{}), std::invalid_argument);
    p.spectrum = 2;
    EXPECT_THROW(m.realize(m.start_system(), p), std::invalid_argument);
}

TEST(OpticalHomotopy, RealizeSetsExtremaSmoothing) {
    const LoadedSystem start = load_residual_system("zweilinser_mechanik.json");
    OpticalHomotopyModel m(start, OpticalHomotopyOptions{});
    HomotopyPoint p = m.start_point();
    EXPECT_FALSE(m.realize(m.start_system(), p).constraints->extrema_smoothing.has_value());
    p.extrema_beta = 2500.0;
    const SystemDescription d = m.realize(m.start_system(), p);
    ASSERT_TRUE(d.constraints->extrema_smoothing.has_value());
    EXPECT_EQ(*d.constraints->extrema_smoothing, (ExtremaSmoothingSpec{2500.0, 2500.0}));
    p.extrema_beta = -1.0;
    EXPECT_THROW((void)m.realize(m.start_system(), p), std::invalid_argument);
}

// Prädiktor des optischen Modells (M11-T3): Sekante über die Variablen, projiziert auf die
// Schranken; andere Topologie → leer.
TEST(OpticalHomotopy, PredictExtrapolatesVariables) {
    const LoadedSystem start = load_residual_system("zweilinser_mechanik.json");
    OpticalHomotopyModel m(start, OpticalHomotopyOptions{});
    const HomotopyPoint p = m.start_point();
    const SystemDescription cur = m.start_system();
    SystemDescription prev = cur;
    prev.surfaces[1].curvature -= 0.001;
    prev.surfaces[2].thickness -= 0.05;
    const std::optional<SystemDescription> out = m.predict(prev, cur, p, 2.0);
    ASSERT_TRUE(out.has_value());
    EXPECT_NEAR(out->surfaces[1].curvature, cur.surfaces[1].curvature + 0.002, 1e-15);
    EXPECT_NEAR(out->surfaces[2].thickness, cur.surfaces[2].thickness + 0.1, 1e-14);
    EXPECT_EQ(out->surfaces[3].curvature, cur.surfaces[3].curvature); // gleich in beiden
    // Projektion auf die Schranke max = 0,3 der Krümmungen
    SystemDescription far = cur;
    far.surfaces[1].curvature = -0.3; // untere Schranke; die Sekante mit Faktor 10 überschießt
    ASSERT_GT(cur.surfaces[1].curvature, -0.27);
    const std::optional<SystemDescription> clamped = m.predict(far, cur, p, 10.0);
    ASSERT_TRUE(clamped.has_value());
    EXPECT_EQ(clamped->surfaces[1].curvature, 0.3);
    // andere Topologie
    SystemDescription other = cur;
    other.surfaces.pop_back();
    EXPECT_FALSE(m.predict(other, cur, p, 1.0).has_value());
}

// Farbregel (M11-T5c): polychromatische Einzellinse ohne Geisterplatten – ohne Regel kein
// Bedarf, mit Regel Operator B an der Linse (Fläche 1); das Einfügen teilt die Linse ohne
// Strafgewicht einer Platte.
// Achromat-Start als zweiter Zweig (M11-T5e): wie bisher und mit Achromat-Start, der Controller
// wählt (allein brach der Achromat-Start bei dicken Linsen und NA 0,6 die Strahlen).
TEST(OpticalHomotopy, AchromatStartIsSecondBranch) {
    const LoadedSystem start = load_residual_system("einzellinse_achromat_start.json");
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.split.achromat_start = true;
    OpticalHomotopyModel m(start, o);
    const HomotopyNeed need = m.diagnose(m.start_system(), m.start_point());
    ASSERT_EQ(need.recommendation, GhostOperator::B) << need.text;
    const HomotopyInsertion ins = m.insert(m.start_system(), m.start_point(), need);
    ASSERT_EQ(ins.branches.size(), 2U);
    EXPECT_NE(ins.branches[0], ins.branches[1]);
    EXPECT_NE(ins.text.find("2 Zweige"), std::string::npos) << ins.text;
    OpticalHomotopyOptions plain = o;
    plain.split.achromat_start = false;
    OpticalHomotopyModel mp(start, plain);
    EXPECT_EQ(mp.insert(mp.start_system(), mp.start_point(), need).branches.size(), 1U);
}

// Volle Flächentabelle: insert lehnt ab (vorher Absturz von synth mit 59 Flächen + 6 Platten).
TEST(OpticalHomotopy, InsertRefusedWhenTableFull) {
    const LoadedSystem start = load_residual_system("einzellinse_achromat_start.json");
    OpticalHomotopyModel m(start, OpticalHomotopyOptions{});
    SystemDescription big = m.start_system();
    while (static_cast<int>(big.surfaces.size()) + 2 <= kMaxSurfaces) {
        big.surfaces.insert(big.surfaces.begin() + 1, big.surfaces[1]);
    }
    HomotopyNeed need;
    need.recommendation = GhostOperator::B;
    need.split_lens = 1;
    EXPECT_THROW((void)m.insert(big, m.start_point(), need), std::invalid_argument);
}

// Spektrum-Homotopie (M11-T5d): Stufe 0 monochromatisch bei λ_ref, Stufe k skaliert die Abstände
// zu λ_ref mit k/n, letzte Stufe = Ziel.
// Stetige Spektrumbreite im optischen Modell: Start monochromatisch bei λ_ref, Wellenlängen um
// λ_ref skaliert, Breite 1 = Datei; nicht zusammen mit Spektrumstufen.
TEST(OpticalHomotopy, ContinuousSpectrumWidth) {
    const LoadedSystem start = load_residual_system("einzellinse_achromat_start.json");
    OpticalHomotopyOptions o;
    o.spectrum_continuous = true;
    OpticalHomotopyModel m(start, o);
    HomotopyPoint p = m.start_point();
    EXPECT_EQ(p.spectrum_width, 0.0);
    EXPECT_EQ(m.realize(m.start_system(), p).spectrum.wavelengths_nm, std::vector<double>{546.07});
    p.spectrum_width = 0.5;
    const Spectrum half = m.realize(m.start_system(), p).spectrum;
    ASSERT_EQ(half.wavelengths_nm.size(), 2U);
    EXPECT_DOUBLE_EQ(half.wavelengths_nm[0], 546.07 + 0.5 * (480.0 - 546.07));
    EXPECT_DOUBLE_EQ(half.wavelengths_nm[1], 546.07 + 0.5 * (644.0 - 546.07));
    p.spectrum_width = 1.0;
    EXPECT_EQ(m.realize(m.start_system(), p).spectrum, start.system.spectrum);
    p.spectrum_width = 1.2;
    EXPECT_THROW((void)m.realize(m.start_system(), p), std::invalid_argument);
    OpticalHomotopyOptions both = o;
    both.spectra = spectrum_stages(start.system.spectrum, 2);
    OpticalHomotopyModel mb(start, both);
    EXPECT_THROW((void)mb.start_point(), std::invalid_argument);
}

TEST(OpticalHomotopy, SpectrumStages) {
    Spectrum target;
    target.wavelengths_nm = {480.0, 644.0};
    target.weights = {1.0, 2.0};
    target.reference_nm = 546.0;
    const std::vector<Spectrum> st = spectrum_stages(target, 4);
    ASSERT_EQ(st.size(), 5U);
    EXPECT_EQ(st[0].wavelengths_nm, std::vector<double>{546.0});
    EXPECT_EQ(st[0].weights, std::vector<double>{1.0});
    EXPECT_DOUBLE_EQ(st[1].wavelengths_nm[0], 546.0 - 66.0 / 4.0);
    EXPECT_DOUBLE_EQ(st[2].wavelengths_nm[1], 546.0 + 98.0 / 2.0);
    EXPECT_EQ(st[1].weights, target.weights);
    EXPECT_EQ(st[4], target);
    EXPECT_THROW((void)spectrum_stages(target, 0), std::invalid_argument);
}

TEST(OpticalHomotopy, ColorRuleRecommendsSplit) {
    const LoadedSystem start = load_residual_system("einzellinse_achromat_start.json");
    OpticalHomotopyModel plain(start, OpticalHomotopyOptions{});
    EXPECT_EQ(plain.diagnose(plain.start_system(), plain.start_point()).recommendation,
              GhostOperator::None);
    OpticalHomotopyOptions o;
    o.color_rule = true;
    OpticalHomotopyModel m(start, o);
    const HomotopyNeed need = m.diagnose(m.start_system(), m.start_point());
    ASSERT_EQ(need.recommendation, GhostOperator::B) << need.text;
    EXPECT_EQ(need.split_lens, 1);
    EXPECT_NE(need.text.find("Farbregel"), std::string::npos);
    const HomotopyInsertion ins = m.insert(m.start_system(), m.start_point(), need);
    ASSERT_EQ(ins.branches.size(), 1U);
    EXPECT_EQ(ins.plate, -1);
    EXPECT_EQ(ins.branches[0].surfaces.size(), m.start_system().surfaces.size() + 1);
    EXPECT_NE(ins.text.find("Farbregel"), std::string::npos) << ins.text;
    // auch mit Glasvariable (Buchdahl) statt Katalogglas (Zielspezifikation, M6-T6 p4_r0)
    LoadedSystem bd = start;
    const BuchdahlFit fit = fit_buchdahl(start.catalog.at("N-BK7"), 3);
    bd.system.surfaces[1].material = MaterialSpec{MaterialKind::Buchdahl, "", 1.0, fit.n0, fit.nu};
    OpticalHomotopyModel mb(bd, o);
    const HomotopyNeed nb = mb.diagnose(mb.start_system(), mb.start_point());
    EXPECT_EQ(nb.recommendation, GhostOperator::B) << nb.text;
    EXPECT_EQ(nb.split_lens, 1);
    // Nur das RMS-Verhältnis (Seidel-Kriterium aus): λ_ref = 546 nm liegt nicht im Spektrum
    // 480/644 nm, Bezug ist dann eine monochromatische Rechnung bei λ_ref (M6-T6 p4_r0). Bild im
    // paraxialen Fokus bei λ_ref wie nach einer monochromatischen Stufe: RMS 0,32/0,19 Wellen bei
    // 480/644 nm gegen 0,064 monochromatisch
    LoadedSystem focused = start;
    focused.system.surfaces.back().thickness =
        compute_paraxial(focused.system, focused.catalog).back_focal_distance;
    OpticalHomotopyOptions ratio_only = o;
    ratio_only.color_seidel = 1e9;
    OpticalHomotopyModel mr(focused, ratio_only);
    const HomotopyNeed nr = mr.diagnose(mr.start_system(), mr.start_point());
    EXPECT_EQ(nr.recommendation, GhostOperator::B) << nr.text;
    EXPECT_NE(nr.text.find("RMS-Verhältnis"), std::string::npos) << nr.text;
}

// Feldregel (M6-T8): Mehrlinser ohne Geisterplatten mit drei Feldern. Ohne Regel kein Bedarf;
// mit Regel Operator A im Luftspalt mit dem größten mittleren |ȳ/y| (Orakel: Aberrationsbericht
// der Datei), Merit nach dem Einfügen unverändert, Glas der Linse vor dem Spalt bzw. vorgegeben.
TEST(OpticalHomotopy, FieldRuleRecommendsMeniscusAtFieldPosition) {
    LoadedSystem start = load_residual_system("testsystem_mehrlinser.json");
    start.system.target = Target{};
    start.system.target->focal_length = compute_paraxial(start.system, start.catalog).focal_length;
    OpticalHomotopyModel plain(start, OpticalHomotopyOptions{});
    const HomotopyPoint p = plain.start_point();
    EXPECT_EQ(plain.diagnose(plain.start_system(), p).recommendation, GhostOperator::None);

    // Orakel: RMS am Rand gegen Achse und Lage aus dem Bericht
    const WavefrontResult wf = evaluate_wavefront(start);
    double rms_axis = 0.0;
    double rms_edge = 0.0;
    for (int k = 0; k < wf.wavelengths; ++k) {
        rms_axis = std::max(rms_axis, wf.at(0, k).rms_waves);
        rms_edge = std::max(rms_edge, wf.at(wf.fields - 1, k).rms_waves);
    }
    ASSERT_GT(rms_edge, rms_axis) << "Testsystem: Feldfehler sollen überwiegen";
    AberrationOptions ao;
    ao.real_rays = false;
    ao.focal_shift = false;
    const AberrationReport rep = compute_aberrations(start.system, start.catalog, ao);
    int best = -1;
    double best_ratio = -1.0;
    for (std::size_t s = 0; s + 1 < start.system.surfaces.size(); ++s) {
        if (start.system.surfaces[s].material.kind != MaterialKind::Air) {
            continue;
        }
        const double r = 0.5 * (std::abs(rep.surfaces[s].ybar / rep.surfaces[s].y) +
                                std::abs(rep.surfaces[s + 1].ybar / rep.surfaces[s + 1].y));
        if (r > best_ratio) {
            best_ratio = r;
            best = static_cast<int>(s);
        }
    }
    ASSERT_GE(best, 1);

    OpticalHomotopyOptions o;
    o.field_rule = true;
    o.field_ratio = 0.5 * rms_edge / rms_axis; // sicher erfüllt
    o.field_min_rms = 0.0;
    OpticalHomotopyModel m(start, o);
    const HomotopyNeed need = m.diagnose(m.start_system(), p);
    ASSERT_EQ(need.recommendation, GhostOperator::A) << need.text;
    EXPECT_EQ(need.field_gap, best) << need.text;
    EXPECT_EQ(need.field_candidates.front(), best);
    EXPECT_NE(need.text.find("Feldregel"), std::string::npos);
    // Varianten (Einfügen nur mit Verbesserung): die übrigen Spalte in der Reihenfolge der Eignung
    const std::vector<HomotopyNeed> variants = m.insertion_options(m.start_system(), p, need, 3);
    ASSERT_EQ(variants.size(), std::min<std::size_t>(3, need.field_candidates.size()));
    for (std::size_t k = 0; k < variants.size(); ++k) {
        EXPECT_EQ(variants[k].field_gap, need.field_candidates[k]);
    }
    const HomotopyInsertion ins = m.insert(m.start_system(), p, need);
    EXPECT_EQ(ins.op, GhostOperator::A);
    EXPECT_EQ(ins.plate, -1);
    EXPECT_EQ(ins.first_surface, best + 1);
    ASSERT_EQ(ins.branches.size(), 2U); // Sattelzweige wie Operator A
    EXPECT_NE(ins.text.find("(Feldregel)"), std::string::npos) << ins.text;
    const SystemDescription& b = ins.branches[0];
    const auto f = static_cast<std::size_t>(best);
    const MaterialSpec& before = start.system.surfaces[f - 1].material;
    EXPECT_EQ(b.surfaces[f + 1].material.kind, before.kind);
    EXPECT_EQ(b.surfaces[f + 1].material.name, before.name);
    EXPECT_EQ(b.surfaces[f + 1].material.n, before.n);
    EXPECT_EQ(b.surfaces[f + 1].material.n0, before.n0);
    EXPECT_DOUBLE_EQ(b.surfaces[f].thickness, 0.5 * start.system.surfaces[f].thickness);
    EXPECT_EQ(b.surfaces[f + 1].thickness, 0.0);

    // Merit unverändert (ein Zweig, c₀ ohne Störung)
    OpticalHomotopyOptions one = o;
    one.saddle_delta = 0.0;
    OpticalHomotopyModel m1(start, one);
    const HomotopyInsertion ins1 = m1.insert(m1.start_system(), p, need);
    ASSERT_EQ(ins1.branches.size(), 1U);
    EXPECT_DOUBLE_EQ(ins1.branches[0].surfaces[f + 1].curvature,
                     start.system.surfaces[f + 1].curvature);
    const auto merit = [&](const SystemDescription& d) {
        const LoadedSystem l = m1.load(m1.realize(d, p));
        return make_cpu_merit({}, {}, 1)(make_system_batch(l, 1)).at(0);
    };
    EXPECT_LE(relative_change(merit(m1.start_system()), merit(ins1.branches[0])),
              kInsertionMeritRelative);

    // vorgegebenes Glas; Kriterium nicht erfüllt: kein Bedarf
    OpticalHomotopyOptions g = o;
    g.field_glass = MaterialSpec{MaterialKind::Catalog, "N-BK7", 1.0, 1.0, {}};
    OpticalHomotopyModel mg(start, g);
    const HomotopyInsertion insg =
        mg.insert(mg.start_system(), p, mg.diagnose(mg.start_system(), p));
    EXPECT_EQ(insg.branches[0].surfaces[f + 1].material.name, "N-BK7");
    OpticalHomotopyOptions high = o;
    high.field_ratio = 2.0 * rms_edge / rms_axis;
    OpticalHomotopyModel mh(start, high);
    EXPECT_EQ(mh.diagnose(mh.start_system(), p).recommendation, GhostOperator::None);
    OpticalHomotopyOptions bad = o;
    bad.field_ratio = 0.0;
    EXPECT_THROW(OpticalHomotopyModel(start, bad), std::invalid_argument);
}

// Rückbau am optischen Modell (M6-T9): Ein Null-Meniskus (Operator A mit Dicke 0) hat den
// Seidel-Beitrag 0 und ist der erste Kandidat; Rückbau-Homotopie und Entfernen lassen die Merit
// unverändert und ergeben das Ausgangssystem (Orakel: Merit und Nebenbedingungen vor dem
// Einfügen). Während der Homotopie gelten die mechanischen Bedingungen nicht für den Block,
// danach wieder mit Standardauswahl.
TEST(OpticalHomotopy, RemovalOfNullElementRestoresSystem) {
    const LoadedSystem start = homotopy_start();
    OpticalHomotopyModel m(start, {});
    const HomotopyPoint p = m.start_point();
    const SystemDescription& base = m.start_system();
    // Null-Meniskus im Luftspalt hinter der Linse, hinter den Geisterplatten
    const InsertionResult ins =
        insert_meniscus(base, meniscus(2, 1.5, -0.05, catalog_glass("N-BK7")));
    // mechanische Bedingungen mit Standardauswahl wie nach finish_insertion (auch der Meniskus)
    SystemDescription with = ins.system;
    with.constraints = base.constraints;
    ASSERT_EQ(ins.first_new_surface, 3);
    ASSERT_EQ(m.elements(with), 2);
    const auto merit = [&](const SystemDescription& d) {
        const LoadedSystem l = m.load(m.realize(d, p));
        return make_cpu_merit({}, {}, 1)(make_system_batch(l, 1)).at(0);
    };
    const double f0 = merit(base);
    EXPECT_LE(relative_change(f0, merit(with)), kInsertionMeritRelative);

    const std::vector<HomotopyRemoval> c = m.removal_candidates(with, p);
    ASSERT_EQ(c.size(), 2U);
    EXPECT_EQ(c[0].first_surface, 3) << c[0].text;
    EXPECT_EQ(c[0].elements, 1);
    EXPECT_EQ(c[1].first_surface, 1);
    EXPECT_LE(c[0].score, 1e-9 * c[1].score) << c[0].text << " / " << c[1].text;
    EXPECT_NE(c[0].text.find("Element ab Fläche 3 (N-BK7"), std::string::npos) << c[0].text;

    std::vector<std::string> warnings;
    SystemDescription cur = with;
    for (const double f : {0.5, 1.0}) {
        cur = m.collapse(with, cur, c[0], f, warnings);
        EXPECT_LE(relative_change(f0, merit(cur)), kInsertionMeritRelative) << f;
    }
    EXPECT_TRUE(is_null_element(cur, 3));
    // mechanische Bedingungen ohne die Flächen 3, 4 (Spalte 3, 4)
    for (const ConstraintSpec& k : cur.constraints->items) {
        if (!is_mechanical(k.type)) {
            continue;
        }
        for (const int s : constraint_surfaces(cur, k)) {
            EXPECT_TRUE(s < 3 || s > 4) << s;
        }
    }
    const SystemDescription removed = m.remove(with, cur, c[0], warnings);
    EXPECT_TRUE(warnings.empty());
    EXPECT_LE(relative_change(f0, merit(removed)), kInsertionMeritRelative);
    ASSERT_EQ(removed.surfaces.size(), base.surfaces.size());
    for (std::size_t s = 0; s < base.surfaces.size(); ++s) {
        EXPECT_EQ(removed.surfaces[s].curvature, base.surfaces[s].curvature) << s;
        EXPECT_NEAR(removed.surfaces[s].thickness, base.surfaces[s].thickness,
                    kRemovalRoundtripRelative * base.surfaces[s].thickness)
            << s;
    }
    EXPECT_EQ(removed.ghost_plates, base.ghost_plates);
    ASSERT_TRUE(removed.constraints.has_value());
    EXPECT_EQ(removed.constraints->items, base.constraints->items); // Standardauswahl zurück
    EXPECT_EQ(m.elements(removed), 1);
}

TEST(OpticalHomotopy, RealizeScalesGhostPlatesAndWeights) {
    const LoadedSystem start = homotopy_start();
    OpticalHomotopyModel m(start, {});
    HomotopyPoint p = m.start_point();
    EXPECT_DOUBLE_EQ(p.aperture, 0.3);
    p.aperture = 0.45;
    p.ghost_weight = 10.0;
    p.ghost_smoothing = 0.1;
    const SystemDescription d = m.realize(m.start_system(), p);
    EXPECT_DOUBLE_EQ(d.surfaces[0].semi_diameter, 4.5);
    for (std::size_t g = 0; g < d.ghost_plates.size(); ++g) {
        EXPECT_DOUBLE_EQ(d.ghost_plates[g].norm_radius,
                         start.system.ghost_plates[g].norm_radius * 4.5 / 3.0);
    }
    for (std::size_t i = 0; i < d.constraints->items.size(); ++i) {
        const ConstraintSpec& c = d.constraints->items[i];
        if (c.type == ConstraintType::GhostPenalty) {
            const ConstraintSpec& b = start.system.constraints->items[i];
            EXPECT_DOUBLE_EQ(*c.weight, 10.0 * b.weight_or_default());
            EXPECT_DOUBLE_EQ(*c.smoothing, 0.1 * b.smoothing_or_default());
        }
    }
    // Punktquelle ohne NA bzw. kollimiert ohne Sollbrennweite: Fehler
    LoadedSystem bad = start;
    bad.system.target.reset();
    bad.system.constraints->items.erase(bad.system.constraints->items.begin());
    EXPECT_THROW(OpticalHomotopyModel(bad, {}), std::invalid_argument);
}

// Augmented Lagrangian (M11-T2): Fortschreibung am realisierten Punkt (Gewichte × s, Grenzen
// mit Vorhalt), θ in der Basisbeschreibung, wirksam in der nächsten Realisierung.
TEST(OpticalHomotopy, UpdateShiftsAtRealizedPoint) {
    const LoadedSystem start = load_residual_system("zweilinser_mechanik.json");
    OpticalHomotopyModel m(start, OpticalHomotopyOptions{});
    HomotopyPoint p = m.start_point();
    p.penalty_scale = 0.1;
    SystemDescription d = start.system;
    const double change = m.update_shifts(d, p, kDefaultMaxShift);

    const SystemDescription realized = m.realize(start.system, p);
    const LoadedSystem l = m.load(realized);
    const SystemBatchHost b = make_system_batch(l, 1);
    const ShiftUpdate u =
        shift_update(l.system, b, 0, OpticalHomotopyOptions{}.lm.jacobian.quadrature);
    EXPECT_EQ(change, u.max_change);
    EXPECT_GT(change, 0.0);
    SystemDescription expected = start.system;
    apply_shift_update(expected, u);
    EXPECT_EQ(d, expected);
    // realize übernimmt θ unverändert (Reihenfolge der Einträge bleibt)
    const SystemDescription again = m.realize(d, p);
    for (std::size_t i = 0; i < d.constraints->items.size(); ++i) {
        EXPECT_EQ(again.constraints->items[i].shift, d.constraints->items[i].shift);
    }
}

TEST(OpticalHomotopy, ElementsAndOneSided) {
    EXPECT_EQ(glass_elements(homotopy_start().system), 1);
    EXPECT_EQ(glass_elements(load_residual_system("testsystem_mehrlinser.json").system), 4);
    ConstraintSpec c;
    c.type = ConstraintType::FocalLength;
    EXPECT_FALSE(is_one_sided(c));
    c.type = ConstraintType::WorkingDistance;
    c.mode = ConstraintMode::Min;
    EXPECT_TRUE(is_one_sided(c));
    c.type = ConstraintType::Petzval;
    c.mode.reset();
    EXPECT_TRUE(is_one_sided(c));
    c.type = ConstraintType::GhostPenalty;
    EXPECT_FALSE(is_one_sided(c));
    c.type = ConstraintType::GlassHull;
    EXPECT_TRUE(is_one_sided(c));
    c.type = ConstraintType::MinEdgeThickness;
    EXPECT_TRUE(is_one_sided(c));
}

// Operator A aus einer Diagnose mit vorgegebenen Koeffizienten: zwei Zweige am Sattelpunkt,
// innere Homotopie über die Dicke (Luftspalt dahinter kürzer bzw. Mindestspalt), Abschluss mit
// Dickenvariable und Nebenbedingungen auf den neuen Flächen; Merit ohne Sattelstörung
// unverändert.
TEST(OpticalHomotopy, InsertMeniscusStagesAndFinish) {
    const LoadedSystem start = homotopy_start();
    OpticalHomotopyOptions o;
    o.element_thickness = 0.4;
    o.element_stages = 3;
    o.element_max_thickness = 2.0;
    o.min_air_gap = 0.1;
    o.saddle_delta = 0.0; // ein Zweig: Merit unverändert
    OpticalHomotopyModel m(start, o);
    SystemDescription s = m.start_system();
    s.ghost_plates[3].coefficients[1] = 2e-3; // hochbrechendes Glas hinter der Linse, Ordnung 2
    const HomotopyPoint p = m.start_point();
    const HomotopyNeed need = m.diagnose(s, p);
    ASSERT_EQ(need.recommendation, GhostOperator::A) << need.text;
    const HomotopyInsertion ins = m.insert(s, p, need);
    EXPECT_EQ(ins.op, GhostOperator::A);
    ASSERT_EQ(ins.branches.size(), 1U);
    EXPECT_EQ(ins.plate, 3);
    EXPECT_EQ(ins.first_surface, 3);
    EXPECT_EQ(ins.new_surfaces, 2);
    EXPECT_EQ(ins.stages, (std::vector<double>{0.1, 0.2, 0.4}));
    // Merit (wie im LM, am realisierten Punkt) unverändert; Strafgewicht der Platte × 10
    const auto merit = [&](const SystemDescription& d) {
        const LoadedSystem l = m.load(m.realize(d, p));
        return make_cpu_merit({}, insertion_extra(l.system), 1,
                              insertion_trace_extra(l.system))(make_system_batch(l, 1))
            .at(0);
    };
    SystemDescription unscaled = ins.branches[0];
    scale_ghost_penalty(unscaled, 3, 0.1);
    EXPECT_LE(relative_change(merit(s), merit(unscaled)), kInsertionMeritRelative);
    EXPECT_DOUBLE_EQ(*ghost_penalty_weight(ins.branches[0], 3), 10.0 * *ghost_penalty_weight(s, 3));

    SystemDescription b = ins.branches[0];
    const double gap0 = b.surfaces[4].thickness; // 3,0 − 0,5
    EXPECT_DOUBLE_EQ(ins.gap, gap0);
    m.set_stage(b, ins, 0.2);
    EXPECT_DOUBLE_EQ(b.surfaces[3].thickness, 0.2);
    EXPECT_DOUBLE_EQ(b.surfaces[4].thickness, gap0 - 0.2);
    m.set_stage(b, ins, 10.0); // länger als der Spalt: Mindestspalt, das System wird länger
    EXPECT_DOUBLE_EQ(b.surfaces[4].thickness, 0.1);
    m.set_stage(b, ins, 0.4);
    m.finish_insertion(b, ins);
    validate_system(b);
    const VariableSpec& v = b.variables->items.back();
    EXPECT_EQ(v.type, VariableType::Thickness);
    EXPECT_EQ(v.surfaces, std::vector<int>{3});
    EXPECT_EQ(v.min, 0.4);
    EXPECT_EQ(v.max, 2.0);
    // mechanische Bedingungen mit Standardauswahl gelten für die neuen Flächen
    for (const ConstraintSpec& c : b.constraints->items) {
        if (!is_mechanical(c.type)) {
            continue;
        }
        EXPECT_EQ(c.surface_selection, SurfaceSelection::None);
        const std::vector<int> surf = constraint_surfaces(b, c);
        if (c.medium == ConstraintMedium::Glass) {
            EXPECT_NE(std::find(surf.begin(), surf.end(), 3), surf.end());
        } else {
            EXPECT_NE(std::find(surf.begin(), surf.end(), 4), surf.end());
        }
    }
    EXPECT_EQ(m.elements(b), 2);

    // zwei Zweige am Sattelpunkt
    o.saddle_delta = 0.01;
    OpticalHomotopyModel m2(start, o);
    const HomotopyInsertion ins2 = m2.insert(s, p, need);
    ASSERT_EQ(ins2.branches.size(), 2U);
    const double c0 = ins.branches[0].surfaces[3].curvature;
    EXPECT_DOUBLE_EQ(ins2.branches[0].surfaces[3].curvature, c0 + 0.01);
    EXPECT_DOUBLE_EQ(ins2.branches[0].surfaces[4].curvature, c0 - 0.01);
    EXPECT_DOUBLE_EQ(ins2.branches[1].surfaces[3].curvature, c0 - 0.01);
    EXPECT_DOUBLE_EQ(ins2.branches[1].surfaces[4].curvature, c0 + 0.01);

    // Schranken der neuen Krümmungsvariablen enger als |c₀| bzw. c₀ ± δ (M6-T6: c₀ von einer
    // Fläche auf ihrer Schranke): Startkrümmungen bleiben innerhalb, die Zweige laden.
    ASSERT_GT(std::abs(c0), 0.006);
    o.meniscus.curvature_variables.min = -0.006;
    o.meniscus.curvature_variables.max = 0.006;
    OpticalHomotopyModel m3(start, o);
    const HomotopyInsertion ins3 = m3.insert(s, p, need); // Spanne 0,012 < 2δ
    ASSERT_EQ(ins3.branches.size(), 2U);
    for (const SystemDescription& br : ins3.branches) {
        for (int j = 3; j <= 4; ++j) {
            EXPECT_GE(br.surfaces[static_cast<std::size_t>(j)].curvature, -0.006);
            EXPECT_LE(br.surfaces[static_cast<std::size_t>(j)].curvature, 0.006);
        }
        EXPECT_NO_THROW((void)m3.load(m3.realize(br, p)));
    }
    o.meniscus.curvature_variables.max = std::abs(c0) - 0.005; // c₀ außerhalb der Schranken
    o.meniscus.curvature_variables.min = -(std::abs(c0) - 0.005);
    OpticalHomotopyModel m4(start, o);
    const HomotopyInsertion ins4 = m4.insert(s, p, need);
    ASSERT_EQ(ins4.branches.size(), 2U);
    const double lim = std::abs(c0) - 0.005;
    const double c4 = c0 > 0.0 ? lim - 0.01 : -lim + 0.01; // c₀ auf [min + δ, max − δ]
    EXPECT_DOUBLE_EQ(ins4.branches[0].surfaces[3].curvature, c4 + 0.01);
    EXPECT_DOUBLE_EQ(ins4.branches[1].surfaces[3].curvature, c4 - 0.01);
    for (const SystemDescription& br : ins4.branches) {
        EXPECT_NO_THROW((void)m4.load(m4.realize(br, p)));
    }
}

// Ziel am realen System: Geisterkoeffizienten 0, RMS pro Tupel in Wellen.
TEST(OpticalHomotopy, GoalUsesRealSystem) {
    const LoadedSystem start = homotopy_start();
    OpticalHomotopyModel m(start, {});
    SystemDescription s = m.start_system();
    const HomotopyGoal g0 = m.goal(s, m.start_point(), 1.0 / 14.0);
    s.ghost_plates[1].coefficients[1] = 1e-3;
    const HomotopyGoal g1 = m.goal(s, m.start_point(), 1.0 / 14.0);
    ASSERT_EQ(g0.tuples.size(), 1U);
    EXPECT_EQ(g0.max_rms, g1.max_rms); // Geisterplatten zählen nicht
    EXPECT_GT(g0.max_rms, 1.0);        // Einzellinse weit über der Beugungsgrenze
    EXPECT_FALSE(g0.reached);
    EXPECT_DOUBLE_EQ(g0.tuples[0].wavelength_nm, 546.07);
    EXPECT_EQ(g0.tuples[0].failed, 0);
    const HomotopyGoal g2 = m.goal(s, m.start_point(), 100.0);
    EXPECT_TRUE(g2.reached);
}

} // namespace
} // namespace mko::test
