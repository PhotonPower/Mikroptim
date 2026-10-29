// Homotopie-Controller (M6-T3): Steuerlogik mit einem synthetischen Modell (Schrittweite,
// Rücknahme, Einfügen bei Δ < Δ_min und Bedarf, Fortsetzungen, Abbruch, Protokoll) und die
// Bausteine des optischen Modells ohne lange Optimierung (docs/optimierung.md, Abschnitt 7).

#include "homotopy_mock.hpp"
#include "insertion_test_systems.hpp"
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
