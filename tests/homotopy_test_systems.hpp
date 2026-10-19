#pragma once

// Homotopie-Controller (M6-T3, CPU und GPU): Abnahmelauf NA 0,3 → 0,6 auf der Achse und
// Prüfungen (docs/optimierung.md, Abschnitt 7).

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/synth/beam_search.hpp"
#include "mikroptim/synth/homotopy.hpp"
#include "mikroptim/synth/optical_homotopy.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <ostream>
#include <utility>

namespace mko::test {

// Startsystem und Ziel der Abnahme.
inline constexpr const char* kHomotopyStartFile = "einzellinse_homotopie_start.json";
inline constexpr double kHomotopyTargetAperture = 0.6;

inline OpticalHomotopyOptions homotopy_optical_options(HomotopyOptimizer optimizer = {}) {
    OpticalHomotopyOptions o;
    o.lm.max_iterations = 500;
    o.optimizer = std::move(optimizer);
    o.meniscus.curvature_variables.min = -1.0;
    o.meniscus.curvature_variables.max = 1.0;
    return o;
}

inline HomotopyOptions homotopy_options() {
    HomotopyOptions h;
    h.target_aperture = kHomotopyTargetAperture;
    return h;
}

struct HomotopyRun {
    HomotopyResult result;
    SystemDescription realized; // am Endpunkt, ohne Vorhalt
};

inline HomotopyRun homotopy_acceptance_run(HomotopyOptimizer optimizer = {}, bool stream = false) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyModel model(start, homotopy_optical_options(std::move(optimizer)));
    HomotopyOptions h = homotopy_options();
    if (stream) {
        write_homotopy_step_header(std::cout);
        h.on_step = [](const HomotopyStep& s) {
            write_homotopy_step_line(s, std::cout);
            std::cout.flush();
        };
    }
    HomotopyRun run;
    run.result = run_homotopy(model, model.start_system(), model.start_point(), h);
    run.realized = model.realize(run.result.system, run.result.point, false);
    return run;
}

// ---------------------------------------------------------------------------------------------
// Kurzlauf (CI): NA 0,3 → 0,31 mit großzügiger Schwelle, wenigen Zyklen und höchstens zwei
// Elementen. Durchläuft Penalty-Fortsetzung, Aperturschritt, ε- und μ-Fortsetzung, Zielprüfung,
// ein Einfügen (zwei Zweige, zwei Dickenstufen) und den Abbruch bei der höchsten Elementzahl.

inline constexpr double kHomotopySmokeAperture = 0.31;

inline HomotopyResult homotopy_smoke_run(HomotopyOptimizer optimizer = {}) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o = homotopy_optical_options(std::move(optimizer));
    o.lm.max_iterations = 40;
    o.element_stages = 2;
    OpticalHomotopyModel model(start, o);
    HomotopyOptions h;
    h.target_aperture = kHomotopySmokeAperture;
    h.merit_factor = 10.0; // F_max = (10/14)² Wellen²: die Einzellinse mit Geisterplatten genügt
    h.max_elements = 2;
    return run_homotopy(model, model.start_system(), model.start_point(), h);
}

inline void check_homotopy_smoke(const HomotopyResult& r) {
    EXPECT_EQ(r.point.aperture, kHomotopySmokeAperture);
    EXPECT_EQ(r.insertions, 1);
    EXPECT_EQ(r.elements, 2);
    EXPECT_EQ(r.status, HomotopyStatus::MaxElements) << r.message;
    bool aperture = false;
    bool stage = false;
    bool goal = false;
    for (const HomotopyStep& s : r.steps) {
        aperture = aperture || s.kind == HomotopyStepKind::Aperture;
        stage = stage || s.kind == HomotopyStepKind::InsertionStage;
        goal = goal || s.kind == HomotopyStepKind::Goal;
        if (s.solved) {
            EXPECT_TRUE(std::isfinite(s.solve.merit));
            EXPECT_EQ(s.solve.failed_rays, 0);
        }
    }
    EXPECT_TRUE(aperture);
    EXPECT_TRUE(stage);
    EXPECT_TRUE(goal);
}

// Gleiche Entscheidungen (Art, Punkt, Annahme, Aktion, Zweig, Stufe, Elemente) und Merit
// relativ innerhalb `merit_tol` bis zum ersten Einfügen; merit_tol = 0: alles bitgleich
// (Determinismus, auch Merit, Zyklen, Texte). Nach dem ersten Einfügen laufen die Optimierungen
// des Kurzlaufs mit 40 Zyklen unkonvergiert; dort weichen GPU und CPU in F um bis zu ~1 % ab
// (Rundungsunterschiede verstärkt über die Zyklen), verglichen werden nur die Entscheidungen.
inline void expect_same_decisions(const HomotopyResult& a, const HomotopyResult& b,
                                  double merit_tol) {
    EXPECT_EQ(a.status, b.status);
    ASSERT_EQ(a.steps.size(), b.steps.size());
    bool inserted = false;
    for (std::size_t i = 0; i < a.steps.size(); ++i) {
        const HomotopyStep& x = a.steps[i];
        const HomotopyStep& y = b.steps[i];
        EXPECT_EQ(x.kind, y.kind) << i;
        EXPECT_EQ(x.point, y.point) << i;
        EXPECT_EQ(x.delta, y.delta) << i;
        EXPECT_EQ(x.accepted, y.accepted) << i;
        EXPECT_EQ(x.action, y.action) << i;
        EXPECT_EQ(x.branch, y.branch) << i;
        EXPECT_EQ(x.stage, y.stage) << i;
        EXPECT_EQ(x.elements, y.elements) << i;
        EXPECT_EQ(x.solved, y.solved) << i;
        EXPECT_EQ(x.goal.reached, y.goal.reached) << i;
        inserted = inserted || x.kind == HomotopyStepKind::Insertion;
        if (merit_tol == 0.0) {
            EXPECT_EQ(x.operation, y.operation) << i;
            EXPECT_EQ(x.solve.merit, y.solve.merit) << i;
            EXPECT_EQ(x.solve.iterations, y.solve.iterations) << i;
            EXPECT_EQ(x.solve.rms, y.solve.rms) << i;
            EXPECT_EQ(x.goal.max_rms, y.goal.max_rms) << i;
        } else if (x.solved && !inserted) {
            EXPECT_LE(std::abs(x.solve.merit - y.solve.merit), merit_tol * std::abs(x.solve.merit))
                << i;
        }
    }
    if (merit_tol == 0.0) {
        EXPECT_EQ(a.message, b.message);
    }
}

// ---------------------------------------------------------------------------------------------
// Strahlsuche (M6-T4): Kurzlauf wie homotopy_smoke_run mit B Kandidaten.

struct BeamSmokeSettings {
    int width = 2;
    bool group = true;
    int starts = 0;
    bool prefilter = false;
    std::uint64_t seed = 0;
};

inline BeamResult beam_smoke_run(const BeamSmokeSettings& s, HomotopyOptimizer optimizer = {}) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o = homotopy_optical_options(std::move(optimizer));
    o.lm.max_iterations = 40;
    o.element_stages = 2;
    o.group_runs = s.group;
    OpticalHomotopyModel model(start, o);
    BeamOptions b;
    b.homotopy.target_aperture = kHomotopySmokeAperture;
    b.homotopy.merit_factor = 10.0;
    b.homotopy.max_elements = 2;
    b.beam_width = s.width;
    b.perturbed_starts = s.starts;
    b.prefilter = s.prefilter;
    b.seed = s.seed;
    return run_beam_search(model, model.start_system(), model.start_point(), b);
}

// Abnahmelauf mit Strahlsuche: wie homotopy_acceptance_run mit B Kandidaten.
inline BeamResult beam_acceptance_run(int width, int alternatives, HomotopyOptimizer optimizer = {},
                                      bool stream = false) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyModel model(start, homotopy_optical_options(std::move(optimizer)));
    BeamOptions b;
    b.homotopy = homotopy_options();
    b.homotopy.max_steps = 1000;
    b.beam_width = width;
    b.alternatives = alternatives;
    std::function<void(const BeamStep&)> on_step;
    if (stream) {
        write_beam_step_header(std::cout);
        on_step = [](const BeamStep& s) {
            write_beam_step_line(s, std::cout);
            std::cout.flush();
        };
    }
    return run_beam_search(model, model.start_system(), model.start_point(), b, on_step);
}

inline void print_homotopy(const HomotopyResult& r, std::ostream& out) {
    write_homotopy_summary(r, out);
}

inline void check_homotopy_acceptance(const HomotopyResult& r) {
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.point.aperture, kHomotopyTargetAperture);
    EXPECT_TRUE(r.goal.reached);
    EXPECT_LE(r.goal.max_rms, 1.0 / 14.0);
}

} // namespace mko::test
