#pragma once

// Abnahme „Optimierung bleibt innerhalb der Hülle“ (M6-T1, CPU und GPU): Testsystem
// einzellinse_glasvariable.json mit Glasvariable, frei erfundene Testgläser
// (data/glass/synthetisch.json) und Penalty-Fortsetzung über das Gewicht der Glas-Hülle
// (docs/optimierung.md, Abschnitt 2.4).

#include "fd_test_systems.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace mko::test {

// Penalty-Fortsetzung: Stufe s = 0 … kGlassStages − 1 rechnet mit dem Gewicht
// w_s = w_Datei · 10^(s − kGlassStages + 1) (w_Datei = 10⁸, also 10² … 10⁸); jede Stufe startet
// am Ergebnis der vorigen.
inline constexpr int kGlassStages = 7;
inline constexpr int kGlassStageIterations = 1000;
// Kontrollfall ohne Strafe: Zyklen und Mindestabstand des Ergebnisses von der Hülle (normiert).
// Ohne Dispersion (ν₁ = ν₂ = 0) liegt das Glas um mehr als die halbe Spannweite der ν₁ der
// Katalog-Gläser außerhalb; 0,1 ist eine Eigenschaft des Testsystems, keine Rechentoleranz.
inline constexpr int kGlassFreeIterations = 300;
inline constexpr double kGlassFreeDistance = 0.1;

// System nach der Umwandlung des Startglases (N-BK7 → Buchdahl, 3 Terme), Batch, Variablen
// (c[1], c[2], n0[1], nu1[1], nu2[1], w), Hülle aller Gläser der Kataloge (basis.json und
// synthetisch.json, 36 + 3 Gläser, SYN-UV und SYN-IR ausgeschlossen).
struct GlassProblem {
    FdProblem p;
    std::vector<GlassConversion> conversions;
    GlassHull hull;
};

inline GlassProblem glass_problem() {
    const LoadedSystem raw = load_residual_system("einzellinse_glasvariable.json");
    SystemDescription d = raw.system;
    GlassProblem g;
    g.conversions = convert_glass_variables(d, raw.catalog);
    g.p = make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
    g.hull = make_glass_hull(g.p.loaded.catalog);
    return g;
}

// Index des Eintrags glass_hull in constraints.items.
inline std::size_t glass_hull_item(const SystemDescription& system) {
    const std::vector<ConstraintSpec>& items = system.constraints.value().items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (items[i].type == ConstraintType::GlassHull) {
            return i;
        }
    }
    throw std::logic_error("Testsystem ohne glass_hull");
}

// Strafe der Datei mit dem Gewicht `weight`.
inline GlassHullPenalty glass_penalty(const GlassProblem& g, double weight) {
    SystemDescription d = g.p.loaded.system;
    d.constraints->items[glass_hull_item(d)].weight = weight;
    return make_glass_hull_penalty(d, g.p.set, g.hull);
}

// Zusatzresiduen: Brennweite der Datei, dahinter die Glas-Hülle mit Gewicht `weight` (0: ohne
// Strafe, das Residuum bleibt 0).
inline ExtraResiduals glass_extra(const GlassProblem& g, double weight) {
    return concat_extra_residuals(
        {make_paraxial_residuals(make_paraxial_constraints(g.p.loaded.system)),
         make_glass_hull_residuals(glass_penalty(g, weight))});
}

// Glaskoordinaten (n₀, ν₁, ν₂) am Punkt x (Komponenten 2, 3, 4 des Variablenvektors).
inline Point3 glass_at(const GlassProblem& g, std::span<const double> x) {
    SystemBatchHost b = g.p.base;
    g.p.set.apply(b, 0, x);
    return batch_glass_coordinates(b, 0, g.p.set[2].batch_surface);
}

struct GlassStage {
    double weight = 0.0;
    LmRunResult run;
    Point3 glass{};         // am Ende der Stufe
    double distance = 0.0;  // Abstand zur Hülle (normiert)
    double ray_merit = 0.0; // F ohne Strafe (Wellenfront + Brennweite)
};

// F ohne Strafterm am Punkt x: Merit eines Laufs ohne Zyklus.
template <class Optimize>
double merit_without_penalty(const GlassProblem& g, Optimize&& optimize,
                             std::span<const double> x) {
    LmOptions o;
    o.max_iterations = 0;
    o.jacobian.extra = glass_extra(g, 0.0);
    return optimize(o, x).runs.at(0).history.front();
}

// Optimierer: LmResult optimize(const LmOptions&, std::span<const double> x0) mit einem Lauf.
template <class Optimize>
std::vector<GlassStage> glass_continuation(const GlassProblem& g, Optimize&& optimize) {
    const double w_file = g.p.loaded.system.constraints->items[glass_hull_item(g.p.loaded.system)]
                              .weight_or_default();
    std::vector<double> x = g.p.x0;
    std::vector<GlassStage> stages;
    for (int s = 0; s < kGlassStages; ++s) {
        GlassStage stage;
        stage.weight = w_file * std::pow(10.0, s - kGlassStages + 1);
        LmOptions o;
        o.max_iterations = kGlassStageIterations;
        o.jacobian.extra = glass_extra(g, stage.weight);
        const LmResult res = optimize(o, std::span<const double>(x));
        stage.run = res.runs.at(0);
        x = stage.run.x;
        stage.glass = glass_at(g, x);
        stage.distance = g.hull.distance(stage.glass);
        stage.ray_merit = merit_without_penalty(g, optimize, x);
        stages.push_back(stage);
    }
    return stages;
}

} // namespace mko::test
