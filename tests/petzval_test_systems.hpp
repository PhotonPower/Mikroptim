#pragma once

// Abnahme der Petzval-Nebenbedingung (M4-T4, CPU und GPU): Testsystem zweilinser_petzval.json
// und Penalty-Fortsetzung über das Gewicht der Bedingung (docs/optimierung.md, Abschnitt 2.3).

#include "fd_test_systems.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace mko::test {

// Penalty-Fortsetzung: Stufe s = 0 … kPetzvalStages − 1 rechnet mit dem Gewicht
// w_s = w_Datei · 10^(s − kPetzvalStages + 1), also sechs Dekaden unter dem Gewicht der Datei
// beginnend und mit ihm endend; jede Stufe startet am Ergebnis der vorigen.
inline constexpr int kPetzvalStages = 7;
// Zyklen pro Stufe (die erste Stufe vom Startsystem braucht auf der CPU 461, die übrigen 8–18)
inline constexpr int kPetzvalStageIterations = 1000;

// Zweilinser (zweilinser_petzval.json): Sammellinse n = 1,8, Luftspalt, Zerstreuungslinse CaF2,
// f = 20 mm (Gleichung), |κ_P| ≤ 0,01/mm; Gewichte 0,01/mm² bzw. 1.000 mm² für Residuen in mm,
// mit 1/(F·L·λ_ref²) = 1/(2·(546,07 nm)²) auf Residuen in Wellen umgerechnet (M4-T1); Variablen
// alle vier Krümmungen in [−0,12; 0,12], Luftspalt in [1; 12] und Arbeitsabstand in [4; 40].
inline FdProblem petzval_problem() {
    return make_fd_problem(load_residual_system("zweilinser_petzval.json"));
}

// Index des Petzval-Eintrags in constraints.items.
inline std::size_t petzval_item(const SystemDescription& system) {
    const std::vector<ConstraintSpec>& items = system.constraints.value().items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (items[i].type == ConstraintType::Petzval) {
            return i;
        }
    }
    throw std::logic_error("Testsystem ohne Petzval-Bedingung");
}

// Zusatzresiduen der Datei mit dem Gewicht `weight` für die Petzval-Bedingung.
inline ExtraResiduals petzval_extra(const FdProblem& p, double weight) {
    SystemDescription d = p.loaded.system;
    d.constraints->items[petzval_item(d)].weight = weight;
    return make_paraxial_residuals(make_paraxial_constraints(d));
}

// Paraxialdaten bei λ_ref am Punkt x.
inline ParaxialData paraxial_at(const FdProblem& p, std::span<const double> x) {
    SystemBatchHost b = p.base;
    p.set.apply(b, 0, x);
    return batch_paraxial(b, 0);
}

struct PetzvalStage {
    double weight = 0.0;
    LmRunResult run;
    ParaxialData paraxial; // am Ende der Stufe
};

// Optimierer: LmResult optimize(const LmOptions&, std::span<const double> x0) mit einem Lauf.
template <class Optimize>
std::vector<PetzvalStage> petzval_continuation(const FdProblem& p, Optimize&& optimize) {
    const double w_file =
        p.loaded.system.constraints->items[petzval_item(p.loaded.system)].weight_or_default();
    std::vector<double> x = p.x0;
    std::vector<PetzvalStage> stages;
    for (int s = 0; s < kPetzvalStages; ++s) {
        PetzvalStage stage;
        stage.weight = w_file * std::pow(10.0, s - kPetzvalStages + 1);
        LmOptions o;
        o.max_iterations = kPetzvalStageIterations;
        o.jacobian.extra = petzval_extra(p, stage.weight);
        const LmResult res = optimize(o, std::span<const double>(x));
        stage.run = res.runs.at(0);
        x = stage.run.x;
        stage.paraxial = paraxial_at(p, x);
        stages.push_back(stage);
    }
    return stages;
}

} // namespace mko::test
