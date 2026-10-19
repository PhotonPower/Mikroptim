#pragma once

// Abnahme der mechanischen Nebenbedingungen (M4-T3, CPU und GPU): Testsystem
// zweilinser_mechanik.json, Zustand der Bedingungen an einem Punkt und Penalty-Fortsetzung über
// die Gewichte der mechanischen Bedingungen (docs/optimierung.md, Abschnitt 2.5).

#include "fd_test_systems.hpp"
#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace mko::test {

// Penalty-Fortsetzung: Stufe s = 0 … kMechanicalStages − 1 rechnet mit dem Gewicht
// w_s = w_Datei · 10^(s − kMechanicalStages + 1) für alle mechanischen Bedingungen (10² … 10⁶),
// jede Stufe ab dem Ergebnis der vorigen.
inline constexpr int kMechanicalStages = 5;
inline constexpr int kMechanicalStageIterations = 1000;

// Zweilinser (zweilinser_mechanik.json): zwei Sammellinsen n = 1,6, f = 10 mm (Gleichung,
// Gewicht 10⁴/mm²), Pupillenradius 4 mm, Felder 0° und 2°; Variablen vier Krümmungen, zwei
// Mittendicken in [0,5; 2], Luftspalt in [0,1; 10], Arbeitsabstand.
inline FdProblem mechanical_problem() {
    return make_fd_problem(load_residual_system("zweilinser_mechanik.json"));
}

// Systembeschreibung mit dem Gewicht `weight` für alle mechanischen Bedingungen.
inline SystemDescription with_mechanical_weight(SystemDescription d, double weight) {
    for (ConstraintSpec& c : d.constraints.value().items) {
        if (is_mechanical(c.type)) {
            c.weight = weight;
        }
    }
    return d;
}

// Einstellungen der Optimierung: paraxiale Bedingungen und mechanische ohne Strahlbezug als
// Zusatzresiduen, die übrigen mechanischen als trace-abhängige Zusatzresiduen.
inline LmOptions mechanical_options(const SystemDescription& d, int max_iterations) {
    LmOptions o;
    o.max_iterations = max_iterations;
    MechanicalResiduals m = make_mechanical_residuals(make_mechanical_constraints(d));
    o.jacobian.extra = concat_extra_residuals(
        {make_paraxial_residuals(make_paraxial_constraints(d)), std::move(m.geometric)});
    o.jacobian.trace_extra = std::move(m.traced);
    return o;
}

// Zustand aller mechanischen Bedingungen am Punkt x (Extrema auf der CPU mit der Standard-
// quadratur der Optimierung).
struct MechanicalState {
    MechanicalConstraints constraints;
    std::vector<double> value;     // Istgröße (mechanical_value)
    std::vector<double> violation; // g (> 0: verletzt)

    // Größte Verletzung über alle Bedingungen eines Typs (−∞ ohne solche Bedingungen).
    double worst(ConstraintType type) const {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < violation.size(); ++k) {
            if (constraints.items[k].type == type) {
                w = std::max(w, violation[k]);
            }
        }
        return w;
    }
};

inline MechanicalState mechanical_state(const FdProblem& p, std::span<const double> x,
                                        const ResidualSettings& quadrature = {}) {
    MechanicalState s;
    s.constraints = make_mechanical_constraints(p.loaded.system);
    SystemBatchHost b = p.base;
    p.set.apply(b, 0, x);
    const SurfaceExtremaResult e =
        surface_extrema_cpu(b, 0, ResidualLayout(b.topology(), quadrature));
    const VariantExtrema ve = variant_extrema(e, 0);
    for (const MechanicalConstraint& c : s.constraints.items) {
        const double v = mechanical_value(c, b, 0, &ve);
        s.value.push_back(v);
        s.violation.push_back(mechanical_violation(c, v));
    }
    return s;
}

// Brennweite am Punkt x.
inline double focal_length_at(const FdProblem& p, std::span<const double> x) {
    SystemBatchHost b = p.base;
    p.set.apply(b, 0, x);
    return batch_paraxial(b, 0).focal_length;
}

struct MechanicalStage {
    double weight = 0.0;
    LmRunResult run;
    MechanicalState state; // am Ende der Stufe
};

// Optimierer: LmResult optimize(const LmOptions&, std::span<const double> x0) mit einem Lauf.
template <class Optimize>
std::vector<MechanicalStage> mechanical_continuation(const FdProblem& p, Optimize&& optimize) {
    double w_file = 0.0;
    for (const ConstraintSpec& c : p.loaded.system.constraints->items) {
        if (is_mechanical(c.type)) {
            w_file = c.weight_or_default();
        }
    }
    std::vector<double> x = p.x0;
    std::vector<MechanicalStage> stages;
    for (int s = 0; s < kMechanicalStages; ++s) {
        MechanicalStage stage;
        stage.weight = w_file * std::pow(10.0, s - kMechanicalStages + 1);
        const LmOptions o = mechanical_options(
            with_mechanical_weight(p.loaded.system, stage.weight), kMechanicalStageIterations);
        const LmResult res = optimize(o, std::span<const double>(x));
        stage.run = res.runs.at(0);
        x = stage.run.x;
        stage.state = mechanical_state(p, x);
        stages.push_back(stage);
    }
    return stages;
}

// Die mechanischen Typen des Testsystems.
inline const std::vector<ConstraintType>& mechanical_types() {
    static const std::vector<ConstraintType> types{
        ConstraintType::MaxSemiDiameter, ConstraintType::MaxCenterThickness,
        ConstraintType::MinEdgeThickness, ConstraintType::MaxIncidenceAngle};
    return types;
}

// Toleranz der Verletzung eines Typs nach der Fortsetzung (Längen in mm, Einfallswinkel in cos).
inline double mechanical_tolerance(ConstraintType type, double length_tol, double cos_tol) {
    return type == ConstraintType::MaxIncidenceAngle ? cos_tol : length_tol;
}

} // namespace mko::test
