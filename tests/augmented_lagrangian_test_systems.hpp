#pragma once

// Abnahme des Augmented Lagrangian (M11-T2, CPU und GPU): äußere Schleife über LM-Läufe mit
// festem Gewicht und Fortschreibung der Verschiebungen θ nach jedem Lauf, auf den Systemen der
// Petzval- (M4-T4) und Mechanik-Abnahme (M4-T3); docs/optimierung.md, Abschnitt 2.6.

#include "mechanical_test_systems.hpp"
#include "mikroptim/optim/augmented_lagrangian.hpp"
#include "petzval_test_systems.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace mko::test {

// Feste Gewichte der Abnahme (Messung der Gewichte: docs/optimierung.md, Abschnitt 2.6).
// Petzval: w_Datei · 10⁻⁴ = 1,677·10⁵ mm² (drittes Gewicht der heutigen Fortsetzung); mit dem
// kleinsten (10⁻⁶) erreicht die CPU die Toleranz in drei Läufen, auf der GPU bleibt g wegen der
// Genauigkeit des inneren LM knapp darüber stehen; ab 10⁻³ läuft der erste Lauf vom Start ins
// Iterationslimit. Mechanik: w_Datei · 10⁻¹ = 10⁵/mm²; mit dem kleinsten Gewicht der Fortsetzung
// (10²/mm²) konvergieren die Multiplikatoren zu langsam (nach 30 Läufen noch g = 0,024 mm).
inline constexpr double kAlPetzvalWeightFactor = 1e-4;
inline constexpr double kAlMechanicalWeightFactor = 1e-1;
// Höchstens so viele LM-Läufe; Abbruch, wenn max √w |Δθ| ≤ T.
inline constexpr int kAlRounds = 30;

// Abbruchschwelle T = √w · δ aus der Toleranz δ der Verletzung: Bei einer verletzten Bedingung
// (g > 0) ist Δθ = g (ohne Schranke), der Abbruch garantiert also g ≤ δ am Endpunkt.
inline double al_tolerance(double weight, double violation_tolerance) {
    return std::sqrt(weight) * violation_tolerance;
}

struct AlRound {
    LmRunResult run;
    ShiftUpdate update; // Fortschreibung nach dem Lauf (Verletzungen am Endpunkt)
};

struct AlResult {
    std::vector<AlRound> rounds;
    std::vector<double> x;    // Endpunkt
    SystemDescription system; // mit den Verschiebungen nach dem letzten Lauf
    int iterations = 0;       // LM-Zyklen aller Läufe
};

// Äußere Schleife: Lauf k optimiert ab dem Ergebnis von Lauf k − 1 mit den Verschiebungen aus
// `system` (make_options(system) baut die LmOptions), danach θ ← fortgeschrieben. Abbruch nach
// `rounds` Läufen oder wenn max √w |Δθ| ≤ tolerance.
template <class MakeOptions, class Optimize>
AlResult al_loop(const FdProblem& p, SystemDescription system, MakeOptions&& make_options,
                 Optimize&& optimize, double tolerance, int rounds = kAlRounds) {
    AlResult r;
    std::vector<double> x = p.x0;
    for (int k = 0; k < rounds; ++k) {
        const LmOptions o = make_options(system);
        AlRound round;
        round.run = optimize(o, std::span<const double>(x)).runs.at(0);
        x = round.run.x;
        SystemBatchHost b = p.base;
        p.set.apply(b, 0, x);
        round.update = shift_update(system, b, 0, o.jacobian.quadrature);
        apply_shift_update(system, round.update);
        r.iterations += round.run.iterations;
        const bool done = round.update.max_change <= tolerance;
        r.rounds.push_back(std::move(round));
        if (done) {
            break;
        }
    }
    r.x = std::move(x);
    r.system = std::move(system);
    return r;
}

// Petzval-System mit festem Gewicht der Petzval-Bedingung.
inline SystemDescription petzval_al_system(const FdProblem& p) {
    SystemDescription d = p.loaded.system;
    ConstraintSpec& c = d.constraints->items[petzval_item(d)];
    c.weight = c.weight_or_default() * kAlPetzvalWeightFactor;
    return d;
}

inline LmOptions petzval_al_options(const SystemDescription& d) {
    LmOptions o;
    o.max_iterations = kPetzvalStageIterations;
    o.jacobian.extra = make_paraxial_residuals(make_paraxial_constraints(d));
    return o;
}

// Mechanik-System mit festem Gewicht aller mechanischen Bedingungen.
inline SystemDescription mechanical_al_system(const FdProblem& p) {
    double w_file = 0.0;
    for (const ConstraintSpec& c : p.loaded.system.constraints->items) {
        if (is_mechanical(c.type)) {
            w_file = c.weight_or_default();
        }
    }
    return with_mechanical_weight(p.loaded.system, w_file * kAlMechanicalWeightFactor);
}

inline LmOptions mechanical_al_options(const SystemDescription& d) {
    return mechanical_options(d, kMechanicalStageIterations);
}

// LM-Zyklen der heutigen Penalty-Fortsetzung (Summe über die Stufen).
inline int stage_iterations(const std::vector<PetzvalStage>& stages) {
    int n = 0;
    for (const PetzvalStage& s : stages) {
        n += s.run.iterations;
    }
    return n;
}
inline int stage_iterations(const std::vector<MechanicalStage>& stages) {
    int n = 0;
    for (const MechanicalStage& s : stages) {
        n += s.run.iterations;
    }
    return n;
}

} // namespace mko::test
