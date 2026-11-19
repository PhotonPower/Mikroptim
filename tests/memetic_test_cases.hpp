#pragma once

// Testfälle der memetischen Suche (M11-T7, CPU und GPU): feste Systeme, Einstellungen und
// Startwerte der Abnahme und die Vergleichsrechnung mit perturbed_starts bei gleichem Budget.

#include "lm_test_systems.hpp"
#include "mikroptim/optim/memetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace mko::test {

struct MemeticCase {
    FdProblem p;
    LmOptions lm;
    MemeticOptions options;
};

// Mehrlinser mit Brennweitenbedingung (mehrlinser_lm_problem, P = 13): CMA-ES mit λ = 11,
// 8 Generationen à 20 LM-Zyklen pro Individuum (Budget 1.760 Zyklen), σ₀ = 0,1, Startwert 1;
// Lamarck, Elitismus, Schrittweitenpfad aus den verfeinerten Punkten (Standard).
inline MemeticCase mehrlinser_memetic_case() {
    MemeticCase c{mehrlinser_lm_problem(kMehrlinserFocalWeight), {}, {}};
    c.lm.jacobian.extra = paraxial_extra(c.p);
    c.options.method = GlobalSearchMethod::Cma;
    c.options.generations = 8;
    c.options.lm_iterations = 20;
    c.options.sigma = 0.1;
    c.options.seed = 1;
    return c;
}

struct BaselineResult {
    double best_merit = std::numeric_limits<double>::infinity();
    std::int64_t best_failed_rays = 0;
    std::int64_t cycles = 0;
    int runs = 0;
};

// Vergleich: K = ⌈budget/L⌉ Starts aus perturbed_starts (gleichverteilt, relativ σ₀, Startwert
// wie die Suche), gemeinsam mit L Zyklen pro Lauf optimiert; bester Lauf nach memetic_better.
template <class Optimizer>
BaselineResult perturbed_starts_baseline(const MemeticCase& c, std::int64_t budget,
                                         const Optimizer& optimize) {
    BaselineResult b;
    const std::int64_t L = c.options.lm_iterations;
    b.runs = static_cast<int>((budget + L - 1) / L);
    const std::vector<double> x0 =
        perturbed_starts(c.p.set, c.p.x0, b.runs, c.options.sigma, c.options.seed);
    LmOptions lo = c.lm;
    lo.max_iterations = c.options.lm_iterations;
    const LmResult r = optimize(c.p.set, c.p.base, x0, lo);
    const LmRunResult* best = nullptr;
    for (const LmRunResult& run : r.runs) {
        if (best == nullptr || memetic_better(run, *best)) {
            best = &run;
        }
        b.cycles += run.iterations;
    }
    if (best != nullptr) {
        b.best_merit = best->merit;
        b.best_failed_rays = best->failed_rays;
    }
    return b;
}

// Zwei Suchen bitgleich (alle Individuen, Generationen, Zähler).
inline bool same_memetic(const MemeticResult& a, const MemeticResult& b) {
    if (a.individuals.size() != b.individuals.size() ||
        a.generations.size() != b.generations.size() || a.best != b.best || a.cycles != b.cycles ||
        a.budget != b.budget || a.lambda != b.lambda) {
        return false;
    }
    for (std::size_t i = 0; i < a.individuals.size(); ++i) {
        const MemeticIndividual& x = a.individuals[i];
        const MemeticIndividual& y = b.individuals[i];
        if (x.generation != y.generation || x.candidate != y.candidate ||
            x.start.size() != y.start.size() || !same_run(x.run, y.run)) {
            return false;
        }
        for (std::size_t k = 0; k < x.start.size(); ++k) {
            if (!same_bits(x.start[k], y.start[k])) {
                return false;
            }
        }
    }
    for (std::size_t g = 0; g < a.generations.size(); ++g) {
        const MemeticGeneration& x = a.generations[g];
        const MemeticGeneration& y = b.generations[g];
        if (x.cycles != y.cycles || x.prefilter_kept != y.prefilter_kept || x.best != y.best ||
            !same_bits(x.sigma, y.sigma) || !same_bits(x.best_merit, y.best_merit)) {
            return false;
        }
    }
    return true;
}

} // namespace mko::test
