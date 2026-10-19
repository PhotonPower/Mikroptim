// Abnahme der L1-Strafe der Geisterkoeffizienten und des Diagnoseberichts (M4-T6) auf der CPU:
// Exaktheit (ein System, das sein Ziel alleine erreicht, endet mit allen |a| < 1e-8) und Bedarf
// (Einzellinse bei NA 0,5 aktiviert Geisterkoeffizienten, der Bericht nennt Lage, Ordnung und
// Glas); docs/optimierung.md, Abschnitte 3 und 5.

#include "ghost_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mko::test {
namespace {

TEST(GhostOptimization, ExactnessCpu) {
    const FdProblem p = ghost_exact_problem();
    const GhostPenalty pen = ghost_penalty_of(p.loaded.system);
    const ExactnessResult e = ghost_exactness(p, pen, [&](const LmOptions& o) {
        return optimize_lm_cpu(p.set, p.base, p.x0, o).runs.at(0);
    });
    print_exactness("CPU", p, e);
    EXPECT_TRUE(lm_converged(e.run.status)) << to_string(e.run.status);
    EXPECT_EQ(e.run.failed_rays, 0);
    // Abnahme: alle |a| < 1e-8
    EXPECT_LT(max_ghost(p.set, e.run.x), kGhostExactCoefficient);
    // μ oberhalb der Schwelle: |g_k| < μ w_k für alle Koeffizienten
    EXPECT_LT(e.threshold_ratio, 1.0);
    // Restkoeffizienten wie abgeschätzt: a_k = −ε s/√(1 − s²), s = g_k/(μ w_k) (auf 1e-4 ε)
    EXPECT_LE(e.max_stationarity_error, kGhostStationarity);
    // Dieselbe Merit wie das reale System allein bis auf den Gewinn aus der Glättung, der der
    // Vorhersage Σ ε g_k²/(2 μ w_k) folgt (bei exakter L1-Strafe wäre er 0).
    EXPECT_GE(e.smoothing_gain, 0.0);
    EXPECT_LE(std::abs(e.smoothing_gain - e.predicted_gain),
              kGhostSmoothingGain * e.predicted_gain);
}

// Die Restkoeffizienten skalieren mit ε |g|/(μ w_k): Mit ε = 1e-6 statt 1e-7 liegen sie
// zehnmal höher (über 1e-8), mit μ = 100 statt 13,6 wieder darunter (Abschnitt 3).
TEST(GhostOptimization, ResidualCoefficientsScaleWithSmoothingAndWeight) {
    const FdProblem p = ghost_exact_problem();
    struct Case {
        double mu;
        double smoothing;
        bool below;
    };
    for (const Case c : {Case{13.6, 1e-6, false}, Case{100.0, 1e-6, true}}) {
        const GhostPenalty pen = ghost_penalty_of(p.loaded.system, c.mu, c.smoothing);
        const ExactnessResult e = ghost_exactness(p, pen, [&](const LmOptions& o) {
            return optimize_lm_cpu(p.set, p.base, p.x0, o).runs.at(0);
        });
        std::printf(
            "[CPU] μ = %g, ε = %g: max|a| = %.3e, max|g|/(μ w) = %.3e, Stationarität %.2e\n", c.mu,
            c.smoothing, max_ghost(p.set, e.run.x), e.threshold_ratio, e.max_stationarity_error);
        EXPECT_TRUE(lm_converged(e.run.status)) << to_string(e.run.status);
        EXPECT_EQ(max_ghost(p.set, e.run.x) < kGhostExactCoefficient, c.below) << c.mu;
        EXPECT_LE(e.max_stationarity_error, kGhostStationarity);
    }
}

TEST(GhostOptimization, NeedCpu) {
    const FdProblem p = ghost_need_problem();
    const NeedResult n = ghost_need(p, [&](const FdProblem& q, const LmOptions& o) {
        return optimize_lm_cpu(q.set, q.base, q.x0, o).runs.at(0);
    });
    print_need("CPU", p, n);
    check_need(p, n);
}

} // namespace
} // namespace mko::test
