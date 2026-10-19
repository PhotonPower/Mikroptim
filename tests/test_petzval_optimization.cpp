// Abnahme der Petzval-Nebenbedingung (M4-T4) auf der CPU: Zweilinser mit Brennweiten- und
// Petzval-Bedingung, LM mit Penalty-Fortsetzung über das Gewicht → |κ_P| ≤ κ_max bei aktiver
// Bedingung, Brennweite eingehalten (docs/optimierung.md, Abschnitt 2.3).

#include "petzval_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mko::test {
namespace {

TEST(PetzvalOptimization, CpuReachesLimitWithActiveConstraint) {
    const FdProblem p = petzval_problem();
    const ConstraintSpec& spec = p.loaded.system.constraints->items[petzval_item(p.loaded.system)];
    const double kappa_max = spec.max_curvature.value();
    const double f_target = p.loaded.system.target->focal_length.value();

    // Start: Bedingung deutlich verletzt
    const ParaxialData start = paraxial_at(p, p.x0);
    std::printf("[CPU] Start: κ_P = %.10e 1/mm, f = %.10f mm, κ_max = %.4g 1/mm\n",
                start.petzval_curvature, start.focal_length, kappa_max);
    EXPECT_GT(std::abs(start.petzval_curvature), 2.0 * kappa_max);

    const std::vector<PetzvalStage> stages =
        petzval_continuation(p, [&](const LmOptions& o, std::span<const double> x0) {
            return optimize_lm_cpu(p.set, p.base, x0, o);
        });
    ASSERT_EQ(stages.size(), static_cast<std::size_t>(kPetzvalStages));
    for (const PetzvalStage& s : stages) {
        const double excess = std::abs(s.paraxial.petzval_curvature) - kappa_max;
        std::printf("[CPU]   w = %8.1e mm²: F %.6e -> %.10e, %3d Zyklen, %-16s |κ_P| − κ_max = "
                    "%+.3e (relativ %+.3e), w·Überschreitung = %.3e, f − f_s = %+.3e mm\n",
                    s.weight, s.run.history.front(), s.run.merit, s.run.iterations,
                    to_string(s.run.status), excess, excess / kappa_max, s.weight * excess,
                    s.paraxial.focal_length - f_target);
        EXPECT_TRUE(lm_converged(s.run.status)) << s.weight << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
        EXPECT_TRUE(p.set.feasible(s.run.x));
    }

    // Ende: |κ_P| ≤ κ_max bis auf die Restverletzung der Strafe (relativ < 1e-6), die Bedingung
    // ist aktiv (Punkt auf der Grenze) und die Brennweite bleibt eingehalten.
    const PetzvalStage& last = stages.back();
    const double kappa = std::abs(last.paraxial.petzval_curvature);
    EXPECT_LE(kappa, kappa_max * (1.0 + kPetzvalPenaltyRelative));
    EXPECT_GE(kappa, kappa_max * (1.0 - kPetzvalPenaltyRelative));
    EXPECT_LT(std::abs(last.paraxial.focal_length - f_target), kPetzvalFocalLength);
    // Restverletzung sinkt mit dem Gewicht (Strafe ∝ 1/w)
    const double first_excess = std::abs(stages.front().paraxial.petzval_curvature) - kappa_max;
    EXPECT_LT(kappa - kappa_max, 1e-3 * first_excess);

    // Die Bedingung bindet: ohne sie läuft der LM vom Ergebnis weg zu größerem |κ_P| und
    // kleinerer Merit (Lagrange-Multiplikator > 0).
    LmOptions free_options;
    free_options.max_iterations = 20;
    free_options.jacobian.extra = petzval_extra(p, 0.0);
    const LmRunResult free_run = optimize_lm_cpu(p.set, p.base, last.run.x, free_options).runs[0];
    const ParaxialData free_paraxial = paraxial_at(p, free_run.x);
    std::printf("[CPU] ohne Petzval-Bedingung (20 Zyklen vom Ergebnis): F %.10e -> %.10e, κ_P = "
                "%.10e 1/mm\n",
                free_run.history.front(), free_run.merit, free_paraxial.petzval_curvature);
    EXPECT_LT(free_run.merit, free_run.history.front());
    EXPECT_GT(std::abs(free_paraxial.petzval_curvature), kappa_max * 1.01);
}

} // namespace
} // namespace mko::test
