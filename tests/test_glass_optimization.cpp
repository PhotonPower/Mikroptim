// Abnahme „Optimierung bleibt innerhalb der Hülle“ (M6-T1) auf der CPU: Einzellinse mit
// Glasvariable, deren Farblängsfehler nur ohne Dispersion verschwindet (weit außerhalb der Hülle
// der Katalog-Gläser). Mit der Strafe glass_hull und Penalty-Fortsetzung endet das Glas auf dem
// Rand der Hülle, ohne Strafe weit außerhalb (docs/optimierung.md, Abschnitt 2.4).

#include "glass_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace mko::test {
namespace {

TEST(GlassOptimization, CpuStaysInsideHull) {
    const GlassProblem g = glass_problem();
    const auto optimize = [&](const LmOptions& o, std::span<const double> x0) {
        return optimize_lm_cpu(g.p.set, g.p.base, x0, o);
    };
    ASSERT_EQ(g.hull.dimension(), 3);
    ASSERT_EQ(g.conversions.size(), 1u);
    const Point3 start = glass_at(g, g.p.x0);
    std::printf("[CPU] Hülle: %zu Gläser, %zu Ecken; Start (N-BK7) n0 = %.7f nu1 = %.5e nu2 = "
                "%.5e, Abstand %.3e\n",
                g.hull.points().size(), g.hull.hull().vertices.size(), start[0], start[1], start[2],
                g.hull.distance(start));
    EXPECT_EQ(g.hull.distance(start), 0.0); // N-BK7 gehört zur Hülle

    // Kontrollfall ohne Strafe (Gewicht 0): das Glas läuft weit aus der Hülle (ν₁, ν₂ → 0)
    LmOptions free_options;
    free_options.max_iterations = kGlassFreeIterations;
    free_options.jacobian.extra = glass_extra(g, 0.0);
    const LmRunResult free_run = optimize(free_options, g.p.x0).runs.at(0);
    const Point3 free_glass = glass_at(g, free_run.x);
    const double free_distance = g.hull.distance(free_glass);
    std::printf("[CPU] ohne Strafe: F %.6e -> %.10e, %d Zyklen, %s, n0 = %.6f nu1 = %.5e nu2 = "
                "%.5e, Abstand %.4e\n",
                free_run.history.front(), free_run.merit, free_run.iterations,
                to_string(free_run.status), free_glass[0], free_glass[1], free_glass[2],
                free_distance);
    EXPECT_EQ(free_run.failed_rays, 0);
    EXPECT_GT(free_distance, kGlassFreeDistance);

    // Mit Strafe: Penalty-Fortsetzung 10² … 10⁸
    const std::vector<GlassStage> stages = glass_continuation(g, optimize);
    ASSERT_EQ(stages.size(), static_cast<std::size_t>(kGlassStages));
    for (const GlassStage& s : stages) {
        std::printf("[CPU]   w = %8.1e: F %.6e -> %.10e (ohne Strafe %.10e), %4d Zyklen, %-16s "
                    "n0 = %.8f nu1 = %.6e nu2 = %.6e, Abstand %.3e, w·d = %.3e\n",
                    s.weight, s.run.history.front(), s.run.merit, s.ray_merit, s.run.iterations,
                    to_string(s.run.status), s.glass[0], s.glass[1], s.glass[2], s.distance,
                    s.weight * s.distance);
        EXPECT_TRUE(lm_converged(s.run.status)) << s.weight << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
    }
    const GlassStage& last = stages.back();
    // Ergebnis in der Hülle bis auf die Restverletzung der Strafe (∝ 1/w)
    EXPECT_LE(last.distance, kGlassHullPenaltyDistance);
    EXPECT_LT(last.distance, 1e-3 * stages.front().distance);
    // Die Strafe bindet: Das Glas liegt auf dem Rand (Restverletzung > 0), und die Merit ohne
    // Strafterm ist schlechter als im Kontrollfall ohne Strafe.
    EXPECT_GT(last.distance, 0.0);
    EXPECT_GT(last.ray_merit, free_run.merit);
    // nächstes Glas der Hülle (zur Information): das Glas mit der kleinsten Dispersion
    const HullProjection proj = g.hull.project(last.glass);
    std::string nearest;
    double best = 1e300;
    for (const GlassHullPoint& p : g.hull.points()) {
        const Point3 u = g.hull.normalize(p.coordinates);
        const double d =
            std::hypot(u[0] - proj.closest[0], u[1] - proj.closest[1], u[2] - proj.closest[2]);
        if (d < best) {
            best = d;
            nearest = p.name;
        }
    }
    std::printf("[CPU] Ergebnis: Abstand %.3e (Toleranz %.1e), nächstes Glas der Hülle %s "
                "(normierter Abstand %.3e); Merit ohne Strafterm %.6e gegen %.6e ohne Strafe\n",
                last.distance, kGlassHullPenaltyDistance, nearest.c_str(), best, last.ray_merit,
                free_run.merit);
}

} // namespace
} // namespace mko::test
