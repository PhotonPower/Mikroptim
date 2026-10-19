// Abnahme der Petzval-Nebenbedingung (M4-T4) auf der GPU (nur mit MIKROPTIM_ENABLE_CUDA):
// dieselbe Penalty-Fortsetzung wie test_petzval_optimization.cpp mit optimize_lm_gpu, Ergebnis
// gegen die Grenze und gegen den CPU-Lauf.

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "petzval_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mko::test {
namespace {

TEST(CudaPetzvalOptimization, GpuReachesLimitWithActiveConstraint) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const FdProblem p = petzval_problem();
    const double kappa_max =
        p.loaded.system.constraints->items[petzval_item(p.loaded.system)].max_curvature.value();
    const double f_target = p.loaded.system.target->focal_length.value();

    cuda::TraceEngine engine(0);
    const std::vector<PetzvalStage> gpu =
        petzval_continuation(p, [&](const LmOptions& o, std::span<const double> x0) {
            return cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o);
        });
    const std::vector<PetzvalStage> cpu =
        petzval_continuation(p, [&](const LmOptions& o, std::span<const double> x0) {
            return optimize_lm_cpu(p.set, p.base, x0, o);
        });
    ASSERT_EQ(gpu.size(), static_cast<std::size_t>(kPetzvalStages));
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const PetzvalStage& s = gpu[i];
        const double excess = std::abs(s.paraxial.petzval_curvature) - kappa_max;
        std::printf("[GPU]   w = %8.1e mm²: F -> %.10e (CPU %.10e), %3d Zyklen (CPU %3d), %-16s "
                    "|κ_P| − κ_max = %+.3e (relativ %+.3e), f − f_s = %+.3e mm\n",
                    s.weight, s.run.merit, cpu[i].run.merit, s.run.iterations,
                    cpu[i].run.iterations, to_string(s.run.status), excess, excess / kappa_max,
                    s.paraxial.focal_length - f_target);
        EXPECT_TRUE(lm_converged(s.run.status)) << s.weight << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
        EXPECT_TRUE(p.set.feasible(s.run.x));
    }
    const PetzvalStage& last = gpu.back();
    const double kappa = std::abs(last.paraxial.petzval_curvature);
    EXPECT_LE(kappa, kappa_max * (1.0 + kPetzvalPenaltyRelative));
    EXPECT_GE(kappa, kappa_max * (1.0 - kPetzvalPenaltyRelative));
    EXPECT_LT(std::abs(last.paraxial.focal_length - f_target), kPetzvalFocalLength);
    // GPU und CPU am selben beschränkten Optimum (Rundung ändert den Weg, nicht das Ziel)
    EXPECT_LE(std::abs(last.run.merit / cpu.back().run.merit - 1.0), kLmReturnRelative);
}

} // namespace
} // namespace mko::test
