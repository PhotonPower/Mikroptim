// Abnahme „Optimierung bleibt innerhalb der Hülle“ (M6-T1) auf der GPU (nur mit
// MIKROPTIM_ENABLE_CUDA): dieselbe Penalty-Fortsetzung wie test_glass_optimization.cpp mit
// optimize_lm_gpu, Ergebnis gegen die Hülle und gegen den CPU-Lauf.

#include "glass_test_systems.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mko::test {
namespace {

TEST(CudaGlassOptimization, GpuStaysInsideHull) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const GlassProblem g = glass_problem();
    cuda::TraceEngine engine(0);
    const auto gpu_optimize = [&](const LmOptions& o, std::span<const double> x0) {
        return cuda::optimize_lm_gpu(engine, g.p.set, g.p.base, x0, o);
    };
    const auto cpu_optimize = [&](const LmOptions& o, std::span<const double> x0) {
        return optimize_lm_cpu(g.p.set, g.p.base, x0, o);
    };

    // Kontrollfall ohne Strafe
    LmOptions free_options;
    free_options.max_iterations = kGlassFreeIterations;
    free_options.jacobian.extra = glass_extra(g, 0.0);
    const LmRunResult free_run = gpu_optimize(free_options, g.p.x0).runs.at(0);
    const double free_distance = g.hull.distance(glass_at(g, free_run.x));
    std::printf("[GPU] ohne Strafe: F -> %.10e, Abstand %.4e\n", free_run.merit, free_distance);
    EXPECT_GT(free_distance, kGlassFreeDistance);

    const std::vector<GlassStage> gpu = glass_continuation(g, gpu_optimize);
    const std::vector<GlassStage> cpu = glass_continuation(g, cpu_optimize);
    ASSERT_EQ(gpu.size(), static_cast<std::size_t>(kGlassStages));
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const GlassStage& s = gpu[i];
        std::printf("[GPU]   w = %8.1e: F -> %.10e (CPU %.10e), %4d Zyklen (CPU %4d), %-16s "
                    "Abstand %.3e (CPU %.3e)\n",
                    s.weight, s.run.merit, cpu[i].run.merit, s.run.iterations,
                    cpu[i].run.iterations, to_string(s.run.status), s.distance, cpu[i].distance);
        EXPECT_TRUE(lm_converged(s.run.status)) << s.weight << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
    }
    const GlassStage& last = gpu.back();
    EXPECT_LE(last.distance, kGlassHullPenaltyDistance);
    EXPECT_GT(last.ray_merit, free_run.merit);
    // GPU und CPU am selben Optimum auf dem Rand der Hülle
    EXPECT_LE(std::abs(last.run.merit / cpu.back().run.merit - 1.0), kLmReturnRelative);
}

} // namespace
} // namespace mko::test
