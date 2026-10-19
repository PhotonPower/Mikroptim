// Abnahme der L1-Strafe (M4-T6) auf der GPU (nur mit MIKROPTIM_ENABLE_CUDA): dieselben
// Prüfungen wie test_ghost_optimization.cpp mit optimize_lm_gpu (Strafe als Zusatzresiduen auf
// dem Host, Strahlresiduen und Jacobi-Matrix auf dem Gerät).

#include "ghost_test_systems.hpp"
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

TEST(CudaGhostOptimization, ExactnessGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const FdProblem p = ghost_exact_problem();
    const GhostPenalty pen = ghost_penalty_of(p.loaded.system);
    cuda::TraceEngine engine(0);
    const ExactnessResult e = ghost_exactness(p, pen, [&](const LmOptions& o) {
        return cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, o).runs.at(0);
    });
    print_exactness("GPU", p, e);
    EXPECT_TRUE(lm_converged(e.run.status)) << to_string(e.run.status);
    EXPECT_EQ(e.run.failed_rays, 0);
    EXPECT_LT(max_ghost(p.set, e.run.x), kGhostExactCoefficient);
    EXPECT_LT(e.threshold_ratio, 1.0);
    EXPECT_LE(e.max_stationarity_error, kGhostStationarity);
    EXPECT_GE(e.smoothing_gain, 0.0);
    EXPECT_LE(std::abs(e.smoothing_gain - e.predicted_gain),
              kGhostSmoothingGain * e.predicted_gain);
}

TEST(CudaGhostOptimization, NeedGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const FdProblem p = ghost_need_problem();
    cuda::TraceEngine engine(0);
    const NeedResult n = ghost_need(p, [&](const FdProblem& q, const LmOptions& o) {
        return cuda::optimize_lm_gpu(engine, q.set, q.base, q.x0, o).runs.at(0);
    });
    print_need("GPU", p, n);
    check_need(p, n);
}

} // namespace
} // namespace mko::test
