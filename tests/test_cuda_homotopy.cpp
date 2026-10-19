// Homotopie-Controller (M6-T3) mit dem GPU-Optimierer (optimize_lm_gpu; nur mit
// MIKROPTIM_ENABLE_CUDA): Kurzlauf gegen die CPU und die Abnahme NA 0,3 → 0,6
// (docs/optimierung.md, Abschnitt 7).

#include "homotopy_test_systems.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <span>

namespace mko::test {
namespace {

HomotopyOptimizer gpu_optimizer(cuda::TraceEngine& engine) {
    return
        [&engine](const VariableSet& set, const SystemBatchHost& base, std::span<const double> x0,
                  const LmOptions& o) { return cuda::optimize_lm_gpu(engine, set, base, x0, o); };
}

// Kurzlauf wie HomotopySmoke.ShortRunCpu: gleiche Entscheidungen, Merit innerhalb der Toleranz.
TEST(CudaHomotopy, ShortRunMatchesCpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    const HomotopyResult cpu = homotopy_smoke_run({});
    const HomotopyResult gpu = homotopy_smoke_run(gpu_optimizer(engine));
    check_homotopy_smoke(gpu);
    expect_same_decisions(cpu, gpu, kHomotopyCpuGpuMerit);
}

// Abnahme auf der GPU (Laufzeit Minuten; lokal mit --gtest_also_run_disabled_tests).
TEST(CudaHomotopy, DISABLED_ApertureToNa06Gpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    const HomotopyRun run = homotopy_acceptance_run(gpu_optimizer(engine), true);
    print_homotopy(run.result, std::cout);
    check_homotopy_acceptance(run.result);
}

} // namespace
} // namespace mko::test
