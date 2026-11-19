// Memetische Suche auf der GPU (M11-T7, nur mit MIKROPTIM_ENABLE_CUDA): Abnahme am Mehrlinser
// mit optimize_lm_gpu (CMA-ES besser als gestörte Starts bei gleichem Budget an LM-Zyklen),
// zwei Läufe bitgleich, FP32-Vorfilter über TraceEngine::prefilter_variants.

#include "lm_test_systems.hpp"
#include "memetic_test_cases.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/memetic.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <span>
#include <vector>

namespace mko {
namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (cuda::device_count() == 0) {                                                           \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

MemeticOptimizer gpu_optimizer(cuda::TraceEngine& engine) {
    cuda::TraceEngine* e = &engine;
    return [e](const VariableSet& set, const SystemBatchHost& base, std::span<const double> x,
               const LmOptions& o) { return cuda::optimize_lm_gpu(*e, set, base, x, o); };
}

MemeticPrefilterFn gpu_prefilter(cuda::TraceEngine& engine) {
    cuda::TraceEngine* e = &engine;
    return [e](const SystemBatchHost& batch, const ResidualSettings& quadrature,
               const PrefilterOptions& po, std::span<const double> extra) {
        cuda::PrefilterSettings ps;
        ps.quadrature = quadrature;
        ps.options = po;
        ps.extra_merit = extra;
        return e->prefilter_variants(batch, ps);
    };
}

TEST(CudaMemetic, CmaBeatsPerturbedStartsOnMehrlinser) {
    MKO_REQUIRE_GPU();
    cuda::TraceEngine engine(0);
    const MemeticOptimizer opt = gpu_optimizer(engine);
    const test::MemeticCase c = test::mehrlinser_memetic_case();
    const auto t0 = std::chrono::steady_clock::now();
    const MemeticResult cma = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options, opt);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const test::BaselineResult base = test::perturbed_starts_baseline(c, cma.cycles, opt);
    const MemeticResult cma2 = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options, opt);
    std::printf("GPU Mehrlinser: CMA F = %.9e (%lld Zyklen, %.2f s), perturbed_starts F = %.9e "
                "(%lld Zyklen)\n",
                cma.best_individual().run.merit, static_cast<long long>(cma.cycles), seconds,
                base.best_merit, static_cast<long long>(base.cycles));
    EXPECT_GE(base.cycles, cma.cycles);
    EXPECT_LT(cma.best_individual().run.merit, base.best_merit);
    EXPECT_TRUE(test::same_memetic(cma, cma2));
}

TEST(CudaMemetic, PrefilterIsDeterministic) {
    MKO_REQUIRE_GPU();
    cuda::TraceEngine engine(0);
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.options.prefilter = true;
    c.options.generations = 2;
    const MemeticResult a = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options,
                                           gpu_optimizer(engine), gpu_prefilter(engine));
    const MemeticResult b = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options,
                                           gpu_optimizer(engine), gpu_prefilter(engine));
    ASSERT_EQ(a.generations.size(), 2u);
    EXPECT_EQ(a.generations[0].candidates, 2 * a.lambda);
    EXPECT_TRUE(test::same_memetic(a, b));
}

} // namespace
} // namespace mko
