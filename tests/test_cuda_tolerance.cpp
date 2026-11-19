// Monte-Carlo-Tolerierung auf der GPU (M8-T4, nur mit MIKROPTIM_ENABLE_CUDA): Trace über den
// Modus RESIDUALS der TraceEngine und Kompensator über optimize_lm_gpu (mko::cuda::use_gpu).
// GPU gegen CPU ohne Kompensator (nur Rundung) und mit Kompensator (Konvergenzgrenze des LM),
// wiederholte Läufe und andere Tranchengrößen bitgleich, Abnahme mit 10.000 Stichproben.
// Empfindlichkeiten (M10-T3): GPU gegen CPU, wiederholte Läufe bitgleich.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/tolerance.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/tolerance/sensitivity.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

using mko::ToleranceResult;
using mko::ToleranceSampleResult;
using mko::ToleranceSettings;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

struct Analysis {
    mko::LoadedSystem loaded;
    mko::SystemBatchHost batch;
};

Analysis prepare() {
    Analysis a;
    a.loaded = mko::load_system(mko::test::kToleranceSystemFile);
    a.batch = mko::make_system_batch(a.loaded, 1);
    return a;
}

ToleranceResult run(const Analysis& a, const ToleranceSettings& s) {
    return mko::run_tolerance_analysis(a.loaded.system, a.batch, s);
}

bool same_double(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

bool same_sample(const ToleranceSampleResult& a, const ToleranceSampleResult& b) {
    for (std::size_t t = 0; t < a.rms.size(); ++t) {
        if (!same_double(a.rms[t], b.rms[t]) || !same_double(a.strehl[t], b.strehl[t])) {
            return false;
        }
    }
    return a.rms.size() == b.rms.size() && same_double(a.working_distance, b.working_distance) &&
           same_double(a.compensation, b.compensation) &&
           a.compensator_iterations == b.compensator_iterations;
}

// Größte Abweichungen GPU gegen CPU: RMS und Strehl pro Tupel (absolut), Kompensation [mm].
struct Deviation {
    double quality = 0.0;
    double rms_relative = 0.0;
    double compensation = 0.0;
};

Deviation compare(const ToleranceResult& g, const ToleranceResult& c) {
    Deviation d;
    for (std::size_t i = 0; i < g.results.size(); ++i) {
        const ToleranceSampleResult& x = g.results[i];
        const ToleranceSampleResult& y = c.results[i];
        for (std::size_t t = 0; t < x.rms.size(); ++t) {
            d.quality = std::max(
                {d.quality, std::abs(x.rms[t] - y.rms[t]), std::abs(x.strehl[t] - y.strehl[t])});
            d.rms_relative = std::max(d.rms_relative, std::abs(x.rms[t] / y.rms[t] - 1.0));
        }
        d.compensation = std::max(d.compensation, std::abs(x.compensation - y.compensation));
    }
    return d;
}

} // namespace

TEST(CudaTolerance, GpuMatchesCpuWithoutCompensator) {
    MKO_REQUIRE_GPU();
    const Analysis a = prepare();
    ToleranceSettings s;
    s.samples = 500;
    s.seed = 8;
    s.compensator = mko::CompensatorType::None;
    const ToleranceResult cpu = run(a, s);
    mko::cuda::TraceEngine engine(0);
    mko::cuda::use_gpu(s, engine);
    const ToleranceResult gpu = run(a, s);
    const Deviation d = compare(gpu, cpu);
    EXPECT_LT(d.quality, mko::test::kToleranceCpuGpuQuality);
    EXPECT_EQ(d.compensation, 0.0);
    std::printf("GPU gegen CPU ohne Kompensator: RMS/Strehl max. %.3e\n", d.quality);
}

TEST(CudaTolerance, GpuMatchesCpuWithCompensator) {
    MKO_REQUIRE_GPU();
    const Analysis a = prepare();
    ToleranceSettings s;
    s.samples = 500;
    s.seed = 9;
    const ToleranceResult cpu = run(a, s);
    mko::cuda::TraceEngine engine(0);
    mko::cuda::use_gpu(s, engine);
    const ToleranceResult gpu = run(a, s);
    const Deviation d = compare(gpu, cpu);
    EXPECT_LT(d.compensation, mko::test::kToleranceCompensator);
    EXPECT_LT(d.rms_relative, mko::test::kToleranceCompensatedRms);
    std::printf("GPU gegen CPU mit Kompensator: Kompensation max. %.3e mm, RMS relativ %.3e\n",
                d.compensation, d.rms_relative);
}

// Abnahme M8-T4 auf der GPU: 10.000 Stichproben mit Kompensator; ein zweiter Lauf mit anderer
// Tranchengröße liefert für die ersten 3.000 Stichproben bitgleiche Ergebnisse (ADR-0007:
// bitgleich auf demselben Gerät, unabhängig von Tranchen und Batchgröße).
TEST(CudaTolerance, TenThousandSamplesReproducible) {
    MKO_REQUIRE_GPU();
    const Analysis a = prepare();
    mko::cuda::TraceEngine engine(0);
    ToleranceSettings s;
    s.samples = 10000;
    s.seed = 2026;
    mko::cuda::use_gpu(s, engine);
    const ToleranceResult r = run(a, s);
    ASSERT_EQ(r.results.size(), 10000u);
    EXPECT_EQ(r.rms_composite.count, 10000);
    EXPECT_EQ(r.failed_samples, 0);
    ToleranceSettings t = s;
    t.samples = 3000;
    t.chunk = 1234;
    const ToleranceResult q = run(a, t);
    EXPECT_TRUE(same_sample(q.nominal, r.nominal));
    for (std::size_t i = 0; i < q.results.size(); ++i) {
        ASSERT_TRUE(same_sample(q.results[i], r.results[i])) << i;
    }
    std::printf("GPU 10.000 Stichproben: RMS Mittel %.5f, 95 %% %.5f Wellen; Strehl Mittel %.4f, "
                "95 %% %.4f; Ausbeute %.4f\n",
                r.rms_composite.mean, r.rms_composite.p95, r.strehl_min.mean, r.strehl_min.p95,
                r.yield);
}

// Empfindlichkeiten (M10-T3) mit dem Trace auf der GPU: Q bei ±δ und die Jacobi-Terme gegen die
// CPU (Toleranzsatz der Testdatei), zwei Läufe bitgleich.
TEST(CudaSensitivity, GpuMatchesCpuAndIsReproducible) {
    MKO_REQUIRE_GPU();
    const Analysis a = prepare();
    mko::SensitivitySettings s;
    const mko::SensitivityResult cpu = mko::run_sensitivity_analysis(a.loaded.system, a.batch, s);
    mko::cuda::TraceEngine engine(0);
    mko::cuda::use_gpu(s, engine);
    const mko::SensitivityResult gpu = mko::run_sensitivity_analysis(a.loaded.system, a.batch, s);
    const mko::SensitivityResult again = mko::run_sensitivity_analysis(a.loaded.system, a.batch, s);
    ASSERT_EQ(gpu.components.size(), cpu.components.size());
    ASSERT_FALSE(gpu.components.empty());
    EXPECT_LE(std::abs(gpu.q0 - cpu.q0), mko::test::kSensitivityCpuGpuQuality * cpu.q0);
    double q_dev = 0.0;
    double j_dev = 0.0;
    for (std::size_t k = 0; k < cpu.components.size(); ++k) {
        const mko::SensitivityComponent& g = gpu.components[k];
        const mko::SensitivityComponent& c = cpu.components[k];
        const double scale = std::max({std::abs(c.q_plus - cpu.q0), std::abs(c.q_minus - cpu.q0),
                                       std::abs(c.jacobian_linear) + c.jacobian_quadratic});
        const double dq = std::max(std::abs(g.q_plus - c.q_plus), std::abs(g.q_minus - c.q_minus));
        const double dj = std::max(std::abs(g.jacobian_linear - c.jacobian_linear),
                                   std::abs(g.jacobian_quadratic - c.jacobian_quadratic));
        q_dev = std::max(q_dev, dq / cpu.q0);
        j_dev = std::max(j_dev, dj / scale);
        EXPECT_LE(dq, mko::test::kSensitivityCpuGpuQuality * cpu.q0) << k;
        EXPECT_LE(dj, mko::test::kSensitivityCpuGpuJacobian * scale) << k;
        const mko::SensitivityComponent& r = again.components[k];
        EXPECT_TRUE(same_double(g.q_plus, r.q_plus) && same_double(g.q_minus, r.q_minus) &&
                    same_double(g.jacobian_linear, r.jacobian_linear) &&
                    same_double(g.jacobian_quadratic, r.jacobian_quadratic))
            << k;
    }
    std::printf("[ Messung ] Empfindlichkeit GPU gegen CPU: Q relativ %.3e, Jacobi-Terme relativ "
                "zu max |ΔQ| %.3e\n",
                q_dev, j_dev);
}
