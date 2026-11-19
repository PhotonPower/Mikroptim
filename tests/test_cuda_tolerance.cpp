// Monte-Carlo-Tolerierung auf der GPU (M8-T4, nur mit MIKROPTIM_ENABLE_CUDA): Trace über den
// Modus RESIDUALS der TraceEngine und Kompensator über optimize_lm_gpu (mko::cuda::use_gpu).
// GPU gegen CPU ohne Kompensator (nur Rundung) und mit Kompensator (Konvergenzgrenze des LM),
// wiederholte Läufe und andere Tranchengrößen bitgleich, Abnahme mit 10.000 Stichproben.
// Empfindlichkeiten (M10-T3): GPU gegen CPU, wiederholte Läufe bitgleich. Toleranzbewusste
// Optimierung (M11-T6): inneres Backend auf der GPU gegen die CPU.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/tolerance.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/tolerance/sensitivity.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "mikroptim/tolerance/tolerance_aware.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
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

// Toleranzbewusste Optimierung (M11-T6) mit dem inneren Backend auf der GPU am Kittblock
// (test_tolerance_aware.cpp): Teilmerits gegen die CPU, Abnahme ab Lösung A wie auf der CPU
// (Kittfläche c → 0), zwei Läufe bitgleich.
TEST(CudaToleranceAware, PicksInsensitiveSolutionLikeCpu) {
    MKO_REQUIRE_GPU();
    const std::string file =
        std::string(MKO_SOURCE_DIR) + "/data/systems/kittblock_toleranzbewusst.json";
    const mko::LoadedSystem loaded = mko::load_system(file);
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::ToleranceAwareSettings ts;
    ts.alpha = 0.5;
    const mko::ToleranceAwareModel model(loaded.system, base, set, ts);
    mko::LmOptions o;
    o.max_iterations = 200;
    mko::cuda::TraceEngine engine(0);
    const mko::ToleranceInnerFactory gpu = mko::cuda::gpu_tolerance_inner(engine);
    const mko::ToleranceInnerFactory cpu = mko::cpu_tolerance_inner();
    const std::vector<double> xa = set.extract(base, 0);
    const int run0 = 0;
    const std::span<const int> run(&run0, 1);
    mko::ToleranceAwareBackend bc(model, cpu, base, o);
    mko::ToleranceAwareBackend bg(model, gpu, base, o);
    const mko::ToleranceAwareMerits mc = bc.merits(run, xa);
    const mko::ToleranceAwareMerits mg = bg.merits(run, xa);
    ASSERT_EQ(mg.merit.size(), mc.merit.size());
    double dev = 0.0;
    for (std::size_t s = 0; s < mc.merit.size(); ++s) {
        dev = std::max(dev, std::abs(mg.merit[s] - mc.merit[s]) / mc.merit[s]);
    }
    EXPECT_LE(dev, mko::test::kSensitivityCpuGpuQuality);
    const mko::LmResult rc = mko::optimize_lm_tolerance_aware(model, cpu, set, base, xa, o);
    const mko::LmResult rg = mko::optimize_lm_tolerance_aware(model, gpu, set, base, xa, o);
    const mko::LmResult rg2 = mko::optimize_lm_tolerance_aware(model, gpu, set, base, xa, o);
    const mko::LmRunResult& c = rc.runs.at(0);
    const mko::LmRunResult& g = rg.runs.at(0);
    EXPECT_TRUE(mko::lm_converged(g.status)) << mko::to_string(g.status);
    EXPECT_LE(std::abs(g.x[0]), mko::test::kToleranceAwareChosen * xa[0]);
    EXPECT_LE(std::abs(g.merit - c.merit), mko::test::kToleranceAwareNominal * c.merit);
    EXPECT_EQ(g.x, rg2.runs.at(0).x);
    EXPECT_EQ(g.history, rg2.runs.at(0).history);
    std::printf("[ Messung ] toleranzbewusst GPU: Teilmerits gegen CPU %.3e relativ; c CPU %.3e, "
                "GPU %.3e; F_tol CPU %.12e, GPU %.12e\n",
                dev, c.x[0], g.x[0], c.merit, g.merit);
}

// Kippungsabzug (M11-T6b) auf der GPU: Teilmerits und Normalgleichungen (FD und adjungiert)
// gegen die CPU, zwei GPU-Läufe der Optimierung bitgleich. Kittblock mit Dezentrierung,
// Verkippung und Radien der Sammellinse und der Brechzahl des hinteren Teils.
TEST(CudaToleranceAware, TiltRemovalMatchesCpu) {
    MKO_REQUIRE_GPU();
    const std::string file =
        std::string(MKO_SOURCE_DIR) + "/data/systems/kittblock_toleranzbewusst.json";
    mko::SystemDescription sys = mko::load_system(file).system;
    mko::ToleranceSpec dec;
    dec.type = mko::ToleranceType::ElementDecenter;
    dec.element_selection = mko::SurfaceSelection::One;
    dec.elements = {{4, 5}};
    dec.delta = 0.01;
    mko::ToleranceSpec tilt;
    tilt.type = mko::ToleranceType::ElementTilt;
    tilt.element_selection = mko::SurfaceSelection::One;
    tilt.elements = {{4, 5}};
    tilt.arcmin = 2.0;
    mko::ToleranceSpec radius;
    radius.type = mko::ToleranceType::Radius;
    radius.selection = mko::SurfaceSelection::List;
    radius.surfaces = {4, 5};
    radius.relative = 0.002;
    sys.tolerances->items.push_back(dec);
    sys.tolerances->items.push_back(tilt);
    sys.tolerances->items.push_back(radius);
    const mko::LoadedSystem loaded =
        mko::prepare_system(sys, std::filesystem::path(file).parent_path());
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::ToleranceAwareSettings ts;
    ts.alpha = 0.5;
    ts.tilt = mko::ToleranceAwareTilt::Remove;
    const mko::ToleranceAwareModel model(loaded.system, base, set, ts);
    mko::cuda::TraceEngine engine(0);
    const mko::ToleranceInnerFactory gpu = mko::cuda::gpu_tolerance_inner(engine);
    const mko::ToleranceInnerFactory cpu = mko::cpu_tolerance_inner();
    const std::vector<double> x = set.extract(base, 0);
    const auto P = static_cast<std::size_t>(set.size());
    const int run0 = 0;
    const std::span<const int> run(&run0, 1);
    double dev_merit = 0.0;
    double dev_gradient = 0.0;
    for (const mko::JacobianMethod method :
         {mko::JacobianMethod::FiniteDifferences, mko::JacobianMethod::Adjoint}) {
        mko::LmOptions o;
        o.jacobian.quadrature.failure = mko::FailureMode::Hard;
        o.jacobian_method = method;
        mko::ToleranceAwareBackend bc(model, cpu, base, o);
        mko::ToleranceAwareBackend bg(model, gpu, base, o);
        const mko::ToleranceAwareMerits mc = bc.merits(run, x);
        const mko::ToleranceAwareMerits mg = bg.merits(run, x);
        ASSERT_EQ(mg.merit.size(), mc.merit.size());
        for (std::size_t s = 0; s < mc.merit.size(); ++s) {
            dev_merit = std::max(dev_merit, std::abs(mg.merit[s] - mc.merit[s]) / mc.merit[s]);
            EXPECT_EQ(mg.failed[s], mc.failed[s]);
        }
        std::vector<double> ac(P * P), gc(P), ag(P * P), gg(P);
        bc.normal_equations(run, x, ac, gc);
        bg.normal_equations(run, x, ag, gg);
        double scale = 0.0;
        for (std::size_t k = 0; k < P; ++k) {
            scale = std::max(scale, std::abs(gc[k]));
        }
        for (std::size_t k = 0; k < P; ++k) {
            dev_gradient = std::max(dev_gradient, std::abs(gg[k] - gc[k]) / scale);
        }
    }
    EXPECT_LE(dev_merit, mko::test::kSensitivityCpuGpuQuality);
    EXPECT_LE(dev_gradient, mko::test::kSensitivityCpuGpuJacobian);
    mko::LmOptions o;
    o.max_iterations = 20;
    o.jacobian.quadrature.failure = mko::FailureMode::Hard;
    const mko::LmResult rg = mko::optimize_lm_tolerance_aware(model, gpu, set, base, x, o);
    const mko::LmResult rg2 = mko::optimize_lm_tolerance_aware(model, gpu, set, base, x, o);
    EXPECT_EQ(rg.runs.at(0).x, rg2.runs.at(0).x);
    EXPECT_EQ(rg.runs.at(0).history, rg2.runs.at(0).history);
    std::printf("[ Messung ] Kippungsabzug GPU gegen CPU: Teilmerits %.3e relativ, Gradient %.3e "
                "relativ zur größten Komponente (FD und adjungiert)\n",
                dev_merit, dev_gradient);
}
