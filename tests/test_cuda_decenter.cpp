// Dezentrierung und Verkippung auf der GPU (M8-T3, nur mit MIKROPTIM_ENABLE_CUDA): Transformation
// 0 bitgleich zum koaxialen Batch mit voller Pupille in allen Kernel-Modi, GPU gegen CPU
// (RESIDUALS, MERIT mit Max-Reduktionen, EVAL, FP32-Vorfilterung, ADJOINT-Vorwärtsdurchlauf,
// adjungierte Jacobi-Matrix, Einzelstrahl-Trace) für zufällig dezentrierte und verkippte
// Varianten, Determinismus und Spiegelsymmetrie. CPU-Teil: test_decenter.cpp.

#include "decenter_test_systems.hpp"
#include "eval_test_systems.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/adjoint.hpp"
#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/trace_check.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::Precision;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

TraceSettings trace_settings(const ResidualSettings& q, Precision p, bool extrema = true) {
    TraceSettings s;
    s.quadrature = q;
    s.opd = true;
    s.extrema = extrema;
    s.precision = p;
    return s;
}

// Testbatches: zufällig gestörte und dezentrierte Varianten (Variante 0 unbewegt) mit versagten
// Strahlen, Geisterplatten und weichem Versagen, dazu das dezentrierte Testsystem aus der Datei.
struct DecenterCase {
    std::string name;
    SystemBatchHost batch;
};

std::vector<DecenterCase> decenter_cases(int variants) {
    std::vector<DecenterCase> c;
    c.push_back({"Mehrlinser", mko::test::decentered_batch(
                                   mko::test::load_residual_system("testsystem_mehrlinser.json"),
                                   variants, 83200)});
    c.push_back(
        {"Geisterplatten", mko::test::decentered_batch(
                               mko::test::load_residual_system("einzellinse_geisterplatte.json"),
                               variants / 2, 83210)});
    c.push_back(
        {"Weiches Versagen",
         mko::test::decentered_batch(mko::test::load_residual_system("weiches_versagen_start.json"),
                                     variants / 2, 83220, 0.3, 3.0 * mko::test::kDecenterDeg)});
    c.push_back({"Mehrlinser dezentriert (Datei)",
                 mko::test::perturbed_batch(
                     mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json"),
                     variants / 2, 83230)});
    return c;
}

// GPU gegen CPU Strahl für Strahl: Status gleich, OPD und Residuen (in mm OPD) innerhalb
// max(kCpuGpuFp64, kCpuGpuFp64Relative · |x|), in Tupeln mit weich versagten Strahlen
// kCpuGpuSoftTuple als absoluter Anteil; Strafgrößen mit kCpuGpuPenaltyRelative.
struct RayComparison {
    double max_opd = 0.0;
    std::size_t rays = 0;
    std::size_t failed = 0;
    std::size_t violations = 0;
    std::size_t status_mismatch = 0;
};

RayComparison compare_rays(const ResidualResult& gpu, const ResidualResult& cpu,
                           const SystemBatchHost& batch, const ResidualLayout& layout) {
    const std::vector<double> units = mko::test::residual_units(batch, layout);
    const std::vector<double> punits = mko::test::penalty_units(batch, layout);
    RayComparison c;
    for (int v = 0; v < cpu.variants; ++v) {
        for (int f = 0; f < layout.fields(); ++f) {
            for (int l = 0; l < layout.wavelengths(); ++l) {
                const std::int64_t o = layout.offset(f, l);
                bool soft_tuple = false;
                for (int k = 0; k < layout.samples(f); ++k) {
                    soft_tuple = soft_tuple || mko::is_soft_failure(cpu.status_at(v, o + k));
                }
                const double abs_tol =
                    soft_tuple ? mko::test::kCpuGpuSoftTuple : mko::test::kCpuGpuFp64;
                for (int k = 0; k < layout.samples(f); ++k) {
                    const std::int64_t i = o + k;
                    ++c.rays;
                    const TraceStatus sc = cpu.status_at(v, i);
                    c.failed += sc == TraceStatus::Ok ? 0 : 1;
                    if (gpu.status_at(v, i) != sc) {
                        ++c.status_mismatch;
                        continue;
                    }
                    const auto check = [&](double a, double b, double tol_abs, double rel) {
                        double d = std::abs(a - b);
                        if (std::isnan(d)) {
                            d = 1e300;
                        }
                        if (d > std::max(tol_abs, rel * std::abs(b))) {
                            ++c.violations;
                        }
                        return d;
                    };
                    c.max_opd = std::max(c.max_opd, check(gpu.opd_at(v, i), cpu.opd_at(v, i),
                                                          abs_tol, mko::test::kCpuGpuFp64Relative));
                    const double u = units[static_cast<std::size_t>(i)];
                    if (u > 0.0) {
                        check(gpu.residual(v, i) / u, cpu.residual(v, i) / u, abs_tol,
                              mko::test::kCpuGpuFp64Relative);
                    }
                    if (layout.penalty_rows() > 0) {
                        const double pu = punits[static_cast<std::size_t>(i)];
                        if (pu > 0.0) {
                            check(gpu.penalty(v, i) / pu, cpu.penalty(v, i) / pu,
                                  mko::test::kCpuGpuFp64, mko::test::kCpuGpuPenaltyRelative);
                        }
                    }
                }
            }
        }
    }
    return c;
}

double max_column_deviation(std::span<const double> a, std::span<const double> b, int P) {
    return mko::test::jacobian_deviation(a, b, P).max_column;
}

} // namespace

// Abnahme (1) auf der GPU: Transformation 0 (make_non_coaxial) ist in allen Kernel-Modi bitgleich
// zum koaxialen Batch mit voller Pupille: RESIDUALS und MERIT (FP64 und df64, weich und hart, mit
// Max-Reduktionen), EVAL, FP32-Vorfilterung, ADJOINT-Vorwärtsdurchlauf und adjungierte Jacobi.
TEST(CudaDecenter, ZeroTransformIsBitIdenticalInAllModes) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    int compared = 0;
    for (const char* file : {"testsystem_mehrlinser.json", "einzellinse_geisterplatte.json",
                             "weiches_versagen_start.json"}) {
        SCOPED_TRACE(file);
        const SystemBatchHost coax = mko::test::with_full_pupil(
            mko::test::perturbed_batch(mko::test::load_residual_system(file), 40, 83240));
        const SystemBatchHost moved = mko::make_non_coaxial(coax);
        for (const ResidualSettings& q : {ResidualSettings{6, 8, mko::FailureMode::Soft},
                                          ResidualSettings{5, 6, mko::FailureMode::Hard}}) {
            for (const Precision p : {Precision::Fp64, Precision::Df64}) {
                const TraceSettings s = trace_settings(q, p);
                const TraceOutput ra = engine.run(coax, TraceMode::Residuals, s);
                const TraceOutput rb = engine.run(moved, TraceMode::Residuals, s);
                EXPECT_TRUE(same_bytes(ra.residuals.residuals, rb.residuals.residuals));
                EXPECT_TRUE(same_bytes(ra.residuals.opd, rb.residuals.opd));
                EXPECT_TRUE(same_bytes(ra.residuals.status, rb.residuals.status));
                EXPECT_TRUE(same_bytes(ra.residuals.extrema.values, rb.residuals.extrema.values));
                const TraceOutput ma = engine.run(coax, TraceMode::Merit, s);
                const TraceOutput mb = engine.run(moved, TraceMode::Merit, s);
                EXPECT_TRUE(same_bytes(ma.merit.merit, mb.merit.merit));
                EXPECT_TRUE(same_bytes(ma.merit.extrema.values, mb.merit.extrema.values));
                compared += 2;
                const ResidualLayout la(coax.topology(), q);
                const ResidualLayout lb(moved.topology(), q);
                const mko::cuda::DeviceSystemBatch da(coax);
                const mko::cuda::DeviceSystemBatch db(moved);
                mko::cuda::AdjointForwardOptions ao;
                ao.precision = p;
                ao.opd = true;
                ao.states = true;
                const auto fa = mko::cuda::run_adjoint_forward(da, la, ao);
                const auto fb = mko::cuda::run_adjoint_forward(db, lb, ao);
                EXPECT_TRUE(same_bytes(fa.residuals.residuals, fb.residuals.residuals));
                EXPECT_TRUE(same_bytes(fa.states, fb.states));
                EXPECT_TRUE(same_bytes(fa.checksum, fb.checksum));
                ++compared;
            }
            mko::cuda::PrefilterSettings ps;
            ps.quadrature = q;
            const mko::PrefilterResult pa = engine.prefilter_variants(coax, ps);
            const mko::PrefilterResult pb = engine.prefilter_variants(moved, ps);
            EXPECT_TRUE(same_bytes(pa.stats.merit, pb.stats.merit));
            EXPECT_TRUE(same_bytes(pa.stats.optical_length, pb.stats.optical_length));
            EXPECT_TRUE(same_bytes(pa.reject, pb.reject));
            ++compared;
        }
        mko::EvalSettings es;
        es.grid = 17;
        const mko::EvalResult ea = engine.evaluate(coax, es);
        const mko::EvalResult eb = engine.evaluate(moved, es);
        EXPECT_TRUE(same_bytes(ea.opd, eb.opd));
        EXPECT_TRUE(same_bytes(ea.spot_y, eb.spot_y));
        EXPECT_TRUE(same_bytes(ea.field_rays, eb.field_rays));
        ++compared;
    }
    for (const auto& c : mko::test::adjoint_cases()) {
        const mko::test::FdProblem& p = c->p;
        const SystemBatchHost coax = mko::test::with_full_pupil(p.base);
        const SystemBatchHost moved = mko::make_non_coaxial(coax);
        const mko::VariableSet sa(p.loaded.system, coax.topology());
        const mko::VariableSet sb(p.loaded.system, moved.topology());
        mko::AdjointJacobianOptions o = c->options;
        o.extra = {};
        o.trace_extra = {};
        for (const Precision prec : {Precision::Fp64, Precision::Df64}) {
            const auto ja = mko::cuda::adjoint_jacobian_gpu(engine, sa, coax, p.x0, o, prec);
            const auto jb = mko::cuda::adjoint_jacobian_gpu(engine, sb, moved, p.x0, o, prec);
            EXPECT_TRUE(same_bytes(ja.jacobian, jb.jacobian)) << c->name;
            EXPECT_TRUE(same_bytes(ja.residuals, jb.residuals)) << c->name;
            ++compared;
        }
    }
    std::printf("[ Info     ] %d Vergleiche bitgleich\n", compared);
}

// Abnahme (7): GPU (FP64) gegen CPU für dezentrierte und verkippte Varianten in RESIDUALS (weich
// und hart), MERIT (gegen Σ r² der GPU-Residuen und Extrema gegen CPU), ADJOINT-Vorwärtsdurchlauf
// (Residuen bitgleich zu RESIDUALS) und df64 gegen FP64.
TEST(CudaDecenter, ResidualsMeritAndAdjointForwardMatchCpu) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    for (const DecenterCase& c : decenter_cases(400)) {
        for (const ResidualSettings& q : {ResidualSettings{8, 8, mko::FailureMode::Soft},
                                          ResidualSettings{6, 6, mko::FailureMode::Hard}}) {
            SCOPED_TRACE(c.name + (q.failure == mko::FailureMode::Soft ? " weich" : " hart"));
            const ResidualLayout layout(c.batch.topology(), q);
            const ResidualResult cpu = mko::residuals_cpu(c.batch, layout);
            const TraceOutput gpu =
                engine.run(c.batch, TraceMode::Residuals, trace_settings(q, Precision::Fp64));
            const RayComparison rc = compare_rays(gpu.residuals, cpu, c.batch, layout);
            EXPECT_EQ(rc.status_mismatch, 0u);
            EXPECT_EQ(rc.violations, 0u);
            EXPECT_GT(rc.failed, 0u);
            // Max-Reduktionen gegen die CPU (Höhen und Einfallswinkel im System der Fläche)
            const mko::SurfaceExtremaResult ext = mko::surface_extrema_cpu(c.batch, layout);
            double max_ext = 0.0;
            for (std::size_t i = 0; i < ext.values.size(); ++i) {
                const double a = gpu.residuals.extrema.values[i];
                const double b = ext.values[i];
                const double d = std::abs(a - b);
                max_ext = std::max(max_ext, d / std::max(1.0, std::abs(b)));
                EXPECT_LE(d, std::max(mko::test::kCpuGpuFp64,
                                      mko::test::kCpuGpuFp64Relative * std::abs(b)))
                    << "Extremum " << i;
            }
            // MERIT gegen die Quadratsumme der GPU-Residuen
            const TraceOutput merit =
                engine.run(c.batch, TraceMode::Merit, trace_settings(q, Precision::Fp64));
            double max_merit = 0.0;
            for (int v = 0; v < c.batch.variants(); ++v) {
                double sum = 0.0;
                for (std::int64_t i = 0; i < gpu.residuals.trace_rows(); ++i) {
                    sum += gpu.residuals.residual(v, i) * gpu.residuals.residual(v, i);
                }
                const double m = merit.merit.merit[static_cast<std::size_t>(v)];
                const double rel = sum > 0.0 ? std::abs(m - sum) / sum : std::abs(m);
                max_merit = std::max(max_merit, rel);
            }
            EXPECT_LT(max_merit, mko::test::kMeritVsResidualSum);
            EXPECT_TRUE(same_bytes(merit.merit.extrema.values, gpu.residuals.extrema.values));
            // ADJOINT-Vorwärtsdurchlauf (übersetzt mit --fmad=false): Status wie RESIDUALS,
            // Residuen und OPD gegen die CPU mit denselben Schranken
            const mko::cuda::DeviceSystemBatch device(c.batch);
            mko::cuda::AdjointForwardOptions ao;
            ao.opd = true;
            const auto fwd = mko::cuda::run_adjoint_forward(device, layout, ao);
            EXPECT_TRUE(same_bytes(fwd.residuals.status, gpu.residuals.status));
            const RayComparison fc = compare_rays(fwd.residuals, cpu, c.batch, layout);
            EXPECT_EQ(fc.status_mismatch, 0u);
            EXPECT_EQ(fc.violations, 0u);
            // df64 gegen FP64 (OPD pro gültigem Strahl)
            const TraceOutput df =
                engine.run(c.batch, TraceMode::Residuals, trace_settings(q, Precision::Df64));
            double max_df = 0.0;
            for (std::size_t i = 0; i < cpu.opd.size(); ++i) {
                if (static_cast<TraceStatus>(cpu.status[i]) == TraceStatus::Ok &&
                    df.residuals.status[i] == cpu.status[i]) {
                    max_df = std::max(max_df, std::abs(df.residuals.opd[i] - gpu.residuals.opd[i]) /
                                                  std::max(1.0, std::abs(gpu.residuals.opd[i])));
                }
            }
            std::printf("[ Info     ] %s, %s: %zu Strahlen, %zu versagt; max |ΔOPD| GPU−CPU "
                        "%.2e mm, Extrema %.2e, MERIT gegen Σr² %.2e, df64 gegen FP64 %.2e\n",
                        c.name.c_str(), q.failure == mko::FailureMode::Soft ? "weich" : "hart",
                        rc.rays, rc.failed, rc.max_opd, max_ext, max_merit, max_df);
            EXPECT_LT(max_df, mko::test::kDf64VsFp64);
        }
    }
}

// Abnahme (7) für EVAL und die FP32-Vorfilterung: GPU gegen CPU auf dezentrierten Varianten.
TEST(CudaDecenter, EvalAndPrefilterMatchCpu) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    for (const DecenterCase& c : decenter_cases(60)) {
        SCOPED_TRACE(c.name);
        for (const mko::EvalReference ref :
             {mko::EvalReference::ExitPupil, mko::EvalReference::Vertex}) {
            mko::EvalSettings s;
            s.grid = 33;
            s.reference = ref;
            const mko::EvalResult cpu = mko::eval_cpu(c.batch, s);
            const mko::EvalResult gpu = engine.evaluate(c.batch, s);
            const mko::test::EvalComparison cmp = mko::test::compare_eval(
                cpu, gpu, mko::test::kCpuGpuFp64, mko::test::kCpuGpuFp64Relative);
            EXPECT_LE(cmp.ratio, 1.0);
            EXPECT_EQ(cmp.status_mismatch, 0u);
            EXPECT_EQ(cmp.nan_mismatch, 0u);
            std::printf("[ Info     ] EVAL %s, %s: max |ΔOPD| %.2e mm, |ΔSpot| %.2e mm, "
                        "Verhältnis zur Toleranz %.3f, versagt %zu von %zu\n",
                        c.name.c_str(), ref == mko::EvalReference::Vertex ? "Scheitel" : "Pupille",
                        cmp.max_opd, cmp.max_spot, cmp.ratio, cpu.failed_count(),
                        cpu.status.size());
        }
        const ResidualLayout layout(c.batch.topology());
        const mko::PrefilterOptions options;
        const mko::PrefilterStats cpu =
            mko::prefilter_stats_cpu(c.batch, layout, options.margins.boundary);
        const mko::PrefilterResult gpu =
            engine.prefilter_variants(c.batch, mko::cuda::PrefilterSettings{});
        int mismatch = 0;
        for (int v = 0; v < c.batch.variants(); ++v) {
            const auto i = static_cast<std::size_t>(v);
            if (cpu.failed[i] != gpu.stats.failed[i] ||
                cpu.uncertain_ok[i] != gpu.stats.uncertain_ok[i] ||
                cpu.uncertain_failed[i] != gpu.stats.uncertain_failed[i]) {
                ++mismatch;
                continue;
            }
            const double bound = 2.0 * options.margins.opl_error_factor * mko::kFloatUnitRoundoff *
                                     cpu.optical_length[i] *
                                     mko::test::residual_scale_norm(c.batch, layout) +
                                 2.0 * options.margins.merit_relative * std::sqrt(cpu.merit[i]);
            EXPECT_LE(std::abs(std::sqrt(cpu.merit[i]) - std::sqrt(gpu.stats.merit[i])), bound)
                << "Variante " << v;
        }
        EXPECT_LE(mismatch, 1);
    }
}

// Abnahme (7) für die adjungierte Jacobi-Matrix: dezentrierte Testfälle, GPU gegen CPU (Status
// gleich, Trace-Zeilen relativ pro Spalte <= kAdjointCpuGpu) und df64 gegen FP64.
TEST(CudaDecenter, AdjointJacobianMatchesCpu) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    for (const auto& c : mko::test::decentered_adjoint_cases()) {
        const mko::test::FdProblem& p = c->p;
        const int P = p.set.size();
        const mko::AdjointJacobian cpu = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, c->options);
        const mko::AdjointJacobian gpu =
            mko::cuda::adjoint_jacobian_gpu(engine, p.set, p.base, p.x0, c->options);
        const mko::AdjointJacobian gdf = mko::cuda::adjoint_jacobian_gpu(
            engine, p.set, p.base, p.x0, c->options, Precision::Df64);
        ASSERT_EQ(gpu.shape, cpu.shape) << c->name;
        EXPECT_TRUE(same_bytes(gpu.status, cpu.status)) << c->name;
        const std::int64_t rv = cpu.shape.ray_rows;
        const double dev =
            max_column_deviation(mko::test::row_block(mko::test::run_columns(gpu, 0), P, 0, rv),
                                 mko::test::row_block(mko::test::run_columns(cpu, 0), P, 0, rv), P);
        const double ddf =
            max_column_deviation(mko::test::row_block(mko::test::run_columns(gdf, 0), P, 0, rv),
                                 mko::test::row_block(mko::test::run_columns(gpu, 0), P, 0, rv), P);
        std::printf("[ Info     ] %s: J GPU gegen CPU größte Spalte %.2e, df64 gegen FP64 %.2e\n",
                    c->name.c_str(), dev, ddf);
        EXPECT_LT(dev, mko::test::kAdjointCpuGpu) << c->name;
        EXPECT_LT(ddf, mko::test::kAdjointDf64) << c->name;
    }
}

// Abnahme (7), Determinismus: wiederholte Läufe, Tupelpakete und Tranchen bitgleich.
TEST(CudaDecenter, RepeatedRunsAreBitIdentical) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const SystemBatchHost batch = mko::test::decentered_batch(
        mko::test::load_residual_system("testsystem_mehrlinser.json"), 300, 83250);
    const ResidualSettings q{8, 8, mko::FailureMode::Soft};
    for (const Precision p : {Precision::Fp64, Precision::Df64}) {
        TraceSettings s = trace_settings(q, p);
        const TraceOutput r0 = engine.run(batch, TraceMode::Residuals, s);
        const TraceOutput m0 = engine.run(batch, TraceMode::Merit, s);
        for (int rep = 0; rep < 3; ++rep) {
            s.pack_rays = rep == 0 ? mko::cuda::kDefaultPackRays : (rep == 1 ? 1 : 4096);
            const TraceOutput r = engine.run(batch, TraceMode::Residuals, s);
            const TraceOutput m = engine.run(batch, TraceMode::Merit, s);
            EXPECT_TRUE(same_bytes(r.residuals.residuals, r0.residuals.residuals));
            EXPECT_TRUE(same_bytes(r.residuals.opd, r0.residuals.opd));
            EXPECT_TRUE(same_bytes(r.residuals.extrema.values, r0.residuals.extrema.values));
            EXPECT_TRUE(same_bytes(m.merit.merit, m0.merit.merit));
        }
    }
    mko::EvalSettings es;
    es.grid = 31;
    const mko::EvalResult e0 = engine.evaluate(batch, es);
    const mko::EvalResult e1 = engine.evaluate(batch, es);
    EXPECT_TRUE(same_bytes(e0.opd, e1.opd));
    EXPECT_TRUE(same_bytes(e0.spot_x, e1.spot_x));
    // Tranchen: kleiner Speicher erzwingt mehrere Kernelstarts
    mko::cuda::EngineOptions small;
    small.memory_limit_bytes = std::size_t{2} << 20;
    small.reserve.min_bytes = 0;
    small.reserve.fraction = 0.0;
    TraceEngine tranched(0, small);
    const TraceSettings s = trace_settings(q, Precision::Fp64);
    const TraceOutput rt = tranched.run(batch, TraceMode::Residuals, s);
    const TraceOutput r0 = engine.run(batch, TraceMode::Residuals, s);
    EXPECT_GT(rt.info.tranches, 1);
    EXPECT_TRUE(same_bytes(rt.residuals.residuals, r0.residuals.residuals));
}

// Abnahme (8) auf der GPU: Dezentrierung in −x (und −β) liefert die gespiegelte OPD-Karte.
TEST(CudaDecenter, MirroredDecenterGivesMirroredOpdMap) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const SystemBatchHost a = mko::make_system_batch(
        mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json"), 1);
    SystemBatchHost b = a;
    for (int s = 0; s < a.surfaces(); ++s) {
        b.decenter_tilt(0, s, 0) = -a.decenter_tilt(0, s, 0);
        b.decenter_tilt(0, s, 3) = -a.decenter_tilt(0, s, 3);
    }
    mko::EvalSettings es;
    es.grid = 33;
    const mko::EvalResult ra = engine.evaluate(a, es);
    const mko::EvalResult rb = engine.evaluate(b, es);
    const int N = es.grid;
    std::vector<std::int64_t> at(static_cast<std::size_t>(N * N), -1);
    for (std::int64_t k = 0; k < ra.samples(); ++k) {
        const auto ku = static_cast<std::size_t>(k);
        at[static_cast<std::size_t>(ra.grid.row[ku] * N + ra.grid.column[ku])] = k;
    }
    double worst = 0.0;
    for (int f = 0; f < a.fields(); ++f) {
        for (int l = 0; l < a.wavelengths(); ++l) {
            for (std::int64_t k = 0; k < ra.samples(); ++k) {
                const auto ku = static_cast<std::size_t>(k);
                const std::int64_t km = at[static_cast<std::size_t>(ra.grid.row[ku] * N +
                                                                    (N - 1 - ra.grid.column[ku]))];
                worst = std::max(worst, std::abs(ra.opd[ra.ray_index(0, f, l, k)] -
                                                 rb.opd[rb.ray_index(0, f, l, km)]));
            }
        }
    }
    std::printf("[ Info     ] GPU Spiegelung: OPD %.1e mm\n", worst);
    EXPECT_LT(worst, mko::test::kDecenterMirror);
}

// Einzelstrahl-Trace (trace_rays_gpu) durch transformierte Flächen gegen die CPU, mit und ohne
// vorab berechnete Sinus und Kosinus.
TEST(CudaDecenter, SingleRayTraceMatchesCpu) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls =
        mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json");
    std::vector<mko::Ray<double>> in;
    const mko::PupilQuadrature<double> q = mko::make_pupil_quadrature(8, 16, mko::PupilMode::Full);
    for (int k = 0; k < mko::sample_count(q); ++k) {
        const mko::PupilSample<double> s = mko::sample(q, k);
        in.push_back(mko::collimated_ray(4.0, 0.04, s.rho, s.phi));
    }
    for (const bool trig : {true, false}) {
        mko::SurfaceTableView<double> table = ls.table.view();
        if (!trig) {
            table.tilt_trig = nullptr;
        }
        std::vector<mko::Ray<double>> out;
        std::vector<unsigned> status;
        std::vector<int> failed;
        ASSERT_TRUE(mko::cuda::trace_rays_gpu(table, 1, in, out, status, failed));
        double worst = 0.0;
        for (std::size_t i = 0; i < in.size(); ++i) {
            mko::Ray<double> r = in[i];
            const TraceStatus st = mko::trace_ray(ls.table.view(), 1, r, nullptr);
            ASSERT_EQ(static_cast<unsigned>(st), status[i]);
            worst = std::max({worst, mko::norm(r.p - out[i].p), std::abs(r.opl - out[i].opl)});
        }
        std::printf("[ Info     ] Einzelstrahlen GPU gegen CPU (%s): %.1e mm\n",
                    trig ? "tilt_trig" : "msin/mcos", worst);
        EXPECT_LT(worst, mko::test::kCpuGpuFp64);
    }
}

// SystemBatch im Gerätespeicher: Transformationsdaten hoch- und wieder herunterladen; koaxiale
// Batches belegen dafür keinen Speicher.
TEST(CudaDecenter, DeviceBatchRoundTrip) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost b = mko::test::decentered_batch(
        mko::test::load_residual_system("testsystem_mehrlinser.json"), 7, 83260);
    const mko::cuda::DeviceSystemBatch device(b);
    const SystemBatchHost all = device.download(); // ohne buchdahl_coeff (nur Host)
    EXPECT_TRUE(same_bytes(
        std::vector<double>(all.decenter_tilt_data().begin(), all.decenter_tilt_data().end()),
        std::vector<double>(b.decenter_tilt_data().begin(), b.decenter_tilt_data().end())));
    EXPECT_TRUE(all.topology() == b.topology());
    mko::cuda::DeviceSystemBatch slice(0);
    slice.upload(b, 4, 3);
    SystemBatchHost back(b.topology(), 3);
    slice.download(back, 0);
    for (int v = 0; v < 3; ++v) {
        for (int s = 0; s < b.surfaces(); ++s) {
            for (int k = 0; k < 4; ++k) {
                EXPECT_EQ(back.decenter_tilt(v, s, k), b.decenter_tilt(v + 4, s, k));
            }
        }
    }
    const SystemBatchHost coax =
        mko::make_system_batch(mko::test::load_residual_system("testsystem_mehrlinser.json"), 7);
    EXPECT_EQ(mko::make_batch_layout(coax.topology(), 7).decenter_tilt.bytes, 0u);
    EXPECT_EQ(mko::cuda::DeviceSystemBatch(coax).view().decenter_tilt, nullptr);
}
