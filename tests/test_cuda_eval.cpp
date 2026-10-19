// Kernel EVAL (M8-T1) über die TraceEngine (nur mit MIKROPTIM_ENABLE_CUDA): dieselben
// analytischen Orakel wie test_eval.cpp auf der GPU, Abgleich mit der CPU-Referenz (gestörte
// Varianten mit versagten Strahlen, Geisterplatten, Punktquelle), Determinismus, Tranchen und
// Fehlerbehandlung.

#include "eval_test_systems.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/eval.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/eval/eval.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using mko::EvalReference;
using mko::EvalResult;
using mko::EvalSettings;
using mko::SystemBatchHost;
using mko::cuda::EngineOptions;
using mko::cuda::TraceEngine;
using mko::test::kCpuGpuFp64;
using mko::test::kCpuGpuFp64Relative;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

mko::test::EvalFn gpu(TraceEngine& engine) {
    return [&engine](const SystemBatchHost& b, const EvalSettings& s) {
        return engine.evaluate(b, s);
    };
}

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool identical(const EvalResult& a, const EvalResult& b) {
    return same_bytes(a.opd, b.opd) && same_bytes(a.spot_x, b.spot_x) &&
           same_bytes(a.spot_y, b.spot_y) && same_bytes(a.status, b.status) &&
           same_bytes(a.field_rays, b.field_rays) && same_bytes(a.field_status, b.field_status);
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Abnahme (analytische Orakel) auf der GPU
// ---------------------------------------------------------------------------------------------

TEST(CudaEval, AplanaticPointSourceHasVanishingOpd) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    mko::test::check_aplanatic(gpu(engine), "GPU");
}

TEST(CudaEval, SingleLensGridMatchesSeidelAndGaussModel) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    mko::test::check_lens_grid(gpu(engine), "GPU");
}

TEST(CudaEval, GridRmsConvergesToGaussQuadrature) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    mko::test::check_grid_convergence(gpu(engine), "GPU", 1024);
}

TEST(CudaEval, DistortionAndFieldCurvatureOfSingleSurface) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    mko::test::check_single_surfaces(gpu(engine), "GPU");
}

// ---------------------------------------------------------------------------------------------
// GPU gegen CPU
// ---------------------------------------------------------------------------------------------

// (6) Pro Strahl innerhalb max(kCpuGpuFp64, kCpuGpuFp64Relative·|x|) (OPD, Spot, Feldstrahlen),
// gleicher Status und NaN an denselben Stellen; Kennzahlen entsprechend. Gestörte Varianten
// des Mehrlinsers (mit versagten Strahlen), Einzellinse mit Geisterplatten, Punktquelle neben
// der Achse; beide Lagen der Referenzsphäre, volle und halbe Pupille.
TEST(CudaEval, MatchesCpu) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    struct Case {
        std::string name;
        SystemBatchHost batch;
    };
    std::vector<Case> cases;
    cases.push_back({"Mehrlinser, gestört",
                     mko::test::perturbed_batch(
                         mko::test::load_residual_system("testsystem_mehrlinser.json"), 60, 7)});
    cases.push_back(
        {"Einzellinse mit Geisterplatten, gestört",
         mko::test::perturbed_batch(
             mko::test::load_residual_system("einzellinse_geisterplatte.json"), 20, 3)});
    {
        mko::SystemDescription point =
            mko::test::load_residual_system("einzellinse_bk7.json").system;
        point.source.type = mko::SourceType::Point;
        point.source.field_angles_deg.clear();
        point.source.field_weights.clear();
        point.source.numerical_aperture = 0.05;
        point.source.point = {0.0, 2.0, -300.0};
        point.surfaces.back().thickness = 120.0;
        mko::LoadedSystem ls = mko::prepare_system(point, MKO_SOURCE_DIR "/data/systems");
        SystemBatchHost b = mko::make_system_batch(ls, 2);
        b.field_target(0, 0) = -0.8; // Bildpunkt neben der Achse (von Hand)
        b.field_target(1, 0) = -0.81;
        cases.push_back({"Punktquelle neben der Achse", std::move(b)});
    }
    for (const Case& c : cases) {
        for (const EvalReference ref : {EvalReference::ExitPupil, EvalReference::Vertex}) {
            for (const bool half : {false, true}) {
                EvalSettings s;
                s.grid = 33;
                s.reference = ref;
                s.half_pupil = half;
                const EvalResult cpu = mko::eval_cpu(c.batch, s);
                const EvalResult gpu_result = engine.evaluate(c.batch, s);
                const mko::test::EvalComparison cmp =
                    mko::test::compare_eval(cpu, gpu_result, kCpuGpuFp64, kCpuGpuFp64Relative);
                EXPECT_LE(cmp.ratio, 1.0) << c.name;
                EXPECT_EQ(cmp.status_mismatch, 0u) << c.name;
                EXPECT_EQ(cmp.nan_mismatch, 0u) << c.name;
                EXPECT_EQ(gpu_result.reference_curvature, cpu.reference_curvature);
                // Kennzahlen aus beiden Ergebnissen: Momente der OPD und des Spots erben die
                // Toleranz pro Strahl; die Brennlinien teilen kleine Differenzen durch den
                // Winkelunterschied der Nachbarstrahlen (Verstärkung ~ 1/δ).
                double max_rms = 0.0;
                double max_focus = 0.0;
                for (int v = 0; v < c.batch.variants(); ++v) {
                    const auto a = mko::eval_statistics(cpu, v);
                    const auto b = mko::eval_statistics(gpu_result, v);
                    for (std::size_t i = 0; i < a.size(); ++i) {
                        if (!a[i].valid()) {
                            EXPECT_FALSE(b[i].valid());
                            continue;
                        }
                        const double tol_rms =
                            std::max(kCpuGpuFp64, kCpuGpuFp64Relative * a[i].pv_opd);
                        EXPECT_NEAR(b[i].rms_opd, a[i].rms_opd, tol_rms);
                        EXPECT_NEAR(
                            b[i].centroid_y, a[i].centroid_y,
                            std::max(kCpuGpuFp64, kCpuGpuFp64Relative * std::abs(a[i].centroid_y)));
                        max_rms = std::max(max_rms, std::abs(b[i].rms_opd - a[i].rms_opd));
                        if (std::isfinite(a[i].focus_tangential) &&
                            std::isfinite(a[i].focus_sagittal)) {
                            const double tol = mko::test::kEvalFocusGpuCpu *
                                               std::max(1.0, std::abs(a[i].focus_tangential) +
                                                                 std::abs(a[i].focus_sagittal));
                            EXPECT_NEAR(b[i].focus_tangential, a[i].focus_tangential, tol);
                            EXPECT_NEAR(b[i].focus_sagittal, a[i].focus_sagittal, tol);
                            max_focus = std::max(
                                {max_focus, std::abs(b[i].focus_tangential - a[i].focus_tangential),
                                 std::abs(b[i].focus_sagittal - a[i].focus_sagittal)});
                        }
                    }
                }
                std::printf("[ Info     ] GPU gegen CPU, %s, %s%s: max |ΔOPD| = %.3e mm, max "
                            "|ΔSpot| = %.3e mm, max |ΔFeldstrahl| = %.3e, Verhältnis zur "
                            "Toleranz %.3f, max |ΔRMS| = %.3e mm, max |ΔBrennlinie| = %.3e mm, "
                            "versagt %zu von %zu\n",
                            c.name.c_str(),
                            ref == EvalReference::Vertex ? "Scheitel" : "Austrittspupille",
                            half ? ", halbe Pupille" : "", cmp.max_opd, cmp.max_spot, cmp.max_field,
                            cmp.ratio, max_rms, max_focus, cpu.failed_count(), cpu.status.size());
            }
        }
    }
}

// Zwei Läufe mit gleichen Eingaben sind bitgleich (ADR-0007), ebenso über mehrere Tranchen.
TEST(CudaEval, RepeatedRunsAndTranchesAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost b = mko::test::perturbed_batch(
        mko::test::load_residual_system("testsystem_mehrlinser.json"), 50, 21);
    EvalSettings s;
    s.grid = 48;
    TraceEngine engine(0);
    const EvalResult first = engine.evaluate(b, s);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(identical(first, engine.evaluate(b, s))) << "Lauf " << i;
    }
    // Grenze für genau 12 Varianten (Reserve 0): 5 Tranchen, bitgleich zum Lauf in einer
    const std::int64_t n = first.samples();
    EngineOptions o;
    o.reserve = mko::MemoryReserve{0, 0.0};
    o.memory_limit_bytes = mko::eval_device_bytes(b.topology(), 12, n);
    TraceEngine limited(0, o);
    EvalResult split;
    const mko::cuda::TraceRunInfo info = limited.evaluate(b, s, split);
    EXPECT_EQ(info.tranche_variants, 12);
    EXPECT_EQ(info.tranches, 5);
    EXPECT_TRUE(identical(first, split));
    // Puffer der anderen Modi bleiben erhalten, wenn genug Platz ist; Lauf danach unverändert
    mko::cuda::TraceSettings ts;
    const mko::cuda::TraceOutput before = engine.run(b, mko::cuda::TraceMode::Merit, ts);
    EXPECT_TRUE(identical(first, engine.evaluate(b, s)));
    const mko::cuda::TraceOutput after = engine.run(b, mko::cuda::TraceMode::Merit, ts);
    EXPECT_EQ(before.merit.merit, after.merit.merit);
}

// Fehlerbehandlung: ungültige Einstellungen vor jeder Gerätearbeit, leerer Batch, zu wenig
// Speicher; die Engine bleibt benutzbar.
TEST(CudaEval, InvalidUseThrows) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost b = mko::make_system_batch(
        mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json"));
    TraceEngine engine(0);
    EvalSettings s;
    s.grid = 0;
    EXPECT_THROW(engine.evaluate(b, s), std::invalid_argument);
    s.grid = 16;
    s.field_ray_offset = -1.0;
    EXPECT_THROW(engine.evaluate(b, s), std::invalid_argument);
    s.field_ray_offset = 1e-3;
    SystemBatchHost empty = b;
    empty.resize(0);
    const EvalResult none = engine.evaluate(empty, s);
    EXPECT_EQ(none.variants, 0);
    EXPECT_TRUE(none.opd.empty());
    EngineOptions o;
    o.reserve = mko::MemoryReserve{0, 0.0};
    o.memory_limit_bytes = 1024;
    TraceEngine tiny(0, o);
    EXPECT_THROW(tiny.evaluate(b, s), mko::cuda::CudaError);
    EXPECT_NO_THROW(engine.evaluate(b, s));

    mko::cuda::DeviceEval dev;
    EvalResult r;
    EXPECT_THROW(dev.download(r, 0), std::logic_error);
    mko::cuda::DeviceSystemBatch db(0);
    EXPECT_THROW(dev.run(db, mko::make_eval_grid(8, false), s, {}), std::logic_error);
}

// Durchsatz (Information, kein Benchmark): Mehrlinser (F = 3, L = 4), wenige Varianten mit
// großem Raster (wenige Blöcke) und viele Varianten mit kleinem Raster.
TEST(CudaEval, LargeGridThroughput) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    struct Config {
        int variants;
        int grid;
    };
    for (const Config c : {Config{1, 512}, Config{10, 256}, Config{400, 64}}) {
        const SystemBatchHost b = mko::make_system_batch(
            mko::load_system(MKO_SOURCE_DIR "/data/systems/testsystem_mehrlinser.json"),
            c.variants);
        EvalSettings s;
        s.grid = c.grid;
        EvalResult r;
        (void)engine.evaluate(b, s, r); // Aufwärmen; danach hat r schon die richtige Größe
        const auto t0 = std::chrono::steady_clock::now();
        const mko::cuda::TraceRunInfo info = engine.evaluate(b, s, r);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        EXPECT_EQ(r.failed_count(), 0u);
        const double rays = static_cast<double>(r.status.size());
        std::printf("[ Info     ] EVAL N = %d (%lld Punkte), Mehrlinser, %d Varianten (%d Blöcke): "
                    "%.0f Strahlen in %.1f ms (%.2e Strahlen/s mit Upload, Download und Puffern; "
                    "Gerät %.1f MB), %d Tranche(n), %s\n",
                    c.grid, static_cast<long long>(r.samples()), c.variants, c.variants * 12, rays,
                    ms, rays / (ms * 1e-3), static_cast<double>(info.device_bytes) / 1e6,
                    info.tranches, engine.device_name().c_str());
    }
}
