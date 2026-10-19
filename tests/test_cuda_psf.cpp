// Skalare PSF auf der GPU (M8-T2, PsfEngine mit cuFFT; nur mit MIKROPTIM_ENABLE_CUDA): dieselben
// analytischen Orakel wie test_psf.cpp, Abgleich mit der CPU-Referenz (eigene Radix-2-FFT),
// Determinismus, Tranchen, Wiederverwendung von Plan und Puffern, Fehlerbehandlung.

#include "eval_test_systems.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/psf.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "psf_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using mko::EvalResult;
using mko::EvalSettings;
using mko::PsfPupil;
using mko::PsfResult;
using mko::SystemBatchHost;
using mko::cuda::EngineOptions;
using mko::cuda::PsfEngine;
using mko::cuda::TraceEngine;
using mko::test::kPsfCpuGpu;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

mko::test::PsfFn gpu(PsfEngine& engine) {
    return
        [&engine](const std::vector<PsfPupil>& pupils, int m) { return engine.compute(pupils, m); };
}

mko::test::EvalFn gpu_eval(TraceEngine& engine) {
    return [&engine](const SystemBatchHost& b, const EvalSettings& s) {
        return engine.evaluate(b, s);
    };
}

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool identical(const std::vector<PsfResult>& a, const std::vector<PsfResult>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t t = 0; t < a.size(); ++t) {
        if (!same_bytes(a[t].psf, b[t].psf) || !same_bytes(a[t].mtf_sagittal, b[t].mtf_sagittal) ||
            !same_bytes(a[t].mtf_tangential, b[t].mtf_tangential)) {
            return false;
        }
    }
    return true;
}

// Größte absolute Abweichung von PSF und MTF zweier Ergebnislisten (NaN an denselben Stellen).
struct PsfComparison {
    double psf = 0.0;
    double mtf = 0.0;
    double strehl = 0.0;
    std::size_t nan_mismatch = 0;
};

PsfComparison compare(const std::vector<PsfResult>& a, const std::vector<PsfResult>& b) {
    PsfComparison c;
    EXPECT_EQ(a.size(), b.size());
    const auto cmp = [&c](const std::vector<double>& x, const std::vector<double>& y, double& max) {
        EXPECT_EQ(x.size(), y.size());
        for (std::size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
            if (std::isnan(x[i]) || std::isnan(y[i])) {
                c.nan_mismatch += std::isnan(x[i]) != std::isnan(y[i]) ? 1 : 0;
                continue;
            }
            max = std::max(max, std::abs(x[i] - y[i]));
        }
    };
    for (std::size_t t = 0; t < std::min(a.size(), b.size()); ++t) {
        cmp(a[t].psf, b[t].psf, c.psf);
        cmp(a[t].mtf_sagittal, b[t].mtf_sagittal, c.mtf);
        cmp(a[t].mtf_tangential, b[t].mtf_tangential, c.mtf);
        c.strehl = std::max({c.strehl, std::abs(a[t].strehl_fft - b[t].strehl_fft),
                             std::abs(a[t].peak - b[t].peak)});
        EXPECT_EQ(a[t].strehl_origin, b[t].strehl_origin); // dieselbe Host-Funktion
        EXPECT_EQ(a[t].strehl, b[t].strehl);
        EXPECT_EQ(a[t].transmitted, b[t].transmitted);
    }
    return c;
}

// Pupillen mit Aberrationen und Ausfall: Einzellinse a = 5 mm (RMS ≈ 0,12 Wellen) mit Feldern 0°,
// 1°, 2°, alle Tupel der Variante 1 über EVAL auf der CPU, dazu zufällige Ausfälle.
std::vector<PsfPupil> aberrated_pupils(int grid) {
    const mko::test::LensCase lc = mko::test::lens_case(5.0, {0.0, 1.0, 2.0});
    EvalSettings s;
    s.grid = grid;
    const EvalResult er = mko::eval_cpu(lc.batch, s);
    std::vector<PsfPupil> pupils = mko::make_psf_pupils(lc.batch, er, 1);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (std::size_t t = 0; t < pupils.size(); t += 2) {
        for (double& v : pupils[t].opd) {
            if (uni(rng) < 0.05) {
                v = PsfResult::kNaN;
            }
        }
    }
    return pupils;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Abnahme (analytische Orakel) auf der GPU
// ---------------------------------------------------------------------------------------------

TEST(CudaPsf, PerfectPupilFirstZeroAiryAndMtf) {
    MKO_REQUIRE_GPU();
    PsfEngine engine(0);
    mko::test::check_perfect_pupil(gpu(engine), "GPU");
}

TEST(CudaPsf, StrehlMatchesMarechal) {
    MKO_REQUIRE_GPU();
    PsfEngine engine(0);
    mko::test::check_marechal(gpu(engine), "GPU");
}

TEST(CudaPsf, ParsevalWithPartialTransmission) {
    MKO_REQUIRE_GPU();
    PsfEngine engine(0);
    mko::test::check_parseval(gpu(engine), "GPU");
}

TEST(CudaPsf, TiltShiftsPsfAndMirrorsForDivergentWave) {
    MKO_REQUIRE_GPU();
    PsfEngine engine(0);
    mko::test::check_tilt(gpu(engine), "GPU");
}

TEST(CudaPsf, AplanaticPointThroughEvalHasAiryZero) {
    MKO_REQUIRE_GPU();
    TraceEngine trace(0);
    PsfEngine engine(0);
    mko::test::check_aplanatic_psf(gpu_eval(trace), gpu(engine), "GPU");
}

TEST(CudaPsf, SingleLensStrehlMatchesMarechal) {
    MKO_REQUIRE_GPU();
    TraceEngine trace(0);
    PsfEngine engine(0);
    mko::test::check_lens_marechal(gpu_eval(trace), gpu(engine), "GPU");
}

// ---------------------------------------------------------------------------------------------
// GPU gegen CPU, Determinismus, Tranchen
// ---------------------------------------------------------------------------------------------

TEST(CudaPsf, MatchesCpuReference) {
    MKO_REQUIRE_GPU();
    PsfEngine engine(0);
    PsfComparison worst;
    for (const int grid : {32, 64, 128}) {
        const std::vector<PsfPupil> pupils = aberrated_pupils(grid);
        for (const double q : {2.0, 4.0}) {
            const int m = mko::psf_fft_size(grid, q);
            const std::vector<PsfResult> g = engine.compute(pupils, m);
            const std::vector<PsfResult> c = mko::test::psf_cpu_all(pupils, m);
            const PsfComparison d = compare(g, c);
            EXPECT_EQ(d.nan_mismatch, 0u);
            EXPECT_LT(d.psf, kPsfCpuGpu) << "N = " << grid << ", M = " << m;
            EXPECT_LT(d.mtf, kPsfCpuGpu) << "N = " << grid << ", M = " << m;
            EXPECT_LT(d.strehl, kPsfCpuGpu);
            worst.psf = std::max(worst.psf, d.psf);
            worst.mtf = std::max(worst.mtf, d.mtf);
            worst.strehl = std::max(worst.strehl, d.strehl);
        }
    }
    // Spiegelung (divergente Welle) und anisotrope NA
    std::vector<PsfPupil> pupils = aberrated_pupils(48);
    for (PsfPupil& p : pupils) {
        p.flip_x = true;
        p.na_y *= 0.7;
    }
    const PsfComparison d =
        compare(engine.compute(pupils, 128), mko::test::psf_cpu_all(pupils, 128));
    EXPECT_LT(d.psf, kPsfCpuGpu);
    EXPECT_LT(d.mtf, kPsfCpuGpu);
    worst.psf = std::max(worst.psf, d.psf);
    worst.mtf = std::max(worst.mtf, d.mtf);
    std::printf("[ Info     ] GPU gegen CPU (Einzellinse a = 5 mm, 3 Felder × 3 Wellenlängen, 5 %% "
                "Ausfall, N = 32 … 128, Q = 2 und 4): max |ΔPSF| = %.2e, max |ΔMTF| = %.2e, max "
                "|ΔStrehl|, |ΔMaximum| = %.2e\n",
                worst.psf, worst.mtf, worst.strehl);
}

TEST(CudaPsf, RepeatedRunsAndTranchesAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const std::vector<PsfPupil> pupils = aberrated_pupils(32);
    const int m = 128;
    PsfEngine engine(0);
    const std::vector<PsfResult> first = engine.compute(pupils, m);
    EXPECT_EQ(engine.last_run().tranches, 1);
    EXPECT_EQ(engine.last_run().tranche, static_cast<int>(pupils.size()));
    for (int run = 1; run < 10; ++run) {
        EXPECT_TRUE(identical(engine.compute(pupils, m), first)) << "Lauf " << run;
    }
    // Künstliche Speichergrenze: mehrere Tranchen (anderer Plan mit kleinerem Batch)
    EngineOptions opt;
    opt.memory_limit_bytes = std::size_t{3} << 20;
    opt.reserve.min_bytes = 0;
    opt.reserve.fraction = 0.0;
    PsfEngine small(0, opt);
    const std::vector<PsfResult> tranched = small.compute(pupils, m);
    const mko::cuda::PsfRunInfo info = small.last_run();
    EXPECT_GT(info.tranches, 1);
    EXPECT_LE(info.device_bytes, *opt.memory_limit_bytes);
    const PsfComparison d = compare(tranched, first);
    EXPECT_LT(d.psf, kPsfCpuGpu);
    EXPECT_LT(d.mtf, kPsfCpuGpu);
    const bool bitwise = identical(tranched, first);
    std::printf("[ Info     ] Determinismus: 10 Läufe bitgleich; %d Tranchen à %d Pupillen "
                "(cuFFT-Arbeitsspeicher %zu Byte) gegen eine Tranche: %s (max |ΔPSF| = %.2e)\n",
                info.tranches, info.tranche, info.work_bytes,
                bitwise ? "bitgleich" : "nicht bitgleich", d.psf);
    // Wiederholt mit derselben Grenze: bitgleich (gleicher Plan)
    EXPECT_TRUE(identical(small.compute(pupils, m), tranched));
}

TEST(CudaPsf, ReusesBuffersAndRejectsInvalidInput) {
    MKO_REQUIRE_GPU();
    PsfEngine engine(0);
    const std::vector<PsfPupil> pupils = aberrated_pupils(32);
    (void)engine.compute(pupils, 64);
    const std::size_t bytes = engine.device_bytes();
    EXPECT_GT(bytes, 0u);
    (void)engine.compute(pupils, 64);
    EXPECT_EQ(engine.device_bytes(), bytes);
    EXPECT_THROW((void)engine.compute(pupils, 32), std::invalid_argument); // M < 2N
    std::vector<PsfPupil> mixed = pupils;
    mixed.push_back(mko::test::synthetic_pupil(16, 500.0, 0.1, [](double, double) { return 0.0; }));
    EXPECT_THROW((void)engine.compute(mixed, 64), std::invalid_argument);
    EXPECT_TRUE(engine.compute(std::vector<PsfPupil>{}, 64).empty());
    engine.release();
    EXPECT_EQ(engine.device_bytes(), 0u);
    EXPECT_THROW({ PsfEngine e(mko::cuda::device_count()); }, std::invalid_argument);
    // Nicht einmal eine Pupille passt
    EngineOptions opt;
    opt.memory_limit_bytes = std::size_t{1} << 16;
    opt.reserve.min_bytes = 0;
    opt.reserve.fraction = 0.0;
    PsfEngine tiny(0, opt);
    EXPECT_THROW((void)tiny.compute(pupils, 64), mko::cuda::CudaError);
}
