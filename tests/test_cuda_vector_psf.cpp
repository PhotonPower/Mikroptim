// Vektorielle PSF auf der GPU (M8-T5, VectorPsfEngine mit cuFFT; nur mit MIKROPTIM_ENABLE_CUDA):
// dieselben analytischen Orakel wie test_vector_psf.cpp, Abgleich mit der CPU-Referenz (eigene
// Radix-2-FFT) für alle Polarisationen, Determinismus, Tranchen, Wiederverwendung von Plänen
// und Puffern, Fehlerbehandlung.

#include "eval_test_systems.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/cuda/vector_psf.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/vector_psf.hpp"
#include "tolerances.hpp"
#include "vector_psf_test_systems.hpp"

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
using mko::Polarization;
using mko::SystemBatchHost;
using mko::VectorPsfPupil;
using mko::VectorPsfResult;
using mko::VectorPsfSettings;
using mko::cuda::EngineOptions;
using mko::cuda::TraceEngine;
using mko::cuda::VectorPsfEngine;
using mko::test::kPsfCpuGpu;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

mko::test::VectorPsfFn gpu(VectorPsfEngine& engine) {
    return [&engine](const std::vector<VectorPsfPupil>& pupils, int m, const VectorPsfSettings& s) {
        return engine.compute(pupils, m, s);
    };
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

bool identical(const std::vector<VectorPsfResult>& a, const std::vector<VectorPsfResult>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t t = 0; t < a.size(); ++t) {
        if (!same_bytes(a[t].psf, b[t].psf) || !same_bytes(a[t].mtf_sagittal, b[t].mtf_sagittal) ||
            !same_bytes(a[t].mtf_tangential, b[t].mtf_tangential)) {
            return false;
        }
        for (std::size_t c = 0; c < 3; ++c) {
            if (!same_bytes(a[t].component[c], b[t].component[c])) {
                return false;
            }
        }
    }
    return true;
}

// Größte absolute Abweichung von PSF, Komponenten, MTF und Kennzahlen zweier Ergebnislisten.
struct VectorComparison {
    double psf = 0.0;
    double component = 0.0;
    double mtf = 0.0;
    double figures = 0.0; // Strehl (FFT), Maximum, Energie, Anteile, FWHM relativ
    std::size_t nan_mismatch = 0;
};

VectorComparison compare(const std::vector<VectorPsfResult>& a,
                         const std::vector<VectorPsfResult>& b) {
    VectorComparison c;
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
        for (std::size_t q = 0; q < 3; ++q) {
            cmp(a[t].component[q], b[t].component[q], c.component);
            c.figures =
                std::max({c.figures, std::abs(a[t].energy_fraction[q] - b[t].energy_fraction[q]),
                          std::abs(a[t].component_peak[q] - b[t].component_peak[q])});
        }
        cmp(a[t].mtf_sagittal, b[t].mtf_sagittal, c.mtf);
        cmp(a[t].mtf_tangential, b[t].mtf_tangential, c.mtf);
        c.figures =
            std::max({c.figures, std::abs(a[t].strehl_fft - b[t].strehl_fft),
                      std::abs(a[t].peak - b[t].peak), std::abs(a[t].energy - b[t].energy)});
        // direkt ausgewertete Größen: dieselbe Host-Funktion auf derselben Pupille
        EXPECT_EQ(a[t].strehl_origin, b[t].strehl_origin);
        EXPECT_EQ(a[t].strehl, b[t].strehl);
        EXPECT_EQ(a[t].reference_intensity, b[t].reference_intensity);
        EXPECT_EQ(a[t].transmitted, b[t].transmitted);
        EXPECT_EQ(a[t].evanescent, b[t].evanescent);
        // FWHM: gleicher Abtastpunkt des Maximums, dann dieselbe Bisektion
        if (std::isfinite(a[t].fwhm_x) || std::isfinite(b[t].fwhm_x)) {
            c.figures = std::max({c.figures, std::abs(a[t].fwhm_x / b[t].fwhm_x - 1.0),
                                  std::abs(a[t].fwhm_y / b[t].fwhm_y - 1.0)});
        }
    }
    return c;
}

// Pupillen mit Aberrationen und Ausfall: Einzellinse a = 3 mm mit Feldern 0°, 1°, 2° (Hauptstrahl
// geneigt, Eingangsbündel geneigt), alle Tupel der Variante 1 über EVAL auf der CPU, dazu
// zufällige Ausfälle.
std::vector<VectorPsfPupil> aberrated_pupils(int grid) {
    const mko::test::LensCase lc = mko::test::lens_case(3.0, {0.0, 1.0, 2.0});
    EvalSettings s;
    s.grid = grid;
    const EvalResult er = mko::eval_cpu(lc.batch, s);
    std::vector<VectorPsfPupil> pupils = mko::make_vector_psf_pupils(lc.batch, er, 1);
    std::mt19937_64 rng(43);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (std::size_t t = 0; t < pupils.size(); t += 2) {
        for (double& v : pupils[t].pupil.opd) {
            if (uni(rng) < 0.05) {
                v = mko::PsfResult::kNaN;
            }
        }
    }
    return pupils;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Abnahme (analytische Orakel) auf der GPU
// ---------------------------------------------------------------------------------------------

TEST(CudaVectorPsf, SmallNaApproachesScalarPsf) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    mko::test::check_vector_scalar_limit(gpu(engine), "GPU");
}

TEST(CudaVectorPsf, HighNaMatchesRichardsWolfQuadrature) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    mko::test::check_vector_richards_wolf(gpu(engine), "GPU");
}

TEST(CudaVectorPsf, DefocusCompensatesPupilPhase) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    mko::test::check_vector_defocus_sign(gpu(engine), "GPU");
}

TEST(CudaVectorPsf, ParsevalAndDirectEvaluation) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    mko::test::check_vector_parseval(gpu(engine), "GPU");
}

TEST(CudaVectorPsf, SymmetryAndPolarizationRotation) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    mko::test::check_vector_symmetry(gpu(engine), "GPU");
}

TEST(CudaVectorPsf, StrehlFollowsWeightedMarechal) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    mko::test::check_vector_marechal(gpu(engine), "GPU");
}

TEST(CudaVectorPsf, AplanaticPointThroughEvalMatchesQuadrature) {
    MKO_REQUIRE_GPU();
    TraceEngine trace(0);
    VectorPsfEngine engine(0);
    mko::test::check_vector_aplanatic(gpu_eval(trace), gpu(engine), "GPU");
}

// ---------------------------------------------------------------------------------------------
// GPU gegen CPU, Determinismus, Tranchen
// ---------------------------------------------------------------------------------------------

TEST(CudaVectorPsf, MatchesCpuReference) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    VectorComparison worst;
    const auto track = [&worst](const VectorComparison& d) {
        EXPECT_EQ(d.nan_mismatch, 0u);
        EXPECT_LT(d.psf, kPsfCpuGpu);
        EXPECT_LT(d.component, kPsfCpuGpu);
        EXPECT_LT(d.mtf, kPsfCpuGpu);
        EXPECT_LT(d.figures, 1e-12);
        worst.psf = std::max(worst.psf, d.psf);
        worst.component = std::max(worst.component, d.component);
        worst.mtf = std::max(worst.mtf, d.mtf);
        worst.figures = std::max(worst.figures, d.figures);
    };
    const double lambda = mko::test::kPsfLambdaNm * 1e-6;
    for (const int grid : {32, 64}) {
        const std::vector<VectorPsfPupil> pupils = aberrated_pupils(grid);
        for (const Polarization pol : {Polarization::X, Polarization::Y, Polarization::Circular,
                                       Polarization::Unpolarized}) {
            for (const double q : {2.0, 4.0}) {
                const int m = mko::psf_fft_size(grid, q);
                const VectorPsfSettings s =
                    mko::test::polarization_settings(pol, q == 4.0 ? 0.3 * lambda : 0.0);
                track(compare(engine.compute(pupils, m, s),
                              mko::test::vector_psf_cpu_all(pupils, m, s)));
            }
        }
    }
    // hohe NA mit geneigtem Hauptstrahl (Teile der Pupille nicht ausbreitungsfähig), Spiegelung
    // und anisotrope NA
    std::vector<VectorPsfPupil> pupils = aberrated_pupils(48);
    for (VectorPsfPupil& p : pupils) {
        p.pupil.na_x = 0.9;
        p.pupil.na_y = 0.8;
        p.pupil.flip_x = true;
        p.chief_y = 0.25;
    }
    for (const Polarization pol : {Polarization::Circular, Polarization::Unpolarized}) {
        const VectorPsfSettings s = mko::test::polarization_settings(pol, -0.2 * lambda);
        const std::vector<VectorPsfResult> g = engine.compute(pupils, 128, s);
        EXPECT_GT(g[0].evanescent, 0);
        track(compare(g, mko::test::vector_psf_cpu_all(pupils, 128, s)));
    }
    std::printf("[ Info     ] GPU gegen CPU (Einzellinse a = 3 mm, 3 Felder × 3 Wellenlängen, 5 %% "
                "Ausfall, alle Polarisationen, Defokus, N = 32/64, Q = 2/4; hohe NA mit "
                "geneigtem Hauptstrahl, Spiegelung, anisotrope NA): max |ΔPSF| = %.2e, "
                "|ΔKomponenten| = %.2e, |ΔMTF| = %.2e, Kennzahlen %.2e\n",
                worst.psf, worst.component, worst.mtf, worst.figures);
}

TEST(CudaVectorPsf, RepeatedRunsAndTranchesAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const std::vector<VectorPsfPupil> pupils = aberrated_pupils(32);
    const int m = 128;
    const VectorPsfSettings s = mko::test::polarization_settings(Polarization::Unpolarized);
    VectorPsfEngine engine(0);
    const std::vector<VectorPsfResult> first = engine.compute(pupils, m, s);
    EXPECT_EQ(engine.last_run().tranches, 1);
    EXPECT_EQ(engine.last_run().tranche, static_cast<int>(pupils.size()));
    for (int run = 1; run < 10; ++run) {
        EXPECT_TRUE(identical(engine.compute(pupils, m, s), first)) << "Lauf " << run;
    }
    // Künstliche Speichergrenze: mehrere Tranchen (andere Pläne mit kleinerem Batch)
    EngineOptions opt;
    opt.memory_limit_bytes = std::size_t{6} << 20;
    opt.reserve.min_bytes = 0;
    opt.reserve.fraction = 0.0;
    VectorPsfEngine small(0, opt);
    const std::vector<VectorPsfResult> tranched = small.compute(pupils, m, s);
    const mko::cuda::PsfRunInfo info = small.last_run();
    EXPECT_GT(info.tranches, 1);
    EXPECT_LE(info.device_bytes, *opt.memory_limit_bytes);
    const VectorComparison d = compare(tranched, first);
    EXPECT_LT(d.psf, kPsfCpuGpu);
    EXPECT_LT(d.component, kPsfCpuGpu);
    EXPECT_LT(d.mtf, kPsfCpuGpu);
    const bool bitwise = identical(tranched, first);
    std::printf("[ Info     ] Determinismus: 10 Läufe bitgleich; %d Tranchen à %d Pupillen "
                "(cuFFT-Arbeitsspeicher %zu Byte) gegen eine Tranche: %s (max |ΔPSF| = %.2e)\n",
                info.tranches, info.tranche, info.work_bytes,
                bitwise ? "bitgleich" : "nicht bitgleich", d.psf);
    EXPECT_TRUE(identical(small.compute(pupils, m, s), tranched));
}

TEST(CudaVectorPsf, ReusesBuffersAndRejectsInvalidInput) {
    MKO_REQUIRE_GPU();
    VectorPsfEngine engine(0);
    const std::vector<VectorPsfPupil> pupils = aberrated_pupils(32);
    (void)engine.compute(pupils, 64);
    const std::size_t bytes = engine.device_bytes();
    EXPECT_GT(bytes, 0u);
    (void)engine.compute(pupils, 64);
    EXPECT_EQ(engine.device_bytes(), bytes);
    EXPECT_THROW((void)engine.compute(pupils, 32), std::invalid_argument); // M < 2N
    std::vector<VectorPsfPupil> mixed = pupils;
    mixed.push_back(
        mko::test::synthetic_vector_pupil(16, 500.0, 0.1, [](double, double) { return 0.0; }));
    EXPECT_THROW((void)engine.compute(mixed, 64), std::invalid_argument);
    VectorPsfSettings bad;
    bad.defocus = std::nan("");
    EXPECT_THROW((void)engine.compute(pupils, 64, bad), std::invalid_argument);
    EXPECT_TRUE(engine.compute(std::vector<VectorPsfPupil>{}, 64).empty());
    engine.release();
    EXPECT_EQ(engine.device_bytes(), 0u);
    EXPECT_THROW({ VectorPsfEngine e(mko::cuda::device_count()); }, std::invalid_argument);
    // Nicht einmal eine Pupille passt
    EngineOptions opt;
    opt.memory_limit_bytes = std::size_t{1} << 16;
    opt.reserve.min_bytes = 0;
    opt.reserve.fraction = 0.0;
    VectorPsfEngine tiny(0, opt);
    EXPECT_THROW((void)tiny.compute(pupils, 64), mko::cuda::CudaError);
}

TEST(CudaVectorPsf, BatchOverloadMatchesCpu) {
    MKO_REQUIRE_GPU();
    const mko::test::LensCase lc = mko::test::lens_case(2.0, {0.0, 1.5});
    EvalSettings s;
    s.grid = 32;
    const EvalResult er = mko::eval_cpu(lc.batch, s);
    mko::PsfSettings ps;
    ps.padding = 2.0;
    const VectorPsfSettings vs = mko::test::polarization_settings(Polarization::X);
    VectorPsfEngine engine(0);
    const VectorComparison d = compare(engine.compute(lc.batch, er, 1, ps, vs),
                                       mko::vector_psf_cpu(lc.batch, er, 1, ps, vs));
    EXPECT_LT(d.psf, kPsfCpuGpu);
    EXPECT_LT(d.component, kPsfCpuGpu);
}
