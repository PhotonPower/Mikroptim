// Skalare PSF, MTF und Strehl (M8-T2) auf der CPU: Radix-2-FFT gegen die direkte DFT,
// Besselfunktion J₁ des Orakels, Abnahme (erste Nullstelle 0,61 λ/NA, Airy, Beugungs-MTF,
// Strehl gegen Maréchal, Parseval, Kippung) und Ende-zu-Ende über EVAL. Dieselben Prüfungen
// laufen auf der GPU in test_cuda_psf.cpp.

#include "eval_test_systems.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/fft.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/psf_ops.hpp"
#include "psf_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

using mko::EvalResult;
using mko::EvalSettings;
using mko::PsfPupil;
using mko::PsfResult;
using mko::SystemBatchHost;

namespace {

using Complex = std::complex<double>;

mko::test::EvalFn cpu_eval() {
    return [](const SystemBatchHost& b, const EvalSettings& s) { return mko::eval_cpu(b, s); };
}

std::vector<Complex> random_data(std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Complex> v(n);
    for (Complex& c : v) {
        c = {uni(rng), uni(rng)};
    }
    return v;
}

// Direkte DFT mit exp(∓2πi jk/n), Phase über (j·k mod n) exakt reduziert.
std::vector<Complex> direct_dft(const std::vector<Complex>& x, double sign) {
    const std::size_t n = x.size();
    std::vector<Complex> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        Complex s(0.0, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            const double a =
                2.0 * std::numbers::pi * static_cast<double>((j * k) % n) / static_cast<double>(n);
            s += x[j] * Complex(std::cos(a), sign * std::sin(a));
        }
        out[k] = s;
    }
    return out;
}

double max_abs_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// FFT und Orakel
// ---------------------------------------------------------------------------------------------

TEST(Fft, MatchesDirectDftAndRoundTrips) {
    for (const std::size_t n : {1u, 2u, 8u, 64u, 256u}) {
        const std::vector<Complex> x = random_data(n, 7u + static_cast<unsigned>(n));
        std::vector<Complex> f = x;
        mko::fft(f, mko::FftDirection::Forward);
        EXPECT_LT(max_abs_diff(f, direct_dft(x, -1.0)), 1e-12 * static_cast<double>(n)) << n;
        std::vector<Complex> b = x;
        mko::fft(b, mko::FftDirection::Inverse);
        EXPECT_LT(max_abs_diff(b, direct_dft(x, 1.0)), 1e-12 * static_cast<double>(n)) << n;
        // rückwärts nach vorwärts = n-fach
        mko::fft(f, mko::FftDirection::Inverse);
        for (Complex& c : f) {
            c /= static_cast<double>(n);
        }
        EXPECT_LT(max_abs_diff(f, x), 1e-14) << n;
    }
}

TEST(Fft, TwoDimensionalMatchesRowsThenColumns) {
    const int rows = 16;
    const int cols = 32;
    const std::vector<Complex> x = random_data(static_cast<std::size_t>(rows * cols), 11u);
    std::vector<Complex> f = x;
    mko::fft2d(f, rows, cols, mko::FftDirection::Forward);
    // Referenz: 2D-DFT direkt
    double max = 0.0;
    for (int kr = 0; kr < rows; ++kr) {
        for (int kc = 0; kc < cols; ++kc) {
            Complex s(0.0, 0.0);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    const double a = 2.0 * std::numbers::pi *
                                     (static_cast<double>((r * kr) % rows) / rows +
                                      static_cast<double>((c * kc) % cols) / cols);
                    s += x[static_cast<std::size_t>(r * cols + c)] *
                         Complex(std::cos(a), -std::sin(a));
                }
            }
            max = std::max(max, std::abs(s - f[static_cast<std::size_t>(kr * cols + kc)]));
        }
    }
    EXPECT_LT(max, 1e-11);
}

TEST(Fft, RejectsNonPowerOfTwo) {
    std::vector<Complex> x(12);
    EXPECT_THROW(mko::fft(x, mko::FftDirection::Forward), std::invalid_argument);
    EXPECT_THROW(mko::fft2d(x, 3, 4, mko::FftDirection::Forward), std::invalid_argument);
    EXPECT_THROW(mko::fft2d(x, 4, 4, mko::FftDirection::Forward), std::invalid_argument);
    EXPECT_TRUE(mko::is_power_of_two(1));
    EXPECT_FALSE(mko::is_power_of_two(0));
    EXPECT_FALSE(mko::is_power_of_two(6));
}

TEST(PsfOracle, BesselJ1AgainstKnownValues) {
    // Werte aus Abramowitz/Stegun, Tabelle 9.1, und die ersten Nullstellen j₁,ₖ
    EXPECT_NEAR(mko::test::bessel_j1(1.0), 0.4400505857449335, 1e-15);
    EXPECT_NEAR(mko::test::bessel_j1(2.0), 0.5767248077568734, 1e-15);
    EXPECT_NEAR(mko::test::bessel_j1(5.0), -0.3275791375914652, 1e-14);
    EXPECT_NEAR(mko::test::bessel_j1(10.0), 0.04347274616886144, 1e-13);
    EXPECT_NEAR(mko::test::bessel_j1(mko::kBesselJ1FirstZero), 0.0, 1e-15);
    EXPECT_NEAR(mko::test::bessel_j1(7.015586669815619), 0.0, 1e-14);
    EXPECT_NEAR(mko::test::bessel_j1(10.17346813506272), 0.0, 1e-13);
    EXPECT_DOUBLE_EQ(mko::test::airy_intensity(0.0), 1.0);
    EXPECT_NEAR(mko::test::diffraction_mtf(0.0), 1.0, 1e-15);
    EXPECT_NEAR(mko::test::diffraction_mtf(1.0), 0.0, 1e-15);
}

TEST(PsfOps, IndexMappingCentersAndMirrors) {
    const std::int64_t m = 8;
    for (const int sign : {1, -1}) {
        for (std::int64_t i = 0; i < m; ++i) {
            const std::int64_t q = mko::psf_fft_index(i, m, sign);
            EXPECT_EQ(mko::psf_output_index(q, m, sign), i);
        }
    }
    EXPECT_EQ(mko::psf_fft_index(4, m, 1), 0);  // Mitte = Frequenz 0
    EXPECT_EQ(mko::psf_fft_index(5, m, 1), 1);  // +1
    EXPECT_EQ(mko::psf_fft_index(5, m, -1), 7); // gespiegelt: −1
    double re = 1.0;
    double im = 1.0;
    mko::psf_pupil_value(std::nan(""), 1.0, re, im);
    EXPECT_EQ(re, 0.0);
    EXPECT_EQ(im, 0.0);
    mko::psf_pupil_value(0.25e-3, 1e-3, re, im); // Viertelwelle: Phase π/2
    EXPECT_NEAR(re, 0.0, 1e-16);
    EXPECT_NEAR(im, 1.0, 1e-16);
}

// ---------------------------------------------------------------------------------------------
// Einstellungen und Fehler
// ---------------------------------------------------------------------------------------------

TEST(Psf, FftSizeFollowsPadding) {
    EXPECT_EQ(mko::psf_fft_size(64, 4.0), 256);
    EXPECT_EQ(mko::psf_fft_size(64, 2.0), 128);
    EXPECT_EQ(mko::psf_fft_size(100, 4.0), 512);
    EXPECT_EQ(mko::psf_fft_size(33, 2.0), 128); // ≥ 2N = 66
    EXPECT_EQ(mko::psf_fft_size(1, 2.0), 2);
    EXPECT_THROW(mko::psf_fft_size(64, 1.5), std::invalid_argument);
    EXPECT_THROW(mko::psf_fft_size(0, 4.0), std::invalid_argument);
    EXPECT_THROW(mko::psf_fft_size(2048, 8.0), std::invalid_argument); // M = 16384
    EXPECT_EQ(mko::psf_fft_size(2048, 4.0), 8192);
    mko::PsfSettings s;
    s.numerical_aperture = -0.1;
    EXPECT_THROW(mko::check_psf_settings(s), std::invalid_argument);
}

TEST(Psf, PrepareRejectsInvalidInput) {
    const PsfPupil p =
        mko::test::synthetic_pupil(16, 500.0, 0.1, [](double, double) { return 0.0; });
    EXPECT_THROW(mko::prepare_psf_result(p, 16), std::invalid_argument); // < 2N
    EXPECT_THROW(mko::prepare_psf_result(p, 48), std::invalid_argument); // keine Zweierpotenz
    PsfPupil q = p;
    q.na_y = 0.0;
    EXPECT_THROW(mko::prepare_psf_result(q, 32), std::invalid_argument);
    q = p;
    q.opd.pop_back();
    EXPECT_THROW(mko::prepare_psf_result(q, 32), std::invalid_argument);
    const PsfResult r = mko::prepare_psf_result(p, 64);
    EXPECT_EQ(r.psf.size(), 64u * 64u);
    EXPECT_EQ(r.mtf_sagittal.size(), 17u);
    EXPECT_DOUBLE_EQ(r.dx, 500e-6 * 16 / (2.0 * 64 * 0.1));
}

TEST(Psf, PupilFromEvalNeedsNumericalAperture) {
    // Planplatte: afokal, die Feldstrahlen bleiben parallel (∂d/∂y = 0) → NA vorgeben
    const mko::LoadedSystem ls =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/planplatte_bk7.json");
    const SystemBatchHost b = mko::make_system_batch(ls, 1);
    EvalSettings s;
    s.grid = 16;
    const EvalResult er = mko::eval_cpu(b, s);
    EXPECT_THROW(mko::make_psf_pupil(b, er, 0, 0, 0), std::invalid_argument);
    mko::PsfSettings ps;
    ps.numerical_aperture = 0.2;
    const PsfPupil p = mko::make_psf_pupil(b, er, 0, 0, 0, ps);
    EXPECT_EQ(p.na_x, 0.2);
    EXPECT_EQ(p.na_y, 0.2);
    EXPECT_THROW(mko::make_psf_pupil(b, er, 0, 0, 99, ps), std::out_of_range);
}

// ---------------------------------------------------------------------------------------------
// Abnahme
// ---------------------------------------------------------------------------------------------

TEST(Psf, PerfectPupilFirstZeroAiryAndMtf) {
    mko::test::check_perfect_pupil(mko::test::psf_cpu_all, "CPU");
}

TEST(Psf, StrehlMatchesMarechal) {
    mko::test::check_marechal(mko::test::psf_cpu_all, "CPU");
}

TEST(Psf, ParsevalWithPartialTransmission) {
    mko::test::check_parseval(mko::test::psf_cpu_all, "CPU");
}

TEST(Psf, TiltShiftsPsfAndMirrorsForDivergentWave) {
    mko::test::check_tilt(mko::test::psf_cpu_all, "CPU");
}

TEST(Psf, AplanaticPointThroughEvalHasAiryZero) {
    mko::test::check_aplanatic_psf(cpu_eval(), mko::test::psf_cpu_all, "CPU");
}

TEST(Psf, SingleLensStrehlMatchesMarechal) {
    mko::test::check_lens_marechal(cpu_eval(), mko::test::psf_cpu_all, "CPU");
}

TEST(Psf, HalfPupilGivesSamePsf) {
    // Einzellinse mit Feld 1°: halbe Pupille (gespiegelt) gegen volle Pupille
    const mko::test::LensCase lc = mko::test::lens_case(4.0, {0.0, 1.0});
    EvalSettings s;
    s.grid = 48;
    const EvalResult full = mko::eval_cpu(lc.batch, s);
    s.half_pupil = true;
    const EvalResult half = mko::eval_cpu(lc.batch, s);
    const std::vector<PsfResult> a = mko::psf_cpu(lc.batch, full, 1);
    const std::vector<PsfResult> b = mko::psf_cpu(lc.batch, half, 1);
    ASSERT_EQ(a.size(), b.size());
    double max = 0.0;
    for (std::size_t t = 0; t < a.size(); ++t) {
        EXPECT_EQ(a[t].na_y, b[t].na_y);
        for (std::size_t k = 0; k < a[t].psf.size(); ++k) {
            max = std::max(max, std::abs(a[t].psf[k] - b[t].psf[k]));
        }
        EXPECT_NEAR(a[t].strehl, b[t].strehl, 1e-12);
    }
    // OPD bis auf kEvalHalfPupil (mm) gleich: Phase ≈ 2π · 10⁻¹⁴/λ ≈ 10⁻⁷ rad
    EXPECT_LT(max, 1e-6);
    // Feld 1° bei der Mittelwellenlänge: NA aus den Feldstrahlen bestimmt
    const PsfResult& off = a[static_cast<std::size_t>(1 * lc.batch.wavelengths() + 1)];
    EXPECT_GT(off.na_x, 0.0);
    EXPECT_GT(off.na_y, 0.0);
}
