#pragma once

// Analytische Orakel und Prüfungen der skalaren PSF (M8-T2), gemeinsam für die CPU-Referenz
// (test_psf.cpp) und die PsfEngine auf der GPU (test_cuda_psf.cpp). Die Prüfungen nehmen eine
// Rechenfunktion (psf_cpu oder PsfEngine::compute) und melden die gemessenen Größen. Orakel:
// Airy-Funktion [2 J₁(v)/v]² mit eigener Besselfunktion (Potenzreihe), Beugungs-MTF der
// Kreispupille, Strehl der Defokussierung in geschlossener Form, Maréchal-Näherung.

#include "eval_test_systems.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numbers>
#include <random>
#include <vector>

namespace mko::test {

using PsfFn = std::function<std::vector<PsfResult>(const std::vector<PsfPupil>&, int)>;

inline std::vector<PsfResult> psf_cpu_all(const std::vector<PsfPupil>& pupils, int fft_size) {
    std::vector<PsfResult> out;
    for (const PsfPupil& p : pupils) {
        out.push_back(psf_cpu(p, fft_size));
    }
    return out;
}

// ---- Orakel -------------------------------------------------------------------------------

// J₁(x) aus der Potenzreihe Σ (−1)^m (x/2)^(2m+1) / (m! (m+1)!), für |x| <= 12 auf ~1e-14
// genau (größter Summand bei x = 12 etwa 2·10², Auslöschung ~1e-14).
inline double bessel_j1(double x) {
    const double h = 0.5 * x;
    double term = h;
    double sum = term;
    for (int m = 0; m < 200; ++m) {
        term *= -h * h / (static_cast<double>(m + 1) * static_cast<double>(m + 2));
        sum += term;
        if (std::abs(term) < 1e-18 * std::abs(sum) && m > static_cast<int>(h)) {
            break;
        }
    }
    return sum;
}

// Airy-Muster [2 J₁(v)/v]², v = 2π NA r / λ.
inline double airy_intensity(double v) {
    if (v == 0.0) {
        return 1.0;
    }
    const double a = 2.0 * bessel_j1(v) / v;
    return a * a;
}

// Beugungsbegrenzte MTF der Kreispupille, ν = f/f_c.
inline double diffraction_mtf(double nu) {
    if (nu >= 1.0) {
        return 0.0;
    }
    return 2.0 / std::numbers::pi * (std::acos(nu) - nu * std::sqrt(1.0 - nu * nu));
}

// Strehl der stetigen Kreispupille für die Phase β·g(u), u = ρ² gleichverteilt auf [0, 1]
// (Simpson mit 2·10⁴ Intervallen, glatter Integrand: Fehler ~1e-16).
template <class G>
double continuous_strehl(double beta, G g) {
    const int n = 20000;
    double re = 0.0;
    double im = 0.0;
    for (int k = 0; k <= n; ++k) {
        const double u = static_cast<double>(k) / n;
        const double w = (k == 0 || k == n) ? 1.0 : (k % 2 == 1 ? 4.0 : 2.0);
        re += w * std::cos(beta * g(u));
        im += w * std::sin(beta * g(u));
    }
    re /= 3.0 * n;
    im /= 3.0 * n;
    return re * re + im * im;
}

// ---- Synthetische Pupillen ------------------------------------------------------------------

// Pupille auf dem Raster von EVAL (Zellmitten, (2i + 1 − N)² + (2j + 1 − N)² <= N²) mit der
// OPD opd(x, y) [mm] (NaN: keine Transmission), konvergent (keine Spiegelung).
template <class F>
PsfPupil synthetic_pupil(int n, double lambda_nm, double na, F opd) {
    PsfPupil p;
    p.grid = n;
    p.wavelength_nm = lambda_nm;
    p.na_x = na;
    p.na_y = na;
    p.opd.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), PsfResult::kNaN);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const long long a = 2LL * i + 1 - n;
            const long long b = 2LL * j + 1 - n;
            if (a * a + b * b > static_cast<long long>(n) * n) {
                continue;
            }
            ++p.full_count;
            const double x = static_cast<double>(a) / n;
            const double y = static_cast<double>(b) / n;
            p.opd[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(i)] = opd(x, y);
        }
    }
    return p;
}

inline constexpr double kPsfLambdaNm = 587.56;

// Mittelwert, RMS und normierte zentrale Momente der OPD einer Pupille über die
// transmittierenden Punkte.
struct PupilMoments {
    double mean = 0.0;     // [mm]
    double rms = 0.0;      // [mm]
    double skewness = 0.0; // ⟨d³⟩ / rms³, d = OPD − mean
    double kurtosis = 0.0; // ⟨d⁴⟩ / rms⁴
    double moment6 = 0.0;  // ⟨d⁶⟩ / rms⁶
};

inline PupilMoments pupil_moments(const PsfPupil& p) {
    PupilMoments m;
    double n = 0.0;
    for (const double v : p.opd) {
        if (std::isfinite(v)) {
            m.mean += v;
            n += 1.0;
        }
    }
    m.mean /= n;
    double s2 = 0.0;
    double s3 = 0.0;
    double s4 = 0.0;
    double s6 = 0.0;
    for (const double v : p.opd) {
        if (std::isfinite(v)) {
            const double d = v - m.mean;
            s2 += d * d;
            s3 += d * d * d;
            s4 += d * d * d * d;
            s6 += d * d * d * d * d * d;
        }
    }
    m.rms = std::sqrt(s2 / n);
    const double r2 = m.rms * m.rms;
    m.skewness = (s3 / n) / (r2 * m.rms);
    m.kurtosis = (s4 / n) / (r2 * r2);
    m.moment6 = (s6 / n) / (r2 * r2 * r2);
    return m;
}

// Entwicklung von |⟨e^{iφ}⟩|² − exp(−s²) für eine Phase φ mit Mittelwert 0 und RMS s
// (Momente m_k = ⟨φ^k⟩): (1 − m₂/2 + m₄/24 − m₆/720)² + (m₃/6 − m₅/120)² − exp(−s²)
// = s⁴ (κ/12 − 1/4) + s⁶ (1/6 − κ/24 − μ₆/360 + γ²/36) + O(s⁸). Liefert den Koeffizienten von
// s⁴ einschließlich der Korrektur s² · (…).
inline double marechal_coefficient(const PupilMoments& m, double s) {
    return m.kurtosis / 12.0 - 0.25 +
           s * s *
               (1.0 / 6.0 - m.kurtosis / 24.0 - m.moment6 / 360.0 + m.skewness * m.skewness / 36.0);
}

// ---- Erste Nullstelle -----------------------------------------------------------------------

// Erste Nullstelle der PSF entlang der x-Achse (axis = 0) bzw. y-Achse (axis = 1) ab dem
// idealen Bildpunkt [mm]. Für eine symmetrische Pupille ist die Amplitude auf den Achsen reell;
// a = ±√I wechselt an der Nullstelle das Vorzeichen. Das Vorzeichen wird am ersten Minimum
// umgeschaltet (Nullstelle zwischen dem Minimum und seinem kleineren Nachbarn), dann ein
// kubisches Interpolationspolynom durch vier Abtastwerte und Bisektion (Fehler O(h⁴)).
inline double first_zero(const PsfResult& r, int axis) {
    const int m = r.fft_size;
    const auto at = [&](int k) {
        return axis == 0 ? r.psf_at(m / 2 + k, m / 2) : r.psf_at(m / 2, m / 2 + k);
    };
    int mn = 1;
    while (mn + 1 < m / 2 && !(at(mn) <= at(mn - 1) && at(mn) <= at(mn + 1))) {
        ++mn;
    }
    const int left = at(mn - 1) < at(mn + 1) ? mn - 1 : mn;
    double xs[4];
    double ys[4];
    for (int q = 0; q < 4; ++q) {
        const int k = left - 1 + q;
        xs[q] = k;
        ys[q] = (k <= left ? 1.0 : -1.0) * std::sqrt(at(k));
    }
    const auto poly = [&](double x) {
        double s = 0.0;
        for (int a = 0; a < 4; ++a) {
            double w = ys[a];
            for (int b = 0; b < 4; ++b) {
                if (b != a) {
                    w *= (x - xs[b]) / (xs[a] - xs[b]);
                }
            }
            s += w;
        }
        return s;
    };
    double lo = left;
    double hi = left + 1.0;
    for (int it = 0; it < 100; ++it) {
        const double mid = 0.5 * (lo + hi);
        if ((poly(mid) > 0.0) == (poly(lo) > 0.0)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi) * (axis == 0 ? r.dx : r.dy);
}

// ---- Abnahmeprüfungen (für CPU und GPU) -----------------------------------------------------

// (1) Perfektes System bei kleiner NA (synthetische Pupille OPD = 0, NA 0,05, N = 128, Q = 8):
// erste Nullstelle bei j₁,₁/(2π) · λ/NA ≈ 0,61 λ/NA (Abnahme: < 1 %), PSF auf den Achsen gegen
// [2 J₁(v)/v]² bis zum dritten dunklen Ring, Strehl 1, Energie 1, MTF gegen die Beugungs-MTF
// und gegen die direkt gezählte Autokorrelation des Rasters.
inline void check_perfect_pupil(const PsfFn& compute, const char* backend) {
    const double na = 0.05;
    const int n = 128;
    const PsfPupil p = synthetic_pupil(n, kPsfLambdaNm, na, [](double, double) { return 0.0; });
    const int m = psf_fft_size(n, 8.0);
    ASSERT_EQ(m, 1024);
    const PsfResult r = compute({p}, m).at(0);
    const double lambda = kPsfLambdaNm * 1e-6;
    EXPECT_DOUBLE_EQ(r.dx, lambda * n / (2.0 * m * na));
    const double r0 = kBesselJ1FirstZero / (2.0 * std::numbers::pi) * lambda / na;
    EXPECT_NEAR(r.airy_radius, r0, 1e-15);
    double worst_zero = 0.0;
    for (const int axis : {0, 1}) {
        const double z = first_zero(r, axis);
        const double rel = std::abs(z / r0 - 1.0);
        const double rel_061 = std::abs(z / (0.61 * lambda / na) - 1.0);
        EXPECT_LT(rel, kPsfFirstZero) << "Achse " << axis;
        EXPECT_LT(rel_061, kPsfFirstZero) << "Achse " << axis;
        worst_zero = std::max(worst_zero, rel);
        std::printf("[ Info     ] %s, perfekte Pupille NA 0,05, N = %d, M = %d: erste Nullstelle "
                    "%c-Achse %.6e mm = %.5f λ/NA (j₁,₁/(2π) = %.5f; rel. %.2e, gegen 0,61 λ/NA "
                    "rel. %.2e)\n",
                    backend, n, m, axis == 0 ? 'x' : 'y', z, z * na / lambda,
                    kBesselJ1FirstZero / (2.0 * std::numbers::pi), rel, rel_061);
    }
    // Airy auf den Achsen bis v = 10,2 (drei dunkle Ringe)
    double max_airy = 0.0;
    int points = 0;
    for (int k = 0; k < m / 2; ++k) {
        const double v = 2.0 * std::numbers::pi * na * k * r.dx / lambda;
        if (v > 10.2) {
            break;
        }
        const double expected = airy_intensity(v);
        max_airy = std::max({max_airy, std::abs(r.psf_at(m / 2 + k, m / 2) - expected),
                             std::abs(r.psf_at(m / 2, m / 2 + k) - expected),
                             std::abs(r.psf_at(m / 2 - k, m / 2) - expected)});
        points += 3;
    }
    EXPECT_LT(max_airy, kPsfAiry);
    EXPECT_NEAR(r.strehl_fft, 1.0, kPsfFftIdentity);
    EXPECT_NEAR(r.strehl_origin, 1.0, kPsfFftIdentity);
    EXPECT_NEAR(r.strehl, 1.0, kPsfFftIdentity);
    EXPECT_NEAR(r.peak, 1.0, kPsfFftIdentity);
    EXPECT_NEAR(r.energy, 1.0, kPsfFftIdentity);
    EXPECT_EQ(r.transmitted, p.full_count);

    // MTF: gegen die direkt gezählte Autokorrelation (Punkte p mit p + (k, 0) in der Pupille)
    // und gegen (2/π)(acos ν − ν√(1 − ν²))
    double max_count = 0.0;
    double max_analytic = 0.0;
    ASSERT_EQ(r.mtf_sagittal.size(), static_cast<std::size_t>(n + 1));
    for (int k = 0; k <= n; ++k) {
        long long overlap = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i + k < n; ++i) {
                const auto in = [&](int a, int b) {
                    return std::isfinite(p.opd[static_cast<std::size_t>(b) * n + a]);
                };
                overlap += in(i, j) && in(i + k, j) ? 1 : 0;
            }
        }
        const double counted = static_cast<double>(overlap) / p.full_count;
        const auto ku = static_cast<std::size_t>(k);
        max_count = std::max({max_count, std::abs(r.mtf_sagittal[ku] - counted),
                              std::abs(r.mtf_tangential[ku] - counted)});
        const double nu = static_cast<double>(k) / n;
        max_analytic = std::max({max_analytic, std::abs(r.mtf_sagittal[ku] - diffraction_mtf(nu)),
                                 std::abs(r.mtf_tangential[ku] - diffraction_mtf(nu))});
    }
    EXPECT_LT(max_count, kPsfFftIdentity);
    EXPECT_LT(max_analytic, kPsfMtfGrid);
    EXPECT_NEAR(r.cutoff_x(), 2.0 * na / lambda, 1e-9);
    EXPECT_NEAR(r.mtf_sagittal_at(0.5), diffraction_mtf(0.5), kPsfMtfGrid);
    std::printf("[ Info     ] %s, perfekte Pupille: max |PSF − Airy| = %.3e (%d Punkte bis v = "
                "10,2), Strehl FFT − 1 = %.2e, Energie − 1 = %.2e; MTF gegen gezählte "
                "Autokorrelation %.2e, gegen Beugungs-MTF %.3e (f_c = %.3f Lp/mm)\n",
                backend, max_airy, points, r.strehl_fft - 1.0, r.energy - 1.0, max_count,
                max_analytic, r.cutoff_x());
}

// (2) Strehl gegen Maréchal: Defokus √3(2ρ² − 1) und sphärische Aberration √5(6ρ⁴ − 6ρ² + 1)
// mit RMS σ = 0,02 … 0,07 Wellen auf dem Raster N = 256 (M = 512). Strehl aus der FFT gleich der
// direkten Summe (|⟨e^{iφ}⟩|²); gegen exp(−(2πσ)²) mit dem RMS des Rasters relativ < 1 %
// (Maréchal ist eine Näherung mit Fehler s⁴ (κ/12 − 1/4) + O(s⁶), s = 2πσ, κ Kurtosis der
// Phase: κ = 1,8 bzw. 2,14, also −0,10 s⁴ bzw. −0,07 s⁴, bei σ = 0,07 relativ 0,5 % bzw. 0,3 %);
// der gemessene Koeffizient von s⁴ stimmt mit der Entwicklung aus den Momenten der Phase
// überein (marechal_coefficient, Rest O(s⁴)); gegen die stetige Pupille
// (Defokus: sinc²(√3 s), sphärisch: Simpson) bis auf den Rasterfehler.
inline void check_marechal(const PsfFn& compute, const char* backend) {
    const double lambda = kPsfLambdaNm * 1e-6;
    const int n = 256;
    const int m = psf_fft_size(n, 2.0);
    for (const int kind : {0, 1}) {
        const auto g = [kind](double u) {
            return kind == 0 ? std::sqrt(3.0) * (2.0 * u - 1.0)
                             : std::sqrt(5.0) * (6.0 * u * u - 6.0 * u + 1.0);
        };
        std::vector<PsfPupil> pupils;
        const std::vector<double> sigmas = {0.02, 0.035, 0.05, 0.07};
        for (const double sigma : sigmas) {
            pupils.push_back(synthetic_pupil(n, kPsfLambdaNm, 0.05, [&](double x, double y) {
                return sigma * lambda * g(x * x + y * y);
            }));
        }
        const std::vector<PsfResult> res = compute(pupils, m);
        for (std::size_t q = 0; q < sigmas.size(); ++q) {
            const PsfResult& r = res[q];
            const PupilMoments mo = pupil_moments(pupils[q]);
            const double s = 2.0 * std::numbers::pi * mo.rms / lambda;
            const double marechal = std::exp(-s * s);
            const double rel = std::abs(r.strehl_fft / marechal - 1.0);
            const double coefficient = (r.strehl_origin - marechal) / (s * s * s * s);
            const double predicted = marechal_coefficient(mo, s);
            const double exact = continuous_strehl(2.0 * std::numbers::pi * sigmas[q], g);
            EXPECT_NEAR(r.strehl_fft, r.strehl_origin, kPsfFftIdentity);
            EXPECT_NEAR(r.strehl, r.strehl_origin, kPsfFftIdentity);
            EXPECT_LT(rel, kPsfMarechal) << "σ = " << sigmas[q];
            EXPECT_NEAR(coefficient, predicted, kPsfMarechalOrder * s * s * s * s)
                << "σ = " << sigmas[q];
            EXPECT_NEAR(r.strehl_origin, exact, kPsfGridStrehl) << "σ = " << sigmas[q];
            std::printf("[ Info     ] %s, %s σ = %.3f λ (Raster %.5f λ): Strehl FFT %.8f, direkt "
                        "%.8f (Δ %.1e), Maréchal %.8f (rel. %.2e), s⁴-Koeffizient %.4f "
                        "(Entwicklung bis s⁶: %.4f), stetig %.8f (Δ %.1e)\n",
                        backend, kind == 0 ? "Defokus" : "sphärisch", sigmas[q], mo.rms / lambda,
                        r.strehl_fft, r.strehl_origin, r.strehl_fft - r.strehl_origin, marechal,
                        rel, coefficient, predicted, exact, r.strehl_origin - exact);
        }
    }
}

// (3) Energieerhaltung (Parseval) und Kennzahlen bei teilweiser Transmission: zufällige OPD
// (bis 0,3 Wellen) und 10 % ausgefallene Rasterpunkte. Σ PSF · n_voll / M² = transmittierte
// Punkte / n_voll, MTF(0) = 1, Strehl FFT = direkt.
inline void check_parseval(const PsfFn& compute, const char* backend) {
    std::mt19937_64 rng(20261019);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double lambda = kPsfLambdaNm * 1e-6;
    std::vector<PsfPupil> pupils;
    for (int q = 0; q < 3; ++q) {
        pupils.push_back(
            synthetic_pupil(48 + 16 * q, kPsfLambdaNm, 0.1 + 0.2 * q, [&](double x, double y) {
                if (uni(rng) < 0.1) {
                    return PsfResult::kNaN;
                }
                return 0.3 * lambda * (uni(rng) - 0.5 + x * y);
            }));
    }
    // Gleiche Rastergröße für einen Aufruf: pro Pupille einzeln rechnen
    double max_energy = 0.0;
    for (const PsfPupil& p : pupils) {
        const PsfResult r = compute({p}, psf_fft_size(p.grid, 4.0)).at(0);
        const double expected = static_cast<double>(r.transmitted) / p.full_count;
        EXPECT_LT(r.transmitted, p.full_count);
        EXPECT_NEAR(r.energy, expected, kPsfFftIdentity * expected);
        EXPECT_NEAR(r.strehl_fft, r.strehl_origin, kPsfFftIdentity);
        EXPECT_NEAR(r.mtf_sagittal[0], 1.0, kPsfFftIdentity);
        EXPECT_NEAR(r.mtf_tangential[0], 1.0, kPsfFftIdentity);
        EXPECT_NEAR(r.mtf_sagittal.back(), 0.0, kPsfFftIdentity);
        EXPECT_LE(r.peak, 1.0);
        max_energy = std::max(max_energy, std::abs(r.energy / expected - 1.0));
    }
    std::printf("[ Info     ] %s, Parseval (zufällige Pupillen, 10 %% Ausfall): max |Energie / "
                "Transmission − 1| = %.2e\n",
                backend, max_energy);
}

// (4) Kippung und Spiegelung: OPD = t·y verschiebt die PSF um t/NA (konvergent) bzw. −t/NA
// (divergent, flip_y); Maximum und Bezugspunkt von `strehl` dort, Strehl nach Abzug der
// Kippung 1.
inline void check_tilt(const PsfFn& compute, const char* backend) {
    const double lambda = kPsfLambdaNm * 1e-6;
    const double na = 0.1;
    const int n = 64;
    const int m = psf_fft_size(n, 8.0);
    const double dy = lambda * n / (2.0 * m * na);
    const double shift = 10.0 * dy; // zehn Bildpunkte ≈ 0,63 λ/NA (nahe der ersten Nullstelle)
    for (const bool flip : {false, true}) {
        PsfPupil p =
            synthetic_pupil(n, kPsfLambdaNm, na, [&](double, double y) { return shift * na * y; });
        p.flip_y = flip;
        const PsfResult r = compute({p}, m).at(0);
        const double expected = flip ? -shift : shift;
        EXPECT_NEAR(r.peak_y, expected, 1e-9 * dy) << "flip " << flip;
        EXPECT_NEAR(r.peak_x, 0.0, 1e-12);
        EXPECT_NEAR(r.peak, 1.0, kPsfFftIdentity);
        EXPECT_NEAR(r.reference_y, expected, 1e-9 * dy);
        EXPECT_NEAR(r.strehl, 1.0, kPsfFftIdentity);
        EXPECT_LT(r.strehl_origin, 1e-2);
        std::printf("[ Info     ] %s, Kippung %s: Maximum bei y = %+.6e mm (erwartet %+.6e), "
                    "Strehl ohne Kippung %.15f, im idealen Bildpunkt %.3e\n",
                    backend, flip ? "divergent" : "konvergent", r.peak_y, expected, r.strehl,
                    r.strehl_origin);
    }
}

// (5) Ende-zu-Ende über EVAL: aplanatischer Punkt (n = 1,5, sin θ = 0,075 im Glas, Bildraum
// Luft NA' = 0,075/1,5 = 0,05, virtueller Bildpunkt, also divergent), N = 128, Q = 8. NA aus den
// Feldstrahlen exakt (Sinusbedingung), erste Nullstelle < 1 % von 0,61 λ/NA, Strehl 1.
inline void check_aplanatic_psf(const EvalFn& eval, const PsfFn& compute, const char* backend) {
    const AplanaticCase ac = aplanatic_case(1.5, 0.0, 0.075);
    EvalSettings s;
    s.grid = 128;
    const EvalResult er = eval(ac.batch, s);
    ASSERT_EQ(er.failed_count(), 0u);
    const std::vector<PsfPupil> pupils = make_psf_pupils(ac.batch, er, 0);
    ASSERT_EQ(pupils.size(), 1u);
    const PsfPupil& p = pupils[0];
    EXPECT_NEAR(p.na_x, 0.05, kPsfNaFieldRays);
    EXPECT_NEAR(p.na_y, 0.05, kPsfNaFieldRays);
    EXPECT_TRUE(p.flip_x);
    EXPECT_TRUE(p.flip_y);
    const PsfResult r = compute(pupils, psf_fft_size(128, 8.0)).at(0);
    const double lambda = kPsfLambdaNm * 1e-6;
    for (const int axis : {0, 1}) {
        const double z = first_zero(r, axis);
        EXPECT_LT(std::abs(z / (0.61 * lambda / 0.05) - 1.0), kPsfFirstZero);
        EXPECT_LT(std::abs(z / r.airy_radius - 1.0), kPsfFirstZero);
        std::printf("[ Info     ] %s, aplanatischer Punkt über EVAL (NA %.12f / %.12f): erste "
                    "Nullstelle %c %.6e mm, rel. zu j₁,₁/(2π) λ/NA %.2e\n",
                    backend, p.na_x, p.na_y, axis == 0 ? 'x' : 'y', z,
                    std::abs(z / r.airy_radius - 1.0));
    }
    EXPECT_NEAR(r.strehl_fft, 1.0, kPsfFftIdentity);
    EXPECT_NEAR(r.strehl, 1.0, kPsfFftIdentity);
}

// (6) Ende-zu-Ende: Einzellinse bei kleiner Apertur (Bildebene im paraxialen Fokus, sphärische
// Aberration, RMS 0,02 … 0,06 Wellen): Strehl aus der FFT gegen exp(−(2πσ/λ)²) mit dem RMS aus
// eval_statistics (relativ < 1 %), NA aus den Feldstrahlen gegen a/f.
inline void check_lens_marechal(const EvalFn& eval, const PsfFn& compute, const char* backend) {
    for (const double a : {3.0, 3.5, 4.0}) {
        const LensCase lc = lens_case(a);
        EvalSettings s;
        s.grid = 128;
        const EvalResult er = eval(lc.batch, s);
        ASSERT_EQ(er.failed_count(), 0u);
        const int v = 1;
        const int l = 1;
        const std::vector<EvalTupleStats> st = eval_statistics(er, v);
        const PsfPupil p = make_psf_pupil(lc.batch, er, v, 0, l);
        EXPECT_FALSE(p.flip_x);
        EXPECT_FALSE(p.flip_y);
        const PsfResult r = compute({p}, psf_fft_size(128, 4.0)).at(0);
        const double lambda = lc.batch.wavelength_nm(l) * 1e-6;
        const double sigma = st[static_cast<std::size_t>(l)].rms_opd / lambda;
        const double marechal =
            std::exp(-(2.0 * std::numbers::pi * sigma) * (2.0 * std::numbers::pi * sigma));
        EXPECT_NEAR(r.strehl_fft, r.strehl_origin, kPsfFftIdentity);
        EXPECT_LT(std::abs(r.strehl_fft / marechal - 1.0), kPsfMarechal) << "a = " << a;
        std::printf("[ Info     ] %s, Einzellinse a = %.1f mm, NA %.5f: RMS %.4f λ, Strehl FFT "
                    "%.6f, Maréchal %.6f (rel. %.2e), Maximum %.6f bei y = %.2e mm\n",
                    backend, a, p.na_y, sigma, r.strehl_fft, marechal,
                    std::abs(r.strehl_fft / marechal - 1.0), r.peak, r.peak_y);
    }
}

} // namespace mko::test
