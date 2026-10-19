#pragma once

// Analytische Orakel und Prüfungen der vektoriellen PSF (M8-T5), gemeinsam für die CPU-Referenz
// (test_vector_psf.cpp) und die VectorPsfEngine auf der GPU (test_cuda_vector_psf.cpp). Orakel:
// - unabhängige Quadratur des Richards-Wolf-Integrals für die aplanatische Fokussierung einer
//   x-polarisierten, gleichmäßig beleuchteten Pupille (Richards und Wolf 1959; Darstellung wie
//   Novotny und Hecht, "Principles of Nano-Optics", Kap. 3.6): Integrale I₀, I₁, I₂ über θ mit
//   zusammengesetzter Gauß-Legendre-Quadratur und Besselfunktionen J₀, J₁, J₂ aus der
//   Integraldarstellung (Trapezregel über eine Periode, exponentiell konvergent), φ-Integration
//   analytisch;
// - geschlossene Formen der Energieanteile in der Bildebene für die stetige Pupille;
// - die skalare PSF (M8-T2) im Grenzfall kleiner NA.

#include "eval_test_systems.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/vector_psf.hpp"
#include "mikroptim/pupil/gauss_legendre_table.hpp"
#include "psf_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <numbers>
#include <random>
#include <vector>

namespace mko::test {

using VectorPsfFn = std::function<std::vector<VectorPsfResult>(const std::vector<VectorPsfPupil>&,
                                                               int, const VectorPsfSettings&)>;

inline std::vector<VectorPsfResult> vector_psf_cpu_all(const std::vector<VectorPsfPupil>& pupils,
                                                       int fft_size,
                                                       const VectorPsfSettings& settings) {
    std::vector<VectorPsfResult> out;
    for (const VectorPsfPupil& p : pupils) {
        out.push_back(vector_psf_cpu(p, fft_size, settings));
    }
    return out;
}

// ---- Orakel -------------------------------------------------------------------------------

// J_n(x) = (1/2π) ∫ cos(nτ − x sin τ) dτ über eine Periode, Trapezregel mit K = 96 Punkten. Der
// Fehler ist ~J_{K−n}(x), für |x| <= 40 unter 10⁻²⁰ (Rundung ~10⁻¹⁶).
inline double bessel_jn(int n, double x) {
    const int k = 96;
    double sum = 0.0;
    for (int m = 0; m < k; ++m) {
        const double tau = 2.0 * std::numbers::pi * m / k;
        sum += std::cos(n * tau - x * std::sin(tau));
    }
    return sum / k;
}

// Richards-Wolf für die aplanatische Fokussierung einer gleichmäßig beleuchteten,
// x-polarisierten Pupille in ein Medium n (öffnung α = asin(NA/n)), Punkt (ρ, ψ, z) relativ zum
// Fokus, Wellenlänge λ (Vakuum):
//   E_x ∝ I₀ + I₂ cos 2ψ,  E_y ∝ I₂ sin 2ψ,  E_z ∝ −2i I₁ cos ψ,
//   I₀ = ∫ √cos θ sin θ (1 + cos θ) J₀(kρ sin θ) e^{ikz(cos θ − 1)} dθ,
//   I₁ = ∫ √cos θ sin²θ J₁(…) e^{…} dθ,  I₂ = ∫ √cos θ sin θ (1 − cos θ) J₂(…) e^{…} dθ,
// k = 2πn/λ. Die Komponenten sind auf |E_x(0)|² normiert (Maximum der Intensität = 1).
class RichardsWolfOracle {
public:
    RichardsWolfOracle(double na, double lambda_mm, double index = 1.0, int panels = 24)
        : k_(2.0 * std::numbers::pi * index / lambda_mm) {
        const double alpha = std::asin(na / index);
        const int order = 16;
        const int off = detail::gauss_legendre_offset(order);
        for (int p = 0; p < panels; ++p) {
            const double a = alpha * p / panels;
            const double b = alpha * (p + 1) / panels;
            for (int q = 0; q < order; ++q) {
                theta_.push_back(a + (b - a) * detail::kGaussLegendreU[off + q]);
                weight_.push_back((b - a) * detail::kGaussLegendreW[off + q]);
            }
        }
        norm_ = 1.0;
        norm_ = components(0.0, 0.0)[0];
    }

    // {|E_x|², |E_y|², |E_z|²} normiert.
    std::array<double, 3> components(double rho, double psi, double z = 0.0) const {
        std::complex<double> i0;
        std::complex<double> i1;
        std::complex<double> i2;
        for (std::size_t q = 0; q < theta_.size(); ++q) {
            const double c = std::cos(theta_[q]);
            const double s = std::sin(theta_[q]);
            const double x = k_ * rho * s;
            const std::complex<double> ph =
                std::polar(weight_[q] * std::sqrt(c), k_ * z * (c - 1.0));
            i0 += ph * (s * (1.0 + c) * bessel_jn(0, x));
            i1 += ph * (s * s * bessel_jn(1, x));
            i2 += ph * (s * (1.0 - c) * bessel_jn(2, x));
        }
        const std::complex<double> ex = i0 + i2 * std::cos(2.0 * psi);
        const std::complex<double> ey = i2 * std::sin(2.0 * psi);
        const std::complex<double> ez = -2.0 * i1 * std::cos(psi);
        return {std::norm(ex) / norm_, std::norm(ey) / norm_, std::norm(ez) / norm_};
    }

    double intensity(double rho, double psi, double z = 0.0) const {
        const std::array<double, 3> e = components(rho, psi, z);
        return e[0] + e[1] + e[2];
    }

    // Halbwertsbreite in der Fokusebene entlang ψ (Bisektion, Fehler ~10⁻¹⁵ relativ).
    double fwhm(double psi, double lambda_mm) const {
        double lo = 0.0;
        double hi = 2.0 * lambda_mm;
        for (int it = 0; it < 80; ++it) {
            const double mid = 0.5 * (lo + hi);
            (intensity(mid, psi) >= 0.5 ? lo : hi) = mid;
        }
        return lo + hi; // 2 · Mitte
    }

private:
    double k_;
    double norm_ = 1.0;
    std::vector<double> theta_;
    std::vector<double> weight_;
};

// Energieanteile der Komponenten in der Bildebene für die stetige Pupille bei x-Polarisation
// (Parseval: Σ|E_c|² ∝ ∫ W² |R_cx|² ds_x ds_y mit W² = 1/cos θ), c₀ = cos α:
//   Anteil z = (1 − c₀)(2 + c₀)/6,  Anteil y = (1 − c₀)²/24,  Anteil x = Rest.
inline std::array<double, 3> continuous_energy_fractions(double na, double index = 1.0) {
    const double s = na / index;
    const double c0 = std::sqrt(1.0 - s * s);
    const double fz = (1.0 - c0) * (2.0 + c0) / 6.0;
    const double fy = (1.0 - c0) * (1.0 - c0) / 24.0;
    return {1.0 - fy - fz, fy, fz};
}

// Synthetische vektorielle Pupille (Bildraum Luft, Achse, Eingangsbündel entlang z).
template <class F>
VectorPsfPupil synthetic_vector_pupil(int n, double lambda_nm, double na, F opd) {
    VectorPsfPupil vp;
    vp.pupil = synthetic_pupil(n, lambda_nm, na, opd);
    return vp;
}

inline VectorPsfSettings polarization_settings(Polarization p, double defocus = 0.0) {
    VectorPsfSettings s;
    s.polarization = p;
    s.defocus = defocus;
    return s;
}

// ---- Abnahmeprüfungen (für CPU und GPU) -----------------------------------------------------

// (1) Grenzfall kleiner NA: perfekte Pupille, x-Polarisation, N = 128, Q = 4. Die vektorielle
// PSF geht in die skalare (M8-T2) über, max |I_v − I_s| <= kVectorPsfScalarLimit · NA², und die
// Abweichung ist von zweiter Ordnung (Verhältnis NA 0,1 zu 0,05 nahe 4). Anteil von |E_z|² gegen
// die geschlossene Form (≈ NA²/4).
inline void check_vector_scalar_limit(const VectorPsfFn& compute, const char* backend) {
    const int n = 128;
    const int m = psf_fft_size(n, 4.0);
    double deviation[2] = {0.0, 0.0};
    const double nas[2] = {0.05, 0.1};
    for (int q = 0; q < 2; ++q) {
        const VectorPsfPupil vp =
            synthetic_vector_pupil(n, kPsfLambdaNm, nas[q], [](double, double) { return 0.0; });
        const VectorPsfResult r = compute({vp}, m, polarization_settings(Polarization::X)).at(0);
        const PsfResult s = psf_cpu(vp.pupil, m);
        double d = 0.0;
        for (std::size_t k = 0; k < r.psf.size(); ++k) {
            d = std::max(d, std::abs(r.psf[k] - s.psf[k]));
        }
        deviation[q] = d;
        const std::array<double, 3> f = continuous_energy_fractions(nas[q]);
        EXPECT_LT(d, kVectorPsfScalarLimit * nas[q] * nas[q]) << "NA " << nas[q];
        EXPECT_NEAR(r.energy_fraction[2] / f[2], 1.0, 10.0 * kVectorPsfRelative);
        EXPECT_NEAR(r.strehl_fft, 1.0, kVectorPsfIdentity);
        EXPECT_NEAR(r.energy, 1.0, kVectorPsfIdentity);
        std::printf("[ Info     ] %s, kleine NA %.2f (N = %d, M = %d): max |I_vektoriell − "
                    "I_skalar| = %.3e = %.4f NA², Anteil |E_z|² %.4e (stetig %.4e), FWHM x/y = "
                    "%.6f\n",
                    backend, nas[q], n, m, d, d / (nas[q] * nas[q]), r.energy_fraction[2], f[2],
                    r.fwhm_x / r.fwhm_y);
    }
    const double ratio = deviation[1] / deviation[0];
    EXPECT_GT(ratio, 3.6);
    EXPECT_LT(ratio, 4.4);
    std::printf("[ Info     ] %s, Verhältnis der Abweichungen NA 0,1 / 0,05 = %.4f (NA²: 4)\n",
                backend, ratio);
}

// (2) Hohe NA 0,95 (Luft), aberrationsfrei, x-Polarisation, N = 256, Q = 4 (M = 1024): PSF und
// Komponenten auf der x- und y-Achse und der Diagonalen bis 1,5 λ gegen die Quadratur des
// Richards-Wolf-Integrals (absolut < kVectorPsfOracle), Halbwertsbreiten in x und y relativ <
// kVectorPsfRelative, Maximum von |E_z|² und Energieanteile gegen Quadratur bzw. geschlossene
// Form. Defokussierte Ebenen z = 0,5 λ ebenso gegen die Quadratur.
inline void check_vector_richards_wolf(const VectorPsfFn& compute, const char* backend) {
    const double na = 0.95;
    const int n = 256;
    const int m = psf_fft_size(n, 4.0);
    const double lambda = kPsfLambdaNm * 1e-6;
    const RichardsWolfOracle oracle(na, lambda);
    const VectorPsfPupil vp =
        synthetic_vector_pupil(n, kPsfLambdaNm, na, [](double, double) { return 0.0; });
    const std::vector<double> planes = {0.0, 0.5 * lambda};
    std::vector<VectorPsfResult> res;
    for (const double z : planes) {
        res.push_back(compute({vp}, m, polarization_settings(Polarization::X, z)).at(0));
    }
    const VectorPsfResult& r = res[0];
    const int c = m / 2;
    const int reach = static_cast<int>(1.5 * lambda / r.dx);
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        const VectorPsfResult& rp = res[plane];
        const double z = planes[plane];
        double max_total = 0.0;
        double max_comp = 0.0;
        for (int k = 0; k <= reach; ++k) {
            const double rho = k * r.dx;
            // x-Achse (ψ = 0): Intensität, |E_x|², |E_z|²
            const std::array<double, 3> ox = oracle.components(rho, 0.0, z);
            max_total =
                std::max(max_total, std::abs(rp.psf_at(c + k, c) - (ox[0] + ox[1] + ox[2])));
            max_total =
                std::max(max_total, std::abs(rp.psf_at(c - k, c) - (ox[0] + ox[1] + ox[2])));
            max_comp = std::max({max_comp, std::abs(rp.component_at(0, c + k, c) - ox[0]),
                                 std::abs(rp.component_at(2, c + k, c) - ox[2])});
            // y-Achse (ψ = π/2)
            const std::array<double, 3> oy = oracle.components(rho, 0.5 * std::numbers::pi, z);
            max_total =
                std::max(max_total, std::abs(rp.psf_at(c, c + k) - (oy[0] + oy[1] + oy[2])));
            max_comp = std::max(max_comp, std::abs(rp.component_at(0, c, c + k) - oy[0]));
            // Diagonale (ψ = π/4, Abstand √2 k Δx): |E_y|² maximal
            const std::array<double, 3> od =
                oracle.components(std::sqrt(2.0) * rho, 0.25 * std::numbers::pi, z);
            max_total =
                std::max(max_total, std::abs(rp.psf_at(c + k, c + k) - (od[0] + od[1] + od[2])));
            max_comp = std::max({max_comp, std::abs(rp.component_at(0, c + k, c + k) - od[0]),
                                 std::abs(rp.component_at(1, c + k, c + k) - od[1]),
                                 std::abs(rp.component_at(2, c + k, c + k) - od[2])});
        }
        EXPECT_LT(max_total, kVectorPsfOracle) << "z = " << z;
        EXPECT_LT(max_comp, kVectorPsfOracle) << "z = " << z;
        std::printf("[ Info     ] %s, NA 0,95 x-pol., z = %+.3f λ: max |PSF − Richards-Wolf| = "
                    "%.2e, Komponenten %.2e (%d Abstände bis 1,5 λ, x/y/Diagonale)\n",
                    backend, z / lambda, max_total, max_comp, reach + 1);
    }
    // Halbwertsbreiten
    const double fx = oracle.fwhm(0.0, lambda);
    const double fy = oracle.fwhm(0.5 * std::numbers::pi, lambda);
    EXPECT_NEAR(r.fwhm_x / fx, 1.0, kVectorPsfRelative);
    EXPECT_NEAR(r.fwhm_y / fy, 1.0, kVectorPsfRelative);
    EXPECT_GT(r.fwhm_x, 1.3 * r.fwhm_y); // in Polarisationsrichtung gestreckt
    // Maximum von |E_z|² auf der x-Achse (Quadratur: grobe Suche, dann goldener Schnitt)
    const auto ez = [&](double rho) { return oracle.components(rho, 0.0)[2]; };
    int best = 0;
    for (int k = 1; k <= 100; ++k) {
        best = ez(k * 0.01 * lambda) > ez(best * 0.01 * lambda) ? k : best;
    }
    double lo = std::max(0, best - 1) * 0.01 * lambda;
    double hi = (best + 1) * 0.01 * lambda;
    const double golden = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int it = 0; it < 80; ++it) {
        const double a = hi - golden * (hi - lo);
        const double b = lo + golden * (hi - lo);
        if (ez(a) < ez(b)) {
            lo = a;
        } else {
            hi = b;
        }
    }
    const double ez_peak = ez(0.5 * (lo + hi));
    // component_peak ist der größte Abtastwert; |E_z|² ist auf der x-Achse maximal, die
    // Abtastpunkte k Δx dort liegen im Raster.
    double ez_sampled = 0.0;
    for (int k = 0; k <= reach; ++k) {
        ez_sampled = std::max(ez_sampled, ez(k * r.dx));
    }
    EXPECT_NEAR(r.component_peak[2] / ez_sampled, 1.0, kVectorPsfRelative);
    // Energieanteile absolut (in Einheiten der Gesamtenergie)
    const std::array<double, 3> f = continuous_energy_fractions(na);
    for (std::size_t q = 0; q < 3; ++q) {
        EXPECT_NEAR(r.energy_fraction[q], f[q], kVectorPsfRelative) << "Komponente " << q;
    }
    EXPECT_NEAR(r.strehl_fft, 1.0, kVectorPsfIdentity);
    EXPECT_NEAR(r.strehl_origin, 1.0, kVectorPsfIdentity);
    EXPECT_NEAR(r.strehl, 1.0, kVectorPsfIdentity);
    EXPECT_NEAR(r.peak, 1.0, kVectorPsfIdentity);
    EXPECT_NEAR(r.energy, 1.0, kVectorPsfIdentity);
    EXPECT_EQ(r.evanescent, 0);
    std::printf("[ Info     ] %s, NA 0,95 x-pol.: FWHM x = %.6f λ (Quadratur %.6f, rel. %.1e), "
                "FWHM y = %.6f λ (%.6f, rel. %.1e), Verhältnis %.5f; max |E_z|² = %.6f (Quadratur "
                "auf denselben Abtastpunkten %.6f, Maximum der stetigen Funktion %.6f); "
                "Energieanteile x/y/z = %.6f / %.6f / %.6f (stetig %.6f / %.6f / %.6f); "
                "skalarer Airy-Radius %.4f λ\n",
                backend, r.fwhm_x / lambda, fx / lambda, r.fwhm_x / fx - 1.0, r.fwhm_y / lambda,
                fy / lambda, r.fwhm_y / fy - 1.0, r.fwhm_x / r.fwhm_y, r.component_peak[2],
                ez_sampled, ez_peak, r.energy_fraction[0], r.energy_fraction[1],
                r.energy_fraction[2], f[0], f[1], f[2], r.airy_radius / lambda);
    // Defokus z: Intensität im idealen Bildpunkt gegen die Quadratur
    for (std::size_t plane = 1; plane < planes.size(); ++plane) {
        EXPECT_NEAR(res[plane].strehl_origin, oracle.intensity(0.0, 0.0, planes[plane]),
                    kVectorPsfOracle);
    }
}

// (3) Defokus gegen Pupillenphase: OPD = −n' z₀ (s_z − 1) wird in der Ebene z = z₀ genau
// aufgehoben (Strehl 1, PSF perfekt); das legt das Vorzeichen fest (z > 0: Fokus in +z hinter
// der Bildebene, wie die Brennlinien in EVAL). Ohne Defokus ist der Strehl dieser Pupille klein.
inline void check_vector_defocus_sign(const VectorPsfFn& compute, const char* backend) {
    const double na = 0.9;
    const int n = 64;
    const int m = psf_fft_size(n, 4.0);
    const double lambda = kPsfLambdaNm * 1e-6;
    const double z0 = 0.4 * lambda;
    const VectorPsfPupil vp = synthetic_vector_pupil(n, kPsfLambdaNm, na, [&](double x, double y) {
        const double t2 = na * na * (x * x + y * y);
        return z0 * t2 / (1.0 + std::sqrt(1.0 - t2)); // −z₀ (s_z − 1)
    });
    const VectorPsfPupil perfect =
        synthetic_vector_pupil(n, kPsfLambdaNm, na, [](double, double) { return 0.0; });
    const VectorPsfSettings s = polarization_settings(Polarization::Unpolarized, z0);
    const std::vector<VectorPsfResult> r = compute({vp, perfect}, m, s);
    const VectorPsfResult in_focus =
        compute({vp}, m, polarization_settings(Polarization::Unpolarized)).at(0);
    const VectorPsfResult reference =
        compute({perfect}, m, polarization_settings(Polarization::Unpolarized)).at(0);
    EXPECT_NEAR(r[0].strehl, 1.0, kVectorPsfIdentity);
    EXPECT_NEAR(r[0].peak, 1.0, kVectorPsfIdentity);
    double d = 0.0;
    for (std::size_t k = 0; k < r[0].psf.size(); ++k) {
        d = std::max(d, std::abs(r[0].psf[k] - reference.psf[k]));
    }
    EXPECT_LT(d, kVectorPsfIdentity);
    EXPECT_LT(in_focus.strehl, 0.9);
    EXPECT_NEAR(in_focus.strehl, r[1].strehl, kVectorPsfIdentity); // Symmetrie in ±z
    std::printf("[ Info     ] %s, Defokus z₀ = 0,4 λ als Pupillenphase: Strehl in z₀ %.15f, max "
                "|ΔPSF| gegen perfekt %.2e; in der Bildebene %.6f\n",
                backend, r[0].strehl, d, in_focus.strehl);
}

// (4) Energie (Parseval) und Identitäten: zufällige OPD (bis 0,3 λ), 10 % ausgefallene
// Rasterpunkte, NA 0,6 … 0,9, alle Polarisationen. Energie = Σ_trans W² / Σ_voll W² mit
// W = (1 − NA²ρ²)^{−1/4}; Summe der Komponenten = Intensität; Strehl FFT = direkt; direkte
// Auswertung in Bildpunkten = FFT; MTF(0) = 1.
inline void check_vector_parseval(const VectorPsfFn& compute, const char* backend) {
    std::mt19937_64 rng(20261020);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double lambda = kPsfLambdaNm * 1e-6;
    double worst_energy = 0.0;
    double worst_direct = 0.0;
    for (int q = 0; q < 3; ++q) {
        const double na = 0.6 + 0.15 * q;
        const int n = 40 + 12 * q;
        const VectorPsfPupil vp =
            synthetic_vector_pupil(n, kPsfLambdaNm, na, [&](double x, double y) {
                if (uni(rng) < 0.1) {
                    return PsfResult::kNaN;
                }
                return 0.3 * lambda * (uni(rng) - 0.5 + x * y + 0.5 * x);
            });
        double w2_trans = 0.0;
        double w2_full = 0.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const long long a = 2LL * i + 1 - n;
                const long long b = 2LL * j + 1 - n;
                if (a * a + b * b > static_cast<long long>(n) * n) {
                    continue;
                }
                const double rho2 =
                    static_cast<double>(a * a + b * b) / (static_cast<double>(n) * n);
                const double w2 = 1.0 / std::sqrt(1.0 - na * na * rho2);
                w2_full += w2;
                if (std::isfinite(vp.pupil.opd[static_cast<std::size_t>(j) * n + i])) {
                    w2_trans += w2;
                }
            }
        }
        const double expected = w2_trans / w2_full;
        for (const Polarization pol : {Polarization::X, Polarization::Y, Polarization::Circular,
                                       Polarization::Unpolarized}) {
            const VectorPsfSettings s = polarization_settings(pol, 0.1 * lambda * q);
            const int m = psf_fft_size(n, 3.0);
            const VectorPsfResult r = compute({vp}, m, s).at(0);
            EXPECT_NEAR(r.energy / expected, 1.0, kVectorPsfIdentity) << polarization_name(pol);
            worst_energy = std::max(worst_energy, std::abs(r.energy / expected - 1.0));
            EXPECT_NEAR(r.strehl_fft, r.strehl_origin, kVectorPsfIdentity);
            EXPECT_NEAR(r.mtf_sagittal[0], 1.0, kVectorPsfIdentity);
            EXPECT_NEAR(r.mtf_tangential.back(), 0.0, kVectorPsfIdentity);
            EXPECT_NEAR(r.energy_fraction[0] + r.energy_fraction[1] + r.energy_fraction[2], 1.0,
                        kVectorPsfIdentity);
            for (std::size_t k = 0; k < r.psf.size(); k += 97) {
                EXPECT_NEAR(r.component[0][k] + r.component[1][k] + r.component[2][k], r.psf[k],
                            kVectorPsfIdentity);
            }
            // direkte Summe in einigen Bildpunkten (Zeile/Spalte gemischt)
            for (const int off : {0, 3, -5, 11}) {
                const int i = m / 2 + off;
                const int j = m / 2 - 2 * off + 1;
                std::array<double, 3> comp{};
                const double direct = vector_psf_intensity_at(
                    vp, s, r.reference_intensity, (i - m / 2) * r.dx, (j - m / 2) * r.dy, &comp);
                EXPECT_NEAR(direct, r.psf_at(i, j), kVectorPsfIdentity);
                EXPECT_NEAR(comp[2], r.component_at(2, i, j), kVectorPsfIdentity);
                worst_direct = std::max(worst_direct, std::abs(direct - r.psf_at(i, j)));
            }
        }
    }
    std::printf("[ Info     ] %s, Parseval (zufällige Pupillen, 10 %% Ausfall, alle "
                "Polarisationen, Defokus): max |Energie / Σ_trans W² − 1| = %.2e; direkte Summe "
                "gegen FFT max %.2e\n",
                backend, worst_energy, worst_direct);
}

// (5) Symmetrien: Pupille mit sphärischer Aberration (rotationssymmetrisch), NA 0,9, N = 64.
// x-Polarisation: PSF spiegelsymmetrisch zu x und y; y-Polarisation = x-Polarisation um 90°
// gedreht (Raster und Bildpunkte gehen exakt ineinander über); unpolarisiert = (x + y)/2;
// unpolarisiert und zirkular invariant unter 90°-Drehung. Zirkular = unpolarisiert für die
// aberrationsfreie Pupille in der Fokusebene (alle Feldkomponenten reell, der Kreuzterm
// Im(E⁽ˣ⁾*·E⁽ʸ⁾) verschwindet exakt); bei der stetigen rotationssymmetrischen Pupille gilt das
// für jede Phase, auf dem nur vierzählig symmetrischen Raster bleibt ein kleiner Rest (nur
// ausgegeben).
inline void check_vector_symmetry(const VectorPsfFn& compute, const char* backend) {
    const double na = 0.9;
    const int n = 64;
    const int m = psf_fft_size(n, 4.0);
    const double lambda = kPsfLambdaNm * 1e-6;
    const VectorPsfPupil vp = synthetic_vector_pupil(n, kPsfLambdaNm, na, [&](double x, double y) {
        const double u = x * x + y * y;
        return 0.1 * lambda * std::sqrt(5.0) * (6.0 * u * u - 6.0 * u + 1.0);
    });
    const auto run = [&](Polarization p) {
        return compute({vp}, m, polarization_settings(p)).at(0);
    };
    const VectorPsfResult rx = run(Polarization::X);
    const VectorPsfResult ry = run(Polarization::Y);
    const VectorPsfResult rc = run(Polarization::Circular);
    const VectorPsfResult ru = run(Polarization::Unpolarized);
    const VectorPsfPupil perfect =
        synthetic_vector_pupil(n, kPsfLambdaNm, na, [](double, double) { return 0.0; });
    const VectorPsfResult pc =
        compute({perfect}, m, polarization_settings(Polarization::Circular)).at(0);
    const VectorPsfResult pu =
        compute({perfect}, m, polarization_settings(Polarization::Unpolarized)).at(0);
    double mirror = 0.0;
    double rotation = 0.0;
    double average = 0.0;
    double circular = 0.0;
    double circular_grid = 0.0;
    double invariance = 0.0;
    for (int j = 1; j < m; ++j) {
        for (int i = 1; i < m; ++i) {
            const double v = rx.psf_at(i, j);
            mirror = std::max(
                {mirror, std::abs(v - rx.psf_at(m - i, j)), std::abs(v - rx.psf_at(i, m - j))});
            // y-Polarisation im Punkt (ξ, η) = x-Polarisation im Punkt (η, −ξ)
            rotation = std::max(rotation, std::abs(ry.psf_at(i, j) - rx.psf_at(j, m - i)));
            average = std::max(
                average, std::abs(ru.psf_at(i, j) - 0.5 * (rx.psf_at(i, j) + ry.psf_at(i, j))));
            circular = std::max(circular, std::abs(pc.psf_at(i, j) - pu.psf_at(i, j)));
            circular_grid = std::max(circular_grid, std::abs(rc.psf_at(i, j) - ru.psf_at(i, j)));
            invariance = std::max({invariance, std::abs(ru.psf_at(i, j) - ru.psf_at(j, m - i)),
                                   std::abs(rc.psf_at(i, j) - rc.psf_at(j, m - i))});
        }
    }
    EXPECT_LT(mirror, kVectorPsfIdentity);
    EXPECT_LT(rotation, kVectorPsfIdentity);
    EXPECT_LT(average, kVectorPsfIdentity);
    EXPECT_LT(circular, kVectorPsfIdentity);
    EXPECT_LT(invariance, kVectorPsfIdentity);
    EXPECT_NEAR(rx.fwhm_x, ry.fwhm_y, 1e-12 * rx.fwhm_x);
    EXPECT_NEAR(ru.fwhm_x, ru.fwhm_y, 1e-12 * ru.fwhm_x);
    EXPECT_GT(rx.fwhm_x, 1.2 * rx.fwhm_y);
    // Ohne Rotationssymmetrie (Astigmatismus unter 45°, OPD ∝ xy) unterscheiden sich zirkular
    // und unpolarisiert (Händigkeit); nur dokumentiert.
    const VectorPsfPupil astig = synthetic_vector_pupil(
        n, kPsfLambdaNm, na, [&](double x, double y) { return 0.2 * lambda * x * y; });
    const VectorPsfResult ac =
        compute({astig}, m, polarization_settings(Polarization::Circular)).at(0);
    const VectorPsfResult au =
        compute({astig}, m, polarization_settings(Polarization::Unpolarized)).at(0);
    double chiral = 0.0;
    for (std::size_t k = 0; k < ac.psf.size(); ++k) {
        chiral = std::max(chiral, std::abs(ac.psf[k] - au.psf[k]));
    }
    std::printf("[ Info     ] %s, Symmetrien (sphärisch 0,1 λ, NA 0,9): Spiegelung %.1e, Drehung "
                "x→y %.1e, unpolarisiert = (x + y)/2 %.1e, zirkular = unpolarisiert %.1e "
                "(aberrationsfrei; sphärisch Rasterrest %.1e), 90°-Invarianz %.1e; FWHM x/y "
                "(x-pol.) %.5f; Astigmatismus 45°: max |zirkular − "
                "unpolarisiert| = %.2e\n",
                backend, mirror, rotation, average, circular, circular_grid, invariance,
                rx.fwhm_x / rx.fwhm_y, chiral);
}

// Gewichtete Momente der Phase φ = 2π OPD/λ mit den Gewichten a_k = W_k R_xx,k der
// x-polarisierten Pupille (Strehl im Fokus = |Σ a e^{iφ}|² / (Σ a)² bei symmetrischer Phase).
inline double weighted_phase_variance(const VectorPsfPupil& vp) {
    const PsfPupil& p = vp.pupil;
    const int n = p.grid;
    const double lambda = p.wavelength_mm();
    double sw = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const double opd = p.opd[static_cast<std::size_t>(j) * n + i];
            if (!std::isfinite(opd)) {
                continue;
            }
            const double sx = -p.na_x * (2.0 * i + 1 - n) / n;
            const double sy = -p.na_y * (2.0 * j + 1 - n) / n;
            const double sz = std::sqrt(1.0 - sx * sx - sy * sy);
            const double a = (1.0 - sx * sx / (1.0 + sz)) / std::sqrt(sz);
            const double phi = 2.0 * std::numbers::pi * opd / lambda;
            sw += a;
            s1 += a * phi;
            s2 += a * phi * phi;
        }
    }
    const double mean = s1 / sw;
    return s2 / sw - mean * mean;
}

// (6) Strehl bei kleinen Aberrationen, NA 0,95, x-Polarisation: Defokus √3(2ρ² − 1) und
// sphärisch √5(6ρ⁴ − 6ρ² + 1) mit RMS σ = 0,02 … 0,07 λ (gleichgewichtet über das Raster, wie
// M8-T2). Die vektorielle Strehlzahl folgt der Maréchal-Näherung mit der amplitudengewichteten
// Phasenvarianz exp(−σ_w²) (relativ < kPsfMarechal); gegen die skalare Strehlzahl (M8-T2) und
// exp(−(2πσ)²) wird die Tendenz ausgegeben.
inline void check_vector_marechal(const VectorPsfFn& compute, const char* backend) {
    const double na = 0.95;
    const double lambda = kPsfLambdaNm * 1e-6;
    const int n = 128;
    const int m = psf_fft_size(n, 2.0);
    for (const int kind : {0, 1}) {
        const auto g = [kind](double u) {
            return kind == 0 ? std::sqrt(3.0) * (2.0 * u - 1.0)
                             : std::sqrt(5.0) * (6.0 * u * u - 6.0 * u + 1.0);
        };
        std::vector<VectorPsfPupil> pupils;
        const std::vector<double> sigmas = {0.02, 0.035, 0.05, 0.07};
        for (const double sigma : sigmas) {
            pupils.push_back(synthetic_vector_pupil(n, kPsfLambdaNm, na, [&](double x, double y) {
                return sigma * lambda * g(x * x + y * y);
            }));
        }
        const std::vector<VectorPsfResult> res =
            compute(pupils, m, polarization_settings(Polarization::X));
        for (std::size_t q = 0; q < sigmas.size(); ++q) {
            const VectorPsfResult& r = res[q];
            const PsfResult scalar = psf_cpu(pupils[q].pupil, m);
            const PupilMoments mo = pupil_moments(pupils[q].pupil);
            const double s = 2.0 * std::numbers::pi * mo.rms / lambda;
            const double sw2 = weighted_phase_variance(pupils[q]);
            const double marechal_w = std::exp(-sw2);
            EXPECT_NEAR(r.strehl, r.strehl_fft, kVectorPsfIdentity);
            EXPECT_LT(std::abs(r.strehl / marechal_w - 1.0), kPsfMarechal) << "σ = " << sigmas[q];
            std::printf("[ Info     ] %s, NA 0,95 %s σ = %.3f λ: Strehl vektoriell %.6f, "
                        "gewichteter Maréchal exp(−σ_w²) %.6f (σ_w = %.4f λ, rel. %.1e), skalar "
                        "(M8-T2) %.6f, exp(−(2πσ)²) %.6f\n",
                        backend, kind == 0 ? "Defokus" : "sphärisch", sigmas[q], r.strehl,
                        marechal_w, std::sqrt(sw2) / (2.0 * std::numbers::pi),
                        r.strehl / marechal_w - 1.0, scalar.strehl, std::exp(-s * s));
        }
    }
}

// (7) Ende-zu-Ende über EVAL: aplanatischer Punkt (n = 1,5, sin θ = 0,9 im Glas, Bildraum Luft
// mit NA' = 0,6, virtueller Bildpunkt → gespiegelt), N = 256. Pupille aus make_vector_psf_pupil
// (NA aus den Feldstrahlen, Hauptstrahl auf der Achse, Punktquelle → Eingangsbündel entlang z)
// und PSF gegen die Quadratur bei NA 0,6.
inline void check_vector_aplanatic(const EvalFn& eval, const VectorPsfFn& compute,
                                   const char* backend) {
    const AplanaticCase ac = aplanatic_case(1.5, 0.0, 0.9);
    EvalSettings es;
    es.grid = 256;
    const EvalResult er = eval(ac.batch, es);
    ASSERT_EQ(er.failed_count(), 0u);
    const std::vector<VectorPsfPupil> pupils = make_vector_psf_pupils(ac.batch, er, 0);
    ASSERT_EQ(pupils.size(), 1u);
    const VectorPsfPupil& vp = pupils[0];
    EXPECT_NEAR(vp.pupil.na_x, 0.6, kPsfNaFieldRays);
    EXPECT_NEAR(vp.pupil.na_y, 0.6, kPsfNaFieldRays);
    EXPECT_TRUE(vp.pupil.flip_x);
    EXPECT_TRUE(vp.pupil.flip_y);
    EXPECT_EQ(vp.index, 1.0);
    EXPECT_NEAR(vp.chief_x, 0.0, 1e-15);
    EXPECT_NEAR(vp.chief_y, 0.0, 1e-15);
    EXPECT_EQ(vp.input_z, 1.0);
    const int m = psf_fft_size(256, 4.0);
    const VectorPsfResult r = compute(pupils, m, polarization_settings(Polarization::X)).at(0);
    const double lambda = kPsfLambdaNm * 1e-6;
    const RichardsWolfOracle oracle(0.6, lambda);
    const int c = m / 2;
    double max_total = 0.0;
    for (int k = 0; k * r.dx <= 1.5 * lambda; ++k) {
        max_total = std::max(
            {max_total, std::abs(r.psf_at(c + k, c) - oracle.intensity(k * r.dx, 0.0)),
             std::abs(r.psf_at(c, c + k) - oracle.intensity(k * r.dy, 0.5 * std::numbers::pi))});
    }
    EXPECT_LT(max_total, kVectorPsfOracle);
    EXPECT_NEAR(r.fwhm_x / oracle.fwhm(0.0, lambda), 1.0, kVectorPsfRelative);
    EXPECT_NEAR(r.fwhm_y / oracle.fwhm(0.5 * std::numbers::pi, lambda), 1.0, kVectorPsfRelative);
    EXPECT_NEAR(r.strehl, 1.0, 1e-9);
    EXPECT_NEAR(r.strehl_fft, 1.0, 1e-9);
    std::printf("[ Info     ] %s, aplanatischer Punkt über EVAL (NA' %.12f, gespiegelt): max "
                "|PSF − Richards-Wolf| = %.2e, FWHM x/y = %.6f / %.6f λ (Quadratur %.6f / %.6f), "
                "Strehl %.12f\n",
                backend, vp.pupil.na_y, max_total, r.fwhm_x / lambda, r.fwhm_y / lambda,
                oracle.fwhm(0.0, lambda) / lambda,
                oracle.fwhm(0.5 * std::numbers::pi, lambda) / lambda, r.strehl);
}

} // namespace mko::test
