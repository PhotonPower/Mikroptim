// Geisterplatte (M4-T5): Rekursionen, lineare Phase, Gradient gegen finite Differenzen,
// fokussierende Geisterplatte, paraxiale Brechkraft gegen den Grenzwert realer Strahlen,
// Einfügen in die Flächentabelle (Abbildung Dateiflächen ↔ Tabellenflächen).

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/trace.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using mko::GhostPlateParams;
using mko::HostSurfaceTable;
using mko::Ray;
using mko::SurfaceKind;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::kGhostFocusRms;
using mko::test::kGhostGradientFd;
using mko::test::kLengthFp64;
using mko::test::kParaxialLimit;
using mko::test::kParaxialRelative;
using mko::test::kUnitFp64;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";

// Geschlossene Formeln der Legendre-Polynome P_0 … P_5 und ihrer Ableitungen.
double legendre_closed(int k, double x) {
    switch (k) {
    case 0:
        return 1.0;
    case 1:
        return x;
    case 2:
        return (3.0 * x * x - 1.0) / 2.0;
    case 3:
        return (5.0 * x * x * x - 3.0 * x) / 2.0;
    case 4:
        return (35.0 * std::pow(x, 4) - 30.0 * x * x + 3.0) / 8.0;
    case 5:
        return (63.0 * std::pow(x, 5) - 70.0 * x * x * x + 15.0 * x) / 8.0;
    default:
        return NAN;
    }
}

double legendre_derivative_closed(int k, double x) {
    switch (k) {
    case 0:
        return 0.0;
    case 1:
        return 1.0;
    case 2:
        return 3.0 * x;
    case 3:
        return (15.0 * x * x - 3.0) / 2.0;
    case 4:
        return (140.0 * x * x * x - 60.0 * x) / 8.0;
    case 5:
        return (315.0 * std::pow(x, 4) - 210.0 * x * x + 15.0) / 8.0;
    default:
        return NAN;
    }
}

// Flächentabelle Blende (Abstand t0 zur Geisterplatte) → Geisterplatte (Abstand t1 zur
// Bildebene bzw. Referenzsphäre), Luft überall, eine Wellenlänge.
HostSurfaceTable ghost_table(double t0, double t1, const std::vector<double>& coeff,
                             double norm_radius, double n_glass) {
    HostSurfaceTable t;
    t.surfaces = 2;
    t.wavelengths = 1;
    t.type = {SurfaceKind::Stop, SurfaceKind::GhostPlate};
    t.curvature = {0.0, 0.0};
    t.thickness = {t0, t1};
    t.index = {1.0, 1.0};
    t.semi_diameter = {norm_radius, norm_radius};
    t.ghost_plates = 1;
    t.ghost_terms = static_cast<int>(coeff.size());
    t.ghost_coeff = coeff;
    t.ghost_index = {n_glass};
    t.ghost_norm_radius = {norm_radius};
    return t;
}

// Ableitung von f in x0 über zentrale Differenzen mit Richardson-Extrapolation:
// D(h_j) mit h_j = h / 2^j, j = 0 … 4, dann Tableau T[j][m] (Fehler O(h^(2m+2))).
template <class F>
double richardson_derivative(F f, double x0, double h) {
    constexpr int kLevels = 5;
    double T[kLevels][kLevels];
    for (int j = 0; j < kLevels; ++j) {
        const double hj = h / std::ldexp(1.0, j);
        T[j][0] = (f(x0 + hj) - f(x0 - hj)) / (2.0 * hj);
    }
    for (int m = 1; m < kLevels; ++m) {
        const double factor = std::ldexp(1.0, 2 * m) - 1.0; // 4^m − 1
        for (int j = m; j < kLevels; ++j) {
            T[j][m] = T[j][m - 1] + (T[j][m - 1] - T[j - 1][m - 1]) / factor;
        }
    }
    return T[kLevels - 1][kLevels - 1];
}

// Legendre-Koeffizienten c_0 … c_K einer Funktion f(u) auf u ∈ [0, 1] (ξ = 2u − 1) als
// L²-Bestapproximation (kleinste Quadrate im kontinuierlichen Sinn) per Gauß-Legendre-
// Quadratur mit 16 Knoten: c_k = (2k + 1)/2 ∫ f P_k dξ = (2k + 1) Σ_i W_i f(u_i) P_k(ξ_i),
// exakt für Polynome f bis Grad 31 − K in u.
template <class F>
std::vector<double> legendre_fit(F f, int K) {
    const auto q = mko::make_pupil_quadrature<double>(16, 1, mko::PupilMode::Axis);
    std::vector<double> c(static_cast<std::size_t>(K + 1), 0.0);
    for (int i = 0; i < q.rings; ++i) {
        const double fi = f(q.u[i]);
        for (int k = 0; k <= K; ++k) {
            double p = 0.0;
            double dp = 0.0;
            mko::legendre(k, 2.0 * q.u[i] - 1.0, p, dp);
            c[static_cast<std::size_t>(k)] += (2.0 * k + 1.0) * q.w[i] * fi * p;
        }
    }
    return c;
}

// Achsschnittpunkt eines achsparallelen meridionalen Strahls der Höhe h nach dem Trace durch
// die Tabelle (relativ zum Scheitel der letzten Fläche).
double real_axis_crossing(const HostSurfaceTable& t, int l, double h) {
    Ray<double> ray = mko::collimated_ray(h, 0.0, 1.0, 0.0);
    EXPECT_EQ(mko::trace_ray(t.view(), l, ray, nullptr), TraceStatus::Ok);
    return mko::meridional_axis_crossing(ray);
}

} // namespace

// --- Legendre-Rekursionen ------------------------------------------------------------------

TEST(GhostPlate, LegendreRecursionMatchesClosedForms) {
    for (int k = 0; k <= 5; ++k) {
        for (double x = -1.5; x <= 1.5; x += 0.0625) {
            double p = 0.0;
            double dp = 0.0;
            mko::legendre(k, x, p, dp);
            const double pe = legendre_closed(k, x);
            const double dpe = legendre_derivative_closed(k, x);
            EXPECT_NEAR(p, pe, kUnitFp64 * std::max(1.0, std::abs(pe))) << "k=" << k << " x=" << x;
            EXPECT_NEAR(dp, dpe, kUnitFp64 * std::max(1.0, std::abs(dpe)))
                << "k=" << k << " x=" << x;
        }
    }
    // Randwerte bis K = kMaxGhostTerms exakt: P_k(1) = 1, P_k(−1) = (−1)^k,
    // P'_k(±1) = (±1)^(k+1) k (k + 1)/2
    for (int k = 0; k <= mko::kMaxGhostTerms; ++k) {
        double p = 0.0;
        double dp = 0.0;
        mko::legendre(k, 1.0, p, dp);
        EXPECT_EQ(p, 1.0);
        EXPECT_EQ(dp, k * (k + 1) / 2.0);
        mko::legendre(k, -1.0, p, dp);
        EXPECT_EQ(p, k % 2 == 0 ? 1.0 : -1.0);
        EXPECT_EQ(dp, (k % 2 == 1 ? 1.0 : -1.0) * k * (k + 1) / 2.0);
    }
}

// ghost_profile (ein Durchlauf der Rekursionen) gegen die Summe der einzeln ausgewerteten
// Polynome; Koeffizienten 0 am Ende ändern das Ergebnis bitgenau nicht (Auffüllen auf K).
TEST(GhostPlate, ProfileMatchesTermwiseSum) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::vector<double> a(12);
    for (double& x : a) {
        x = 1e-3 * unit(rng);
    }
    for (double u = 0.0; u <= 1.5; u += 0.03125) {
        double t = 0.0;
        double dt = 0.0;
        for (int k = 1; k <= 12; ++k) {
            double p = 0.0;
            double dp = 0.0;
            mko::legendre(k, 2.0 * u - 1.0, p, dp);
            t += a[static_cast<std::size_t>(k - 1)] * p;
            dt += a[static_cast<std::size_t>(k - 1)] * dp;
        }
        const mko::GhostProfile<double> prof = mko::ghost_profile(a.data(), 12, u);
        EXPECT_NEAR(prof.t, t, 1e-15);
        EXPECT_NEAR(prof.dt_du, 2.0 * dt, 1e-14);
        std::vector<double> padded = a;
        padded.resize(20, 0.0);
        const mko::GhostProfile<double> prof_padded = mko::ghost_profile(padded.data(), 20, u);
        EXPECT_EQ(prof_padded.t, prof.t);
        EXPECT_EQ(prof_padded.dt_du, prof.dt_du);
    }
    // Ohne Koeffizienten: t = 0
    EXPECT_EQ(mko::ghost_profile(a.data(), 0, 0.3).t, 0.0);
}

// --- Abnahme: lineare Phase ------------------------------------------------------------------

// Φ = a·y lenkt exakt um Δ(n d_y) = a ab (d_x unverändert, |d'| = 1), OPL wächst um Φ.
TEST(GhostPlate, LinearPhaseDeflectsExactly) {
    double max_dev = 0.0;
    for (const double n : {1.0, 1.33, 1.7}) {
        for (const double a : {-0.3, -1e-3, 1e-6, 0.05, 0.4}) {
            for (const double theta : {-0.4, 0.0, 0.2}) {
                for (const double phi_dir : {0.0, 0.9, 2.5}) {
                    Ray<double> ray;
                    ray.p = Vec3<double>{0.7, -1.3, 0.0};
                    const double s = std::sin(theta);
                    ray.d =
                        Vec3<double>{s * std::sin(phi_dir), s * std::cos(phi_dir), std::cos(theta)};
                    ray.opl = 12.5;
                    const Ray<double> before = ray;
                    const double phase = a * ray.p.y;
                    ASSERT_EQ(mko::phase_surface_step(ray, phase, 0.0, a, n, n), TraceStatus::Ok);
                    const double dev = std::abs(n * ray.d.y - n * before.d.y - a);
                    max_dev = std::max(max_dev, dev);
                    EXPECT_LE(dev, kUnitFp64) << "n=" << n << " a=" << a;
                    EXPECT_NEAR(ray.d.x, before.d.x, kUnitFp64);
                    EXPECT_NEAR(mko::norm(ray.d), 1.0, kUnitFp64);
                    EXPECT_GT(ray.d.z, 0.0);
                    EXPECT_EQ(ray.opl, before.opl + phase);
                    EXPECT_EQ(ray.p.x, before.p.x);
                    EXPECT_EQ(ray.p.y, before.p.y);
                }
            }
        }
    }
    // Brechende Ebene mit Phase: n' d'_y = n d_y + a
    Ray<double> ray{{0.0, 1.0, 0.0}, {0.0, 0.3, std::sqrt(1.0 - 0.09)}, 0.0};
    ASSERT_EQ(mko::phase_surface_step(ray, 0.02, 0.0, 0.02, 1.0, 1.5), TraceStatus::Ok);
    EXPECT_NEAR(1.5 * ray.d.y, 0.3 + 0.02, kUnitFp64);
    EXPECT_NEAR(mko::norm(ray.d), 1.0, kUnitFp64);
    // Zu starke Phase: Radikand < 0 → TIR (evaneszent)
    Ray<double> strong{{0.0, 1.0, 0.0}, {0.0, 0.5, std::sqrt(0.75)}, 0.0};
    EXPECT_EQ(mko::phase_surface_step(strong, 0.6, 0.0, 0.6, 1.0, 1.0), TraceStatus::TIR);
    RecordProperty("max_deviation_linear_phase", std::to_string(max_dev));
    std::printf("[ Info     ] lineare Phase: max |Δ(n d_y) − a| = %.2e\n", max_dev);
}

// --- Abnahme: Gradient gegen finite Differenzen ---------------------------------------------

TEST(GhostPlate, ProfileGradientMatchesFiniteDifferences) {
    std::mt19937_64 rng(4711);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    double worst = 0.0;
    int points = 0;
    for (const int K : {1, 2, 3, 5, 8, 12, 16}) {
        for (const double rg : {0.8, 3.0, 12.0}) {
            std::vector<double> a(static_cast<std::size_t>(K));
            for (double& x : a) {
                x = 1e-3 * unit(rng);
            }
            const auto t_of = [&](double x, double y) {
                return mko::ghost_profile(a.data(), K, (x * x + y * y) / (rg * rg)).t;
            };
            // Schrittweite relativ zur Strukturbreite r_g / (2K) des Profils; kleinste Stufe h/16
            const double h = 0.2 * rg / K;
            double worst_k = 0.0;
            for (int i = 0; i < 40; ++i) {
                const double x = 1.1 * rg * unit(rng);
                const double y = 1.1 * rg * unit(rng);
                const mko::GhostProfile<double> prof =
                    mko::ghost_profile(a.data(), K, (x * x + y * y) / (rg * rg));
                // ∇t = (dt/du) (2/r_g²) (x, y) = (4/r_g²) Σ a_k P'_k (x, y)
                const double gx = prof.dt_du * 2.0 / (rg * rg) * x;
                const double gy = prof.dt_du * 2.0 / (rg * rg) * y;
                const double fx = richardson_derivative([&](double s) { return t_of(s, y); }, x, h);
                const double fy = richardson_derivative([&](double s) { return t_of(x, s); }, y, h);
                const double g = std::hypot(gx, gy);
                if (g == 0.0) {
                    continue;
                }
                const double rel = std::hypot(fx - gx, fy - gy) / g;
                worst_k = std::max(worst_k, rel);
                ++points;
                EXPECT_LT(rel, kGhostGradientFd)
                    << "K=" << K << " r_g=" << rg << " (x, y) = (" << x << ", " << y << ")";
            }
            worst = std::max(worst, worst_k);
        }
    }
    RecordProperty("max_relative_gradient_error", std::to_string(worst));
    std::printf("[ Info     ] Gradient gegen Richardson-FD: %d Punkte, max. relativ %.2e\n", points,
                worst);
}

// Ablenkung an der Geisterplatte im Trace gegen die Formeln von numerik.md, Abschnitt 6,
// unabhängig ausgewertet (Legendre-Summe Term für Term).
TEST(GhostPlate, TraceStepMatchesFormula) {
    const std::vector<double> a{-2e-3, 6e-4, 1e-4, -3e-5};
    const double rg = 4.0;
    const double nG = 1.62;
    const HostSurfaceTable t = ghost_table(2.5, 30.0, a, rg, nG);
    const double theta = 0.05;
    Ray<double> ray = mko::collimated_ray(3.0, theta, 0.8, 0.6);
    const Ray<double> start = ray;
    int failed = 0;
    ASSERT_EQ(mko::trace_ray(t.view(), 0, ray, nullptr, &failed), TraceStatus::Ok);
    EXPECT_EQ(failed, -1);
    // Erwartet: Transfer um 2,5 mm entlang d, dann Phase und Ablenkung
    const double s0 = 2.5 / start.d.z;
    const double x = start.p.x + s0 * start.d.x;
    const double y = start.p.y + s0 * start.d.y;
    const double u = (x * x + y * y) / (rg * rg);
    double tt = 0.0;
    double sum_dp = 0.0;
    for (int k = 1; k <= 4; ++k) {
        tt += a[static_cast<std::size_t>(k - 1)] * legendre_closed(k, 2.0 * u - 1.0);
        sum_dp += a[static_cast<std::size_t>(k - 1)] * legendre_derivative_closed(k, 2.0 * u - 1.0);
    }
    const double gx = (nG - 1.0) * 4.0 / (rg * rg) * sum_dp * x;
    const double gy = (nG - 1.0) * 4.0 / (rg * rg) * sum_dp * y;
    EXPECT_NEAR(ray.p.x, x, kLengthFp64);
    EXPECT_NEAR(ray.p.y, y, kLengthFp64);
    EXPECT_EQ(ray.p.z, 0.0);
    EXPECT_NEAR(ray.opl, start.opl + s0 + (nG - 1.0) * tt, kLengthFp64);
    EXPECT_NEAR(ray.d.x, start.d.x + gx, kUnitFp64);
    EXPECT_NEAR(ray.d.y, start.d.y + gy, kUnitFp64);
    EXPECT_NEAR(mko::norm(ray.d), 1.0, kUnitFp64);

    // Geisterplatte ohne Daten in der Tabelle: nicht unterstützt
    HostSurfaceTable bare = t;
    bare.ghost_plates = 0;
    Ray<double> r2 = start;
    EXPECT_EQ(mko::trace_ray(bare.view(), 0, r2, nullptr, &failed), TraceStatus::Unsupported);
    EXPECT_EQ(failed, 1);
    // Viel zu starkes Profil: Radikand < 0 an der Geisterplatte → TIR an Fläche 1
    HostSurfaceTable strong = t;
    strong.ghost_coeff = {-5.0, 0.0, 0.0, 0.0};
    Ray<double> r3 = mko::collimated_ray(3.0, 0.0, 1.0, 0.0);
    EXPECT_EQ(mko::trace_ray(strong.view(), 0, r3, nullptr, &failed), TraceStatus::TIR);
    EXPECT_EQ(failed, 1);
}

// --- Abnahme: fokussierende Geisterplatte -----------------------------------------------------

// Kollimiertes Bündel, Geisterplatte direkt hinter der Blende (Abstand 0) in Luft, Brennweite
// F = 10 mm, Pupillenradius a mit a / √(a² + F²) = 0,3 (NA 0,3 im Bildraum). Ideale Phase für
// den Brennpunkt (0, 0, F): Φ(r) + √(r² + F²) = const, also Φ(r) = −(√(r² + F²) − F)
// (Strahlen werden zur Achse hin abgelenkt: dΦ/dr = −r/√(r² + F²) = −sin θ). Dickenprofil
// t = Φ / (n_G − 1), Legendre-Fit in u = (r/a)², r_g = a; der Kolben (P_0) entfällt.
// Referenzsphäre um den Brennpunkt durch den Scheitel der Geisterplatte.
TEST(GhostPlate, FittedFocusingPhaseIsDiffractionLimitedAtNa03) {
    const double F = 10.0;
    const double na = 0.3;
    const double a = F * na / std::sqrt(1.0 - na * na);
    const double nG = 1.5;
    const auto phase = [&](double u) {
        const double r2 = a * a * u;
        return -(std::sqrt(r2 + F * F) - F);
    };
    const auto q = mko::make_pupil_quadrature<double>(10, 8, mko::PupilMode::Axis);
    const int n = mko::sample_count(q);
    double rms_accepted = 0.0;
    for (const int K : {4, 6, 8, 10, 12, 14, 16}) {
        const std::vector<double> c = legendre_fit(phase, K);
        std::vector<double> coeff;
        for (int k = 1; k <= K; ++k) {
            coeff.push_back(c[static_cast<std::size_t>(k)] / (nG - 1.0));
        }
        const HostSurfaceTable t = ghost_table(0.0, F, coeff, a, nG);
        std::vector<double> opl(static_cast<std::size_t>(n));
        std::vector<double> w(static_cast<std::size_t>(n));
        std::vector<TraceStatus> st(static_cast<std::size_t>(n));
        std::vector<double> opd(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            const mko::PupilSample<double> s = mko::sample(q, k);
            Ray<double> ray = mko::collimated_ray(a, 0.0, s.rho, s.phi);
            st[static_cast<std::size_t>(k)] =
                mko::trace_to_reference_sphere(t.view(), 0, ray, Vec3<double>{0.0, 0.0, F});
            opl[static_cast<std::size_t>(k)] = ray.opl;
            w[static_cast<std::size_t>(k)] = s.weight;
        }
        const mko::OpdStatistics<double> stat =
            mko::opd_statistics(opl.data(), w.data(), st.data(), n, opd.data());
        ASSERT_EQ(stat.failed, 0);
        double max_opd = 0.0;
        for (const double e : opd) {
            max_opd = std::max(max_opd, std::abs(e));
        }
        std::printf("[ Info     ] NA 0,3, K = %2d: RMS-OPD = %.3e mm, max |OPD| = %.3e mm\n", K,
                    stat.rms, max_opd);
        if (K == 12) {
            rms_accepted = stat.rms;
            EXPECT_LT(stat.rms, kGhostFocusRms);
        }
    }
    RecordProperty("rms_opd_K12_mm", std::to_string(rms_accepted));

    // Nur dokumentiert: kleines Feld (0,5°), Referenz im idealen Bildpunkt (0, F tan θ, F).
    // Die Phase ist nur für die Achse ideal (Koma der Phasenplatte).
    const std::vector<double> c = legendre_fit(phase, 12);
    std::vector<double> coeff;
    for (int k = 1; k <= 12; ++k) {
        coeff.push_back(c[static_cast<std::size_t>(k)] / (nG - 1.0));
    }
    const HostSurfaceTable t = ghost_table(0.0, F, coeff, a, nG);
    const auto qh = mko::make_pupil_quadrature<double>(10, 8, mko::PupilMode::Half);
    const int nh = mko::sample_count(qh);
    const double theta = 0.5 * std::acos(-1.0) / 180.0;
    std::vector<double> opl(static_cast<std::size_t>(nh)), w(static_cast<std::size_t>(nh));
    std::vector<TraceStatus> st(static_cast<std::size_t>(nh));
    for (int k = 0; k < nh; ++k) {
        const mko::PupilSample<double> s = mko::sample(qh, k);
        Ray<double> ray = mko::collimated_ray(a, theta, s.rho, s.phi);
        st[static_cast<std::size_t>(k)] = mko::trace_to_reference_sphere(
            t.view(), 0, ray, Vec3<double>{0.0, F * std::tan(theta), F});
        opl[static_cast<std::size_t>(k)] = ray.opl;
        w[static_cast<std::size_t>(k)] = s.weight;
    }
    const auto field =
        mko::opd_statistics(opl.data(), w.data(), st.data(), nh, static_cast<double*>(nullptr));
    std::printf("[ Info     ] Feld 0,5° (nur dokumentiert): RMS-OPD = %.3e mm\n", field.rms);
}

// --- Abnahme: paraxiale Brechkraft ------------------------------------------------------------

// φ_G = −2 (n_G − 1) t₂ gegen den Achsschnittpunkt realer Strahlen für h → 0 (dünnes Element
// in Luft: hintere Schnittweite = Brennweite = 1/φ_G).
TEST(GhostPlate, ParaxialPowerMatchesRealRayLimit) {
    struct Case {
        std::vector<double> a;
        double rg, nG;
    };
    const Case cases[] = {{{-2e-3, 1e-3, -5e-4, 2e-4}, 5.0, 1.6},
                          {{3e-3}, 2.0, 1.45},
                          {{0.0, 4e-4, 0.0, -1e-4, 0.0, 2e-5}, 3.0, 1.8},
                          {{1e-3, -2e-3, 1e-3, 3e-4, -2e-4, 1e-4, 5e-5, -2e-5}, 8.0, 1.5}};
    for (const Case& cs : cases) {
        const int K = static_cast<int>(cs.a.size());
        const double t2 = mko::ghost_profile_t2(cs.a.data(), K, cs.rg);
        // t₂ = dt/d(r²) bei r = 0 = (1/r_g²) dt/du(0)
        EXPECT_NEAR(t2, mko::ghost_profile(cs.a.data(), K, 0.0).dt_du / (cs.rg * cs.rg),
                    1e-15 * std::abs(t2) + 1e-300);
        const double phi = mko::ghost_plate_power(cs.a.data(), K, cs.rg, cs.nG, 1.0);
        EXPECT_EQ(phi, -2.0 * (cs.nG - 1.0) * t2);
        const double f = 1.0 / phi;
        const HostSurfaceTable t = ghost_table(0.0, 50.0, cs.a, cs.rg, cs.nG);
        // Längsaberration ∝ h²: Verhältnis bei h und h/2 gegen 4
        double previous = real_axis_crossing(t, 0, 0.002 * cs.rg) - f;
        for (const double h : {0.001 * cs.rg, 0.0005 * cs.rg, 0.00025 * cs.rg}) {
            const double diff = real_axis_crossing(t, 0, h) - f;
            EXPECT_NEAR(previous / diff / 4.0, 1.0, mko::test::kAberrationScaling) << "h = " << h;
            previous = diff;
        }
        const double limit = real_axis_crossing(t, 0, 1e-7 * cs.rg);
        std::printf("[ Info     ] φ_G = %.6e 1/mm, 1/φ_G = %.6f mm, Achsschnittpunkt (h = "
                    "1e-7 r_g) − 1/φ_G = %.2e mm\n",
                    phi, f, limit - f);
        EXPECT_LT(std::abs(limit - f), kParaxialLimit) << "f = " << f;
    }
}

// Geisterplatte im y-nu-Trace des Gesamtsystems (einzellinse_geisterplatte.json): hintere
// Schnittweite gegen den Grenzwert realer Strahlen durch die Flächentabelle mit
// Geisterplatten; Petzvalbeitrag φ_G/n_G; Tabelle und Katalog liefern dieselben Zahlen.
TEST(GhostPlate, ParaxialSystemTraceIncludesGhostPlates) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    ASSERT_EQ(ls.system.ghost_plates.size(), 2u);
    const int l_ref = 1; // 587,56 nm
    ASSERT_EQ(ls.system.spectrum.wavelengths_nm[l_ref], ls.system.spectrum.reference_nm);
    const mko::ParaxialData p = mko::compute_paraxial(ls.system, ls.catalog);
    const mko::ParaxialData p_table = mko::compute_paraxial(ls.system, ls.index, l_ref);
    EXPECT_EQ(p.focal_length, p_table.focal_length);
    EXPECT_EQ(p.back_focal_distance, p_table.back_focal_distance);
    EXPECT_EQ(p.petzval_sum, p_table.petzval_sum);

    const double limit = real_axis_crossing(ls.table, l_ref, 1e-4);
    EXPECT_LT(std::abs(limit - p.back_focal_distance), kParaxialLimit)
        << "real " << limit << ", paraxial " << p.back_focal_distance;

    mko::SystemDescription bare = ls.system;
    bare.ghost_plates.clear();
    const mko::ParaxialData p0 = mko::compute_paraxial(bare, ls.catalog);
    EXPECT_GT(std::abs(p.back_focal_distance - p0.back_focal_distance), 0.1)
        << "Geisterplatten ohne Wirkung";
    double petzval = 0.0;
    for (std::size_t g = 0; g < ls.system.ghost_plates.size(); ++g) {
        const mko::GhostPlate& gp = ls.system.ghost_plates[g];
        const double nG = ls.catalog.at(gp.glass.name).index(ls.system.spectrum.reference_nm);
        const double phi =
            mko::ghost_plate_power(gp.coefficients.data(), static_cast<int>(gp.coefficients.size()),
                                   gp.norm_radius, nG, 1.0);
        petzval += phi / nG;
    }
    EXPECT_NEAR(p.petzval_sum - p0.petzval_sum, petzval,
                kParaxialRelative * std::abs(p0.petzval_sum));
    std::printf("[ Info     ] Schnittweite mit/ohne Geisterplatten: %.6f / %.6f mm, Grenzwert "
                "realer Strahlen − paraxial = %.2e mm\n",
                p.back_focal_distance, p0.back_focal_distance, limit - p.back_focal_distance);

    // Geisterplatten mit Koeffizienten 0 ändern die Paraxialdaten nur um Rundung (geteilter
    // Luftspalt).
    mko::SystemDescription zero = ls.system;
    for (mko::GhostPlate& gp : zero.ghost_plates) {
        std::fill(gp.coefficients.begin(), gp.coefficients.end(), 0.0);
    }
    const mko::ParaxialData pz = mko::compute_paraxial(zero, ls.catalog);
    EXPECT_NEAR(pz.focal_length, p0.focal_length, kParaxialRelative * p0.focal_length);
    EXPECT_NEAR(pz.back_focal_distance, p0.back_focal_distance,
                kParaxialRelative * p0.back_focal_distance);
    EXPECT_EQ(pz.petzval_sum, p0.petzval_sum);
}

// --- Flächentabelle und Abbildung --------------------------------------------------------------

TEST(GhostPlate, SurfaceMapAndTableFromFile) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    const mko::SurfaceMap map = mko::make_surface_map(ls.system);
    EXPECT_EQ(map.file_surfaces(), 3);
    EXPECT_EQ(map.table_surfaces(), 5);
    EXPECT_EQ(map.table_of_file, (std::vector<int>{0, 3, 4}));
    EXPECT_EQ(map.file_of_table, (std::vector<int>{0, -1, -1, 1, 2}));
    EXPECT_EQ(map.ghost_of_table, (std::vector<int>{-1, 0, 1, -1, -1}));
    EXPECT_EQ(map.table_of_ghost, (std::vector<int>{1, 2}));
    EXPECT_EQ(map.file_surface(5), 3); // Referenzsphäre
    EXPECT_EQ(map.table_surface(3), 5);
    EXPECT_EQ(map.file_surface(2), -1);
    EXPECT_THROW((void)map.file_surface(6), std::out_of_range);
    EXPECT_THROW((void)map.table_surface(-1), std::out_of_range);
    EXPECT_EQ(mko::table_surface_count(ls.system), 5);

    const HostSurfaceTable& t = ls.table;
    ASSERT_EQ(t.surfaces, 5);
    EXPECT_EQ(t.type, (std::vector<SurfaceKind>{SurfaceKind::Stop, SurfaceKind::GhostPlate,
                                                SurfaceKind::GhostPlate, SurfaceKind::Sphere,
                                                SurfaceKind::Sphere}));
    EXPECT_EQ(t.thickness, (std::vector<double>{1.0, 2.5, 1.5, 6.0, 98.23}));
    EXPECT_EQ(t.curvature[1], 0.0);
    EXPECT_EQ(t.curvature[3], 0.00967);
    const int L = t.wavelengths;
    for (int l = 0; l < L; ++l) {
        EXPECT_EQ(t.index[static_cast<std::size_t>(1 * L + l)], 1.0);
        EXPECT_EQ(t.index[static_cast<std::size_t>(2 * L + l)], 1.0);
        const double lambda = ls.system.spectrum.wavelengths_nm[static_cast<std::size_t>(l)];
        EXPECT_EQ(t.index[static_cast<std::size_t>(3 * L + l)],
                  ls.catalog.at("N-BK7").index(lambda));
        EXPECT_EQ(t.ghost_index[static_cast<std::size_t>(0 * L + l)],
                  ls.catalog.at("CAF2").index(lambda));
        EXPECT_EQ(t.ghost_index[static_cast<std::size_t>(1 * L + l)],
                  ls.catalog.at("N-BK7").index(lambda));
    }
    EXPECT_EQ(t.ghost_plates, 2);
    EXPECT_EQ(t.ghost_terms, 6);
    EXPECT_EQ(t.ghost_coeff, (std::vector<double>{-0.002, 0.0005, -0.0001, 2e-05, 0.0, 0.0, 0.0,
                                                  0.0002, 0.0, -5e-05, 0.0, 1e-05}));
    EXPECT_EQ(t.ghost_norm_radius, (std::vector<double>{10.0, 10.0}));
    EXPECT_EQ(t.semi_diameter[0], 10.0);

    // Mehrere Spalte, gleiche Abstände, Platte am Anfang und am Ende eines Spalts
    mko::SystemDescription sys = mko::read_system_file(kSystems + "testsystem_mehrlinser.json");
    mko::GhostPlate g;
    g.glass.kind = mko::MaterialKind::Constant;
    g.glass.n = 1.5;
    g.norm_radius = 4.0;
    g.coefficients = {1e-4};
    for (const auto& [after, distance] :
         std::vector<std::pair<int, double>>{{0, 0.0}, {0, 0.0}, {0, 3.0}, {3, 0.4}, {5, 2.0}}) {
        g.after_surface = after;
        g.distance = distance;
        sys.ghost_plates.push_back(g);
    }
    const mko::LoadedSystem lm = mko::prepare_system(sys, MKO_SOURCE_DIR);
    const mko::SurfaceMap mm = mko::make_surface_map(sys);
    EXPECT_EQ(mm.table_of_ghost, (std::vector<int>{1, 2, 3, 7, 10}));
    EXPECT_EQ(mm.table_of_file, (std::vector<int>{0, 4, 5, 6, 8, 9, 11, 12}));
    EXPECT_EQ(lm.table.surfaces, 13);
    EXPECT_EQ(lm.table.thickness[0], 0.0);
    EXPECT_EQ(lm.table.thickness[1], 0.0);
    EXPECT_EQ(lm.table.thickness[2], 3.0);
    EXPECT_EQ(lm.table.thickness[3], 0.0);
    EXPECT_EQ(lm.table.thickness[6], 0.4);
    EXPECT_EQ(lm.table.thickness[7], 1.0 - 0.4);
    EXPECT_EQ(lm.table.thickness[9], 2.0);
    EXPECT_EQ(lm.table.thickness[10], 0.0);
}

// Geisterplatten mit Koeffizienten 0 ändern den Trace nur um Rundung (Luftspalt geteilt).
TEST(GhostPlate, ZeroCoefficientsLeaveTraceUnchanged) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "einzellinse_geisterplatte.json");
    for (mko::GhostPlate& gp : sys.ghost_plates) {
        std::fill(gp.coefficients.begin(), gp.coefficients.end(), 0.0);
    }
    const mko::LoadedSystem with = mko::prepare_system(sys, MKO_SOURCE_DIR);
    sys.ghost_plates.clear();
    const mko::LoadedSystem without = mko::prepare_system(sys, MKO_SOURCE_DIR);
    const auto q = mko::make_pupil_quadrature(8, 6, mko::PupilMode::Half);
    const mko::CpuTraceResult a = mko::trace_system_cpu(with.system, with.table, q);
    const mko::CpuTraceResult b = mko::trace_system_cpu(without.system, without.table, q);
    ASSERT_EQ(a.rays.size(), b.rays.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.rays.size(); ++i) {
        ASSERT_EQ(a.status[i], TraceStatus::Ok);
        ASSERT_EQ(b.status[i], TraceStatus::Ok);
        max_diff = std::max({max_diff, std::abs(a.rays[i].opl - b.rays[i].opl),
                             std::abs(a.rays[i].p.y - b.rays[i].p.y)});
        EXPECT_NEAR(a.rays[i].d.y, b.rays[i].d.y, kUnitFp64);
    }
    EXPECT_LT(max_diff, kLengthFp64);
}

// Mit Geisterplatten: Wellenfront wird verändert, die Auswertung läuft durch (RMS endlich).
TEST(GhostPlate, WavefrontOfFileSystem) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    const mko::WavefrontResult w = mko::evaluate_wavefront(ls);
    EXPECT_EQ(w.failed_count(), 0u);
    mko::SystemDescription bare = ls.system;
    bare.ghost_plates.clear();
    const mko::WavefrontResult w0 =
        mko::evaluate_wavefront(mko::prepare_system(bare, MKO_SOURCE_DIR));
    for (std::size_t i = 0; i < w.maps.size(); ++i) {
        EXPECT_TRUE(std::isfinite(w.maps[i].rms));
        EXPECT_NE(w.maps[i].rms, w0.maps[i].rms);
    }
}

TEST(GhostPlate, TableErrors) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    // Brechzahltabelle ohne Gläser der Geisterplatten
    mko::IndexTable no_ghost = ls.index;
    no_ghost.ghost_plates = 0;
    no_ghost.ghost_n.clear();
    EXPECT_THROW((void)mko::make_surface_table(ls.system, no_ghost), std::invalid_argument);
    EXPECT_THROW((void)mko::compute_paraxial(ls.system, no_ghost, 0), std::invalid_argument);
    // Zu viele Tabellenflächen (Datei 3 + 62 Geisterplatten > kMaxSurfaces)
    mko::SystemDescription many = ls.system;
    while (static_cast<int>(many.surfaces.size() + many.ghost_plates.size()) <= mko::kMaxSurfaces) {
        many.ghost_plates.push_back(many.ghost_plates.back());
    }
    EXPECT_THROW((void)mko::prepare_system(many, MKO_SOURCE_DIR), std::invalid_argument);
    // Unsortierte Platten: make_surface_map lehnt ab (validate_system ebenso)
    mko::SystemDescription unsorted = ls.system;
    std::swap(unsorted.ghost_plates[0], unsorted.ghost_plates[1]);
    EXPECT_THROW((void)mko::make_surface_map(unsorted), std::invalid_argument);
    EXPECT_THROW(mko::validate_system(unsorted), mko::FormatError);
    // Unbekanntes Glas einer Geisterplatte beim Laden
    mko::SystemDescription unknown = ls.system;
    unknown.ghost_plates[1].glass.name = "N-BK8";
    EXPECT_THROW((void)mko::prepare_system(unknown, MKO_SOURCE_DIR), mko::FormatError);
}
