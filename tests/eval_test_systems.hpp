#pragma once

// Testsysteme, analytische Orakel und Prüfungen des Modus EVAL (M8-T1), gemeinsam für die
// CPU-Referenz (test_eval.cpp) und den Kernel (test_cuda_eval.cpp). Die Prüfungen nehmen eine
// Auswertefunktion (eval_cpu oder TraceEngine::evaluate) und melden die gemessenen Größen.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/eval_ray.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

namespace mko::test {

using EvalFn = std::function<EvalResult(const SystemBatchHost&, const EvalSettings&)>;

inline constexpr double kEvalDeg = std::numbers::pi / 180.0;

inline LoadedSystem load_eval_system(const std::string& json) {
    return prepare_system(read_system_json(json), MKO_SOURCE_DIR);
}

// ---- Aplanatische Punkte (perfekte Kugelwelle) ------------------------------------------------

// Punktquelle im Glas (n) im aplanatischen Punkt einer Kugelfläche (R = 3 mm, Übergang Glas →
// Luft, Aufbau wie Wavefront.AplanaticPointSourceHasVanishingRms): Die gebrochene Welle ist eine
// Kugelwelle um den virtuellen Bildpunkt img. Richtung u = (0, u_y, −1)/|…| (x = 0, damit die
// halbe Pupille erlaubt ist); Bildpunkt M = (0, img.y, img.z) über field_target und den
// Arbeitsabstand. Brechzahlen bei λ_ref für die Paraxialrechnung (Austrittspupille) gesetzt.
struct AplanaticCase {
    SystemBatchHost batch;
    Vec3<double> image;
};

inline AplanaticCase aplanatic_case(double n, double uy, double sin_max = 0.95) {
    const double Rabs = 3.0;
    const double c = -1.0 / Rabs;
    const double t0 = 6.0;
    const Vec3<double> C{0.0, 0.0, 1.0 / c};
    Vec3<double> u{0.0, uy, -1.0};
    u = (1.0 / norm(u)) * u;
    const Vec3<double> src = C + (Rabs / n) * u;
    const Vec3<double> img = C + (n * Rabs) * u;
    BatchTopology t;
    t.surfaces = 2;
    t.wavelengths = 1;
    t.fields = 1;
    t.source = SourceKind::Point;
    t.type = {SurfaceKind::Plane, SurfaceKind::Sphere};
    SystemBatchHost b(t, 1);
    b.wavelength_nm(0) = 587.56;
    b.wavelength_weight(0) = 1.0;
    b.field_weight(0) = 1.0;
    b.reference_wavelength_nm = 587.56;
    b.source_point = {src.x, src.y, src.z + t0};
    b.source_na = sin_max * n;
    b.curvature(0, 1) = c;
    b.thickness(0, 0) = t0;
    b.thickness(0, 1) = img.z;
    b.index(0, 0, 0) = n;
    b.index(0, 1, 0) = 1.0;
    b.paraxial_index(0, 0) = n;
    b.paraxial_index(0, 1) = 1.0;
    b.pupil_radius(0) = 1.0;
    b.field_target(0, 0) = img.y;
    b.validate_values();
    return {std::move(b), img};
}

// ---- Einzelflächen mit bekannter Verzeichnung und Feldkrümmung ---------------------------------

// Kugelfläche Luft → Glas n' = 1,5, kollimiertes Licht, Bildebene im paraxialen Fokus, Felder
// 0°, 1°, 5°, 10°:
// - Blende im Scheitel (c = 0,02, R = 50 mm, f = 100 mm, f' = 150 mm): Der Hauptstrahl trifft
//   den Scheitel unter I = θ, sin θ' = sin θ / n', Bildhöhe f' tan θ', Verzeichnung
//   n' tan θ' / tan θ − 1 gegen f tan θ (exakt).
// - Blende im Krümmungsmittelpunkt (c = −0,05, R = −20 mm, f = −40 mm, virtuelles Bild bei
//   f' = −60 mm): Der Hauptstrahl läuft durch C, trifft senkrecht (I = 0) und schneidet die
//   Bildebene bei f tan θ: keine Verzeichnung, kein Astigmatismus, Bildschale = Petzvalschale
//   (Kugel um C mit Radius |f|).
inline constexpr double kSurfaceIndex = 1.5;

inline std::string single_surface_json(bool stop_at_center) {
    const std::string stop_t = stop_at_center ? "20.0" : "0.0";
    const std::string c = stop_at_center ? "-0.05" : "0.02";
    const std::string w = stop_at_center ? "-60.0" : "150.0";
    return R"({"format": "mikroptim.system/1", "name": "Einzelfläche", "units": "mm",
        "glass_catalogs": [],
        "spectrum": {"wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56},
        "source": {"type": "collimated", "field_angles_deg": [0.0, 1.0, 5.0, 10.0],
                   "field_weights": [1.0, 1.0, 1.0, 1.0]},
        "surfaces": [
          {"type": "stop", "semi_diameter": 1.0, "thickness": )" +
           stop_t + R"(, "material": "AIR"},
          {"type": "sphere", "curvature": )" +
           c + R"(, "thickness": )" + w +
           R"(, "material": {"model": "constant", "n": 1.5}, "semi_diameter": 30.0}]})";
}

inline SystemBatchHost single_surface_batch(bool stop_at_center) {
    return make_system_batch(load_eval_system(single_surface_json(stop_at_center)));
}

// Coddington-Gleichungen für kollimiertes Licht (t = s = ∞) an einer Kugelfläche mit Radius R:
// Abstände der tangentialen und sagittalen Brennlinie entlang des gebrochenen Hauptstrahls vom
// Auftreffpunkt, n' cos² I' / t' = (n' cos I' − n cos I) / R, n' / s' = (…) / R.
struct CoddingtonFoci {
    double tangential;
    double sagittal;
};

inline CoddingtonFoci coddington_collimated(double n, double n2, double R, double cos_i,
                                            double cos_i2) {
    const double k = (n2 * cos_i2 - n * cos_i) / R;
    return {n2 * cos_i2 * cos_i2 / k, n2 / k};
}

// Erwartete Feldgrößen der Einzelflächen (z der Brennlinien relativ zur Bildebene, Hauptstrahl
// relativ zum idealen Bildpunkt f tan θ).
struct SurfaceFieldExpectation {
    double chief_y;
    double focus_tangential;
    double focus_sagittal;
};

inline SurfaceFieldExpectation single_surface_expectation(bool stop_at_center, double theta) {
    const double n2 = kSurfaceIndex;
    if (!stop_at_center) {
        const double R = 50.0;
        const double f = R / (n2 - 1.0);
        const double w = n2 * f;
        const double sin_i2 = std::sin(theta) / n2;
        const double cos_i2 = std::sqrt((1.0 - sin_i2) * (1.0 + sin_i2));
        const CoddingtonFoci c = coddington_collimated(1.0, n2, R, std::cos(theta), cos_i2);
        return {w * sin_i2 / cos_i2 - f * std::tan(theta), c.tangential * cos_i2 - w,
                c.sagittal * cos_i2 - w};
    }
    const double R = -20.0;
    const double f = R / (n2 - 1.0);
    const double w = n2 * f;
    const CoddingtonFoci c = coddington_collimated(1.0, n2, R, 1.0, 1.0);
    const double pz = R + std::abs(R) * std::cos(theta); // Auftreffpunkt (Scheitelkoordinaten)
    return {0.0, pz + c.tangential * std::cos(theta) - w, pz + c.sagittal * std::cos(theta) - w};
}

// ---- Einzellinse bei kleiner Apertur (sphärische Aberration, M1-T7) ---------------------------

// einzellinse_bk7.json mit Blendenradius a und den Feldern `field_angles_deg` (Standard nur das
// Achsfeld), drei Varianten: Variante l hat
// die Bildebene im paraxialen Fokus der Wellenlänge l (aus einem unabhängigen paraxialen
// Randstrahl), ausgewertet wird das Tupel (v = l, f = 0, l). Seidel: W₀₄₀ = S_I / 8 mit
// S_I = −Σ A² y Δ(u/n), A = n (y c + u) (wie Wavefront.SeidelSphericalAberrationOfSingleLens).
struct LensCase {
    SystemBatchHost batch;
    std::vector<double> w040; // [L]
};

inline LensCase lens_case(double a, std::vector<double> field_angles_deg = {0.0}) {
    LoadedSystem ls = load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    ls.system.source.field_weights.assign(field_angles_deg.size(), 1.0);
    ls.system.source.field_angles_deg = std::move(field_angles_deg);
    const int S = ls.table.surfaces;
    const int L = ls.table.wavelengths;
    LensCase out{make_system_batch(ls.system, ls.index, L), {}};
    for (int l = 0; l < L; ++l) {
        double y = a;
        double u = 0.0;
        double s1 = 0.0;
        for (int s = 1; s < S; ++s) {
            const auto su = static_cast<std::size_t>(s);
            y += ls.table.thickness[su - 1] * u;
            const double nb = ls.index.at(s - 1, l);
            const double na = ls.index.at(s, l);
            const double cs = ls.table.curvature[su];
            const double A = nb * (y * cs + u);
            const double up = (nb * u - y * cs * (na - nb)) / na;
            s1 += -A * A * y * (up / na - u / nb);
            u = up;
        }
        out.batch.pupil_radius(l) = a;
        out.batch.thickness(l, S - 1) = -y / u;
        out.w040.push_back(s1 / 8.0);
    }
    out.batch.validate_values();
    return out;
}

// Wellenfront desselben Tupels mit eval_ray an den Stützstellen der Gauß-Quadratur auf der
// Achse (Fächer ρ_i = √u_i, x = 0), gleiche Referenzsphäre: unabhängige Integration derselben
// Wellenfrontfunktion. Daraus die Projektion auf verschobene Legendre-Polynome P_m(2u − 1),
// m <= degree (exakt für die Quadratur, solange 2·degree < 2N), als stetiges Modell der OPD.
struct GaussWavefront {
    double rms = 0.0;
    std::vector<double> legendre; // c_m

    double model(double u) const {
        const double xi = 2.0 * u - 1.0;
        double p0 = 1.0;
        double p1 = xi;
        double s = legendre[0] * p0 + (legendre.size() > 1 ? legendre[1] * p1 : 0.0);
        for (std::size_t m = 1; m + 1 < legendre.size(); ++m) {
            const double md = static_cast<double>(m);
            const double p2 = ((2.0 * md + 1.0) * xi * p1 - md * p0) / (md + 1.0);
            s += legendre[m + 1] * p2;
            p0 = p1;
            p1 = p2;
        }
        return s;
    }
};

inline GaussWavefront gauss_wavefront(const SystemBatchHost& batch, int v, int f, int l,
                                      EvalReference reference, int rings, int degree) {
    const SystemBatchView<double> view = batch.view();
    const SurfaceTableView<double> table = view.surface_table(v);
    const Vec3<double> center = batch_image_point(view, table, v, f);
    const EvalReferenceGeometry geo = eval_reference_geometry(batch, reference);
    const double kappa = geo.curvature[static_cast<std::size_t>(v * batch.fields() + f)];
    const PupilQuadrature<double> q = make_pupil_quadrature(rings, 1, PupilMode::Axis);
    const int n = sample_count(q);
    std::vector<double> opl(static_cast<std::size_t>(n)), w(opl.size()), opd(opl.size()),
        u(opl.size());
    std::vector<TraceStatus> st(opl.size());
    for (int k = 0; k < n; ++k) {
        const auto ku = static_cast<std::size_t>(k);
        const PupilSample<double> s = sample(q, k);
        const EvalRay<double> r =
            eval_ray(view, table, l, v, f, l, 0.0, s.rho, center, reference, kappa);
        st[ku] = r.status;
        opl[ku] = r.opl;
        w[ku] = s.weight;
        u[ku] = s.rho * s.rho;
    }
    GaussWavefront g;
    const OpdStatistics<double> stats =
        opd_statistics(opl.data(), w.data(), st.data(), n, opd.data());
    EXPECT_EQ(stats.failed, 0);
    g.rms = stats.rms;
    for (int m = 0; m <= degree; ++m) {
        double c = 0.0;
        for (int k = 0; k < n; ++k) {
            GaussWavefront unit;
            unit.legendre.assign(static_cast<std::size_t>(m + 1), 0.0);
            unit.legendre.back() = 1.0;
            const auto ku = static_cast<std::size_t>(k);
            c += w[ku] * opd[ku] * unit.model(u[ku]);
        }
        g.legendre.push_back((2.0 * m + 1.0) * c);
    }
    return g;
}

// Gewichteter Fit OPD ≈ Σ_m c_m ρ^(2m), m = 0 … terms − 1, über die gültigen Rasterpunkte eines
// Tupels (Normalgleichungen, Gauß-Elimination mit Spaltenpivot; terms <= 4).
inline std::vector<double> radial_fit(const EvalResult& r, int v, int f, int l, int terms) {
    std::vector<double> G(static_cast<std::size_t>(terms * terms), 0.0);
    std::vector<double> h(static_cast<std::size_t>(terms), 0.0);
    for (std::int64_t k = 0; k < r.samples(); ++k) {
        const std::size_t i = r.ray_index(v, f, l, k);
        if (static_cast<TraceStatus>(r.status[i]) != TraceStatus::Ok) {
            continue;
        }
        const EvalPoint& p = r.grid.points[static_cast<std::size_t>(k)];
        const double u = p.x * p.x + p.y * p.y;
        std::vector<double> phi(static_cast<std::size_t>(terms), 1.0);
        for (int m = 1; m < terms; ++m) {
            phi[static_cast<std::size_t>(m)] = phi[static_cast<std::size_t>(m - 1)] * u;
        }
        for (int a = 0; a < terms; ++a) {
            for (int b = 0; b < terms; ++b) {
                G[static_cast<std::size_t>(a * terms + b)] +=
                    p.weight * phi[static_cast<std::size_t>(a)] * phi[static_cast<std::size_t>(b)];
            }
            h[static_cast<std::size_t>(a)] +=
                p.weight * phi[static_cast<std::size_t>(a)] * r.opd[i];
        }
    }
    const auto at = [&](int a, int b) -> double& {
        return G[static_cast<std::size_t>(a * terms + b)];
    };
    for (int c = 0; c < terms; ++c) {
        int piv = c;
        for (int a = c + 1; a < terms; ++a) {
            if (std::abs(at(a, c)) > std::abs(at(piv, c))) {
                piv = a;
            }
        }
        for (int b = 0; b < terms; ++b) {
            std::swap(at(c, b), at(piv, b));
        }
        std::swap(h[static_cast<std::size_t>(c)], h[static_cast<std::size_t>(piv)]);
        for (int a = c + 1; a < terms; ++a) {
            const double q = at(a, c) / at(c, c);
            for (int b = c; b < terms; ++b) {
                at(a, b) -= q * at(c, b);
            }
            h[static_cast<std::size_t>(a)] -= q * h[static_cast<std::size_t>(c)];
        }
    }
    std::vector<double> x(static_cast<std::size_t>(terms), 0.0);
    for (int a = terms - 1; a >= 0; --a) {
        double s = h[static_cast<std::size_t>(a)];
        for (int b = a + 1; b < terms; ++b) {
            s -= at(a, b) * x[static_cast<std::size_t>(b)];
        }
        x[static_cast<std::size_t>(a)] = s / at(a, a);
    }
    return x;
}

// ---- Vergleich zweier Ergebnisse --------------------------------------------------------------

// Größtes Verhältnis |a − b| / max(abs, rel · |a|) über alle gültigen Werte (OPD, Spot,
// Feldstrahlen) und Zahl der Strahlen mit verschiedenem Status. NaN muss auf beiden Seiten an
// denselben Stellen stehen.
struct EvalComparison {
    double ratio = 0.0;     // <= 1: innerhalb der Toleranz
    double max_opd = 0.0;   // größte Abweichung der OPD [mm]
    double max_spot = 0.0;  // größte Abweichung des Spots [mm]
    double max_field = 0.0; // größte Abweichung der Feldstrahl-Werte
    std::size_t status_mismatch = 0;
    std::size_t nan_mismatch = 0;
};

inline EvalComparison compare_eval(const EvalResult& a, const EvalResult& b, double abs_tol,
                                   double rel_tol) {
    EvalComparison c;
    const auto cmp = [&](const std::vector<double>& x, const std::vector<double>& y, double& max) {
        EXPECT_EQ(x.size(), y.size());
        for (std::size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
            if (std::isnan(x[i]) || std::isnan(y[i])) {
                c.nan_mismatch += std::isnan(x[i]) != std::isnan(y[i]) ? 1 : 0;
                continue;
            }
            const double d = std::abs(x[i] - y[i]);
            max = std::max(max, d);
            c.ratio = std::max(c.ratio, d / std::max(abs_tol, rel_tol * std::abs(x[i])));
        }
    };
    cmp(a.opd, b.opd, c.max_opd);
    cmp(a.spot_x, b.spot_x, c.max_spot);
    cmp(a.spot_y, b.spot_y, c.max_spot);
    cmp(a.field_rays, b.field_rays, c.max_field);
    EXPECT_EQ(a.status.size(), b.status.size());
    for (std::size_t i = 0; i < std::min(a.status.size(), b.status.size()); ++i) {
        c.status_mismatch += a.status[i] != b.status[i] ? 1 : 0;
    }
    for (std::size_t i = 0; i < std::min(a.field_status.size(), b.field_status.size()); ++i) {
        c.status_mismatch += a.field_status[i] != b.field_status[i] ? 1 : 0;
    }
    return c;
}

// ---- Abnahmeprüfungen (für CPU und GPU) -------------------------------------------------------

// (1) Perfekte Kugelwelle: OPD-Karte ≈ 0 auf dem ganzen Raster (max |OPD| < kAplanaticRms),
// Spot ≈ 0 (alle Strahlen schneiden die Bildebene im Bildpunkt), Brennlinien in der
// Bildebene; für beide Lagen der Referenzsphäre, volle und halbe Pupille.
inline void check_aplanatic(const EvalFn& eval, const char* backend) {
    double max_opd = 0.0;
    double max_spot = 0.0;
    double max_focus = 0.0;
    std::size_t rays = 0;
    for (const double n : {1.5, 1.8}) {
        for (const double uy : {0.0, -0.15}) {
            const AplanaticCase ac = aplanatic_case(n, uy);
            for (const EvalReference ref : {EvalReference::ExitPupil, EvalReference::Vertex}) {
                for (const bool half : {false, true}) {
                    EvalSettings s;
                    s.grid = 64;
                    s.half_pupil = half;
                    s.reference = ref;
                    const EvalResult r = eval(ac.batch, s);
                    ASSERT_EQ(r.failed_count(), 0u);
                    for (std::size_t i = 0; i < r.opd.size(); ++i) {
                        max_opd = std::max(max_opd, std::abs(r.opd[i]));
                        max_spot = std::max(max_spot, std::hypot(r.spot_x[i], r.spot_y[i]));
                    }
                    rays += r.opd.size();
                    const EvalTupleStats st = eval_statistics(r, 0)[0];
                    EXPECT_LT(st.rms_opd, kAplanaticRms);
                    max_focus = std::max({max_focus, std::abs(st.focus_tangential),
                                          std::abs(st.focus_sagittal), std::abs(st.chief_y)});
                }
            }
        }
    }
    EXPECT_LT(max_opd, kAplanaticRms);
    EXPECT_LT(max_spot, kRayPointFp64);
    EXPECT_LT(max_focus, kEvalFocusFp64);
    std::printf("[ Info     ] %s, aplanatische Punkte (sin θ = 0,95, N = 64): max |OPD| = %.3e "
                "mm, max Spotradius = %.3e mm, max |Brennlinie|, |Hauptstrahl| = %.3e mm (%zu "
                "Strahlen)\n",
                backend, max_opd, max_spot, max_focus, rays);
}

// (2) Einzellinse bei kleiner Apertur (NA 0,049 und 0,025, wie M1-T7), Achsfeld, Bildebene im
// paraxialen Fokus: Die OPD jedes Rasterpunkts stimmt mit dem stetigen Modell aus der
// Gauß-Quadratur (Legendre bis Grad 8 in u = ρ², nach Kolbenabzug auf dem Raster) überein, und
// der Fit c₀ + c₁ρ² + c₂ρ⁴ + c₃ρ⁶ liefert c₂ = −W₀₄₀ (Seidel, Modellfehler höherer Ordnung
// wie M1-T7) und c₁ ≈ 0 (kein Defokus). Beide Lagen der Referenzsphäre.
inline void check_lens_grid(const EvalFn& eval, const char* backend) {
    for (const double a : {5.0, 2.5}) {
        const LensCase lc = lens_case(a);
        for (const EvalReference ref : {EvalReference::ExitPupil, EvalReference::Vertex}) {
            EvalSettings s;
            s.grid = 96;
            s.reference = ref;
            const EvalResult r = eval(lc.batch, s);
            ASSERT_EQ(r.failed_count(), 0u);
            for (int l = 0; l < lc.batch.wavelengths(); ++l) {
                const int v = l;
                const GaussWavefront g = gauss_wavefront(lc.batch, v, 0, l, ref, 16, 8);
                double mean = 0.0;
                for (std::int64_t k = 0; k < r.samples(); ++k) {
                    const EvalPoint& p = r.grid.points[static_cast<std::size_t>(k)];
                    mean += p.weight * g.model(p.x * p.x + p.y * p.y);
                }
                double max_dev = 0.0;
                for (std::int64_t k = 0; k < r.samples(); ++k) {
                    const EvalPoint& p = r.grid.points[static_cast<std::size_t>(k)];
                    const double expected = g.model(p.x * p.x + p.y * p.y) - mean;
                    max_dev =
                        std::max(max_dev, std::abs(r.opd[r.ray_index(v, 0, l, k)] - expected));
                }
                EXPECT_LT(max_dev, kReferenceSphereFp64) << "a=" << a << " l=" << l;
                const std::vector<double> c = radial_fit(r, v, 0, l, 4);
                const double w040 = lc.w040[static_cast<std::size_t>(l)];
                const double rel_seidel = std::abs(c[2] + w040) / std::abs(w040);
                EXPECT_LT(rel_seidel, kSeidelRelative) << "a=" << a << " l=" << l;
                EXPECT_LT(std::abs(c[1]), kSeidelRelative * std::abs(w040)) << "a=" << a;
                std::printf("[ Info     ] %s, %s, a = %.1f mm, l = %d: max |OPD − Modell| = "
                            "%.3e mm, ρ⁴-Koeffizient gegen −W040 rel. %.2e, ρ²/W040 = %.2e\n",
                            backend, ref == EvalReference::Vertex ? "Scheitel" : "Austrittspupille",
                            a, l, max_dev, rel_seidel, c[1] / w040);
            }
        }
    }
}

// (2) Konvergenz des Rasters: Der RMS der OPD auf dem N×N-Raster konvergiert mit N gegen den
// RMS der Gauß-Quadratur (16 Ringe, auf der Achse exakt bis Grad 31 in u). Der Rasterfehler ist
// ein reiner Quadraturfehler: Er stimmt mit dem RMS des stetigen Modells auf denselben
// Rasterpunkten überein (Abweichung < kEvalGridModel relativ) und fällt wie die Abweichung der
// Gitterpunktzahl vom Flächeninhalt (Gaußsches Kreisproblem), Schranke
// kEvalGridConvergence·N^(−4/3).
inline void check_grid_convergence(const EvalFn& eval, const char* backend, int max_grid) {
    const LensCase lc = lens_case(5.0);
    const int v = 1;
    const int l = 1;
    for (const EvalReference ref : {EvalReference::ExitPupil, EvalReference::Vertex}) {
        const GaussWavefront g = gauss_wavefront(lc.batch, v, 0, l, ref, 16, 8);
        for (int N = 16; N <= max_grid; N *= 2) {
            EvalSettings s;
            s.grid = N;
            s.reference = ref;
            const EvalResult r = eval(lc.batch, s);
            const EvalTupleStats st = eval_statistics(r, v)[static_cast<std::size_t>(l)];
            // RMS des Modells auf den Rasterpunkten (exakte Momente des Rasters)
            double mean = 0.0;
            for (const EvalPoint& p : r.grid.points) {
                mean += p.weight * g.model(p.x * p.x + p.y * p.y);
            }
            double sq = 0.0;
            for (const EvalPoint& p : r.grid.points) {
                const double e = g.model(p.x * p.x + p.y * p.y) - mean;
                sq += p.weight * e * e;
            }
            const double rms_model = std::sqrt(sq);
            const double rel_gauss = std::abs(st.rms_opd - g.rms) / g.rms;
            const double rel_model = std::abs(st.rms_opd - rms_model) / g.rms;
            const double bound =
                kEvalGridConvergence * std::pow(static_cast<double>(N), -4.0 / 3.0);
            EXPECT_LT(rel_model, kEvalGridModel) << "N=" << N;
            EXPECT_LT(rel_gauss, bound) << "N=" << N;
            std::printf("[ Info     ] %s, %s, N = %3d (%6lld Punkte): RMS = %.9e mm, Gauß %.9e, "
                        "rel. Rasterfehler %.2e (Schranke %.2e), gegen Modell auf dem Raster "
                        "%.2e\n",
                        backend, ref == EvalReference::Vertex ? "Scheitel" : "Austrittspupille", N,
                        static_cast<long long>(r.samples()), st.rms_opd, g.rms, rel_gauss, bound,
                        rel_model);
        }
    }
}

// (3) und (5) Verzeichnung und Feldkrümmung der Einzelflächen gegen die exakten Formeln.
inline void check_single_surfaces(const EvalFn& eval, const char* backend) {
    for (const bool center : {false, true}) {
        const SystemBatchHost b = single_surface_batch(center);
        EvalSettings s;
        s.grid = 32;
        const EvalResult r = eval(b, s);
        ASSERT_EQ(r.failed_count(), 0u);
        const std::vector<EvalTupleStats> st = eval_statistics(r, 0);
        double max_chief = 0.0;
        double max_focus = 0.0;
        double max_distortion = 0.0;
        for (int f = 0; f < b.fields(); ++f) {
            const double theta = b.field_angle(0, f);
            const SurfaceFieldExpectation e = single_surface_expectation(center, theta);
            const EvalTupleStats& t = st[static_cast<std::size_t>(f)];
            EXPECT_NEAR(t.chief_x, 0.0, kLengthFp64);
            EXPECT_NEAR(t.chief_y, e.chief_y, kLengthFp64) << "f=" << f;
            const double scale = std::max(1.0, std::abs(e.focus_tangential));
            EXPECT_NEAR(t.focus_tangential, e.focus_tangential, kEvalFocusRelative * scale)
                << "f=" << f;
            EXPECT_NEAR(t.focus_sagittal, e.focus_sagittal, kEvalFocusRelative * scale)
                << "f=" << f;
            max_chief = std::max(max_chief, std::abs(t.chief_y - e.chief_y));
            max_focus = std::max({max_focus, std::abs(t.focus_tangential - e.focus_tangential),
                                  std::abs(t.focus_sagittal - e.focus_sagittal)});
            if (f > 0) {
                const double expected = 100.0 * e.chief_y / t.image_height;
                // Hauptstrahl auf kLengthFp64 genau, in Prozent der Bildhöhe
                EXPECT_NEAR(t.distortion_chief, expected,
                            100.0 * kLengthFp64 / std::abs(t.image_height));
                max_distortion = std::max(max_distortion, std::abs(t.distortion_chief - expected));
                std::printf("[ Info     ] %s, Blende %s, θ = %4.1f°: Verzeichnung %+.6f %% "
                            "(exakt %+.6f %%), z_t = %+.6e mm (Coddington %+.6e), z_s = %+.6e "
                            "mm (%+.6e)\n",
                            backend, center ? "in C" : "im Scheitel", theta / kEvalDeg,
                            t.distortion_chief, expected, t.focus_tangential, e.focus_tangential,
                            t.focus_sagittal, e.focus_sagittal);
            } else {
                EXPECT_TRUE(std::isnan(t.distortion_chief));
            }
        }
        if (center) {
            // Astigmatismusfrei: Bildschale = Petzvalschale, z ≈ κ_P h² / 2 mit κ_P = −n' P
            // (docs/numerik.md, Abschnitt 9) für kleine Felder (Modellfehler O(θ²) relativ).
            const ParaxialData p =
                compute_paraxial(load_eval_system(single_surface_json(true)).system,
                                 load_eval_system(single_surface_json(true)).catalog);
            const EvalTupleStats& t1 = st[1];
            const double sag = 0.5 * p.petzval_curvature * t1.image_height * t1.image_height;
            EXPECT_NEAR(t1.astigmatism(), 0.0, kEvalFocusRelative * std::abs(sag));
            EXPECT_NEAR(t1.focus_sagittal / sag, 1.0, kEvalPetzvalModel);
            std::printf("[ Info     ] %s, Blende in C, θ = 1°: z_s / (κ_P h²/2) = %.8f, z_t − z_s "
                        "= %.3e mm\n",
                        backend, t1.focus_sagittal / sag, t1.astigmatism());
        }
        std::printf("[ Info     ] %s, Blende %s: max |Δ Hauptstrahl| = %.3e mm, max |Δ "
                    "Verzeichnung| = %.3e %%, max |Δ Brennlinie| = %.3e mm\n",
                    backend, center ? "in C" : "im Scheitel", max_chief, max_distortion, max_focus);
    }
}

} // namespace mko::test
