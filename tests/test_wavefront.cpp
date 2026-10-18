// Referenzsphäre, OPD und RMS-Wellenfront (M1-T7). Orakel: exakte Kugelwellen (Kosinussatz für
// den Abstand Punkt–Sphäre entlang eines Strahls), aplanatische Punkte einer Kugelfläche und die
// Seidel-Summe S_I aus einem unabhängigen paraxialen Randstrahl.

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/trace.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

using mko::HostSurfaceTable;
using mko::PupilMode;
using mko::Ray;
using mko::SurfaceKind;
using mko::SystemDescription;
using mko::TraceStatus;
using mko::Vec3;
using mko::WavefrontMap;
using mko::WavefrontOptions;
using mko::WavefrontResult;
using mko::test::kAplanaticRms;
using mko::test::kLengthFp64;
using mko::test::kReferenceSphereFp64;
using mko::test::kSeidelRelative;
using mko::test::kUnitFp64;

namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

// Tabelle aus Spalten mit einer Wellenlänge; index[s] = Medium nach Fläche s.
HostSurfaceTable make_table(std::vector<SurfaceKind> type, std::vector<double> curvature,
                            std::vector<double> thickness, std::vector<double> index) {
    HostSurfaceTable t;
    t.surfaces = static_cast<int>(type.size());
    t.wavelengths = 1;
    t.type = std::move(type);
    t.curvature = std::move(curvature);
    t.thickness = std::move(thickness);
    t.index = std::move(index);
    t.semi_diameter.assign(t.type.size(), 10.0);
    return t;
}

// Systembeschreibung mit Punktquelle für eine Tabelle (nicht validiert: Fläche 0 liegt im Glas,
// was nur über die Tabellen-API geht, numerik.md, Abschnitt 4). Der Tracer liest nur Quelle,
// Spektrum und die Flächenzahl.
SystemDescription point_source_system(const HostSurfaceTable& table, Vec3<double> q, double na) {
    SystemDescription s;
    s.name = "Punktquelle";
    s.spectrum.wavelengths_nm = {587.56};
    s.spectrum.weights = {1.0};
    s.spectrum.reference_nm = 587.56;
    s.source.type = mko::SourceType::Point;
    s.source.point = {q.x, q.y, q.z};
    s.source.numerical_aperture = na;
    s.surfaces.resize(static_cast<std::size_t>(table.surfaces));
    return s;
}

mko::LoadedSystem load_test_system(const std::string& json) {
    return mko::prepare_system(mko::read_system_json(json), MKO_SOURCE_DIR);
}

// Abstand von P entlang einer Richtung mit cos α = cos_a (Winkel zur Achse) bis zur Sphäre vom
// Radius R um M = P + δz·ẑ, P innerhalb der Sphäre (|δz| < R). Kosinussatz im Dreieck P, M, X:
// R² = s² + δz² − 2 s δz cos α  →  s = δz cos α + √(R² − δz² sin² α).
double distance_to_sphere(double dz, double cos_a, double R) {
    const double sin2 = (1.0 - cos_a) * (1.0 + cos_a);
    return dz * cos_a + std::sqrt(R * R - dz * dz * sin2);
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Referenzsphäre
// ---------------------------------------------------------------------------------------------

// Konvergente Kugelwelle: Strahlen auf einer gekrümmten Fläche laufen exakt auf P = (0, 0, w0)
// zu, OPL(X) = C − n |P − X| (Wellenfront um P). Referenzzentrum M = P + δz·ẑ, R_ref = |M|.
// Exakt: OPL auf der Referenzsphäre = C − n σ, σ = Abstand von P in Richtung −d bis zur Sphäre
// (Kosinussatz mit cos α = −d_z, Winkel zwischen −d und P → M = +ẑ).
TEST(ReferenceSphere, ConvergingSphericalWaveWithShiftedCenter) {
    const double w0 = 50.0;
    const double n = 1.33;
    const double C = 100.0;
    const double c_surf = 0.02; // Startpunkte auf einer Sphäre z = c r² / (1 + √(1 − c² r²))
    const Vec3<double> P{0.0, 0.0, w0};
    double max_err = 0.0;
    double max_on_sphere = 0.0;
    int count = 0;
    for (const double dz : {-3.0, -0.2, 0.0, 0.2, 3.0}) {
        const Vec3<double> M{0.0, 0.0, w0 + dz};
        const double R = w0 + dz;
        for (const double r : {0.0, 1.0, 5.0, 12.0, 20.0}) {
            for (const double phi : {0.0, 0.7, 2.0, 4.0}) {
                const double z = c_surf * r * r / (1.0 + std::sqrt(1.0 - c_surf * c_surf * r * r));
                const Vec3<double> p{r * std::sin(phi), r * std::cos(phi), z};
                const Vec3<double> to_p = P - p;
                const double len = mko::norm(to_p);
                Ray<double> ray{p, (1.0 / len) * to_p, C - n * len};
                const double dzc = ray.d.z;
                double s = 0.0;
                ASSERT_EQ(mko::reference_sphere_step(ray, M, n, &s), TraceStatus::Ok);
                const double sigma = distance_to_sphere(dz, -dzc, R);
                const double expected = C - n * sigma;
                EXPECT_NEAR(ray.opl, expected, kReferenceSphereFp64)
                    << "dz=" << dz << " r=" << r << " phi=" << phi;
                const double on_sphere = std::abs(mko::norm(ray.p - M) - R);
                EXPECT_LT(on_sphere, kLengthFp64);
                max_err = std::max(max_err, std::abs(ray.opl - expected));
                max_on_sphere = std::max(max_on_sphere, on_sphere);
                ++count;
            }
        }
    }
    std::printf("[ Info     ] Konvergente Kugelwelle: max |OPL - exakt| = %.3e mm, max Abstand "
                "von der Referenzsphäre = %.3e mm (%d Strahlen)\n",
                max_err, max_on_sphere, count);
}

// Abnahme M1-T7: divergente Kugelwelle aus einer Punktquelle (homogenes Medium, zwei Ebenen),
// Referenzzentrum um δz gegen den Quellpunkt verschoben, über das ganze System-API
// (evaluate_wavefront). Die Bildebene liegt virtuell vor der letzten Fläche (M_z < 0).
// OPL bis zur Referenzsphäre exakt n · s(α) (distance_to_sphere), OPD = OPL − Σ W OPL.
TEST(ReferenceSphere, DivergingSphericalWaveWithShiftedCenterThroughSystem) {
    const double n = 1.5;
    const double t0 = 5.0;
    const Vec3<double> q{0.0, 0.0, 0.0};      // Quelle, Koordinaten von Fläche 0
    const Vec3<double> P{0.0, 0.0, q.z - t0}; // Quelle in Koordinaten der letzten Fläche
    const double na = 0.9;                    // sin α_max = 0,6
    double max_opl = 0.0;
    double max_opd = 0.0;
    for (const double dz : {-2.0, -0.5, 0.5, 2.0}) {
        const Vec3<double> M{0.0, 0.0, P.z + dz};
        const double R = std::abs(M.z);
        const HostSurfaceTable table =
            make_table({SurfaceKind::Plane, SurfaceKind::Plane}, {0.0, 0.0}, {t0, M.z}, {n, n});
        const SystemDescription sys = point_source_system(table, q, na);
        WavefrontOptions opt;
        opt.pupil_mode = PupilMode::Full;
        opt.rings = 12;
        opt.spokes = 10;
        const auto points = mko::ideal_image_points(sys, table, std::nan(""));
        ASSERT_EQ(points.size(), 1u);
        EXPECT_EQ(points[0].z, M.z);
        const WavefrontResult res = mko::evaluate_wavefront(sys, table, points, opt);
        const WavefrontMap& m = res.at(0, 0);
        ASSERT_EQ(m.failed, 0);
        // Erwartete OPL und Kolben
        std::vector<double> expected(static_cast<std::size_t>(m.samples()));
        double piston = 0.0;
        for (int k = 0; k < m.samples(); ++k) {
            const auto& s = m.pupil[static_cast<std::size_t>(k)];
            const Ray<double> start = mko::point_source_ray(q, na / n, s.rho, s.phi);
            expected[static_cast<std::size_t>(k)] = n * distance_to_sphere(dz, start.d.z, R);
            piston += s.weight * expected[static_cast<std::size_t>(k)];
        }
        for (int k = 0; k < m.samples(); ++k) {
            const auto ku = static_cast<std::size_t>(k);
            EXPECT_NEAR(m.opl[ku], expected[ku], kReferenceSphereFp64) << "dz=" << dz;
            EXPECT_NEAR(m.opd[ku], expected[ku] - piston, kReferenceSphereFp64) << "dz=" << dz;
            max_opl = std::max(max_opl, std::abs(m.opl[ku] - expected[ku]));
            max_opd = std::max(max_opd, std::abs(m.opd[ku] - (expected[ku] - piston)));
        }
        // Verschobenes Zentrum: die OPD ist nicht null (Defokus), sonst wäre der Test leer.
        EXPECT_GT(m.rms, 1e-3);
    }
    std::printf("[ Info     ] Divergente Kugelwelle, verschobenes Zentrum: max |OPL - exakt| = "
                "%.3e mm, max |OPD - exakt| = %.3e mm\n",
                max_opl, max_opd);
}

// Strahl läuft nach der Brechung an der letzten Fläche rückwärts (numerik.md §11): Glas
// n = 1,5 vor einer konkaven Fläche (c = −0,2) in Luft, steiler Strahl (82°). Der Schnittpunkt
// liegt bei |c·y| ≈ 0,65 und k ≈ 0,024 (keine Totalreflexion), danach ist d_z ≈ −0,53.
// trace_ray meldet Ok (kein Transfer mehr), trace_to_reference_sphere muss Backward liefern.
TEST(ReferenceSphere, BackwardRayAfterLastSurfaceIsReported) {
    const HostSurfaceTable table =
        make_table({SurfaceKind::Plane, SurfaceKind::Sphere}, {0.0, -0.2}, {1.0, 10.0}, {1.5, 1.0});
    const Ray<double> start{
        {0.0, 4.8, 0.0}, {0.0, std::sin(82.0 * kDeg), std::cos(82.0 * kDeg)}, 0.0};

    Ray<double> traced = start;
    ASSERT_EQ(mko::trace_ray(table.view(), 0, traced, nullptr), TraceStatus::Ok);
    ASSERT_LT(traced.d.z, -0.5);

    Ray<double> ray = start;
    int failed = 0;
    EXPECT_EQ(
        mko::trace_to_reference_sphere(table.view(), 0, ray, Vec3<double>{0.0, 0.0, 10.0}, &failed),
        TraceStatus::Backward);
    EXPECT_EQ(failed, table.surfaces);
}

// Strahl verfehlt die Referenzsphäre: Status Missed, Strahl unverändert; im Systemtrace zählt
// die Referenzsphäre als Fläche S.
TEST(ReferenceSphere, MissedSphereLeavesRayUnchanged) {
    Ray<double> ray{{5.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 2.0};
    const Ray<double> before = ray;
    EXPECT_EQ(mko::reference_sphere_step(ray, Vec3<double>{0.0, 0.0, 1.0}, 1.0),
              TraceStatus::Missed);
    EXPECT_EQ(ray.p.x, before.p.x);
    EXPECT_EQ(ray.opl, before.opl);

    // Strahl läuft vom Zentrum weg und liegt außerhalb der Sphäre: zugewandte Kappe verfehlt.
    Ray<double> away{{0.0, 3.0, 0.0}, {0.0, 1.0, 0.0}, 0.0};
    EXPECT_EQ(mko::reference_sphere_step(away, Vec3<double>{0.0, 0.0, 1.0}, 1.0),
              TraceStatus::Missed);

    // Achsstrahl im Scheitel: s_ref = 0 für reelle und virtuelle Bildpunkte.
    for (const double w : {10.0, -10.0}) {
        Ray<double> axial{{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 1.0};
        double s = -1.0;
        EXPECT_EQ(mko::reference_sphere_step(axial, Vec3<double>{0.0, 0.0, w}, 1.0, &s),
                  TraceStatus::Ok);
        EXPECT_EQ(s, 0.0);
        EXPECT_EQ(axial.opl, 1.0);
    }

    const HostSurfaceTable table =
        make_table({SurfaceKind::Plane, SurfaceKind::Plane}, {0.0, 0.0}, {1.0, 1.0}, {1.0, 1.0});
    Ray<double> r{{5.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 0.0};
    int failed = 0;
    EXPECT_EQ(
        mko::trace_to_reference_sphere(table.view(), 0, r, Vec3<double>{0.0, 0.0, 1.0}, &failed),
        TraceStatus::Missed);
    EXPECT_EQ(failed, table.surfaces);
}

// ---------------------------------------------------------------------------------------------
// Kolbenabzug und RMS
// ---------------------------------------------------------------------------------------------

TEST(Wavefront, OpdStatisticsExcludesFailedRaysAndRenormalizes) {
    const std::vector<double> opl{10.0, 10.5, 99.0, 9.5, 10.25};
    const std::vector<double> w{0.1, 0.2, 0.3, 0.25, 0.15};
    const std::vector<TraceStatus> st{TraceStatus::Ok, TraceStatus::Ok, TraceStatus::Missed,
                                      TraceStatus::Ok, TraceStatus::Ok};
    std::vector<double> opd(opl.size(), -1.0);
    const auto s = mko::opd_statistics(opl.data(), w.data(), st.data(), 5, opd.data());
    const double vw = 0.1 + 0.2 + 0.25 + 0.15;
    const double piston = (0.1 * 10.0 + 0.2 * 10.5 + 0.25 * 9.5 + 0.15 * 10.25) / vw;
    double ss = 0.0;
    for (const int k : {0, 1, 3, 4}) {
        const auto ku = static_cast<std::size_t>(k);
        ss += w[ku] * (opl[ku] - piston) * (opl[ku] - piston);
        EXPECT_NEAR(opd[ku], opl[ku] - piston, kLengthFp64);
    }
    EXPECT_EQ(s.failed, 1);
    EXPECT_NEAR(s.valid_weight, vw, kUnitFp64);
    EXPECT_NEAR(s.piston, piston, kLengthFp64);
    EXPECT_NEAR(s.rms, std::sqrt(ss / vw), kLengthFp64);
    EXPECT_EQ(opd[2], 0.0); // maskiert

    // Ohne gültigen Strahl: Gewicht 0, Kolben und RMS 0.
    const std::vector<TraceStatus> none(5, TraceStatus::TIR);
    const auto e = mko::opd_statistics(opl.data(), w.data(), none.data(), 5, opd.data());
    EXPECT_EQ(e.failed, 5);
    EXPECT_EQ(e.valid_weight, 0.0);
    EXPECT_EQ(e.rms, 0.0);
    EXPECT_EQ(opd[0], 0.0);

    // Kippung bleibt erhalten: OPL = α y (y = ρ cos φ) hat über die halbe Pupille den Kolben
    // ⟨y⟩ = 0 und ⟨y²⟩ = 1/4 (beides exakt mit der Quadratur), also RMS = α/2. Würde die
    // Kippung entfernt, wäre der RMS 0.
    const auto q = mko::make_pupil_quadrature(6, 4, PupilMode::Half);
    const double alpha = 1e-3;
    std::vector<double> tilt, wt;
    std::vector<TraceStatus> ok;
    for (int k = 0; k < mko::sample_count(q); ++k) {
        const auto smp = mko::sample(q, k);
        tilt.push_back(alpha * smp.rho * std::cos(smp.phi));
        wt.push_back(smp.weight);
        ok.push_back(TraceStatus::Ok);
    }
    const auto t = mko::opd_statistics(tilt.data(), wt.data(), ok.data(),
                                       static_cast<int>(tilt.size()), nullptr);
    EXPECT_NEAR(t.piston / alpha, 0.0, mko::test::kQuadrature);
    EXPECT_NEAR(t.rms / alpha, 0.5, mko::test::kQuadrature);
}

// ---------------------------------------------------------------------------------------------
// Aplanatischer Fall (Abnahme M1-T7)
// ---------------------------------------------------------------------------------------------

// Punktquelle im Glas (n) im Abstand R/n vom Krümmungsmittelpunkt C einer Kugelfläche,
// Übergang Glas → Luft (Aufbau wie Trace.AplanaticPointsOfSphere). Die gebrochene Welle ist
// eine Kugelwelle um den virtuellen Bildpunkt im Abstand n·R von C; mit ihm als Zentrum der
// Referenzsphäre verschwindet die OPD. Arbeitsabstand w = z-Koordinate des Bildpunkts (< 0,
// virtuell). Apertur: sin θ = 0,95 im Glas; dazu Quellen außerhalb der Achse mit
// vorgegebenem Bildpunkt (volle Pupille).
TEST(Wavefront, AplanaticPointSourceHasVanishingRms) {
    const double Rabs = 3.0;
    const double c = -1.0 / Rabs;
    const double t0 = 6.0;
    const Vec3<double> C{0.0, 0.0, 1.0 / c}; // Koordinaten der Kugelfläche
    const Vec3<double> shift{0.0, 0.0, t0};  // Kugelfläche → Fläche 0
    double max_rms = 0.0;
    double max_opd = 0.0;
    int rays = 0;
    for (const double n : {1.5, 1.8}) {
        for (Vec3<double> u : {Vec3<double>{0.0, 0.0, -1.0}, Vec3<double>{0.1, 0.2, -1.0},
                               Vec3<double>{0.0, -0.15, -1.0}}) {
            u = (1.0 / mko::norm(u)) * u;
            const Vec3<double> src = C + (Rabs / n) * u;
            const Vec3<double> img = C + (n * Rabs) * u;
            const HostSurfaceTable table = make_table({SurfaceKind::Plane, SurfaceKind::Sphere},
                                                      {0.0, c}, {t0, img.z}, {n, 1.0});
            const SystemDescription sys = point_source_system(table, src + shift, 0.95 * n);
            WavefrontOptions opt;
            opt.rings = 12;
            opt.spokes = 12;
            std::vector<Vec3<double>> points = mko::ideal_image_points(sys, table, std::nan(""));
            ASSERT_EQ(points.size(), 1u);
            const bool on_axis = (u.x == 0.0 && u.y == 0.0);
            if (on_axis) {
                // Ideale Bildpunkt-Regel: (0, 0, w) ist der exakte Bildpunkt.
                EXPECT_EQ(points[0].x, img.x);
                EXPECT_EQ(points[0].y, img.y);
                EXPECT_EQ(points[0].z, img.z);
                opt.pupil_mode = PupilMode::Full; // windschiefe Strahlen mitprüfen
            } else {
                points[0] = img;
            }
            const WavefrontResult res = mko::evaluate_wavefront(sys, table, points, opt);
            const WavefrontMap& m = res.at(0, 0);
            ASSERT_EQ(m.failed, 0);
            ASSERT_TRUE(m.valid());
            EXPECT_LT(m.rms, kAplanaticRms) << "n=" << n << " u=(" << u.x << ", " << u.y << ")";
            max_rms = std::max(max_rms, m.rms);
            for (const double v : m.opd) {
                max_opd = std::max(max_opd, std::abs(v));
            }
            rays += m.samples();
        }
    }
    std::printf("[ Info     ] Aplanatische Punkte: max RMS-OPD = %.3e mm, max |OPD| = %.3e mm "
                "(%d Strahlen, sin θ <= 0,95)\n",
                max_rms, max_opd, rays);
}

// ---------------------------------------------------------------------------------------------
// Seidel-Näherung (Abnahme M1-T7)
// ---------------------------------------------------------------------------------------------

// Sphärische Aberration der Einzellinse (einzellinse_bk7.json, nur Feld 0) bei kleiner Apertur
// gegen die Seidel-Summe S_I (Welford): paraxialer Randstrahl (y₀ = a, u₀ = 0), pro Fläche
// A = n (y c + u), S_I = −Σ A² y Δ(u/n), W₀₄₀ = S_I / 8. Bildebene im paraxialen Fokus der
// jeweiligen Wellenlänge (aus demselben Randstrahl), sodass kein Defokus auftritt.
//
// Vorzeichen: Die OPD hier ist OPL − Kolben. Bei Unterkorrektur (S_I > 0, Randstrahlen
// schneiden die Achse vor dem paraxialen Fokus) ist die Wellenfront am Rand der Referenzsphäre
// voraus, die OPL am Rand also kürzer: OPD = −W₀₄₀ ρ⁴ + Kolben. Das Vorzeichen wird getrennt
// über den Achsschnittpunkt des reellen Randstrahls plausibilisiert.
//
// Erwartung nach Kolbenabzug: RMS = |W₀₄₀| √(⟨ρ⁸⟩ − ⟨ρ⁴⟩²) = |W₀₄₀| √(1/5 − 1/9) = |W₀₄₀| √(4/45).
// Zusätzlich Fit OPD = A ρ⁴ + B ρ⁶ + K über die Stützstellen: A ist der Seidel-Anteil ohne den
// Beitrag höherer Ordnung.
TEST(Wavefront, SeidelSphericalAberrationOfSingleLens) {
    mko::LoadedSystem ls = mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    ls.system.source.field_angles_deg = {0.0};
    ls.system.source.field_weights = {1.0};
    const int S = ls.table.surfaces;
    const int L = ls.table.wavelengths;
    for (const double a : {5.0, 2.5}) {
        ls.system.surfaces[0].semi_diameter = a;
        ls.table.semi_diameter[0] = a;
        for (int l = 0; l < L; ++l) {
            // Unabhängiger paraxialer Randstrahl und Seidel-Summe
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
            const double bfd = -y / u;
            const double w040 = s1 / 8.0;
            const double na_image = std::sin(std::atan(-u)); // NA im Bildraum (Luft)
            ASSERT_LE(na_image, 0.05);

            ls.table.thickness[static_cast<std::size_t>(S - 1)] = bfd;
            ls.system.surfaces.back().thickness = bfd;
            const std::vector<Vec3<double>> points{{0.0, 0.0, bfd}};
            const WavefrontResult res = mko::evaluate_wavefront(ls.system, ls.table, points);
            const WavefrontMap& m = res.at(0, l);
            ASSERT_EQ(m.quadrature.mode, PupilMode::Axis);
            ASSERT_EQ(m.failed, 0);

            // RMS gegen Seidel
            const double rms_seidel = std::abs(w040) * std::sqrt(4.0 / 45.0);
            const double rel_rms = std::abs(m.rms - rms_seidel) / rms_seidel;
            EXPECT_LT(rel_rms, kSeidelRelative) << "a=" << a << " l=" << l;

            // Gewichteter Fit OPD = A ρ⁴ + B ρ⁶ + K (Normalgleichungen 3×3, Cramer)
            double G[3][3] = {};
            double h[3] = {};
            for (int k = 0; k < m.samples(); ++k) {
                const auto& smp = m.pupil[static_cast<std::size_t>(k)];
                const double r2 = smp.rho * smp.rho;
                const double phi[3] = {r2 * r2, r2 * r2 * r2, 1.0};
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        G[i][j] += smp.weight * phi[i] * phi[j];
                    }
                    h[i] += smp.weight * phi[i] * m.opd[static_cast<std::size_t>(k)];
                }
            }
            auto det3 = [](const double M3[3][3]) {
                return M3[0][0] * (M3[1][1] * M3[2][2] - M3[1][2] * M3[2][1]) -
                       M3[0][1] * (M3[1][0] * M3[2][2] - M3[1][2] * M3[2][0]) +
                       M3[0][2] * (M3[1][0] * M3[2][1] - M3[1][1] * M3[2][0]);
            };
            double GA[3][3];
            for (int i = 0; i < 3; ++i) {
                GA[i][0] = h[i];
                GA[i][1] = G[i][1];
                GA[i][2] = G[i][2];
            }
            const double A_fit = det3(GA) / det3(G);
            const double rel_fit = std::abs(A_fit + w040) / std::abs(w040);
            EXPECT_LT(rel_fit, kSeidelRelative) << "a=" << a << " l=" << l;

            // Vorzeichen: unterkorrigiert ⇔ reeller Randstrahl schneidet die Achse vor dem
            // paraxialen Fokus ⇔ S_I > 0 ⇔ OPD am Rand (ρ = 1) < OPD auf der Achse.
            Ray<double> marginal =
                mko::source_ray(ls.system, ls.table, 0, l, mko::PupilSample<double>{1.0, 0.0, 0.0});
            ASSERT_EQ(mko::trace_ray(ls.table.view(), l, marginal, nullptr), TraceStatus::Ok);
            const double crossing = mko::meridional_axis_crossing(marginal);
            EXPECT_GT(s1, 0.0);
            EXPECT_LT(crossing, bfd);
            EXPECT_LT(A_fit, 0.0);
            EXPECT_LT(m.opd.back(), m.opd.front());

            std::printf("[ Info     ] Seidel a = %.2f mm, λ = %.2f nm, NA = %.4f: W040 = %.6e mm, "
                        "RMS = %.6e mm (Seidel %.6e, rel. %.2e), Fit ρ⁴ = %.6e (rel. %.2e)\n",
                        a, m.wavelength_nm, na_image, w040, m.rms, rms_seidel, rel_rms, A_fit,
                        rel_fit);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Regeln der Host-API
// ---------------------------------------------------------------------------------------------

// f_soll: target.focal_length, sonst paraxiale Brennweite bei reference_nm; M_f = (0, f tanθ, w).
TEST(Wavefront, FocalLengthRuleAndIdealImagePoints) {
    mko::LoadedSystem ls = mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    const double f_par = mko::compute_paraxial(ls.system, ls.catalog).focal_length;
    EXPECT_EQ(mko::target_focal_length(ls.system, ls.catalog), f_par);
    EXPECT_NEAR(f_par, 101.05, mko::test::kDocumentedValue);

    const auto pts = mko::ideal_image_points(ls.system, ls.table, f_par);
    ASSERT_EQ(pts.size(), 3u);
    for (std::size_t f = 0; f < pts.size(); ++f) {
        EXPECT_EQ(pts[f].x, 0.0);
        EXPECT_EQ(pts[f].y, f_par * std::tan(ls.system.source.field_angles_deg[f] * kDeg));
        EXPECT_EQ(pts[f].z, 99.05);
    }
    const WavefrontResult res = mko::evaluate_wavefront(ls);
    EXPECT_EQ(res.focal_length, f_par);
    EXPECT_EQ(res.at(2, 1).image_point.y, pts[2].y);

    ls.system.target = mko::Target{};
    ls.system.target->focal_length = 3.29;
    EXPECT_EQ(mko::target_focal_length(ls.system, ls.catalog), 3.29);
    EXPECT_EQ(mko::evaluate_wavefront(ls).at(1, 0).image_point.y, 3.29 * std::tan(1.0 * kDeg));

    // Afokales System (Planplatte) ohne Sollbrennweite: Fehler.
    const mko::LoadedSystem plate =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/planplatte_bk7.json");
    EXPECT_THROW(mko::target_focal_length(plate.system, plate.catalog), std::invalid_argument);
    EXPECT_THROW(mko::evaluate_wavefront(plate), std::invalid_argument);
}

// Quadratur: Feld θ = 0 → Achsfächer (N Stützstellen), sonst halbe Pupille (N·M); OPD in Wellen
// bei λ_ref; Ablage maps[f·L + l]; zwei Läufe bitgleich.
TEST(Wavefront, PupilModeRuleUnitsAndLayout) {
    const mko::LoadedSystem ls =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    const WavefrontResult res = mko::evaluate_wavefront(ls);
    ASSERT_EQ(res.fields, 3);
    ASSERT_EQ(res.wavelengths, 3);
    ASSERT_EQ(res.maps.size(), 9u);
    EXPECT_EQ(res.reference_nm, 587.56);
    EXPECT_EQ(res.failed_count(), 0u);
    const double lambda_mm = res.reference_nm * 1e-6; // wie in der Implementierung
    for (int f = 0; f < res.fields; ++f) {
        for (int l = 0; l < res.wavelengths; ++l) {
            const WavefrontMap& m = res.at(f, l);
            EXPECT_EQ(m.field, f);
            EXPECT_EQ(m.wavelength, l);
            EXPECT_EQ(m.quadrature.mode, f == 0 ? PupilMode::Axis : PupilMode::Half);
            EXPECT_EQ(m.samples(), f == 0 ? 10 : 80);
            EXPECT_TRUE(m.valid());
            EXPECT_NEAR(m.valid_weight, 1.0, kUnitFp64);
            EXPECT_EQ(m.rms_waves, m.rms / lambda_mm);
            double sum_w = 0.0;
            double mean = 0.0;
            for (int k = 0; k < m.samples(); ++k) {
                const auto ku = static_cast<std::size_t>(k);
                EXPECT_EQ(m.status[ku], TraceStatus::Ok);
                EXPECT_EQ(m.failed_surface[ku], -1);
                EXPECT_EQ(m.opd_waves[ku], m.opd[ku] / lambda_mm);
                sum_w += m.pupil[ku].weight;
                mean += m.pupil[ku].weight * m.opd[ku];
            }
            EXPECT_NEAR(sum_w, 1.0, kUnitFp64);
            EXPECT_NEAR(mean, 0.0, kLengthFp64); // Kolben entfernt
        }
    }
    WavefrontOptions opt;
    opt.rings = 6;
    opt.spokes = 3;
    const WavefrontResult small = mko::evaluate_wavefront(ls, opt);
    EXPECT_EQ(small.at(0, 0).samples(), 6);
    EXPECT_EQ(small.at(1, 0).samples(), 18);

    const WavefrontResult again = mko::evaluate_wavefront(ls);
    for (std::size_t i = 0; i < res.maps.size(); ++i) {
        EXPECT_EQ(res.maps[i].rms, again.maps[i].rms);
        EXPECT_EQ(res.maps[i].opd, again.maps[i].opd);
    }

    // Punktquelle: Achse → Axis, x = 0 → Half, sonst Full
    SystemDescription p;
    p.source.type = mko::SourceType::Point;
    EXPECT_EQ(mko::wavefront_pupil_mode(p, 0), PupilMode::Axis);
    p.source.point = {0.0, 0.3, 0.0};
    EXPECT_EQ(mko::wavefront_pupil_mode(p, 0), PupilMode::Half);
    p.source.point = {0.1, 0.0, 0.0};
    EXPECT_EQ(mko::wavefront_pupil_mode(p, 0), PupilMode::Full);
    EXPECT_THROW(mko::wavefront_pupil_mode(p, 1), std::out_of_range);
}

// Hartes Versagen: Strahlen, die die Linse verfehlen, fallen aus Kolben und RMS heraus
// (Gewichte renormiert), ihre Werte sind NaN, die versagende Fläche wird gemeldet.
TEST(Wavefront, FailedRaysAreExcludedAndReported) {
    // Stark gekrümmte Fläche (R = 6 mm), Blendenradius 8 mm: äußere Ringe verfehlen.
    const mko::LoadedSystem ls = load_test_system(
        R"({"format": "mikroptim.system/1", "name": "Versagen", "units": "mm",
            "glass_catalogs": [],
            "spectrum": {"wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56},
            "source": {"type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0]},
            "target": {"focal_length": 18.0},
            "surfaces": [
              {"type": "stop", "semi_diameter": 8.0, "thickness": 2.0, "material": "AIR"},
              {"type": "sphere", "curvature": 0.16666666666666666, "thickness": 18.0,
               "material": {"model": "constant", "n": 1.5}, "semi_diameter": 9.0}]})");
    const WavefrontResult res = mko::evaluate_wavefront(ls);
    const WavefrontMap& m = res.at(0, 0);
    ASSERT_GT(m.failed, 0);
    ASSERT_LT(m.failed, m.samples());
    EXPECT_EQ(res.failed_count(), static_cast<std::size_t>(m.failed));
    EXPECT_TRUE(m.valid());
    EXPECT_LT(m.valid_weight, 1.0);
    EXPECT_TRUE(std::isfinite(m.rms));
    double vw = 0.0;
    double mean = 0.0;
    for (int k = 0; k < m.samples(); ++k) {
        const auto ku = static_cast<std::size_t>(k);
        const double h = 8.0 * m.pupil[ku].rho;
        if (h > 6.0) {
            EXPECT_EQ(m.status[ku], TraceStatus::Missed);
            EXPECT_EQ(m.failed_surface[ku], 1);
            EXPECT_TRUE(std::isnan(m.opd[ku]));
            EXPECT_TRUE(std::isnan(m.opd_waves[ku]));
        } else if (m.status[ku] == TraceStatus::Ok) {
            vw += m.pupil[ku].weight;
            mean += m.pupil[ku].weight * m.opd[ku];
        }
    }
    EXPECT_NEAR(vw, m.valid_weight, kUnitFp64);
    EXPECT_NEAR(mean / vw, 0.0, kLengthFp64);
}
