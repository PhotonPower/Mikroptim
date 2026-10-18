// Sequenzieller Trace (M1-T5): analytische Orakel für trace_ray und den CPU-Tracer.

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/trace.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <numbers>
#include <string>
#include <vector>

using mko::HostSurfaceTable;
using mko::PupilMode;
using mko::Ray;
using mko::SurfaceKind;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::kLengthFp64;
using mko::test::kRayPointFp64;
using mko::test::kRayReversalFp64;
using mko::test::kUnitFp64;

namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

Vec3<double> cross(Vec3<double> a, Vec3<double> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Tabelle aus Spalten; index zeilenweise [s * L + l].
HostSurfaceTable make_table(std::vector<SurfaceKind> type, std::vector<double> curvature,
                            std::vector<double> thickness, std::vector<double> index,
                            int wavelengths) {
    HostSurfaceTable t;
    t.surfaces = static_cast<int>(type.size());
    t.wavelengths = wavelengths;
    t.type = std::move(type);
    t.curvature = std::move(curvature);
    t.thickness = std::move(thickness);
    t.index = std::move(index);
    t.semi_diameter.assign(t.type.size(), 10.0);
    return t;
}

// Gespiegeltes System für den Rückwärtstrace: Fläche 0 ist die Bildebene, danach die Flächen
// S−1 … 0 mit negierter Krümmung; Medien und Dicken entsprechend verschoben. Koordinaten
// werden durch die Drehung um die y-Achse (x, y, z) → (−x, y, −z) abgebildet.
HostSurfaceTable reversed_table(const HostSurfaceTable& f) {
    const int S = f.surfaces;
    const int L = f.wavelengths;
    HostSurfaceTable r;
    r.surfaces = S + 1;
    r.wavelengths = L;
    auto push_index_row = [&](int s) {
        for (int l = 0; l < L; ++l) {
            r.index.push_back(s < 0 ? 1.0 : f.index[static_cast<std::size_t>(s * L + l)]);
        }
    };
    r.type.push_back(SurfaceKind::Plane);
    r.curvature.push_back(0.0);
    r.thickness.push_back(f.thickness[static_cast<std::size_t>(S - 1)]);
    push_index_row(S - 1);
    for (int j = S - 1; j >= 0; --j) {
        const auto ju = static_cast<std::size_t>(j);
        r.type.push_back(f.type[ju] == SurfaceKind::Sphere ? SurfaceKind::Sphere
                                                           : SurfaceKind::Plane);
        r.curvature.push_back(-f.curvature[ju]);
        r.thickness.push_back(j >= 1 ? f.thickness[ju - 1] : 0.0);
        push_index_row(j - 1);
    }
    r.semi_diameter.assign(r.type.size(), 10.0);
    return r;
}

Vec3<double> rotate_y(Vec3<double> v) {
    return {-v.x, v.y, -v.z};
}

double max_abs(Vec3<double> a, Vec3<double> b) {
    return std::max({std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.z - b.z)});
}

std::string single_surface_json(double curvature, double n) {
    return R"({"format": "mikroptim.system/1", "name": "Einzelfläche", "units": "mm",
      "glass_catalogs": [],
      "spectrum": {"wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56},
      "source": {"type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0]},
      "surfaces": [
        {"type": "stop", "semi_diameter": 8.0, "thickness": 3.0, "material": "AIR"},
        {"type": "sphere", "curvature": )" +
           std::to_string(curvature) + R"(, "thickness": 20.0,
         "material": {"model": "constant", "n": )" +
           std::to_string(n) + R"(}, "semi_diameter": 9.0}]})";
}

// Frei erfundenes Testsystem aus vier Linsen (ein Kittglied) mit drei Katalog-Gläsern und
// einem konstanten Glas; keine Daten eines realen Objektivs.
constexpr const char* kTripletJson = R"({
  "format": "mikroptim.system/1", "name": "Testsystem Strahlumkehr", "units": "mm",
  "glass_catalogs": ["data/glass/basis.json"],
  "spectrum": {"wavelengths_nm": [486.13, 587.56, 656.27], "weights": [1, 1, 1],
               "reference_nm": 587.56},
  "source": {"type": "collimated", "field_angles_deg": [0.0, 3.0, -5.0],
             "field_weights": [1, 1, 1]},
  "surfaces": [
    {"type": "stop", "semi_diameter": 4.0, "thickness": 5.0, "material": "AIR"},
    {"type": "sphere", "curvature": 0.04, "thickness": 3.5, "material": "CAF2", "semi_diameter": 7},
    {"type": "sphere", "curvature": -0.05, "thickness": 1.2, "material": "N-BK7", "semi_diameter": 7},
    {"type": "sphere", "curvature": -0.01, "thickness": 2.0, "material": "AIR", "semi_diameter": 7},
    {"type": "sphere", "curvature": -0.06, "thickness": 1.5, "material": "F_SILICA", "semi_diameter": 7},
    {"type": "plane", "thickness": 1.0, "material": "AIR", "semi_diameter": 7},
    {"type": "sphere", "curvature": 0.07, "thickness": 2.5,
     "material": {"model": "constant", "n": 1.72}, "semi_diameter": 7},
    {"type": "sphere", "curvature": -0.03, "thickness": 20.0, "material": "AIR", "semi_diameter": 7}
  ]
})";

mko::LoadedSystem load_test_system(const std::string& json) {
    return mko::prepare_system(mko::read_system_json(json), MKO_SOURCE_DIR);
}

} // namespace

// Planplatte: OPL von der Startwellenfront bis zur Bildebene in geschlossener Form,
// OPL = p₀·d₀ + (t₀ + t₂)/cos θ + n t₁ / cos θ_g mit sin θ_g = sin θ / n.
TEST(Trace, PlaneParallelPlateMatchesClosedForm) {
    const mko::LoadedSystem ls =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/planplatte_bk7.json");
    const auto q = mko::make_pupil_quadrature(8, 12, PupilMode::Full);
    const mko::CpuTraceResult res = mko::trace_system_cpu(ls.system, ls.table, q);
    ASSERT_EQ(res.fields, 2);
    ASSERT_EQ(res.failed_count(), 0u);

    const auto view = ls.table.view();
    const double a = ls.system.surfaces[0].semi_diameter;
    const double t0 = ls.system.surfaces[0].thickness;
    const double t1 = ls.system.surfaces[1].thickness;
    const double t2 = ls.system.surfaces[2].thickness;
    double max_opl = 0.0;
    double max_pos = 0.0;
    for (int f = 0; f < res.fields; ++f) {
        const double th = ls.system.source.field_angles_deg[static_cast<std::size_t>(f)] * kDeg;
        for (int l = 0; l < res.wavelengths; ++l) {
            const double n = ls.index.at(1, l);
            const double sin_g = std::sin(th) / n;
            const double cos_g = std::sqrt(1.0 - sin_g * sin_g);
            for (int k = 0; k < res.samples; ++k) {
                const std::size_t i = res.at(f, l, k);
                ASSERT_EQ(res.failed_surface[i], -1);
                Ray<double> ray = res.rays[i];
                ASSERT_EQ(mko::transfer_to_image_plane(view, l, ray), TraceStatus::Ok);
                const auto& s = res.pupil[static_cast<std::size_t>(k)];
                const double x0 = a * s.rho * std::sin(s.phi);
                const double y0 = a * s.rho * std::cos(s.phi);
                const double opl = y0 * std::sin(th) + (t0 + t2) / std::cos(th) + n * t1 / cos_g;
                const double y = y0 + (t0 + t2) * std::tan(th) + t1 * sin_g / cos_g;
                max_opl = std::max(max_opl, std::abs(ray.opl - opl));
                max_pos = std::max({max_pos, std::abs(ray.p.y - y), std::abs(ray.p.x - x0)});
                EXPECT_NEAR(ray.opl, opl, kLengthFp64) << "f=" << f << " k=" << k;
                EXPECT_NEAR(ray.p.y, y, kLengthFp64);
                EXPECT_NEAR(ray.p.x, x0, kLengthFp64);
                EXPECT_EQ(ray.p.z, 0.0);
            }
        }
    }
    std::printf("[ Info     ] Planplatte: max |OPL - Formel| = %.3e mm, max Ortsfehler = %.3e mm "
                "(%d Strahlen)\n",
                max_opl, max_pos, res.fields * res.wavelengths * res.samples);
}

// Einzelfläche über das ganze System-API (JSON → Lader → CPU-Tracer) gegen den exakten
// trigonometrischen Strahl (Kingslake): sin I = h/R, sin I' = (n/n') sin I, U' = I − I',
// L' = R + R sin I' / sin U'; dazu OPL bis zum Schnittpunkt = t₀ + Pfeilhöhe.
TEST(Trace, SingleSurfaceThroughSystemApiMatchesTrigonometricRay) {
    for (const double R : {10.0, -10.0, 25.0}) {
        const double n2 = 1.5;
        const mko::LoadedSystem ls = load_test_system(single_surface_json(1.0 / R, n2));
        const auto q = mko::make_pupil_quadrature(12, 1, PupilMode::Axis);
        const mko::CpuTraceResult res = mko::trace_system_cpu(ls.system, ls.table, q);
        ASSERT_EQ(res.failed_count(), 0u);
        const double a = ls.system.surfaces[0].semi_diameter;
        const double c = ls.table.curvature[1]; // gerundete Krümmung wie im Tracer
        const double Rc = 1.0 / c;
        double max_err = 0.0;
        for (int k = 0; k < res.samples; ++k) {
            const Ray<double>& ray = res.rays[res.at(0, 0, k)];
            const double h = a * res.pupil[static_cast<std::size_t>(k)].rho;
            const double sinI = h / Rc;
            const double sinIp = sinI / n2;
            const double U = std::asin(sinI) - std::asin(sinIp);
            const double Lp = Rc + Rc * sinIp / std::sin(U);
            const double sag = Rc - std::copysign(std::sqrt(Rc * Rc - h * h), Rc);
            EXPECT_EQ(ray.p.x, 0.0);
            EXPECT_NEAR(ray.p.y, h, kLengthFp64);
            EXPECT_NEAR(ray.p.z, sag, kLengthFp64);
            EXPECT_NEAR(ray.opl, 3.0 + sag, kLengthFp64);
            EXPECT_NEAR(mko::meridional_axis_crossing(ray), Lp, kLengthFp64)
                << "R = " << R << ", h = " << h;
            EXPECT_NEAR(mko::norm(ray.d), 1.0, kUnitFp64);
            max_err = std::max(max_err, std::abs(mko::meridional_axis_crossing(ray) - Lp));
        }
        std::printf("[ Info     ] Einzelfläche R = %g: max |L' - Kingslake| = %.3e mm\n", R,
                    max_err);
    }
}

// Aplanatische Punkte: Punktquelle im Glas (n) im Abstand R/n vom Krümmungsmittelpunkt C einer
// Kugelfläche, Übergang Glas → Luft. Alle gebrochenen Strahlen zeigen rückwärts verlängert auf
// den virtuellen Bildpunkt im Abstand n·R von C (gleiche Seite). Apertur: halber
// Öffnungswinkel θ der Quelle im Glas bis sin θ = 0,95; das schließt NA = n sin θ <= 0,95 ein.
// Zusätzlich ist die OPL vom Quellpunkt bis zum Schnittpunkt gleich dem Abstand zum virtuellen
// Punkt (die gebrochene Welle ist eine Kugelwelle um ihn).
TEST(Trace, AplanaticPointsOfSphere) {
    const double Rabs = 3.0;
    const double c = -1.0 / Rabs; // Mittelpunkt auf der −z-Seite, im Glas
    const double t0 = 6.0;        // Scheitelabstand Fläche 0 (Ebene im Glas) → Kugelfläche
    const std::vector<double> n_glass{1.5, 1.8};
    const int L = static_cast<int>(n_glass.size());
    // Startfläche 0: Ebene, deren Medium Glas ist (Startmedium der Punktquelle).
    const HostSurfaceTable table = make_table({SurfaceKind::Plane, SurfaceKind::Sphere}, {0.0, c},
                                              {t0, 10.0}, {n_glass[0], n_glass[1], 1.0, 1.0}, L);
    const auto view = table.view();
    const Vec3<double> C{0.0, 0.0, 1.0 / c}; // Mittelpunkt in Koordinaten der Kugelfläche
    const Vec3<double> shift{0.0, 0.0, t0};  // Kugelfläche → Fläche 0

    // Quellrichtungen u vom Mittelpunkt aus: auf der Achse und schräg (Windschiefe Strahlen).
    const std::vector<Vec3<double>> dirs{{0.0, 0.0, -1.0}, {0.1, 0.2, -1.0}, {-0.15, 0.0, -1.0}};
    const auto q = mko::make_pupil_quadrature(10, 12, PupilMode::Full);
    std::vector<mko::PupilSample<double>> samples;
    for (int k = 0; k < mko::sample_count(q); ++k) {
        samples.push_back(mko::sample(q, k));
    }
    for (int j = 0; j < 12; ++j) { // Rand der Apertur: sin θ = 0,95 exakt
        samples.push_back({1.0, j * std::numbers::pi / 6.0, 0.0});
    }
    const double sin_max = 0.95;

    double max_dist = 0.0;
    double max_opl = 0.0;
    int traced = 0;
    for (int l = 0; l < L; ++l) {
        const double n = n_glass[static_cast<std::size_t>(l)];
        for (Vec3<double> u : dirs) {
            u = (1.0 / mko::norm(u)) * u;
            const Vec3<double> src = C + (Rabs / n) * u; // Koordinaten der Kugelfläche
            const Vec3<double> img = C + (n * Rabs) * u;
            for (const auto& s : samples) {
                Ray<double> ray = mko::point_source_ray(src + shift, sin_max, s.rho, s.phi);
                int failed = 0;
                ASSERT_EQ(mko::trace_ray(view, l, ray, nullptr, &failed), TraceStatus::Ok);
                ASSERT_EQ(failed, -1);
                const Vec3<double> w = img - ray.p;
                const double dist = mko::norm(cross(w, ray.d)) / mko::norm(ray.d);
                EXPECT_LT(dist, kRayPointFp64) << "n=" << n << " rho=" << s.rho;
                EXPECT_LT(mko::dot(w, ray.d), 0.0); // virtueller Punkt liegt rückwärts
                const double opl_err = std::abs(ray.opl - mko::norm(w));
                EXPECT_LT(opl_err, kLengthFp64);
                max_dist = std::max(max_dist, dist);
                max_opl = std::max(max_opl, opl_err);
                ++traced;
            }
        }
    }
    std::printf("[ Info     ] Aplanatische Punkte: max Abstand Gerade-Punkt = %.3e mm, "
                "max |OPL - |PQ|| = %.3e mm (%d Strahlen, sin θ <= 0,95)\n",
                max_dist, max_opl, traced);
}

// Strahlumkehr: vorwärts bis zur Bildebene, im gespiegelten System zurück bis zur Blende.
TEST(Trace, RayReversalThroughMultiLensSystem) {
    const mko::LoadedSystem ls = load_test_system(kTripletJson);
    const HostSurfaceTable rev = reversed_table(ls.table);
    const auto fwd_view = ls.table.view();
    const auto rev_view = rev.view();
    const auto q = mko::make_pupil_quadrature(6, 8, PupilMode::Full);

    double max_pos = 0.0;
    double max_opl = 0.0;
    double max_dir = 0.0;
    for (int f = 0; f < mko::source_field_count(ls.system); ++f) {
        for (int l = 0; l < ls.table.wavelengths; ++l) {
            for (int k = 0; k < mko::sample_count(q); ++k) {
                const Ray<double> start =
                    mko::source_ray(ls.system, ls.table, f, l, mko::sample(q, k));
                Ray<double> ray = start;
                ASSERT_EQ(mko::trace_ray(fwd_view, l, ray, nullptr), TraceStatus::Ok);
                ASSERT_EQ(mko::transfer_to_image_plane(fwd_view, l, ray), TraceStatus::Ok);
                const double opl_fwd = ray.opl - start.opl;

                Ray<double> back{rotate_y(ray.p), -1.0 * rotate_y(ray.d), 0.0};
                back.p.z = 0.0;
                int failed = 0;
                ASSERT_EQ(mko::trace_ray(rev_view, l, back, nullptr, &failed), TraceStatus::Ok);
                const Vec3<double> p_end = rotate_y(back.p);
                const Vec3<double> d_end = -1.0 * rotate_y(back.d);
                max_pos = std::max(max_pos, max_abs(p_end, start.p));
                max_dir = std::max(max_dir, max_abs(d_end, start.d));
                max_opl = std::max(max_opl, std::abs(back.opl - opl_fwd));
                EXPECT_LT(max_abs(p_end, start.p), kRayReversalFp64);
                EXPECT_LT(max_abs(d_end, start.d), kUnitFp64);
                EXPECT_LT(std::abs(back.opl - opl_fwd), kRayReversalFp64);
            }
        }
    }
    std::printf("[ Info     ] Strahlumkehr: max Ortsfehler = %.3e mm, max Richtungsfehler = "
                "%.3e, max OPL-Fehler = %.3e mm\n",
                max_pos, max_dir, max_opl);
}

// Aufzeichnung: Einträge liegen auf den Flächen, letzter Eintrag ist der Endzustand.
TEST(Trace, RecordStoresStateAfterEachSurface) {
    const mko::LoadedSystem ls = load_test_system(kTripletJson);
    const auto view = ls.table.view();
    const auto q = mko::make_pupil_quadrature(4, 4, PupilMode::Half);
    for (int k = 0; k < mko::sample_count(q); ++k) {
        const Ray<double> start = mko::source_ray(ls.system, ls.table, 1, 0, mko::sample(q, k));
        Ray<double> ray = start;
        mko::TraceRecord<double> rec{};
        ASSERT_EQ(mko::trace_ray(view, 0, ray, &rec), TraceStatus::Ok);
        ASSERT_EQ(rec.count, ls.table.surfaces);
        EXPECT_EQ(max_abs(rec.p[0], start.p), 0.0);
        EXPECT_EQ(rec.opl[0], start.opl);
        EXPECT_EQ(max_abs(rec.p[rec.count - 1], ray.p), 0.0);
        EXPECT_EQ(max_abs(rec.d[rec.count - 1], ray.d), 0.0);
        EXPECT_EQ(rec.opl[rec.count - 1], ray.opl);
        for (int s = 1; s < rec.count; ++s) {
            const double cs = ls.table.curvature[static_cast<std::size_t>(s)];
            const Vec3<double> p = rec.p[s];
            // Sphärengleichung c (x² + y² + z²) − 2 z = 0
            EXPECT_NEAR(cs * mko::dot(p, p) - 2.0 * p.z, 0.0, kLengthFp64);
            EXPECT_GT(rec.opl[s], rec.opl[s - 1]);
            EXPECT_NEAR(mko::norm(rec.d[s]), 1.0, kUnitFp64);
        }
    }
}

TEST(Trace, StatusMissedAtFirstSurface) {
    const HostSurfaceTable t =
        make_table({SurfaceKind::Stop, SurfaceKind::Sphere}, {0.0, 0.2}, {1.0, 5.0}, {1.0, 1.5}, 1);
    Ray<double> ray = mko::collimated_ray(6.0, 0.0, 1.0, 0.0); // Höhe 6 > R = 5
    mko::TraceRecord<double> rec{};
    int failed = 0;
    EXPECT_EQ(mko::trace_ray(t.view(), 0, ray, &rec, &failed), TraceStatus::Missed);
    EXPECT_EQ(failed, 1);
    EXPECT_EQ(rec.count, 1);
}

TEST(Trace, StatusTotalInternalReflection) {
    // Punktquelle im Glas (Startfläche 0 mit Glas), Planfläche Glas → Luft, θ = 50° > 41,8°.
    const HostSurfaceTable t =
        make_table({SurfaceKind::Plane, SurfaceKind::Plane}, {0.0, 0.0}, {1.0, 5.0}, {1.5, 1.0}, 1);
    Ray<double> ray =
        mko::point_source_ray(Vec3<double>{0.0, 0.0, 0.0}, std::sin(50.0 * kDeg), 1.0, 0.0);
    int failed = 0;
    EXPECT_EQ(mko::trace_ray(t.view(), 0, ray, nullptr, &failed), TraceStatus::TIR);
    EXPECT_EQ(failed, 1);
}

TEST(Trace, StatusBackward) {
    const HostSurfaceTable t =
        make_table({SurfaceKind::Stop, SurfaceKind::Plane}, {0.0, 0.0}, {1.0, 5.0}, {1.0, 1.5}, 1);
    for (const Vec3<double> d : {Vec3<double>{0.0, 1.0, 0.0}, Vec3<double>{0.0, 0.6, -0.8},
                                 Vec3<double>{0.0, 0.0, std::nan("")}}) {
        Ray<double> ray{{0.0, 0.0, 0.0}, d, 0.0};
        int failed = 0;
        EXPECT_EQ(mko::trace_ray(t.view(), 0, ray, nullptr, &failed), TraceStatus::Backward);
        EXPECT_EQ(failed, 1);
    }
    // Ende des Trace: rückwärts laufender Strahl wird nicht zur Bildebene transportiert.
    Ray<double> ray{{0.0, 0.0, 0.0}, {0.0, 0.6, -0.8}, 0.0};
    EXPECT_EQ(mko::transfer_to_image_plane(t.view(), 0, ray), TraceStatus::Backward);
}

TEST(Trace, StatusUnsupportedSurfaceKind) {
    const HostSurfaceTable t = make_table({SurfaceKind::Stop, SurfaceKind::GhostPlate}, {0.0, 0.0},
                                          {1.0, 5.0}, {1.0, 1.0}, 1);
    Ray<double> ray = mko::collimated_ray(1.0, 0.0, 0.5, 0.0);
    int failed = 0;
    EXPECT_EQ(mko::trace_ray(t.view(), 0, ray, nullptr, &failed), TraceStatus::Unsupported);
    EXPECT_EQ(failed, 1);
}

// CPU-Tracer: Strahlen, die eine spätere Fläche verfehlen, werden mit Fläche und Status
// gemeldet; die übrigen laufen durch.
TEST(Trace, CpuTracerReportsFailingSurface) {
    const std::string json = R"({"format": "mikroptim.system/1", "name": "Versagen",
      "units": "mm", "glass_catalogs": [],
      "spectrum": {"wavelengths_nm": [550.0], "weights": [1.0], "reference_nm": 550.0},
      "source": {"type": "collimated", "field_angles_deg": [0.0, 4.0], "field_weights": [1, 1]},
      "surfaces": [
        {"type": "stop", "semi_diameter": 8.0, "thickness": 1.0, "material": "AIR"},
        {"type": "sphere", "curvature": 0.01, "thickness": 2.0,
         "material": {"model": "constant", "n": 1.5}, "semi_diameter": 9.0},
        {"type": "sphere", "curvature": -0.2, "thickness": 10.0, "material": "AIR",
         "semi_diameter": 5.0}]})";
    const mko::LoadedSystem ls = load_test_system(json);
    const auto q = mko::make_pupil_quadrature(8, 6, PupilMode::Half);
    const mko::CpuTraceResult res = mko::trace_system_cpu(ls.system, ls.table, q);
    ASSERT_EQ(res.rays.size(), static_cast<std::size_t>(2 * 1 * 48));
    std::size_t ok = 0;
    std::size_t failed = 0;
    std::size_t missed = 0;
    for (std::size_t i = 0; i < res.rays.size(); ++i) {
        if (res.status[i] == TraceStatus::Ok) {
            EXPECT_EQ(res.failed_surface[i], -1);
            ++ok;
        } else {
            // knapp unter dem Rand der Kugel (R = 5 mm) Totalreflexion, darüber verfehlt
            EXPECT_TRUE(res.status[i] == TraceStatus::Missed || res.status[i] == TraceStatus::TIR)
                << mko::trace_status_name(res.status[i]);
            EXPECT_EQ(res.failed_surface[i], 2);
            ++failed;
            missed += res.status[i] == TraceStatus::Missed ? 1 : 0;
        }
    }
    EXPECT_GT(ok, 0u);
    EXPECT_GT(missed, 0u);
    EXPECT_EQ(res.failed_count(), failed);
    // Innerster Ring (ρ klein) kommt auf der Achse immer durch.
    EXPECT_EQ(res.status[res.at(0, 0, 0)], TraceStatus::Ok);
}

TEST(Trace, CollimatedStartRay) {
    const double a = 3.125;
    const double th = 7.0 * kDeg;
    const Ray<double> r = mko::collimated_ray(a, th, 0.6, 0.4);
    EXPECT_NEAR(r.p.x, a * 0.6 * std::sin(0.4), kLengthFp64);
    EXPECT_NEAR(r.p.y, a * 0.6 * std::cos(0.4), kLengthFp64);
    EXPECT_EQ(r.p.z, 0.0);
    EXPECT_NEAR(r.d.y, std::sin(th), kUnitFp64);
    EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
    EXPECT_NEAR(r.opl, r.p.y * std::sin(th), kLengthFp64); // OPL₀ = p₀·d₀
    const Ray<double> axis = mko::collimated_ray(a, 0.0, 1.0, 0.0);
    EXPECT_EQ(axis.opl, 0.0);
    EXPECT_EQ(axis.p.x, 0.0);
    EXPECT_EQ(axis.p.y, a);
}

TEST(Trace, PointSourceStartRay) {
    const Ray<double> r = mko::point_source_ray(Vec3<double>{0.5, -1.0, -2.0}, 0.8, 1.0, 0.3);
    EXPECT_EQ(r.opl, 0.0);
    EXPECT_EQ(r.p.z, -2.0);
    EXPECT_NEAR(std::hypot(r.d.x, r.d.y), 0.8, kUnitFp64);
    EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
    EXPECT_GT(r.d.z, 0.0);
}
