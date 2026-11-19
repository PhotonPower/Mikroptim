// Referenzsphäre unabhängig von der letzten Fläche (M6-T7, ADR-0017; docs/numerik.md,
// Abschnitt 7): Schritt auf die Sphäre um den Bildpunkt gegen den Schritt von EVAL, analytische
// Orakel (reine Defokussierung einer Punktquelle, ideal fokussierende Geisterplatte mit
// Hilfsebene im Brennpunkt = Falle F12), Unabhängigkeit von einer Hilfsebene hinter dem System,
// κ aus der paraxialen Austrittspupille, Merit gegen evaluate_wavefront, weiches Versagen,
// Dateiformat und Kompatibilität (Dateien ohne Feld rechnen wie bisher).

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/eval_ray.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using mko::HostSurfaceTable;
using mko::ReferenceSphere;
using mko::ReferenceSphereKind;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::SurfaceKind;
using mko::SystemBatchHost;
using mko::SystemDescription;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::kReferenceSphereFp64;

namespace {

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

Vec3<double> unit(Vec3<double> v) {
    const double n = std::sqrt(mko::dot(v, v));
    return Vec3<double>{v.x / n, v.y / n, v.z / n};
}

ReferenceSphere<double> image_reference(Vec3<double> center, double kappa) {
    return ReferenceSphere<double>{center, kappa,
                                   kappa == 0.0 ? ReferenceSphereKind::ImageInfinite
                                                : ReferenceSphereKind::ImageExitPupil};
}

// Erwarteter Beitrag n·(s + σR) des Schritts auf die Sphäre um M mit Krümmung κ für einen
// Strahl durch den Punkt `x` mit Richtung `d` (dieselbe Größe für jeden Punkt der Geraden).
double sphere_step(Vec3<double> x, Vec3<double> d, Vec3<double> m, double kappa) {
    const Vec3<double> q = x - m;
    const double b = -mko::dot(q, d);
    const Vec3<double> c{q.y * d.z - q.z * d.y, q.z * d.x - q.x * d.z, q.x * d.y - q.y * d.x};
    const double t = -mko::dot(c, c);
    return b - kappa * t / (std::sqrt(1.0 + kappa * kappa * t) + 1.0);
}

// Systemdatei mit einer Hilfsebene in Luft hinter der letzten Fläche: Die letzte Fläche
// bekommt den Abstand `fraction`·w, die neue Ebene den Rest, die Bildebene bleibt im Raum fest.
// Variablen und Nebenbedingungen entfallen (sie zählen Flächen).
SystemDescription with_dummy_plane(SystemDescription d, double fraction) {
    mko::Surface& last = d.surfaces.back();
    const double w = last.thickness;
    mko::Surface plane;
    plane.type = mko::SurfaceType::Plane;
    plane.thickness = w * (1.0 - fraction);
    plane.material = last.material;
    plane.semi_diameter = 1000.0;
    last.thickness = w * fraction;
    d.surfaces.push_back(plane);
    d.variables.reset();
    d.constraints.reset();
    return d;
}

mko::LoadedSystem prepared(SystemDescription d, ReferenceSphereKind kind) {
    d.reference_sphere = kind;
    return mko::prepare_system(std::move(d), std::filesystem::path(mko::test::kResidualSystems));
}

// Tabelle aus Ebenen in Luft (Blende, dann Ebenen) mit den gegebenen Dicken.
HostSurfaceTable air_planes(const std::vector<double>& thickness) {
    HostSurfaceTable t;
    t.surfaces = static_cast<int>(thickness.size());
    t.wavelengths = 1;
    t.type.assign(thickness.size(), SurfaceKind::Plane);
    t.type[0] = SurfaceKind::Stop;
    t.curvature.assign(thickness.size(), 0.0);
    t.thickness = thickness;
    t.index.assign(thickness.size(), 1.0);
    t.semi_diameter.assign(thickness.size(), 1000.0);
    return t;
}

// L²-Projektion von f(u) auf Legendre-Polynome P_k(2u − 1), k = 0 … K (16 Gauß-Knoten), wie in
// test_ghost_plate.cpp.
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

} // namespace

// Der Schritt auf die Sphäre um den Bildpunkt (image_sphere_step) liefert dieselbe OPL wie der
// Schritt durch die Austrittspupille des Modus EVAL (exit_pupil_sphere_step, M8-T1), bitgleich
// und mit gleichem Status; die Position ist der Fußpunkt des Lots von M auf den Strahl.
TEST(ImageReference, StepEqualsExitPupilStepOfEval) {
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    int ok = 0;
    int missed = 0;
    double worst_foot = 0.0;
    for (int i = 0; i < 20000; ++i) {
        const Vec3<double> center{0.3 * u(rng), 2.0 * u(rng), 5.0 + 20.0 * std::abs(u(rng))};
        const double kappa = i % 4 == 0 ? 0.0 : u(rng) * (i % 3 == 0 ? 2.0 : 0.05);
        const mko::Ray<double> r{{3.0 * u(rng), 3.0 * u(rng), 0.5 * u(rng)},
                                 unit({0.4 * u(rng), 0.4 * u(rng), 1.0}),
                                 10.0 * u(rng)};
        mko::Ray<double> a = r;
        mko::Ray<double> b = r;
        const TraceStatus sa = mko::image_sphere_step(a, center, kappa, 1.3);
        const TraceStatus sb = mko::exit_pupil_sphere_step(b, center, kappa, 1.3);
        ASSERT_EQ(sa, sb);
        if (sa != TraceStatus::Ok) {
            ++missed;
            continue;
        }
        ++ok;
        ASSERT_TRUE(same_bits(a.opl, b.opl)) << a.opl << " " << b.opl;
        worst_foot = std::max(worst_foot, std::abs(mko::dot(a.p - center, r.d)));
    }
    std::printf("[ Info     ] %d Strahlen auf der Sphäre, %d verfehlt, |(p − M)·d| <= %.2e mm\n",
                ok, missed, worst_foot);
    EXPECT_GT(ok, 10000);
    EXPECT_GT(missed, 100);
    EXPECT_LT(worst_foot, kReferenceSphereFp64);
}

// Reine Defokussierung (analytisch): Eine Punktquelle q in Luft vor Ebenen in Luft erzeugt eine
// exakte Kugelwelle um q. Mit der Referenzsphäre um M ≠ q ist die OPL bis zur Sphäre
// (bis auf die für das Feld gleiche Konstante −σR) für den Strahl mit Richtung d
// b − κt/(√(1 + κ²t) + 1) mit b = (M − q)·d und t = −|(q − M) × d|², für κ = 0 also
// (M − q)·d = Δz cos θ + h_y d_y (Defokus und Kippung). Alle drei Arten der Sphäre um den
// Bildpunkt; das Ergebnis hängt nicht von der Zahl der Ebenen hinter der Blende ab.
TEST(ImageReference, PureDefocusOfPointSourceIsAnalytic) {
    const Vec3<double> q_stop{0.0, 0.3, -20.0};
    const double sin_max = 0.4;
    double worst = 0.0;
    int rays = 0;
    for (const std::vector<double>& th :
         {std::vector<double>{5.0, 4.0, 7.0}, std::vector<double>{9.0, 7.0},
          std::vector<double>{5.0, 4.0, 2.5, 4.5}}) {
        const HostSurfaceTable t = air_planes(th);
        double z_last = 0.0; // Lage der letzten Fläche in Koordinaten der Blende
        for (std::size_t s = 0; s + 1 < th.size(); ++s) {
            z_last += th[s];
        }
        const Vec3<double> q{q_stop.x, q_stop.y, q_stop.z - z_last};
        const Vec3<double> m{0.0, 0.12, th.back()}; // Bildpunkt 16 mm hinter der Blende
        for (const double kappa : {0.0, 1.0 / 40.0, -1.0 / 55.0}) {
            for (double rho = 0.0; rho <= 1.0; rho += 0.125) {
                for (double phi = 0.0; phi < 6.28; phi += 0.7) {
                    mko::Ray<double> ray = mko::point_source_ray(q_stop, sin_max, rho, phi);
                    const Vec3<double> d = ray.d;
                    int failed = 0;
                    const TraceStatus st = mko::trace_to_reference_sphere(
                        t.view(), 0, ray, image_reference(m, kappa), &failed);
                    ASSERT_EQ(st, TraceStatus::Ok);
                    ASSERT_EQ(failed, -1);
                    const double expected = sphere_step(q, d, m, kappa);
                    worst = std::max(worst, std::abs(ray.opl - expected));
                    ++rays;
                }
            }
        }
    }
    std::printf("[ Info     ] %d Strahlen: max |OPL − analytisch| = %.2e mm\n", rays, worst);
    EXPECT_LT(worst, kReferenceSphereFp64);
}

// Falle F12 (M6-T6): Eine ideal fokussierende Geisterplatte (Brennweite F = 10 mm, NA 0,3,
// gefittete Phase K = 12 wie GhostPlate.FittedFocusingPhaseIsDiffractionLimitedAtNa03) mit
// einer Hilfsebene im Brennpunkt und der Bildebene Δ = 2,7 mm dahinter. Mit der alten
// Konvention geht die Sphäre durch den Scheitel der Hilfsebene, also durch den Brennpunkt: Alle
// Strahlen treffen sie dort, die OPD ist ≈ 0, obwohl das Bild um Δ defokussiert ist. Mit der
// Sphäre um den Bildpunkt ist die OPD der analytische Defokus: für den idealen Strahl durch den
// Brennpunkt P mit Richtung d und q = P − M = (0, 0, −Δ) gilt b = Δ d_z und
// t = −Δ² (d_x² + d_y²); für κ = 0 also OPD = Δ (cos θ − Mittel).
TEST(ImageReference, FocusInAuxiliaryPlaneIsNoLongerHidden) {
    const double F = 10.0;
    const double na = 0.3;
    const double a = F * na / std::sqrt(1.0 - na * na);
    const double nG = 1.5;
    const double delta = 2.7;
    const auto phase = [&](double u) { return -(std::sqrt(a * a * u + F * F) - F); };
    const std::vector<double> c = legendre_fit(phase, 12);
    HostSurfaceTable t;
    t.surfaces = 3;
    t.wavelengths = 1;
    t.type = {SurfaceKind::Stop, SurfaceKind::GhostPlate, SurfaceKind::Plane};
    t.curvature = {0.0, 0.0, 0.0};
    t.thickness = {0.0, F, delta};
    t.index = {1.0, 1.0, 1.0};
    t.semi_diameter = {a, a, 100.0};
    t.ghost_plates = 1;
    t.ghost_terms = 12;
    for (int k = 1; k <= 12; ++k) {
        t.ghost_coeff.push_back(c[static_cast<std::size_t>(k)] / (nG - 1.0));
    }
    t.ghost_index = {nG};
    t.ghost_norm_radius = {a};
    const auto q = mko::make_pupil_quadrature<double>(10, 8, mko::PupilMode::Axis);
    const int n = mko::sample_count(q);
    const Vec3<double> m{0.0, 0.0, delta};
    // Paraxiale Austrittspupille: Bild der Blende (in der Geisterplatte) = die Platte selbst,
    // F vor der Hilfsebene; Radius bis zur Bildebene F + Δ
    const double kappa_ep = 1.0 / (F + delta);
    struct Run {
        const char* name;
        ReferenceSphere<double> ref;
    };
    const Run runs[] = {
        {"letzte Fläche", ReferenceSphere<double>{m, 0.0, ReferenceSphereKind::LastSurface}},
        {"Bildpunkt, unendlich", image_reference(m, 0.0)},
        {"Bildpunkt, Austrittspupille", image_reference(m, kappa_ep)}};
    double rms[3] = {0.0, 0.0, 0.0};
    for (int r = 0; r < 3; ++r) {
        std::vector<double> opl(static_cast<std::size_t>(n));
        std::vector<double> w(static_cast<std::size_t>(n));
        std::vector<double> expected(static_cast<std::size_t>(n));
        std::vector<TraceStatus> st(static_cast<std::size_t>(n));
        std::vector<double> opd(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            const mko::PupilSample<double> s = mko::sample(q, k);
            mko::Ray<double> ray = mko::collimated_ray(a, 0.0, s.rho, s.phi);
            const double h = a * s.rho;
            const Vec3<double> d_ideal = unit({-h * std::sin(s.phi), -h * std::cos(s.phi), F});
            expected[ku] = sphere_step(Vec3<double>{0.0, 0.0, 0.0}, d_ideal, m, runs[r].ref.kappa);
            st[ku] = mko::trace_to_reference_sphere(t.view(), 0, ray, runs[r].ref);
            opl[ku] = ray.opl;
            w[ku] = s.weight;
        }
        const mko::OpdStatistics<double> stat =
            mko::opd_statistics(opl.data(), w.data(), st.data(), n, opd.data());
        ASSERT_EQ(stat.failed, 0) << runs[r].name;
        rms[r] = stat.rms;
        if (r == 0) {
            std::printf("[ Info     ] %s: RMS-OPD %.3e mm (Falle: Defokus versteckt)\n",
                        runs[r].name, stat.rms);
            continue;
        }
        std::vector<double> e_opd(static_cast<std::size_t>(n));
        std::vector<TraceStatus> ok(static_cast<std::size_t>(n), TraceStatus::Ok);
        const mko::OpdStatistics<double> e_stat =
            mko::opd_statistics(expected.data(), w.data(), ok.data(), n, e_opd.data());
        double worst = 0.0;
        for (std::size_t k = 0; k < opd.size(); ++k) {
            worst = std::max(worst, std::abs(opd[k] - e_opd[k]));
        }
        std::printf("[ Info     ] %s: RMS-OPD %.6e mm, analytisch %.6e mm, max |ΔOPD| = %.2e mm\n",
                    runs[r].name, stat.rms, e_stat.rms, worst);
        EXPECT_LT(worst, kReferenceSphereFp64) << runs[r].name;
    }
    EXPECT_LT(rms[0], mko::test::kGhostFocusRms);
    EXPECT_GT(rms[1], 1e4 * rms[0]);
    EXPECT_GT(rms[2], 1e4 * rms[0]);
}

// Kern der Falle F12: Eine Hilfsebene in Luft hinter dem System (die Bildebene bleibt im Raum
// fest) ändert die OPD mit der Sphäre um den Bildpunkt nicht, für beide Radien; κ ist gleich,
// weil Bildpunkt und Austrittspupille um denselben Betrag wandern. Mit der alten Konvention
// ändert sie sich (anderer Radius).
TEST(ImageReference, AuxiliaryPlaneBehindSystemDoesNotChangeOpd) {
    for (const char* file :
         {"einzellinse_bk7.json", "testsystem_mehrlinser.json", "einzellinse_geisterplatte.json"}) {
        const SystemDescription base = mko::test::load_residual_system(file).system;
        for (const ReferenceSphereKind kind :
             {ReferenceSphereKind::LastSurface, ReferenceSphereKind::ImageInfinite,
              ReferenceSphereKind::ImageExitPupil}) {
            const SystemBatchHost a = mko::make_system_batch(prepared(base, kind), 1);
            ASSERT_EQ(a.topology().reference_sphere, kind);
            double worst_opd = 0.0;
            double worst_kappa = 0.0;
            for (const double fraction : {0.2, 0.75}) {
                const SystemBatchHost b =
                    mko::make_system_batch(prepared(with_dummy_plane(base, fraction), kind), 1);
                const ResidualResult ra = mko::residuals_cpu(a, ResidualLayout(a.topology()));
                const ResidualResult rb = mko::residuals_cpu(b, ResidualLayout(b.topology()));
                ASSERT_EQ(ra.opd.size(), rb.opd.size());
                for (std::size_t i = 0; i < ra.opd.size(); ++i) {
                    ASSERT_EQ(ra.status[i], rb.status[i]);
                    worst_opd = std::max(worst_opd, std::abs(ra.opd[i] - rb.opd[i]));
                }
                if (kind == ReferenceSphereKind::LastSurface) {
                    continue;
                }
                for (int f = 0; f < a.fields(); ++f) {
                    const double ka = a.reference_curvature(0, f);
                    const double kb = b.reference_curvature(0, f);
                    worst_kappa = std::max(worst_kappa, std::abs(ka - kb) / std::max(ka, 1e-300));
                }
            }
            std::printf("[ Info     ] %-32s %-17s max |ΔOPD| = %.2e mm, κ relativ %.1e\n", file,
                        mko::reference_sphere_name(kind), worst_opd, worst_kappa);
            if (kind == ReferenceSphereKind::LastSurface) {
                EXPECT_GT(worst_opd, 1e3 * kReferenceSphereFp64) << file;
            } else {
                EXPECT_LT(worst_opd, kReferenceSphereFp64) << file;
                EXPECT_LT(worst_kappa, mko::test::kParaxialRelative) << file;
            }
        }
    }
}

// κ beim Aufbau: Austrittspupille wie EVAL (eval_reference_geometry, bitgleich), unendlich 0,
// letzte Fläche ohne Speicher. Es ist eine feste Größe des Batches: VariableSet::apply ändert es
// nicht, update_reference_curvature rechnet es für die geänderte Variante neu.
TEST(ImageReference, CurvatureFromParaxialExitPupilIsFrozen) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    {
        const SystemBatchHost b = mko::make_system_batch(ls, 2);
        EXPECT_EQ(b.topology().reference_sphere, ReferenceSphereKind::LastSurface);
        EXPECT_TRUE(b.reference_curvature_data().empty());
        EXPECT_EQ(b.view().reference_curvature, nullptr);
    }
    {
        const SystemBatchHost b =
            mko::make_system_batch(prepared(ls.system, ReferenceSphereKind::ImageInfinite), 2);
        ASSERT_EQ(b.reference_curvature_data().size(), 2u * static_cast<std::size_t>(b.fields()));
        for (const double k : b.reference_curvature_data()) {
            EXPECT_EQ(k, 0.0);
        }
    }
    const mko::LoadedSystem ep = prepared(ls.system, ReferenceSphereKind::ImageExitPupil);
    SystemBatchHost b = mko::test::perturbed_batch(ep, 20, 99);
    // perturbed_batch kopiert Variante 0: κ ist eingefroren
    for (int v = 0; v < b.variants(); ++v) {
        for (int f = 0; f < b.fields(); ++f) {
            ASSERT_TRUE(same_bits(b.reference_curvature(v, f), b.reference_curvature(0, f)));
        }
    }
    mko::update_reference_curvature(b);
    const mko::EvalReferenceGeometry geo =
        mko::eval_reference_geometry(b, mko::EvalReference::ExitPupil);
    int differs = 0;
    for (int v = 0; v < b.variants(); ++v) {
        for (int f = 0; f < b.fields(); ++f) {
            const double k = b.reference_curvature(v, f);
            ASSERT_TRUE(same_bits(k, geo.curvature[static_cast<std::size_t>(v * b.fields() + f)]))
                << v << " " << f;
            differs += same_bits(k, b.reference_curvature(0, f)) ? 0 : 1;
        }
    }
    EXPECT_GT(differs, 10);
    // VariableSet::apply (auch Arbeitsabstand) ändert κ nicht
    const mko::LoadedSystem lv =
        prepared(mko::test::load_residual_system("einzellinse_bk7_variablen.json").system,
                 ReferenceSphereKind::ImageExitPupil);
    ASSERT_TRUE(lv.system.variables.has_value());
    SystemBatchHost bv = mko::make_system_batch(lv, 1);
    const mko::VariableSet set(lv.system, bv.topology());
    std::vector<double> x = set.extract(bv, 0);
    for (double& xi : x) {
        xi *= 1.01;
    }
    const std::vector<double> before(bv.reference_curvature_data().begin(),
                                     bv.reference_curvature_data().end());
    ASSERT_FALSE(before.empty());
    set.apply(bv, 0, x);
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_TRUE(same_bits(before[i], bv.reference_curvature_data()[i]));
    }
}

// Die Merit mit der Sphäre um den Bildpunkt ist das gewichtete Mittel der RMS² aus
// evaluate_wavefront derselben Datei (M4-T1, beide folgen dem Feld reference_sphere).
TEST(ImageReference, MeritEqualsWeightedSumOfWavefrontRms) {
    for (const ReferenceSphereKind kind :
         {ReferenceSphereKind::ImageInfinite, ReferenceSphereKind::ImageExitPupil}) {
        for (const mko::test::WeightedCase& c : mko::test::weighted_cases()) {
            SCOPED_TRACE(c.name);
            const mko::LoadedSystem ls = prepared(c.system.system, kind);
            const mko::WavefrontResult wf = mko::evaluate_wavefront(ls);
            ASSERT_EQ(wf.failed_count(), 0u);
            std::vector<double> rms_waves;
            for (const mko::WavefrontMap& m : wf.maps) {
                rms_waves.push_back(m.rms_waves);
            }
            const double expected = mko::test::weighted_rms_merit(ls.system, rms_waves);
            const SystemBatchHost batch = mko::make_system_batch(ls, 1);
            const mko::MeritResult m =
                mko::merit_cpu(batch, ResidualLayout(batch.topology()), false);
            const double rel = std::abs(m.merit[0] - expected) / expected;
            EXPECT_LT(rel, mko::test::kWeightedMerit) << mko::reference_sphere_name(kind);
            std::printf("[ Info     ] %-17s %-46s F = %.10e, relativ %.1e\n",
                        mko::reference_sphere_name(kind), c.name.c_str(), m.merit[0], rel);
        }
    }
}

// Verfehlt der Strahl die Sphäre (1 + κ²t < 0, nur bei endlichem Radius): hart Missed (Strahl
// unverändert), weich SoftMissed mit Strafe −(1 + κ²t) und Fortsetzung mit √(1 + κ²t) := 0,
// also n·(b − κt) (stetig an der Grenze). NaN bleibt hart.
TEST(ImageReference, MissedSphereHardAndSoft) {
    const Vec3<double> m{0.0, 0.0, 5.0};
    const double kappa = 1.0 / 2.0;                                  // R = 2 mm
    const mko::Ray<double> r{{0.0, 3.0, 0.0}, {0.0, 0.0, 1.0}, 1.0}; // Abstand 3 mm > R
    const double b = 5.0;
    const double t = -9.0;
    const double disc = 1.0 + kappa * kappa * t;
    ASSERT_LT(disc, 0.0);
    mko::Ray<double> hard = r;
    EXPECT_EQ(mko::image_sphere_step(hard, m, kappa, 1.25), TraceStatus::Missed);
    EXPECT_TRUE(same_bits(hard.opl, r.opl));
    EXPECT_TRUE(same_bits(hard.p.y, r.p.y));
    mko::Ray<double> soft = r;
    mko::SoftFailure<double> sf{0.0, TraceStatus::Ok};
    double s_out = 0.0;
    EXPECT_EQ(mko::image_sphere_step(soft, m, kappa, 1.25, &s_out, &sf), TraceStatus::SoftMissed);
    EXPECT_EQ(sf.first, TraceStatus::SoftMissed);
    EXPECT_DOUBLE_EQ(sf.penalty, -disc);
    EXPECT_DOUBLE_EQ(s_out, b - kappa * t);
    EXPECT_DOUBLE_EQ(soft.opl, 1.0 + 1.25 * (b - kappa * t));
    // An der Grenze stetig: R = 3 mm berührt
    mko::Ray<double> touch = r;
    ASSERT_EQ(mko::image_sphere_step(touch, m, 1.0 / 3.0, 1.25, &s_out), TraceStatus::Ok);
    EXPECT_NEAR(s_out, b + 3.0, 1e-12);
    // NaN
    mko::Ray<double> bad = r;
    bad.d.z = std::nan("");
    EXPECT_EQ(mko::image_sphere_step(bad, m, kappa, 1.25, nullptr, &sf), TraceStatus::Missed);
    // Im Trace: versagende Fläche S
    const HostSurfaceTable tab = air_planes({1.0, 5.0});
    mko::Ray<double> ray = mko::collimated_ray(3.0, 0.0, 1.0, 0.0);
    int failed = 0;
    EXPECT_EQ(
        mko::trace_to_reference_sphere(tab.view(), 0, ray, image_reference(m, kappa), &failed),
        TraceStatus::Missed);
    EXPECT_EQ(failed, 2);
}

// Kompatibilität: Datei ohne Feld = letzte Fläche (bitgleich zur ausdrücklichen Angabe);
// Feld einlesen, schreiben (nur wenn gesetzt), ungültige Werte mit Pfad ablehnen.
TEST(ImageReference, FileFormatAndCompatibility) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    ASSERT_FALSE(ls.system.reference_sphere.has_value());
    const std::string text = mko::write_system_json(ls.system);
    EXPECT_EQ(text.find("reference_sphere"), std::string::npos);
    const SystemBatchHost plain = mko::make_system_batch(ls, 3);
    const SystemBatchHost explicit_old =
        mko::make_system_batch(prepared(ls.system, ReferenceSphereKind::LastSurface), 3);
    EXPECT_TRUE(plain == explicit_old);
    const ResidualResult r0 = mko::residuals_cpu(plain, ResidualLayout(plain.topology()));
    const ResidualResult r1 =
        mko::residuals_cpu(explicit_old, ResidualLayout(explicit_old.topology()));
    for (std::size_t i = 0; i < r0.residuals.size(); ++i) {
        ASSERT_TRUE(same_bits(r0.residuals[i], r1.residuals[i]));
    }
    for (const ReferenceSphereKind kind :
         {ReferenceSphereKind::LastSurface, ReferenceSphereKind::ImageExitPupil,
          ReferenceSphereKind::ImageInfinite}) {
        SystemDescription d = ls.system;
        d.reference_sphere = kind;
        const std::string t = mko::write_system_json(d);
        EXPECT_NE(t.find(std::string("\"reference_sphere\": \"") +
                         mko::reference_sphere_name(kind) + "\""),
                  std::string::npos);
        const SystemDescription back = mko::read_system_json(t);
        ASSERT_TRUE(back.reference_sphere.has_value());
        EXPECT_EQ(*back.reference_sphere, kind);
        EXPECT_TRUE(back == d);
        EXPECT_EQ(mko::write_system_json(back), t);
    }
    std::string bad = text;
    bad.insert(bad.find("\"surfaces\""), "\"reference_sphere\": \"vertex\",\n  ");
    try {
        (void)mko::read_system_json(bad);
        ADD_FAILURE() << "ungültiger Wert angenommen";
    } catch (const mko::FormatError& e) {
        EXPECT_EQ(e.json_path(), "reference_sphere");
    }
    std::string num = text;
    num.insert(num.find("\"surfaces\""), "\"reference_sphere\": 1,\n  ");
    EXPECT_THROW((void)mko::read_system_json(num), mko::FormatError);
}

// EVAL mit `infinite`: κ = 0 für alle Felder, keine Austrittspupille; die OPD des Rasterpunkts
// ist die des Schritts image_sphere_step mit κ = 0.
TEST(ImageReference, EvalInfiniteReference) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost b = mko::make_system_batch(ls, 1);
    const mko::EvalReferenceGeometry geo =
        mko::eval_reference_geometry(b, mko::EvalReference::Infinite);
    for (const double k : geo.curvature) {
        EXPECT_EQ(k, 0.0);
    }
    EXPECT_TRUE(std::isnan(geo.exit_pupil_position[0]));
    mko::EvalSettings s;
    s.grid = 16;
    s.reference = mko::EvalReference::Infinite;
    const mko::EvalResult e = mko::eval_cpu(b, s);
    // Gegen die Merit-Konvention image_infinite auf denselben Strahlen: Hauptstrahl jedes Tupels
    const SystemBatchHost bi =
        mko::make_system_batch(prepared(ls.system, ReferenceSphereKind::ImageInfinite), 1);
    const mko::SystemBatchView<double> view = bi.view();
    const mko::SurfaceTableView<double> table = view.surface_table(0);
    for (int f = 0; f < b.fields(); ++f) {
        for (int l = 0; l < b.wavelengths(); ++l) {
            const ReferenceSphere<double> ref = mko::batch_reference(view, table, 0, f);
            ASSERT_EQ(ref.kind, ReferenceSphereKind::ImageInfinite);
            const mko::EvalRay<double> er =
                mko::eval_ray(view, table, l, 0, f, l, 0.0, 0.0, ref.center, s.reference, 0.0);
            mko::Ray<double> ray = mko::eval_source_ray(view, 0, f, l, 0.0, 0.0);
            ASSERT_EQ(mko::trace_to_reference_sphere(table, l, ray, ref), TraceStatus::Ok);
            EXPECT_TRUE(same_bits(er.opl, ray.opl));
        }
    }
    EXPECT_EQ(e.failed_count(), 0u);
}
