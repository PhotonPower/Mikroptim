// Aufzeichnung des Vorwärtsdurchlaufs für den adjungierten Modus (M5-T1, CPU;
// include/mikroptim/trace/adjoint_record.hpp, docs/architektur.md, Abschnitt "Modus ADJOINT:
// Vorwärtsdurchlauf mit Aufzeichnung"): Die Aufzeichnung ändert den Trace nicht (bitgleich), jeder
// gespeicherte Zustand stimmt mit einem unabhängigen Trace Fläche für Fläche überein, die Zweige
// des weichen Versagens stehen an den erwarteten Flächen, das Neu-Vorwärtsrechnen aus
// Checkpoints ist bitgleich, und die lokalen Zwischengrößen folgen aus dem minimalen Satz
// (Neuberechnung gegen geometrische Beziehungen). Flächentypen Stop, Plane, Sphere, GhostPlate,
// harter und weicher Modus, Referenzsphäre.

#include "adjoint_test_tools.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/trace.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using mko::AdjointRecord;
using mko::CheckpointRecord;
using mko::HostSurfaceTable;
using mko::Ray;
using mko::SoftFailure;
using mko::SurfaceKind;
using mko::SurfaceTableView;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::BatchRay;
using mko::test::kLengthFp64;
using mko::test::kReferenceSphereFp64;
using mko::test::kUnitFp64;

namespace {

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool same_bits(const Vec3<double>& a, const Vec3<double>& b) {
    return same_bits(a.x, b.x) && same_bits(a.y, b.y) && same_bits(a.z, b.z);
}

bool same_bits(const Ray<double>& a, const Ray<double>& b) {
    return same_bits(a.p, b.p) && same_bits(a.d, b.d) && same_bits(a.opl, b.opl);
}

Vec3<double> cross(const Vec3<double>& a, const Vec3<double>& b) {
    return Vec3<double>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double max_abs(const Vec3<double>& a) {
    return std::max({std::abs(a.x), std::abs(a.y), std::abs(a.z)});
}

// Frei erfundenes Prüfsystem mit allen Flächentypen (L = 2): Blende, Sphäre in Glas, Ebene,
// Geisterplatte, Kittfläche, Sphäre, Geisterplatte, Ebene in Glas, letzte Sphäre; zwei
// Geisterplatten mit K = 4. `strong_ghost`: erste Geisterplatte mit großem Koeffizienten (am Rand
// negative Radikanden).
HostSurfaceTable mixed_table(bool strong_ghost = false) {
    HostSurfaceTable t;
    t.surfaces = 9;
    t.wavelengths = 2;
    t.type = {SurfaceKind::Stop,       SurfaceKind::Sphere, SurfaceKind::Plane,
              SurfaceKind::GhostPlate, SurfaceKind::Sphere, SurfaceKind::Sphere,
              SurfaceKind::GhostPlate, SurfaceKind::Plane,  SurfaceKind::Sphere};
    t.curvature = {0.0, 0.04, 0.0, 0.0, -0.03, 0.02, 0.0, 0.0, -0.01};
    t.thickness = {2.0, 3.0, 1.0, 2.0, 2.5, 1.5, 1.0, 2.0, 40.0};
    t.index = {1.0,  1.0, 1.5, 1.49, 1.0, 1.0, 1.0,  1.0, 1.7,
               1.69, 1.0, 1.0, 1.0,  1.0, 1.6, 1.59, 1.0, 1.0};
    t.semi_diameter.assign(9, 10.0);
    t.ghost_plates = 2;
    t.ghost_terms = 4;
    t.ghost_coeff = {strong_ghost ? 5.0 : 0.01, -0.004, 0.001, 0.0005, -0.008, 0.002, 0.0, -0.0003};
    t.ghost_index = {1.5, 1.49, 1.6, 1.59};
    t.ghost_norm_radius = {6.0, 6.0};
    return t;
}

// Strahlen durch das Prüfsystem: kollimiert mit Pupillenradius a (a = 30 > 1/c₁ = 25: Strahlen
// verfehlen Fläche 1 bzw. totalreflektieren), drei Feldwinkel, beide Wellenlängen.
std::vector<BatchRay> mixed_rays(const HostSurfaceTable& t, double a) {
    std::vector<BatchRay> rays;
    const auto q = mko::make_pupil_quadrature<double>(6, 6, mko::PupilMode::Full);
    for (const double theta : {0.0, 0.03, 0.08}) {
        for (int l = 0; l < 2; ++l) {
            for (int k = 0; k < mko::sample_count(q); ++k) {
                const mko::PupilSample<double> s = mko::sample(q, k);
                BatchRay r;
                r.l = l;
                r.k = k;
                r.table = t.view();
                r.start = mko::collimated_ray(a, theta, s.rho, s.phi);
                r.center = Vec3<double>{0.0, 40.0 * std::tan(theta), t.thickness.back()};
                rays.push_back(r);
            }
        }
    }
    return rays;
}

// Alle Prüffälle: Batches mit gestörten Varianten (auch extreme mit versagten Strahlen) und das
// Prüfsystem mit allen Flächentypen. Die Tabellen der Strahlen zeigen in die Batches bzw.
// Tabellen dieser Struktur (einmal aufgebaut, lebt bis zum Ende des Programms).
struct RayCases {
    std::vector<mko::SystemBatchHost> batches;
    std::vector<HostSurfaceTable> tables;
    std::vector<std::string> names;
    std::vector<std::vector<BatchRay>> rays;
};

const RayCases& ray_cases() {
    static const RayCases cases = [] {
        RayCases c;
        c.batches.push_back(mko::test::perturbed_batch(
            mko::test::load_residual_system("testsystem_mehrlinser.json"), 40, 20261019));
        c.names.emplace_back("Mehrlinser, 40 Varianten");
        c.batches.push_back(mko::test::perturbed_batch(
            mko::test::load_residual_system("einzellinse_geisterplatte.json"), 40, 20261020));
        c.names.emplace_back("Einzellinse mit Geisterplatten, 40 Varianten");
        c.batches.push_back(mko::make_system_batch(
            mko::test::load_residual_system("weiches_versagen_start.json"), 1));
        c.names.emplace_back("weiches_versagen_start.json");
        c.batches.push_back(mko::test::many_surface_batch(6, 20261021));
        c.names.emplace_back("S = 64, 6 Varianten");
        for (const mko::SystemBatchHost& b : c.batches) {
            std::vector<BatchRay> list;
            const mko::ResidualLayout layout(b.topology(), mko::ResidualSettings{6, 6});
            mko::test::for_each_batch_ray(b, layout, [&](const BatchRay& r) { list.push_back(r); });
            c.rays.push_back(std::move(list));
        }
        c.tables.push_back(mixed_table(false));
        c.tables.push_back(mixed_table(true));
        c.rays.push_back(mixed_rays(c.tables[0], 4.0));
        c.names.emplace_back("Prüfsystem alle Flächentypen, a = 4");
        c.rays.push_back(mixed_rays(c.tables[0], 30.0));
        c.names.emplace_back("Prüfsystem alle Flächentypen, a = 30");
        c.rays.push_back(mixed_rays(c.tables[1], 8.0));
        c.names.emplace_back("Prüfsystem, starke Geisterplatte, a = 8");
        return c;
    }();
    return cases;
}

// Ergebnis eines Strahls bis zur Referenzsphäre.
struct Traced {
    TraceStatus status = TraceStatus::Ok;
    int failed = -2;
    Ray<double> ray{};
    SoftFailure<double> soft{0.0, TraceStatus::Ok};
};

template <class Recorder>
Traced trace(const BatchRay& r, bool soft_mode, Recorder* recorder) {
    Traced t;
    t.ray = r.start;
    mko::NoSurfaceObserver* const none = nullptr;
    t.status = mko::trace_to_reference_sphere(r.table, r.l, t.ray, r.center, &t.failed, none,
                                              soft_mode ? &t.soft : nullptr, recorder);
    return t;
}

bool same_result(const Traced& a, const Traced& b) {
    return a.status == b.status && a.failed == b.failed && same_bits(a.ray, b.ray) &&
           same_bits(a.soft.penalty, b.soft.penalty) && a.soft.first == b.soft.first;
}

} // namespace

// Die Aufzeichnung ändert nichts: Status, versagende Fläche, Endzustand (Ort, Richtung, OPL auf
// der Referenzsphäre) und Akkumulator des weichen Versagens sind mit jeder Aufzeichnung
// (TraceRecord, AdjointRecord in allen passenden Kapazitätsklassen, CheckpointRecord mit
// k = 1, 4, 8) bitgleich zum Trace ohne Aufzeichnung, im harten und im weichen Modus.
TEST(AdjointRecord, RecordingLeavesTraceBitIdentical) {
    const RayCases& cases = ray_cases();
    auto trace_record = std::make_unique<mko::TraceRecord<double>>();
    auto full64 = std::make_unique<AdjointRecord<double, 64>>();
    auto full32 = std::make_unique<AdjointRecord<double, 32>>();
    auto full16 = std::make_unique<AdjointRecord<double, 16>>();
    auto check8 = std::make_unique<CheckpointRecord<double, 64, 8>>();
    auto check4 = std::make_unique<CheckpointRecord<double, 64, 4>>();
    auto check1 = std::make_unique<CheckpointRecord<double, 16, 1>>();
    std::size_t rays = 0;
    std::size_t soft_rays = 0;
    std::size_t hard_rays = 0;
    for (std::size_t c = 0; c < cases.rays.size(); ++c) {
        for (const bool soft_mode : {false, true}) {
            for (const BatchRay& r : cases.rays[c]) {
                const int S = r.table.surfaces;
                mko::NoTraceRecorder* const no_record = nullptr;
                const Traced plain = trace(r, soft_mode, no_record);
                ++rays;
                soft_rays += mko::is_soft_failure(plain.status) ? 1u : 0u;
                hard_rays += mko::is_hard_failure(plain.status) ? 1u : 0u;
                ASSERT_TRUE(same_result(trace(r, soft_mode, trace_record.get()), plain))
                    << cases.names[c];
                ASSERT_TRUE(same_result(trace(r, soft_mode, full64.get()), plain));
                ASSERT_TRUE(same_result(trace(r, soft_mode, check8.get()), plain));
                ASSERT_TRUE(same_result(trace(r, soft_mode, check4.get()), plain));
                if (S <= 32) {
                    ASSERT_TRUE(same_result(trace(r, soft_mode, full32.get()), plain));
                }
                if (S <= 16) {
                    ASSERT_TRUE(same_result(trace(r, soft_mode, full16.get()), plain));
                    ASSERT_TRUE(same_result(trace(r, soft_mode, check1.get()), plain));
                }
                // trace_ray (ohne Referenzsphäre) mit und ohne TraceRecord
                Ray<double> a = r.start;
                Ray<double> b = r.start;
                SoftFailure<double> sa{0.0, TraceStatus::Ok};
                SoftFailure<double> sb{0.0, TraceStatus::Ok};
                int fa = -2;
                int fb = -2;
                mko::NoSurfaceObserver* const none = nullptr;
                const TraceStatus ta =
                    mko::trace_ray(r.table, r.l, a, nullptr, &fa, none, soft_mode ? &sa : nullptr);
                const TraceStatus tb = mko::trace_ray(r.table, r.l, b, trace_record.get(), &fb,
                                                      none, soft_mode ? &sb : nullptr);
                ASSERT_EQ(ta, tb);
                ASSERT_EQ(fa, fb);
                ASSERT_TRUE(same_bits(a, b));
                ASSERT_TRUE(same_bits(sa.penalty, sb.penalty));
            }
        }
    }
    std::printf("[ Info     ] %zu Strahlen (harter und weicher Modus), davon %zu weich und %zu "
                "hart versagt: mit und ohne Aufzeichnung bitgleich\n",
                rays, soft_rays, hard_rays);
    EXPECT_GT(soft_rays, 100u);
    EXPECT_GT(hard_rays, 100u);
}

// Jeder gespeicherte Zustand gegen einen unabhängigen Trace Fläche für Fläche (eigene Schleife
// über propagate_surface mit eigener Zählung der Geisterplatten, dann reference_sphere_step):
// p[s], d[s] und branch[s] bitgleich, count = Fläche des harten Versagens bzw. S, Schritt auf
// die Referenzsphäre (s_ref, Zweig) bitgleich.
TEST(AdjointRecord, StatesMatchIndependentStepByStepTrace) {
    const RayCases& cases = ray_cases();
    auto rec = std::make_unique<AdjointRecord<double>>();
    std::size_t states = 0;
    std::size_t branches[4] = {0, 0, 0, 0};
    std::size_t kinds[4] = {0, 0, 0, 0};
    std::size_t image_soft = 0;
    for (std::size_t c = 0; c < cases.rays.size(); ++c) {
        for (const bool soft_mode : {false, true}) {
            for (const BatchRay& r : cases.rays[c]) {
                const SurfaceTableView<double>& t = r.table;
                const int S = t.surfaces;
                const int L = t.wavelengths;
                const Traced traced = trace(r, soft_mode, rec.get());
                ASSERT_EQ(rec->soft_mode, soft_mode);
                ASSERT_TRUE(same_bits(rec->p[0], r.start.p));
                ASSERT_TRUE(same_bits(rec->d[0], r.start.d));
                ASSERT_EQ(rec->branch[0], mko::kBranchRegular);
                Ray<double> ray = r.start;
                SoftFailure<double> soft{0.0, TraceStatus::Ok};
                int ghost = 0;
                int passed = S;
                for (int s = 1; s < S; ++s) {
                    const SurfaceKind kind = t.type[s];
                    mko::GhostPlateParams<double> plate{};
                    if (kind == SurfaceKind::GhostPlate) {
                        plate.coeff = t.ghost_coeff + ghost * t.ghost_terms;
                        plate.terms = t.ghost_terms;
                        plate.norm_radius = t.ghost_norm_radius[ghost];
                        plate.n_glass = t.ghost_index[ghost * L + r.l];
                    }
                    unsigned char branch = 0xff;
                    const TraceStatus st = mko::propagate_surface(
                        ray, t.thickness[s - 1], kind, t.curvature[s], t.index[(s - 1) * L + r.l],
                        t.index[s * L + r.l], kind == SurfaceKind::GhostPlate ? &plate : nullptr,
                        soft_mode ? &soft : nullptr, &branch);
                    if (st != TraceStatus::Ok) {
                        passed = s;
                        break;
                    }
                    ghost += kind == SurfaceKind::GhostPlate ? 1 : 0;
                    ASSERT_TRUE(same_bits(rec->p[s], ray.p)) << cases.names[c] << " s=" << s;
                    ASSERT_TRUE(same_bits(rec->d[s], ray.d)) << cases.names[c] << " s=" << s;
                    ASSERT_EQ(rec->branch[s], branch) << cases.names[c] << " s=" << s;
                    ++states;
                    ++branches[branch & 3];
                    ++kinds[static_cast<int>(kind)];
                }
                ASSERT_EQ(rec->count, passed) << cases.names[c];
                if (passed < S || !(ray.d.z > 0.0)) {
                    ASSERT_TRUE(mko::is_hard_failure(traced.status));
                    continue;
                }
                double s_ref = 0.0;
                const TraceStatus ref = mko::reference_sphere_step(
                    ray, r.center, t.index[(S - 1) * L + r.l], &s_ref, soft_mode ? &soft : nullptr);
                if (mko::is_hard_failure(ref)) {
                    ASSERT_EQ(traced.status, ref);
                    continue;
                }
                ASSERT_TRUE(same_bits(rec->image_s, s_ref));
                ASSERT_EQ(rec->image_branch, ref == TraceStatus::SoftMissed ? mko::kBranchSoftMissed
                                                                            : mko::kBranchRegular);
                image_soft += ref == TraceStatus::SoftMissed ? 1u : 0u;
                ASSERT_TRUE(same_bits(ray.opl, traced.ray.opl));
                ASSERT_TRUE(same_bits(soft.penalty, traced.soft.penalty));
            }
        }
    }
    std::printf("[ Info     ] %zu Zustände geprüft; Zweige regulär %zu, SoftMissed %zu, SoftTIR "
                "%zu, beide %zu; Flächen Stop %zu, Plane %zu, Sphere %zu, GhostPlate %zu; "
                "Referenzsphäre weich verfehlt %zu\n",
                states, branches[0], branches[1], branches[2], branches[3], kinds[0], kinds[1],
                kinds[2], kinds[3], image_soft);
    EXPECT_GT(branches[1], 0u);
    EXPECT_GT(branches[2], 0u);
    EXPECT_GT(kinds[1], 0u);
    EXPECT_GT(kinds[2], 0u);
    EXPECT_GT(kinds[3], 0u);
}

// Zweige des weichen Versagens an bekannten Flächen (konstruierte Strahlen): verfehlte Sphäre,
// Totalreflexion an einer Ebene, beides an derselben Fläche, negativer Radikand an einer
// Geisterplatte, verfehlte Referenzsphäre. Im harten Modus endet die Aufzeichnung vor der Fläche.
TEST(AdjointRecord, SoftBranchesAtKnownSurfaces) {
    auto rec = std::make_unique<AdjointRecord<double, 16>>();
    auto legacy = std::make_unique<mko::TraceRecord<double>>();
    const auto make = [](std::vector<SurfaceKind> type, std::vector<double> c,
                         std::vector<double> t, std::vector<double> n) {
        HostSurfaceTable h;
        h.surfaces = static_cast<int>(type.size());
        h.wavelengths = 1;
        h.type = std::move(type);
        h.curvature = std::move(c);
        h.thickness = std::move(t);
        h.index = std::move(n);
        h.semi_diameter.assign(h.type.size(), 10.0);
        return h;
    };
    struct Case {
        const char* name;
        HostSurfaceTable table;
        Ray<double> start;
        int surface;          // Fläche mit dem weichen Ereignis
        unsigned char branch; // erwarteter Zweig
    };
    std::vector<Case> list;
    // (a) Sphäre R = 5 mm in Höhe 6 mm verfehlt, μ < 1: nur SoftMissed
    list.push_back({"verfehlte Sphäre",
                    make({SurfaceKind::Stop, SurfaceKind::Sphere, SurfaceKind::Plane},
                         {0.0, 0.2, 0.0}, {1.0, 2.0, 20.0}, {1.0, 1.5, 1.0}),
                    mko::collimated_ray(6.0, 0.0, 1.0, 0.0), 1, mko::kBranchSoftMissed});
    // (b) Punktquelle in Glas, Ebene Glas → Luft unter 50° (> 41,8°): SoftTIR
    list.push_back(
        {"Totalreflexion an einer Ebene",
         make({SurfaceKind::Plane, SurfaceKind::Plane, SurfaceKind::Plane}, {0.0, 0.0, 0.0},
              {1.0, 5.0, 5.0}, {1.5, 1.0, 1.0}),
         mko::point_source_ray(Vec3<double>{0.0, 0.0, 0.0}, std::sin(0.8726646), 1.0, 0.0), 1,
         mko::kBranchSoftTIR});
    // (c) Sphäre Glas → Luft verfehlt: am fortgesetzten Punkt cos I = 0, μ = 1,5 > 1 → beides
    list.push_back({"verfehlt und Totalreflexion",
                    make({SurfaceKind::Plane, SurfaceKind::Sphere, SurfaceKind::Plane},
                         {0.0, 0.2, 0.0}, {1.0, 2.0, 20.0}, {1.5, 1.0, 1.0}),
                    mko::collimated_ray(6.0, 0.0, 1.0, 0.0), 1,
                    static_cast<unsigned char>(mko::kBranchSoftMissed | mko::kBranchSoftTIR)});
    const HostSurfaceTable strong = mixed_table(true);
    for (Case& cs : list) {
        for (const bool soft_mode : {false, true}) {
            BatchRay r;
            r.table = cs.table.view();
            r.start = cs.start;
            r.center = Vec3<double>{0.0, 0.0, cs.table.thickness.back()};
            const Traced t = trace(r, soft_mode, rec.get());
            if (!soft_mode) {
                EXPECT_TRUE(mko::is_hard_failure(t.status)) << cs.name;
                EXPECT_EQ(rec->count, cs.surface) << cs.name;
                continue;
            }
            ASSERT_GT(rec->count, cs.surface) << cs.name;
            EXPECT_EQ(rec->branch[cs.surface], cs.branch) << cs.name;
            for (int s = 0; s < cs.surface; ++s) {
                EXPECT_EQ(rec->branch[s], mko::kBranchRegular) << cs.name << " s=" << s;
            }
            Ray<double> ray = cs.start;
            SoftFailure<double> soft{0.0, TraceStatus::Ok};
            mko::trace_ray(r.table, 0, ray, legacy.get(), nullptr,
                           static_cast<mko::NoSurfaceObserver*>(nullptr), &soft);
            EXPECT_EQ(legacy->branch[cs.surface], cs.branch) << cs.name;
            if (!mko::is_hard_failure(t.status)) {
                EXPECT_EQ(t.failed, cs.surface) << cs.name; // erstes weiches Ereignis
            }
        }
    }
    // (d) negativer Radikand an der ersten Geisterplatte (Fläche 3) des Prüfsystems mit starker
    // Geisterplatte: Strahl am Rand (die Ablenkung ∝ Höhe·dt/du übersteigt 1)
    {
        BatchRay r;
        r.table = strong.view();
        r.start = mko::collimated_ray(8.0, 0.0, 1.0, 0.0);
        r.center = Vec3<double>{0.0, 0.0, 40.0};
        const Traced t = trace(r, true, rec.get());
        ASSERT_GE(rec->count, 4);
        EXPECT_EQ(rec->branch[3], mko::kBranchSoftTIR);
        EXPECT_EQ(rec->d[3].z, 0.0);
        EXPECT_TRUE(mko::is_hard_failure(t.status)); // d_z = 0: Backward an Fläche 4
        EXPECT_EQ(rec->count, 4);
        EXPECT_EQ(t.failed, 4);
        const Traced hard = trace(r, false, rec.get());
        EXPECT_EQ(hard.status, TraceStatus::TIR);
        EXPECT_EQ(rec->count, 3);
    }
    // (e) Referenzsphäre verfehlt: Strahl parallel zur Achse in Höhe 10 mm, Sphäre um (0, 0, 1)
    // mit Radius 1 (b² − g < 0): s_ref = b, Zweig SoftMissed
    {
        const HostSurfaceTable t =
            make({SurfaceKind::Stop, SurfaceKind::Plane}, {0.0, 0.0}, {1.0, 1.0}, {1.0, 1.0});
        BatchRay r;
        r.table = t.view();
        r.start = mko::collimated_ray(10.0, 0.0, 1.0, 0.0);
        r.center = Vec3<double>{0.0, 0.0, 1.0};
        const Traced soft = trace(r, true, rec.get());
        EXPECT_EQ(soft.status, TraceStatus::SoftMissed);
        EXPECT_EQ(soft.failed, 2);
        EXPECT_EQ(rec->count, 2);
        EXPECT_EQ(rec->image_branch, mko::kBranchSoftMissed);
        // b = −(p − M)·d mit p = (0, 10, 0) in Koordinaten der letzten Fläche, d = (0, 0, 1)
        EXPECT_EQ(rec->image_s, 1.0);
        const Traced hard = trace(r, false, rec.get());
        EXPECT_EQ(hard.status, TraceStatus::Missed);
        EXPECT_EQ(rec->image_branch, mko::kBranchRegular);
        EXPECT_EQ(rec->image_s, 0.0);
    }
}

// Neu-Vorwärtsrechnen aus Checkpoints: Jeder Abschnitt liefert bitgleich die Zustände der
// vollständigen Aufzeichnung, sein Endzustand (Strahl mit OPL, Fortschritt, Akkumulator) ist
// bitgleich zum nächsten Checkpoint bzw. zum Ende von trace_ray, und die Folge der Zustände in
// absteigender Reihenfolge (for_each_state_reverse) ist für beide Formen dieselbe.
TEST(AdjointRecord, CheckpointReplayIsBitIdentical) {
    const RayCases& cases = ray_cases();
    auto full = std::make_unique<AdjointRecord<double, 64>>();
    auto ck8 = std::make_unique<CheckpointRecord<double, 64, 8>>();
    auto ck4 = std::make_unique<CheckpointRecord<double, 64, 4>>();
    auto ck64 = std::make_unique<CheckpointRecord<double, 64, 64>>();
    auto ck32 = std::make_unique<CheckpointRecord<double, 32, 8>>();
    std::size_t segments = 0;
    std::size_t rays = 0;
    struct State {
        int s;
        Vec3<double> p;
        Vec3<double> d;
        unsigned char b;
    };
    const auto check = [&](const BatchRay& r, bool soft_mode, const auto& rec, const char* name) {
        using Rec = std::remove_cvref_t<decltype(rec)>;
        constexpr int k = Rec::interval();
        ASSERT_EQ(rec.count, full->count) << name;
        ASSERT_EQ(rec.soft_mode, soft_mode);
        // Ende von trace_ray (vor der Referenzsphäre) als Vergleich für den letzten Abschnitt
        Ray<double> end_ray = r.start;
        SoftFailure<double> end_soft{0.0, TraceStatus::Ok};
        mko::NoTraceRecorder* const no_record = nullptr;
        mko::NoSurfaceObserver* const none = nullptr;
        const TraceStatus end_status = mko::trace_ray_recorded(
            r.table, r.l, end_ray, no_record, nullptr, none, soft_mode ? &end_soft : nullptr);
        mko::SegmentStates<double, k> seg{};
        const int last = (rec.count - 1) / k;
        for (int c = 0; c <= last; ++c) {
            mko::TraceCheckpoint<double> end{};
            ASSERT_EQ(mko::replay_segment(r.table, r.l, rec, c, seg, &end), TraceStatus::Ok);
            ++segments;
            ASSERT_EQ(seg.first, c * k);
            ASSERT_EQ(seg.count, std::min(k, rec.count - c * k));
            for (int i = 0; i < seg.count; ++i) {
                const int s = seg.first + i;
                ASSERT_TRUE(same_bits(seg.p[i], full->p[s])) << name << " s=" << s;
                ASSERT_TRUE(same_bits(seg.d[i], full->d[s])) << name << " s=" << s;
            }
            ASSERT_EQ(end.at.surface, seg.first + seg.count - 1);
            if (c < last) {
                // Vom Endzustand des Abschnitts eine Fläche weiter: bitgleich zum nächsten
                // Checkpoint (Strahl mit OPL, Fortschritt, Akkumulator)
                mko::TraceCheckpoint<double> cont = end;
                ASSERT_EQ(mko::trace_segment(r.table, r.l, cont.ray, cont.at, (c + 1) * k + 1,
                                             no_record, nullptr, none,
                                             soft_mode ? &cont.soft : nullptr),
                          TraceStatus::Ok);
                // (min nur gegen eine falsche Warnung von GCC für k = C: dann ist c < last nie)
                const mko::TraceCheckpoint<double>& next =
                    rec.checkpoint[std::min(c + 1, Rec::checkpoints() - 1)];
                ASSERT_TRUE(same_bits(cont.ray, next.ray)) << name;
                ASSERT_EQ(cont.at.surface, next.at.surface);
                ASSERT_EQ(cont.at.ghost, next.at.ghost);
                ASSERT_EQ(cont.at.soft_surface, next.at.soft_surface);
                ASSERT_TRUE(same_bits(cont.soft.penalty, next.soft.penalty));
                ASSERT_EQ(cont.soft.first, next.soft.first);
            } else if (rec.count == r.table.surfaces && !mko::is_hard_failure(end_status)) {
                ASSERT_TRUE(same_bits(end.ray, end_ray)) << name;
                ASSERT_EQ(end.at.surface, r.table.surfaces - 1);
                ASSERT_TRUE(same_bits(end.soft.penalty, end_soft.penalty));
                ASSERT_EQ(end.soft.first, end_soft.first);
            }
        }
        std::vector<State> a;
        std::vector<State> b;
        mko::for_each_state_reverse(r.table, r.l, *full,
                                    [&](int s, const Vec3<double>& p, const Vec3<double>& d,
                                        unsigned char br) { a.push_back({s, p, d, br}); });
        mko::for_each_state_reverse(r.table, r.l, rec,
                                    [&](int s, const Vec3<double>& p, const Vec3<double>& d,
                                        unsigned char br) { b.push_back({s, p, d, br}); });
        ASSERT_EQ(a.size(), b.size());
        ASSERT_EQ(a.size(), static_cast<std::size_t>(full->count));
        for (std::size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(a[i].s, full->count - 1 - static_cast<int>(i));
            ASSERT_EQ(a[i].s, b[i].s);
            ASSERT_TRUE(same_bits(a[i].p, b[i].p));
            ASSERT_TRUE(same_bits(a[i].d, b[i].d));
            ASSERT_EQ(a[i].b, b[i].b);
        }
    };
    for (std::size_t c = 0; c < cases.rays.size(); ++c) {
        for (const bool soft_mode : {false, true}) {
            for (const BatchRay& r : cases.rays[c]) {
                const Traced ref = trace(r, soft_mode, full.get());
                ASSERT_TRUE(same_result(trace(r, soft_mode, ck8.get()), ref));
                ASSERT_TRUE(same_result(trace(r, soft_mode, ck4.get()), ref));
                ASSERT_TRUE(same_result(trace(r, soft_mode, ck64.get()), ref));
                check(r, soft_mode, *ck8, "k = 8");
                check(r, soft_mode, *ck4, "k = 4");
                check(r, soft_mode, *ck64, "k = 64");
                if (r.table.surfaces <= 32) {
                    ASSERT_TRUE(same_result(trace(r, soft_mode, ck32.get()), ref));
                    check(r, soft_mode, *ck32, "C = 32, k = 8");
                }
                for (int s = 0; s < full->count; ++s) {
                    ASSERT_EQ(ck8->branch[s], full->branch[s]);
                }
                ASSERT_TRUE(same_bits(ck8->image_s, full->image_s));
                ASSERT_EQ(ck8->image_branch, full->image_branch);
                ++rays;
            }
        }
    }
    std::printf("[ Info     ] %zu Strahlen, %zu Abschnitte neu vorwärts gerechnet: bitgleich\n",
                rays, segments);
}

// Neuberechnung der lokalen Zwischengrößen aus dem minimalen Satz (Zustand nach Fläche s − 1):
// Ausgang, Status und Zweig bitgleich zur Aufzeichnung; die Zwischengrößen erfüllen die
// geometrischen Beziehungen (Tangentialebene, Sphärengleichung, Einheitsnormale, Snellius in
// Vektorform, √k = d'·N, Fortsetzungen D < 0 bzw. k < 0 mit cos I = 0 bzw. d'·N = 0,
// Geisterplatte gegen einzeln ausgewertete Legendre-Polynome, Referenzsphäre |p − M| = R_ref).
TEST(AdjointRecord, RecomputedLocalQuantitiesFromMinimalSet) {
    const RayCases& cases = ray_cases();
    auto rec = std::make_unique<AdjointRecord<double>>();
    std::size_t steps = 0;
    std::size_t checked = 0;
    std::size_t references = 0;
    double worst_sphere = 0.0;
    double worst_snell = 0.0;
    double worst_ghost = 0.0;
    double worst_reference = 0.0;
    for (std::size_t c = 0; c < cases.rays.size(); ++c) {
        for (const bool soft_mode : {false, true}) {
            for (const BatchRay& r : cases.rays[c]) {
                const SurfaceTableView<double>& t = r.table;
                const int L = t.wavelengths;
                const Traced traced = trace(r, soft_mode, rec.get());
                // Geometrische Beziehungen nur für Strahlen mit moderaten Zahlen (gültig oder
                // weich versagt); bitgleiche Ausgänge für alle aufgezeichneten Flächen.
                const bool relations = mko::in_piston(traced.status);
                int ghost = 0;
                for (int s = 1; s < rec->count; ++s) {
                    ASSERT_EQ(ghost, mko::ghost_index_before(t, s));
                    const mko::SurfaceStepState<double> st = mko::recompute_surface_step(
                        t, r.l, s, ghost, rec->p[s - 1], rec->d[s - 1], soft_mode);
                    ghost += t.type[s] == SurfaceKind::GhostPlate ? 1 : 0;
                    ++steps;
                    ASSERT_EQ(st.status, TraceStatus::Ok) << cases.names[c] << " s=" << s;
                    ASSERT_EQ(st.branch, rec->branch[s]) << cases.names[c] << " s=" << s;
                    ASSERT_TRUE(same_bits(st.p_out, rec->p[s])) << cases.names[c] << " s=" << s;
                    ASSERT_TRUE(same_bits(st.d_out, rec->d[s])) << cases.names[c] << " s=" << s;
                    ASSERT_EQ(st.p_plane.z, 0.0);
                    if (!relations) {
                        continue;
                    }
                    ++checked;
                    const Vec3<double> shifted{st.p_in.x, st.p_in.y, st.p_in.z - st.t};
                    const Vec3<double> plane = shifted + st.s0 * st.d_in;
                    const double scale = 1.0 + max_abs(st.p_in) + std::abs(st.t);
                    EXPECT_LT(std::abs(plane.x - st.p_plane.x), kLengthFp64 * scale);
                    EXPECT_LT(std::abs(plane.y - st.p_plane.y), kLengthFp64 * scale);
                    if (st.kind == SurfaceKind::GhostPlate) {
                        EXPECT_TRUE(same_bits(st.p_out, st.p_plane));
                        const int g = mko::ghost_index_before(t, s);
                        const double rg = t.ghost_norm_radius[g];
                        const double x = st.p_plane.x;
                        const double y = st.p_plane.y;
                        const double u = (x * x + y * y) / (rg * rg);
                        double sum_p = 0.0;
                        double sum_dp = 0.0;
                        for (int k = 1; k <= t.ghost_terms; ++k) {
                            double p = 0.0;
                            double dp = 0.0;
                            mko::legendre(k, 2.0 * u - 1.0, p, dp);
                            sum_p += t.ghost_coeff[g * t.ghost_terms + k - 1] * p;
                            sum_dp += t.ghost_coeff[g * t.ghost_terms + k - 1] * dp;
                        }
                        const double n_g = t.ghost_index[g * L + r.l];
                        const double gscale = 1.0 + std::abs(sum_p) + std::abs(sum_dp);
                        EXPECT_NEAR(st.u, u, kUnitFp64 * (1.0 + u));
                        EXPECT_NEAR(st.profile_t, sum_p, kUnitFp64 * gscale);
                        EXPECT_NEAR(st.profile_dt_du, 2.0 * sum_dp, kUnitFp64 * gscale);
                        EXPECT_NEAR(st.phase, (n_g - st.n_before) * sum_p, kUnitFp64 * gscale);
                        const double gx = (n_g - st.n_before) * 4.0 / (rg * rg) * sum_dp * x;
                        EXPECT_NEAR(st.grad_x, gx, kUnitFp64 * gscale * (1.0 + std::abs(x)));
                        const double e = std::abs(st.n_after * st.d_out.x -
                                                  (st.n_before * st.d_in.x + st.grad_x));
                        worst_ghost = std::max(worst_ghost, e);
                        EXPECT_LT(e, kUnitFp64 * gscale * (1.0 + std::abs(x)));
                        if ((st.branch & mko::kBranchSoftTIR) != 0) {
                            EXPECT_LT(st.radicand, 0.0);
                            EXPECT_EQ(st.d_out.z, 0.0);
                        } else {
                            EXPECT_TRUE(same_bits(st.root_radicand, st.d_out.z));
                        }
                        continue;
                    }
                    // Stop, Plane, Sphere
                    const double cs = st.c;
                    const Vec3<double> q = st.p_out;
                    EXPECT_NEAR(mko::norm(st.normal), 1.0, kUnitFp64);
                    EXPECT_TRUE(same_bits(st.cos_i, mko::dot(st.d_in, st.normal)));
                    const Vec3<double> hit = st.p_plane + st.s * st.d_in;
                    EXPECT_LT(max_abs(hit - q), kLengthFp64 * (1.0 + max_abs(q)));
                    if ((st.branch & mko::kBranchSoftMissed) != 0) {
                        EXPECT_LT(st.D, 0.0);
                        EXPECT_TRUE(same_bits(st.s, st.B / cs));
                        EXPECT_NEAR(st.cos_i, 0.0, kUnitFp64);
                    } else {
                        EXPECT_GE(st.D, 0.0);
                        const double sphere = cs * mko::dot(q, q) - 2.0 * q.z;
                        worst_sphere = std::max(worst_sphere, std::abs(sphere));
                        EXPECT_LT(std::abs(sphere),
                                  kLengthFp64 * (1.0 + std::abs(cs) * mko::dot(q, q)));
                        EXPECT_NEAR(st.root_d * st.root_d, st.D, kUnitFp64 * (1.0 + st.D));
                    }
                    if ((st.branch & mko::kBranchSoftTIR) != 0) {
                        EXPECT_LT(st.k, 0.0);
                        EXPECT_NEAR(mko::dot(st.d_out, st.normal), 0.0, kUnitFp64);
                    } else {
                        EXPECT_GE(st.k, 0.0);
                        EXPECT_NEAR(st.root_k, mko::dot(st.d_out, st.normal), kUnitFp64);
                        const Vec3<double> e = st.n_before * cross(st.d_in, st.normal) -
                                               st.n_after * cross(st.d_out, st.normal);
                        worst_snell = std::max(worst_snell, max_abs(e));
                        EXPECT_LT(max_abs(e), kUnitFp64 * st.n_after);
                    }
                }
                if (rec->count < t.surfaces || !mko::in_piston(traced.status)) {
                    continue;
                }
                const int last = t.surfaces - 1;
                const mko::ReferenceStepState<double> rs = mko::recompute_reference_step(
                    rec->p[last], rec->d[last], r.center, t.index[last * L + r.l], soft_mode);
                ++references;
                ASSERT_TRUE(mko::in_piston(rs.status));
                ASSERT_TRUE(same_bits(rs.s, rec->image_s));
                ASSERT_EQ(rs.branch, rec->image_branch);
                ASSERT_TRUE(same_bits(rs.p_out.y, traced.ray.p.y));
                const double radius = mko::norm(r.center);
                const double quad = rs.s * rs.s - 2.0 * rs.b * rs.s + rs.g;
                if (rs.branch == mko::kBranchRegular) {
                    const double e = std::abs(mko::norm(rs.p_out - r.center) - radius);
                    worst_reference = std::max(worst_reference, e / radius);
                    EXPECT_LT(e, kReferenceSphereFp64 * radius);
                    EXPECT_LT(std::abs(quad), kLengthFp64 * radius * radius);
                    EXPECT_NEAR(rs.root * rs.root, rs.disc, kUnitFp64 * radius * radius);
                } else {
                    EXPECT_LT(rs.disc, 0.0);
                    EXPECT_TRUE(same_bits(rs.s, rs.b));
                }
            }
        }
    }
    std::printf("[ Info     ] %zu Flächenschritte neu berechnet (bitgleich), %zu gegen "
                "Beziehungen geprüft: Sphärengleichung %.2e, Snellius %.2e, Geisterplatte %.2e; "
                "%zu Schritte auf die Referenzsphäre, |p − M| − R relativ %.2e\n",
                steps, checked, worst_sphere, worst_snell, worst_ghost, references,
                worst_reference);
}

// Speicherbedarf pro Strahl (Zahlen für docs/architektur.md): (6·8 + 1)·C Byte plus Kopf für die
// vollständige Aufzeichnung, C/k Checkpoints zu je sizeof(TraceCheckpoint) plus C Zweige.
TEST(AdjointRecord, MemoryPerRay) {
    static_assert(std::is_trivially_copyable_v<AdjointRecord<double, 64>>);
    static_assert(std::is_trivially_copyable_v<CheckpointRecord<double, 64, 8>>);
    static_assert(std::is_trivially_copyable_v<mko::SegmentStates<double, 8>>);
    const auto check_full = [](std::size_t bytes, std::size_t capacity) {
        EXPECT_GE(bytes, 49 * capacity);
        EXPECT_LE(bytes, 49 * capacity + 32);
    };
    check_full(sizeof(AdjointRecord<double, 16>), 16);
    check_full(sizeof(AdjointRecord<double, 32>), 32);
    check_full(sizeof(AdjointRecord<double, 64>), 64);
    EXPECT_EQ(sizeof(mko::TraceCheckpoint<double>), 88u); // Ray 56 + Fortschritt 12 + Strafe 16
    EXPECT_LE(sizeof(CheckpointRecord<double, 64, 8>), 8 * 88u + 64u + 32u);
    std::printf("[ Info     ] Bytes pro Strahl: AdjointRecord C = 16/32/64: %zu/%zu/%zu; "
                "CheckpointRecord k = 8, C = 16/32/64: %zu/%zu/%zu; SegmentStates k = 8: %zu; "
                "TraceRecord: %zu\n",
                sizeof(AdjointRecord<double, 16>), sizeof(AdjointRecord<double, 32>),
                sizeof(AdjointRecord<double, 64>), sizeof(CheckpointRecord<double, 16, 8>),
                sizeof(CheckpointRecord<double, 32, 8>), sizeof(CheckpointRecord<double, 64, 8>),
                sizeof(mko::SegmentStates<double, 8>), sizeof(mko::TraceRecord<double>));
    EXPECT_EQ(mko::record_capacity_class(2), 16);
    EXPECT_EQ(mko::record_capacity_class(16), 16);
    EXPECT_EQ(mko::record_capacity_class(17), 32);
    EXPECT_EQ(mko::record_capacity_class(32), 32);
    EXPECT_EQ(mko::record_capacity_class(33), 64);
    EXPECT_EQ(mko::record_capacity_class(mko::kMaxSurfaces), 64);
}
