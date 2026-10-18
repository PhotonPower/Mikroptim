// Merit und Max-Reduktionen pro Variante (M2-T3, CPU-Teil): merit_cpu gegen die CPU-Residuen,
// Beobachter von trace_ray gegen TraceRecord, Strahlhöhe und Einfallswinkel gegen
// geschlossene Formeln, Versagensbehandlung.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/trace.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>

using mko::MeritResult;
using mko::PupilQuadrature;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::SurfaceExtremaResult;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::kLengthFp64;
using mko::test::kUnitFp64;

namespace {

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// Einzellinse mit den angegebenen Feldwinkeln [Grad].
mko::LoadedSystem lens_with_fields(std::vector<double> angles_deg) {
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    mko::SystemDescription d = lens.system;
    d.source.field_weights.assign(angles_deg.size(), 1.0);
    d.source.field_angles_deg = std::move(angles_deg);
    return mko::prepare_system(d, std::filesystem::path(mko::test::kResidualSystems));
}

} // namespace

// merit_cpu ist die Quadratsumme der CPU-Residuen in der dokumentierten Reihenfolge
// (pro Tupel aufsteigend in k, Tupel f-major); versagte Strahlen wie residuals_cpu.
TEST(MeritCpu, EqualsSumOfCpuResiduals) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 60, 8008);
    const ResidualLayout layout(batch.topology());
    const MeritResult m = mko::merit_cpu(batch, layout, false);
    const ResidualResult r = mko::residuals_cpu(batch, layout);
    ASSERT_EQ(m.variants, 60);
    EXPECT_EQ(m.rays_per_variant, layout.rays_per_variant());
    EXPECT_TRUE(m.extrema.empty());
    std::int64_t total_failed = 0;
    for (int v = 0; v < batch.variants(); ++v) {
        double sum = 0.0;
        std::int64_t failed = 0;
        for (int f = 0; f < layout.fields(); ++f) {
            for (int l = 0; l < layout.wavelengths(); ++l) {
                double tuple = 0.0;
                for (int k = 0; k < layout.samples(f); ++k) {
                    const std::int64_t i = layout.offset(f, l) + k;
                    tuple += r.residual(v, i) * r.residual(v, i);
                    failed += r.status_at(v, i) != TraceStatus::Ok ? 1 : 0;
                }
                sum += tuple;
            }
        }
        EXPECT_TRUE(same_bits(m.merit[static_cast<std::size_t>(v)], sum)) << "v = " << v;
        EXPECT_EQ(m.failed[static_cast<std::size_t>(v)], failed) << "v = " << v;
        total_failed += failed;
        // Einzelaufruf liefert dasselbe
        if (v % 20 == 0) {
            const MeritResult single = mko::merit_cpu(batch, v, layout, true);
            EXPECT_TRUE(same_bits(single.merit[0], sum));
            EXPECT_EQ(single.extrema.variants, 1);
            EXPECT_EQ(single.extrema.surfaces, batch.surfaces());
        }
    }
    EXPECT_GT(total_failed, 0) << "Versagenspfad nicht geprüft";
    EXPECT_THROW((void)mko::merit_cpu(batch, 60, layout), std::out_of_range);
    EXPECT_THROW((void)mko::surface_extrema_cpu(batch, -1, layout), std::out_of_range);
}

// Der Beobachter ändert den Trace nicht (bitgleicher Endzustand) und liefert pro Fläche
// dieselben Werte, die aus TraceRecord folgen (p[s], d[s − 1], Normale aus der Krümmung).
TEST(SurfaceExtrema, ObserverMatchesTraceRecord) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const mko::SurfaceTableView<double> table = ls.table.view();
    const int S = table.surfaces;
    const auto record = std::make_unique<mko::TraceRecord<double>>();
    const auto observed = std::make_unique<mko::RaySurfaceRecord<double>>();
    for (const double theta : {0.0, 0.03, 0.05}) {
        for (const double rho : {0.0, 0.4, 0.95}) {
            const mko::Ray<double> start =
                mko::collimated_ray(4.0, theta, rho, 0.7); // a = Blendenradius
            mko::Ray<double> plain = start;
            mko::Ray<double> with_record = start;
            mko::Ray<double> with_observer = start;
            const TraceStatus s0 = mko::trace_ray(table, 1, plain, nullptr);
            const TraceStatus s1 = mko::trace_ray(table, 1, with_record, record.get());
            int failed = 0;
            const TraceStatus s2 =
                mko::trace_ray(table, 1, with_observer, nullptr, &failed, observed.get());
            ASSERT_EQ(s0, TraceStatus::Ok);
            ASSERT_EQ(s1, s0);
            ASSERT_EQ(s2, s0);
            EXPECT_EQ(failed, -1);
            EXPECT_TRUE(same_bits(with_observer.opl, plain.opl));
            EXPECT_TRUE(same_bits(with_observer.p.y, plain.p.y));
            EXPECT_TRUE(same_bits(with_observer.d.z, plain.d.z));
            ASSERT_EQ(record->count, S);
            for (int s = 0; s < S; ++s) {
                const Vec3<double> p = record->p[s];
                const double c =
                    s == 0 ? 0.0 : mko::surface_curvature(table.type[s], table.curvature[s]);
                const Vec3<double> d = s == 0 ? record->d[0] : record->d[s - 1];
                const Vec3<double> n{-c * p.x, -c * p.y, 1.0 - c * p.z};
                EXPECT_TRUE(same_bits(observed->height_sq[s], p.x * p.x + p.y * p.y)) << s;
                EXPECT_TRUE(same_bits(observed->cos_incidence[s], mko::dot(d, n))) << s;
                // Normale ist ein Einheitsvektor, cos I also ein Kosinus
                EXPECT_NEAR(mko::norm(n), 1.0, kUnitFp64) << s;
                EXPECT_LE(observed->cos_incidence[s], 1.0 + kUnitFp64);
            }
            EXPECT_TRUE(same_bits(observed->cos_incidence[0], start.d.z));
        }
    }
}

// Geschlossene Formeln (kollimiertes Bündel auf der Achse, Einzellinse): An der Blende ist die
// größte Höhe a·ρ_N (äußerster Ring) und cos I = 1; an der ersten Sphäre (achsparallele
// Strahlen) dieselbe Höhe und cos I = √(1 − (c·h)²). Mit Feldwinkeln bis 2° ist cos I an der
// Blende cos 2°.
TEST(SurfaceExtrema, CollimatedBeamMatchesClosedForm) {
    {
        const SystemBatchHost batch = mko::make_system_batch(lens_with_fields({0.0}), 1);
        const ResidualLayout layout(batch.topology());
        const SurfaceExtremaResult e = mko::surface_extrema_cpu(batch, 0, layout);
        const PupilQuadrature<double>& q = layout.quadrature(0);
        const double h = batch.pupil_radius(0) * std::sqrt(q.u[q.rings - 1]);
        const double c = batch.curvature(0, 1);
        EXPECT_NEAR(e.max_height(0, 0), h, kLengthFp64);
        EXPECT_EQ(e.min_cos_incidence(0, 0), 1.0);
        EXPECT_EQ(e.max_incidence_angle(0, 0), 0.0);
        EXPECT_NEAR(e.max_height(0, 1), h, kLengthFp64);
        EXPECT_NEAR(e.min_cos_incidence(0, 1), std::sqrt(1.0 - c * h * c * h), kUnitFp64);
        EXPECT_NEAR(e.max_incidence_angle(0, 1), std::asin(c * h), 1e-12);
        // Hinter der Bikonvexlinse konvergiert das Bündel: kleinere Höhe an Fläche 2
        EXPECT_LT(e.max_height(0, 2), h);
        EXPECT_GT(e.max_height(0, 2), 0.9 * h);
    }
    {
        const SystemBatchHost batch = mko::make_system_batch(lens_with_fields({0.0, 1.0, 2.0}), 1);
        const ResidualLayout layout(batch.topology());
        const SurfaceExtremaResult e = mko::surface_extrema_cpu(batch, 0, layout);
        EXPECT_NEAR(e.min_cos_incidence(0, 0), std::cos(2.0 * std::acos(-1.0) / 180.0), kUnitFp64);
        const PupilQuadrature<double>& q = layout.quadrature(0);
        EXPECT_NEAR(e.max_height(0, 0), batch.pupil_radius(0) * std::sqrt(q.u[q.rings - 1]),
                    kLengthFp64);
    }
}

// Versagte Strahlen tragen nicht bei: Variante ohne gültigen Strahl behält die neutralen
// Elemente (Höhe 0, cos I = 1) und hat die Merit 0; bei teilweisem Versagen ist die
// Blendenhöhe das Maximum von a·ρ_k über die gültigen Strahlen.
TEST(SurfaceExtrema, FailedRaysDoNotContribute) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost batch = mko::make_system_batch(ls, 3);
    batch.pupil_radius(1) = 1e4;   // alle Strahlen verfehlen die Linse (R = 103 mm)
    batch.pupil_radius(2) = 110.0; // äußere Ringe verfehlen
    const ResidualLayout layout(batch.topology());
    const MeritResult m = mko::merit_cpu(batch, layout, true);
    const ResidualResult r = mko::residuals_cpu(batch, layout);
    const int S = batch.surfaces();

    EXPECT_EQ(m.failed[0], 0);
    EXPECT_EQ(m.failed[1], layout.rays_per_variant());
    EXPECT_EQ(m.merit[1], 0.0);
    for (int s = 0; s < S; ++s) {
        EXPECT_EQ(m.extrema.max_height(1, s), 0.0);
        EXPECT_EQ(m.extrema.min_cos_incidence(1, s), 1.0);
    }

    EXPECT_GT(m.failed[2], 0);
    EXPECT_LT(m.failed[2], layout.rays_per_variant());
    double stop = 0.0;
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            for (int k = 0; k < layout.samples(f); ++k) {
                if (r.status_at(2, layout.offset(f, l) + k) == TraceStatus::Ok) {
                    stop = std::max(stop, batch.pupil_radius(2) * mko::sample(q, k).rho);
                }
            }
        }
    }
    EXPECT_NEAR(m.extrema.max_height(2, 0), stop, kLengthFp64);
    EXPECT_LT(m.extrema.max_height(2, 0), 110.0 * std::sqrt(layout.quadrature(0).u[9]));
    // Gültige Strahlen schneiden die Sphäre R = 1/c: Höhe < R
    EXPECT_LT(m.extrema.max_height(2, 1), 1.0 / batch.curvature(2, 1));
    EXPECT_GT(m.extrema.max_incidence_angle(2, 1), m.extrema.max_incidence_angle(0, 1));
}
