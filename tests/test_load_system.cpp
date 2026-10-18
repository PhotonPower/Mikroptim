// Lader und Flächentabelle (M1-T5): Systemdatei → Glaskataloge → Brechzahl- und Flächentabelle.

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

using mko::LoadedSystem;
using mko::SurfaceKind;
using mko::test::kIndexCatalog;
using mko::test::kUnitFp64;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";

} // namespace

TEST(LoadSystem, PlaneParallelPlate) {
    const LoadedSystem ls = mko::load_system(kSystems + "planplatte_bk7.json");
    EXPECT_TRUE(ls.warnings.empty());
    ASSERT_EQ(ls.catalog_files.size(), 1u);
    EXPECT_TRUE(ls.catalog.contains("N-BK7"));
    const auto& t = ls.table;
    ASSERT_EQ(t.surfaces, 3);
    ASSERT_EQ(t.wavelengths, 1);
    EXPECT_EQ(t.type[0], SurfaceKind::Stop);
    EXPECT_EQ(t.type[1], SurfaceKind::Plane);
    EXPECT_EQ(t.type[2], SurfaceKind::Plane);
    EXPECT_EQ(t.thickness[0], 2.0);
    EXPECT_EQ(t.thickness[1], 10.0);
    EXPECT_EQ(t.thickness[2], 20.0);
    EXPECT_EQ(t.semi_diameter[0], 5.0);
    EXPECT_EQ(t.index[0], 1.0);
    EXPECT_NEAR(t.index[1], 1.51680, kIndexCatalog); // N-BK7 bei n_d
    EXPECT_EQ(t.index[2], 1.0);

    const auto v = t.view();
    EXPECT_EQ(v.surfaces, 3);
    EXPECT_EQ(v.wavelengths, 1);
    EXPECT_EQ(v.index, t.index.data());
    EXPECT_EQ(v.type[1], SurfaceKind::Plane);
}

TEST(LoadSystem, SingleLens) {
    const LoadedSystem ls = mko::load_system(kSystems + "einzellinse_bk7.json");
    EXPECT_TRUE(ls.warnings.empty());
    const auto& t = ls.table;
    ASSERT_EQ(t.surfaces, 3);
    ASSERT_EQ(t.wavelengths, 3);
    EXPECT_EQ(t.type[1], SurfaceKind::Sphere);
    EXPECT_EQ(t.curvature[0], 0.0);
    EXPECT_EQ(t.curvature[1], 0.00967);
    EXPECT_EQ(t.curvature[2], -0.00967);
    EXPECT_EQ(t.thickness[2], 99.05);
    for (int l = 0; l < 3; ++l) {
        EXPECT_EQ(t.index[static_cast<std::size_t>(0 * 3 + l)], 1.0);
        EXPECT_EQ(t.index[static_cast<std::size_t>(1 * 3 + l)], ls.index.at(1, l));
        EXPECT_EQ(t.index[static_cast<std::size_t>(2 * 3 + l)], 1.0);
    }
    // Dispersion: n(F) > n(d) > n(C)
    EXPECT_GT(ls.index.at(1, 0), ls.index.at(1, 1));
    EXPECT_GT(ls.index.at(1, 1), ls.index.at(1, 2));
    EXPECT_NEAR(ls.index.at(1, 1), 1.51680, kIndexCatalog);

    // Ganzes System auf der CPU: 3 Felder × 3 Wellenlängen × Stützstellen, kein Versagen.
    const auto q = mko::make_pupil_quadrature(10, 8, mko::PupilMode::Half);
    const mko::CpuTraceResult res = mko::trace_system_cpu(ls.system, ls.table, q);
    EXPECT_EQ(res.fields, 3);
    EXPECT_EQ(res.wavelengths, 3);
    EXPECT_EQ(res.samples, 80);
    EXPECT_EQ(res.rays.size(), 720u);
    EXPECT_EQ(res.failed_count(), 0u);
    for (const auto& r : res.rays) {
        EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
    }
}

TEST(LoadSystem, UnknownGlassIsReportedWithPath) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "planplatte_bk7.json");
    sys.surfaces[1].material.name = "UNOBTAINIUM";
    try {
        (void)mko::prepare_system(sys, MKO_SOURCE_DIR);
        FAIL() << "FormatError erwartet";
    } catch (const mko::FormatError& e) {
        EXPECT_EQ(e.json_path(), "surfaces[1].material");
    }
}

TEST(LoadSystem, MissingCatalogIsReported) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "planplatte_bk7.json");
    sys.glass_catalogs = {"data/glass/gibt_es_nicht.json"};
    EXPECT_THROW((void)mko::prepare_system(sys, MKO_SOURCE_DIR), mko::FormatError);
}

TEST(LoadSystem, WavelengthOutsideGlassRangeIsReported) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "planplatte_bk7.json");
    sys.spectrum.wavelengths_nm = {3000.0}; // N-BK7 gilt bis 2500 nm
    EXPECT_THROW((void)mko::prepare_system(sys, MKO_SOURCE_DIR), mko::GlassError);
}

TEST(LoadSystem, InvalidDescriptionIsRejected) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "planplatte_bk7.json");
    sys.surfaces[0].type = mko::SurfaceType::Plane; // Blende fehlt
    EXPECT_THROW((void)mko::prepare_system(sys, MKO_SOURCE_DIR), mko::FormatError);
}

TEST(SurfaceTable, RejectsTooManySurfaces) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "planplatte_bk7.json");
    while (sys.surfaces.size() <= static_cast<std::size_t>(mko::kMaxSurfaces)) {
        sys.surfaces.push_back(sys.surfaces.back());
    }
    EXPECT_THROW((void)mko::prepare_system(sys, MKO_SOURCE_DIR), std::invalid_argument);
}

TEST(SurfaceTable, RejectsMismatchedIndexTable) {
    const LoadedSystem ls = mko::load_system(kSystems + "planplatte_bk7.json");
    mko::IndexTable wrong = ls.index;
    wrong.surfaces = 2;
    wrong.n.resize(2);
    EXPECT_THROW((void)mko::make_surface_table(ls.system, wrong), std::invalid_argument);
    mko::IndexTable empty;
    empty.surfaces = 3;
    EXPECT_THROW((void)mko::make_surface_table(ls.system, empty), std::invalid_argument);
}

TEST(CpuTrace, PointSourceStartRays) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + "einzellinse_bk7.json");
    sys.source.type = mko::SourceType::Point;
    sys.source.field_angles_deg.clear();
    sys.source.field_weights.clear();
    sys.source.point = {0.0, 0.0, -50.0};
    sys.source.numerical_aperture = 0.1;
    const LoadedSystem ls = mko::prepare_system(sys, MKO_SOURCE_DIR);
    EXPECT_EQ(mko::source_field_count(ls.system), 1);
    const auto q = mko::make_pupil_quadrature(6, 4, mko::PupilMode::Half);
    for (int k = 0; k < mko::sample_count(q); ++k) {
        const auto s = mko::sample(q, k);
        const auto r = mko::source_ray(ls.system, ls.table, 0, 1, s);
        EXPECT_EQ(r.opl, 0.0);
        EXPECT_EQ(r.p.z, -50.0);
        EXPECT_NEAR(std::hypot(r.d.x, r.d.y), 0.1 * s.rho, kUnitFp64); // n = 1 (Luft)
    }
    const mko::CpuTraceResult res = mko::trace_system_cpu(ls.system, ls.table, q);
    EXPECT_EQ(res.fields, 1);
    EXPECT_EQ(res.failed_count(), 0u);
    EXPECT_THROW((void)mko::source_ray(ls.system, ls.table, 1, 0, mko::sample(q, 0)),
                 std::out_of_range);

    sys.source.numerical_aperture = 1.0; // NA >= n
    const LoadedSystem bad = mko::prepare_system(sys, MKO_SOURCE_DIR);
    EXPECT_THROW((void)mko::source_ray(bad.system, bad.table, 0, 0, mko::sample(q, 0)),
                 std::invalid_argument);
}
