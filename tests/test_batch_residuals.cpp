// Residuenvektor pro Variante (M2-T2, CPU-Teil): Ablage ResidualLayout, Pupillenarten als
// Eigenschaft des Batches, CPU-Referenz residuals_cpu gegen evaluate_wavefront (M1-T7) und
// Versagensbehandlung.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "residual_test_systems.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using mko::BatchTopology;
using mko::PupilMode;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SurfaceKind;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::WavefrontMap;
using mko::WavefrontOptions;
using mko::WavefrontResult;

namespace {

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

BatchTopology lens_topology(std::vector<PupilMode> modes) {
    BatchTopology t;
    t.surfaces = 3;
    t.wavelengths = 3;
    t.fields = static_cast<int>(modes.size());
    t.type = {SurfaceKind::Stop, SurfaceKind::Sphere, SurfaceKind::Sphere};
    t.pupil_mode = std::move(modes);
    return t;
}

// Vergleicht residuals_cpu (Variante v von `batch`) mit evaluate_wavefront für `system`:
// Status gleich, OPD gültiger Strahlen bitgleich, Residuum = √(W) · OPD bitgleich, versagte
// Strahlen OPD = Residuum = 0. Liefert die Zahl versagter Strahlen.
std::size_t expect_matches_wavefront(const mko::LoadedSystem& system, const SystemBatchHost& batch,
                                     int v, const ResidualSettings& settings) {
    WavefrontOptions opt;
    opt.rings = settings.rings;
    opt.spokes = settings.spokes;
    const WavefrontResult wf = mko::evaluate_wavefront(system, opt);
    const ResidualLayout layout(batch.topology(), settings);
    const ResidualResult r = mko::residuals_cpu(batch, v, layout);
    EXPECT_EQ(r.variants, 1);
    EXPECT_EQ(r.rays_per_variant, layout.rays_per_variant());
    EXPECT_EQ(r.stride, layout.rays_per_variant());
    EXPECT_EQ(wf.fields, layout.fields());
    EXPECT_EQ(wf.wavelengths, layout.wavelengths());
    std::size_t failed = 0;
    for (int f = 0; f < layout.fields(); ++f) {
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const WavefrontMap& m = wf.at(f, l);
            EXPECT_EQ(m.samples(), layout.samples(f));
            EXPECT_EQ(m.quadrature.mode, layout.pupil_mode(f));
            const std::int64_t o = layout.offset(f, l);
            for (int k = 0; k < m.samples(); ++k) {
                const auto ku = static_cast<std::size_t>(k);
                const std::int64_t i = o + k;
                if (r.status_at(0, i) != m.status[ku]) {
                    ADD_FAILURE() << "Status verschieden: " << f << " " << l << " " << k;
                    continue;
                }
                if (m.status[ku] == TraceStatus::Ok) {
                    EXPECT_TRUE(same_bits(r.opd_at(0, i), m.opd[ku]))
                        << f << " " << l << " " << k << ": " << r.opd_at(0, i) << " vs "
                        << m.opd[ku];
                    EXPECT_TRUE(
                        same_bits(r.residual(0, i), mko::residual_weight(m.pupil[ku]) * m.opd[ku]));
                } else {
                    ++failed;
                    EXPECT_EQ(r.opd_at(0, i), 0.0);
                    EXPECT_EQ(r.residual(0, i), 0.0);
                }
            }
        }
    }
    EXPECT_EQ(failed, wf.failed_count());
    EXPECT_EQ(failed, r.failed_count());
    return failed;
}

// System mit geänderter Beschreibung neu laden (Glaskataloge relativ zu data/systems/).
mko::LoadedSystem modified(mko::SystemDescription system) {
    return mko::prepare_system(std::move(system),
                               std::filesystem::path(mko::test::kResidualSystems));
}

// Dasselbe System mit Punktquelle in q (lokal zu Fläche 0), NA 0,05.
mko::SystemDescription with_point_source(mko::SystemDescription system, std::array<double, 3> q) {
    system.source.type = mko::SourceType::Point;
    system.source.field_angles_deg.clear();
    system.source.field_weights.clear();
    system.source.numerical_aperture = 0.05;
    system.source.point = q;
    return system;
}

} // namespace

// ---- Ablage -------------------------------------------------------------------------------

TEST(ResidualLayout, OffsetsFollowFieldMajorOrder) {
    // Einzellinse: Feld 0 auf der Achse (N = 10), Felder 1, 2 halbe Pupille (N · M = 80)
    const ResidualLayout a(lens_topology({PupilMode::Axis, PupilMode::Half, PupilMode::Half}));
    EXPECT_EQ(a.fields(), 3);
    EXPECT_EQ(a.wavelengths(), 3);
    EXPECT_EQ(a.axis_fields(), 1);
    EXPECT_EQ(a.samples(0), 10);
    EXPECT_EQ(a.samples(1), 80);
    EXPECT_EQ(a.offset(0, 0), 0);
    EXPECT_EQ(a.offset(0, 2), 20);
    EXPECT_EQ(a.offset(1, 0), 30);
    EXPECT_EQ(a.offset(1, 1), 110);
    EXPECT_EQ(a.offset(2, 0), 270);
    EXPECT_EQ(a.offset(2, 2), 430);
    EXPECT_EQ(a.rays_per_variant(), 510);
    EXPECT_EQ(a.rays_per_variant(), mko::rays_per_variant(3, 3, 80, 1, 10));
    ASSERT_EQ(a.plan().size(), 3u);
    EXPECT_EQ(a.plan()[1].offset, 30);
    EXPECT_EQ(a.plan()[1].samples, 80);
    EXPECT_EQ(a.plan()[0].mode, PupilMode::Axis);
    EXPECT_EQ(a.quadrature(0).spokes, 1);
    EXPECT_EQ(a.quadrature(2).spokes, 8);

    // Achsfeld nicht an erster Stelle, volle Pupille, andere Quadratur: die Ablage folgt der
    // Feldreihenfolge, die Gesamtzahl bleibt die Formel von rays_per_variant.
    const ResidualSettings s{6, 5};
    const ResidualLayout b(lens_topology({PupilMode::Half, PupilMode::Axis, PupilMode::Full}), s);
    EXPECT_EQ(b.samples(0), 30);
    EXPECT_EQ(b.samples(1), 6);
    EXPECT_EQ(b.samples(2), 30);
    EXPECT_EQ(b.offset(1, 0), 90);
    EXPECT_EQ(b.offset(1, 2), 102);
    EXPECT_EQ(b.offset(2, 1), 138);
    EXPECT_EQ(b.rays_per_variant(), mko::rays_per_variant(3, 3, 30, 1, 6));
    EXPECT_EQ(b.quadrature(2).mode, PupilMode::Full);
    // Jede Stützstelle hat genau einen Platz: Tupel liegen lückenlos hintereinander
    std::int64_t next = 0;
    for (int f = 0; f < b.fields(); ++f) {
        for (int l = 0; l < b.wavelengths(); ++l) {
            EXPECT_EQ(b.offset(f, l), next);
            next += b.samples(f);
        }
    }
    EXPECT_EQ(next, b.rays_per_variant());
}

TEST(ResidualLayout, InvalidInputThrows) {
    BatchTopology t = lens_topology({});
    t.fields = 3;
    EXPECT_NO_THROW(mko::validate_topology(t)); // nicht festgelegt ist eine gültige Topologie
    EXPECT_THROW(ResidualLayout{t}, std::invalid_argument);
    t.pupil_mode = {PupilMode::Axis, PupilMode::Half}; // zwei Einträge für drei Felder
    EXPECT_THROW(mko::validate_topology(t), std::invalid_argument);
    EXPECT_THROW(ResidualLayout{t}, std::invalid_argument);
    t.pupil_mode = {PupilMode::Axis, PupilMode::Half, static_cast<PupilMode>(7)};
    EXPECT_THROW(mko::validate_topology(t), std::invalid_argument);

    const BatchTopology ok = lens_topology({PupilMode::Axis, PupilMode::Half, PupilMode::Half});
    EXPECT_THROW(ResidualLayout(ok, ResidualSettings{0, 8}), std::invalid_argument);
    EXPECT_THROW(ResidualLayout(ok, ResidualSettings{17, 8}), std::invalid_argument);
    EXPECT_THROW(ResidualLayout(ok, ResidualSettings{10, 0}), std::invalid_argument);
    EXPECT_THROW(ResidualLayout(ok, ResidualSettings{16, 1 << 30}), std::invalid_argument);

    const ResidualLayout layout(ok);
    EXPECT_THROW((void)layout.offset(3, 0), std::out_of_range);
    EXPECT_THROW((void)layout.offset(0, 3), std::out_of_range);
    EXPECT_THROW((void)layout.samples(-1), std::out_of_range);
    EXPECT_NO_THROW(mko::check_residual_layout(ok, layout));
    BatchTopology other = ok;
    other.pupil_mode[1] = PupilMode::Full;
    EXPECT_THROW(mko::check_residual_layout(other, layout), std::invalid_argument);
    other = ok;
    other.wavelengths = 2;
    EXPECT_THROW(mko::check_residual_layout(other, layout), std::invalid_argument);
}

TEST(ResidualLayout, OutputLayoutHasOptionalOpd) {
    const BatchTopology t = lens_topology({PupilMode::Axis, PupilMode::Half, PupilMode::Half});
    mko::OutputRequest out;
    out.rays_per_variant = 510;
    out.merit = false;
    out.surface_max = false;
    const mko::OutputLayout without = mko::make_output_layout(t, 10, out);
    EXPECT_EQ(without.opd.bytes, 0u);
    out.opd = true;
    const mko::OutputLayout with = mko::make_output_layout(t, 10, out);
    EXPECT_EQ(with.opd.bytes, 10u * 510u * sizeof(double));
    EXPECT_EQ(with.opd.offset % mko::kBatchAlignment, 0u);
    EXPECT_EQ(with.total_bytes, without.total_bytes + mko::align_bytes(10u * 510u * 8u));
}

// ---- Pupillenart als Eigenschaft des Batches ---------------------------------------------

TEST(ResidualLayout, TopologyPupilModesFollowWavefrontRule) {
    for (const char* file :
         {"einzellinse_bk7.json", "testsystem_mehrlinser.json", "planplatte_bk7.json"}) {
        const mko::LoadedSystem ls = mko::test::load_residual_system(file);
        const BatchTopology t = mko::topology_of(ls.system, ls.index);
        ASSERT_EQ(t.pupil_mode.size(), static_cast<std::size_t>(t.fields)) << file;
        for (int f = 0; f < t.fields; ++f) {
            EXPECT_EQ(t.pupil_mode[static_cast<std::size_t>(f)],
                      mko::wavefront_pupil_mode(ls.system, f))
                << file << ", Feld " << f;
        }
        EXPECT_EQ(mko::make_system_batch(ls, 2).topology(), t);
    }
    // Punktquelle: Achse → Axis, x = 0 → Half, sonst Full
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    for (const auto& [point, mode] :
         {std::pair{std::array<double, 3>{0.0, 0.0, -80.0}, PupilMode::Axis},
          std::pair{std::array<double, 3>{0.0, 1.0, -80.0}, PupilMode::Half},
          std::pair{std::array<double, 3>{0.5, 1.0, -80.0}, PupilMode::Full}}) {
        const mko::LoadedSystem ls = modified(with_point_source(lens.system, point));
        const BatchTopology t = mko::topology_of(ls.system, ls.index);
        ASSERT_EQ(t.pupil_mode.size(), 1u);
        EXPECT_EQ(t.pupil_mode[0], mode);
    }
}

TEST(ResidualLayout, VariantsMustNotLeaveAxisField) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost b = mko::make_system_batch(ls, 3);
    const ResidualLayout layout(b.topology());
    EXPECT_NO_THROW(mko::check_pupil_modes(b));
    EXPECT_NO_THROW(b.validate_values());
    // Ein Feldpunkt (Half) darf auf die Achse wandern (Feld-Homotopie), das Achsfeld nicht weg.
    b.field_angle(1, 1) = 0.0;
    EXPECT_NO_THROW(mko::check_pupil_modes(b));
    EXPECT_NO_THROW((void)mko::residuals_cpu(b, 1, layout));
    b.field_angle(2, 0) = 1e-3;
    EXPECT_THROW(mko::check_pupil_modes(b), std::invalid_argument);
    EXPECT_NO_THROW(mko::check_pupil_modes(b, 0, 2));
    EXPECT_THROW(mko::check_pupil_modes(b, 2, 1), std::invalid_argument);
    EXPECT_THROW(mko::check_pupil_modes(b, 2, 2), std::out_of_range);
    EXPECT_THROW(b.validate_values(), std::invalid_argument);
    EXPECT_THROW((void)mko::residuals_cpu(b, 2, layout), std::invalid_argument);
    EXPECT_THROW((void)mko::residuals_cpu(b, layout), std::invalid_argument);
    EXPECT_NO_THROW((void)mko::residuals_cpu(b, 0, layout));
    EXPECT_THROW((void)mko::residuals_cpu(b, 3, layout), std::out_of_range);

    // Punktquelle: Quellpunkt muss zur Pupillenart passen
    SystemBatchHost pb =
        mko::make_system_batch(modified(with_point_source(ls.system, {0.0, 0.0, -80.0})), 1);
    EXPECT_NO_THROW(mko::check_pupil_modes(pb));
    pb.source_point[1] = 0.5;
    EXPECT_THROW(mko::check_pupil_modes(pb), std::invalid_argument);
}

// ---- CPU-Referenz gegen M1-T7 ------------------------------------------------------------

// Ungestörtes System: residuals_cpu ist bitgleich zu evaluate_wavefront (OPD) für alle Felder
// und Wellenlängen, für Standard- und andere Quadratur.
TEST(ResidualsCpu, MatchesEvaluateWavefront) {
    for (const char* file : {"einzellinse_bk7.json", "testsystem_mehrlinser.json"}) {
        SCOPED_TRACE(file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(file);
        const SystemBatchHost b = mko::make_system_batch(ls, 2);
        EXPECT_EQ(expect_matches_wavefront(ls, b, 0, ResidualSettings{}), 0u);
        EXPECT_EQ(expect_matches_wavefront(ls, b, 1, ResidualSettings{6, 5}), 0u);
        EXPECT_EQ(expect_matches_wavefront(ls, b, 0, ResidualSettings{16, 12}), 0u);
    }
}

// Punktquelle auf und neben der Achse (Axis, Half, Full).
TEST(ResidualsCpu, MatchesEvaluateWavefrontForPointSource) {
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    for (const std::array<double, 3> q :
         {std::array<double, 3>{0.0, 0.0, -300.0}, std::array<double, 3>{0.0, 2.0, -300.0},
          std::array<double, 3>{1.0, 2.0, -300.0}}) {
        mko::SystemDescription p = with_point_source(lens.system, q);
        p.surfaces.back().thickness = 120.0;
        const mko::LoadedSystem ls = modified(p);
        const SystemBatchHost b = mko::make_system_batch(ls, 1);
        EXPECT_EQ(expect_matches_wavefront(ls, b, 0, ResidualSettings{}), 0u);
    }
}

// Hartes Versagen wie in M1-T7: Blendenradius größer als der Krümmungsradius der Linse, die
// äußeren Ringe verfehlen die Fläche. Kolben und OPD der übrigen sind identisch mit
// evaluate_wavefront (renormierte Gewichte), versagte Strahlen haben Residuum 0.
TEST(ResidualsCpu, FailedRaysAreMaskedLikeWavefront) {
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    mko::SystemDescription sys = lens.system;
    sys.surfaces[0].semi_diameter = 115.0; // R = 103 mm
    const mko::LoadedSystem ls = modified(sys);
    const SystemBatchHost b = mko::make_system_batch(ls, 1);
    const std::size_t failed = expect_matches_wavefront(ls, b, 0, ResidualSettings{});
    EXPECT_GT(failed, 0u);
    const ResidualLayout layout(b.topology());
    EXPECT_LT(failed, static_cast<std::size_t>(layout.rays_per_variant()));
}

// Alle Strahlen eines Tupels versagen: Kolben undefiniert → OPD und Residuen 0, Status gesetzt.
TEST(ResidualsCpu, AllRaysFailedGiveZeroResiduals) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost b = mko::make_system_batch(ls, 2);
    b.pupil_radius(1) = 1e4; // jeder Strahl liegt außerhalb der Sphäre (R = 103 mm)
    const ResidualLayout layout(b.topology());
    const ResidualResult r = mko::residuals_cpu(b, 1, layout);
    ASSERT_EQ(r.failed_count(), static_cast<std::size_t>(layout.rays_per_variant()));
    for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
        EXPECT_NE(r.status_at(0, i), TraceStatus::Ok);
        EXPECT_EQ(r.residual(0, i), 0.0);
        EXPECT_EQ(r.opd_at(0, i), 0.0);
    }
}

// Überladung für alle Varianten = Einzelaufrufe (gleiche Ablage wie der Download vom Gerät).
TEST(ResidualsCpu, AllVariantsMatchSingleCalls) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost b = mko::test::perturbed_batch(ls, 40, 7);
    const ResidualLayout layout(b.topology());
    const ResidualResult all = mko::residuals_cpu(b, layout);
    ASSERT_EQ(all.variants, 40);
    ASSERT_EQ(all.residuals.size(), 40u * static_cast<std::size_t>(layout.rays_per_variant()));
    std::size_t failed = 0;
    for (int v = 0; v < b.variants(); ++v) {
        const ResidualResult one = mko::residuals_cpu(b, v, layout);
        failed += one.failed_count();
        for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
            ASSERT_TRUE(same_bits(all.residual(v, i), one.residual(0, i)));
            ASSERT_TRUE(same_bits(all.opd_at(v, i), one.opd_at(0, i)));
            ASSERT_EQ(all.status_at(v, i), one.status_at(0, i));
        }
    }
    EXPECT_EQ(failed, all.failed_count());
}

// RMS pro Tupel aus OPD und Status (Grundlage für `mikroptim trace --gpu`, M2-T5): für das
// Ergebnis von residuals_cpu bitgleich zu evaluate_wavefront, auch mit versagten Strahlen und
// für Tupel ohne gültigen Strahl (rms NaN).
TEST(ResidualsCpu, TupleWavefrontsMatchEvaluateWavefront) {
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    mko::SystemDescription clipped = lens.system;
    clipped.surfaces[0].semi_diameter = 115.0; // äußere Ringe verfehlen die Linse (R = 103 mm)
    mko::SystemDescription blocked = lens.system;
    blocked.surfaces[0].semi_diameter = 1e4; // alle Strahlen verfehlen
    const mko::LoadedSystem systems[] = {
        mko::test::load_residual_system("testsystem_mehrlinser.json"), lens, modified(clipped),
        modified(blocked)};
    std::size_t failed_total = 0;
    int invalid_tuples = 0;
    int index = 0;
    for (const mko::LoadedSystem& ls : systems) {
        SCOPED_TRACE(index++);
        for (const ResidualSettings settings : {ResidualSettings{}, ResidualSettings{6, 5}}) {
            WavefrontOptions opt;
            opt.rings = settings.rings;
            opt.spokes = settings.spokes;
            const WavefrontResult wf = mko::evaluate_wavefront(ls, opt);
            const SystemBatchHost b = mko::make_system_batch(ls, 2);
            const ResidualLayout layout(b.topology(), settings);
            const ResidualResult r = mko::residuals_cpu(b, layout);
            for (int v = 0; v < 2; ++v) {
                const std::vector<mko::TupleWavefront> t = mko::tuple_wavefronts(r, layout, v);
                ASSERT_EQ(t.size(), wf.maps.size());
                for (std::size_t i = 0; i < t.size(); ++i) {
                    const WavefrontMap& m = wf.maps[i];
                    EXPECT_EQ(t[i].field, m.field);
                    EXPECT_EQ(t[i].wavelength, m.wavelength);
                    EXPECT_EQ(t[i].samples, m.samples());
                    EXPECT_EQ(t[i].failed, m.failed);
                    EXPECT_TRUE(same_bits(t[i].valid_weight, m.valid_weight));
                    EXPECT_EQ(t[i].valid(), m.valid());
                    if (m.valid()) {
                        EXPECT_TRUE(same_bits(t[i].rms, m.rms)) << t[i].rms << " " << m.rms;
                    } else {
                        EXPECT_TRUE(std::isnan(t[i].rms));
                        ++invalid_tuples;
                    }
                    failed_total += static_cast<std::size_t>(t[i].failed);
                }
            }
        }
    }
    EXPECT_GT(failed_total, 0u);
    EXPECT_GT(invalid_tuples, 0);
}

TEST(ResidualsCpu, TupleWavefrontsInvalidInputThrows) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost b = mko::make_system_batch(ls, 1);
    const ResidualLayout layout(b.topology());
    ResidualResult r = mko::residuals_cpu(b, layout);
    EXPECT_THROW((void)mko::tuple_wavefronts(r, layout, 1), std::out_of_range);
    EXPECT_THROW((void)mko::tuple_wavefronts(r, layout, -1), std::out_of_range);
    const ResidualLayout other(b.topology(), ResidualSettings{6, 5});
    EXPECT_THROW((void)mko::tuple_wavefronts(r, other, 0), std::invalid_argument);
    r.opd.clear();
    EXPECT_THROW((void)mko::tuple_wavefronts(r, layout, 0), std::invalid_argument);
}
