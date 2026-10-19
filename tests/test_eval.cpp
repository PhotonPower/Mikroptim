// Modus EVAL (M8-T1) auf der CPU: Raster, Referenzsphäre durch die Austrittspupille,
// CPU-Referenz und Kennzahlen. Analytische Orakel: aplanatische Punkte (perfekte Kugelwelle),
// Seidel-Koeffizient der Einzellinse und Gauß-Quadratur (M1-T7), exakte Verzeichnung und
// Coddington-Gleichungen einer Einzelfläche, Petzvalschale, paraxiale Bildhöhe. Dieselben
// Prüfungen laufen auf der GPU in test_cuda_eval.cpp.

#include "eval_test_systems.hpp"
#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/eval_ray.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using mko::EvalGrid;
using mko::EvalReference;
using mko::EvalResult;
using mko::EvalSettings;
using mko::EvalTupleStats;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::kLengthFp64;
using mko::test::kUnitFp64;

namespace {

const mko::test::EvalFn kCpu = [](const SystemBatchHost& b, const EvalSettings& s) {
    return mko::eval_cpu(b, s);
};

const mko::LoadedSystem& lc_system() {
    static const mko::LoadedSystem ls =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    return ls;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Raster, Einstellungen, Speicher
// ---------------------------------------------------------------------------------------------

TEST(EvalGrid, PointsWeightsAndSymmetry) {
    // N = 1: nur die Pupillenmitte; N = 2: vier Punkte (±1/2, ±1/2)
    const EvalGrid g1 = mko::make_eval_grid(1, false);
    ASSERT_EQ(g1.samples(), 1);
    EXPECT_EQ(g1.points[0].x, 0.0);
    EXPECT_EQ(g1.points[0].y, 0.0);
    EXPECT_EQ(g1.points[0].weight, 1.0);
    const EvalGrid g2 = mko::make_eval_grid(2, false);
    ASSERT_EQ(g2.samples(), 4);
    EXPECT_EQ(g2.points[0].x, -0.5);
    EXPECT_EQ(g2.points[0].y, -0.5);
    EXPECT_EQ(g2.points[3].x, 0.5);
    EXPECT_EQ(g2.points[3].y, 0.5);

    for (const int N : {3, 8, 17, 64, 255}) {
        const EvalGrid full = mko::make_eval_grid(N, false);
        const EvalGrid half = mko::make_eval_grid(N, true);
        EXPECT_EQ(full.full_count, full.samples());
        EXPECT_EQ(half.full_count, full.full_count);
        double sum_full = 0.0;
        for (std::size_t k = 0; k < full.points.size(); ++k) {
            const auto& p = full.points[k];
            const long long ci = 2LL * full.column[k] + 1 - N;
            const long long rj = 2LL * full.row[k] + 1 - N;
            EXPECT_LE(ci * ci + rj * rj, static_cast<long long>(N) * N);
            EXPECT_EQ(p.x, (2.0 * full.column[k] + 1.0 - N) / N);
            EXPECT_EQ(p.y, (2.0 * full.row[k] + 1.0 - N) / N);
            sum_full += p.weight;
            if (k > 0) { // zeilenweise
                EXPECT_TRUE(full.row[k] > full.row[k - 1] || (full.row[k] == full.row[k - 1] &&
                                                              full.column[k] > full.column[k - 1]));
            }
        }
        double sum_half = 0.0;
        int zero_column = 0;
        for (const auto& p : half.points) {
            EXPECT_GE(p.x, 0.0);
            sum_half += p.weight;
            zero_column += p.x == 0.0 ? 1 : 0;
            EXPECT_EQ(p.weight, (p.x > 0.0 ? 2.0 : 1.0) / full.full_count);
        }
        EXPECT_NEAR(sum_full, 1.0, mko::test::kEvalWeightSum);
        EXPECT_NEAR(sum_half, 1.0, mko::test::kEvalWeightSum);
        EXPECT_EQ(2 * half.samples() - zero_column, full.samples());
        EXPECT_EQ(zero_column > 0, N % 2 == 1);
        // Spiegelsymmetrie des vollen Rasters (x → −x exakt)
        for (std::size_t k = 0; k < full.points.size(); ++k) {
            const auto mirror =
                std::find_if(full.points.begin(), full.points.end(), [&](const mko::EvalPoint& q) {
                    return q.x == -full.points[k].x && q.y == full.points[k].y;
                });
            EXPECT_NE(mirror, full.points.end());
            if (N > 17) {
                break; // eine Stichprobe genügt für große N
            }
        }
        std::printf("[ Info     ] Raster N = %3d: %6lld Punkte (π N²/4 = %.1f), halbe Pupille "
                    "%lld\n",
                    N, static_cast<long long>(full.samples()), 0.25 * 3.14159265358979 * N * N,
                    static_cast<long long>(half.samples()));
    }
    EXPECT_THROW(mko::make_eval_grid(0, false), std::invalid_argument);
    EXPECT_THROW(mko::make_eval_grid(mko::kMaxEvalGrid + 1, false), std::invalid_argument);
}

TEST(EvalSettings, InvalidSettingsThrow) {
    EvalSettings s;
    EXPECT_NO_THROW(mko::check_eval_settings(s));
    s.grid = 0;
    EXPECT_THROW(mko::check_eval_settings(s), std::invalid_argument);
    s.grid = 32;
    s.field_ray_offset = 0.0;
    EXPECT_THROW(mko::check_eval_settings(s), std::invalid_argument);
    s.field_ray_offset = 0.6;
    EXPECT_THROW(mko::check_eval_settings(s), std::invalid_argument);
    s.field_ray_offset = 1e-3;
    s.reference = static_cast<EvalReference>(7);
    EXPECT_THROW(mko::check_eval_settings(s), std::invalid_argument);
    s.reference = EvalReference::Vertex;

    // Halbe Pupille bei einer Punktquelle mit x ≠ 0: keine Symmetrie
    mko::test::AplanaticCase ac = mko::test::aplanatic_case(1.5, 0.0);
    s.half_pupil = true;
    EXPECT_NO_THROW(mko::check_eval_settings(s, ac.batch));
    ac.batch.source_point[0] = 0.1;
    EXPECT_THROW(mko::check_eval_settings(s, ac.batch), std::invalid_argument);
    EXPECT_THROW(mko::eval_cpu(ac.batch, s), std::invalid_argument);

    // Austrittspupille ohne Brechzahlen bei λ_ref (von Hand gebauter Batch): Fehler
    mko::test::AplanaticCase bad = mko::test::aplanatic_case(1.5, 0.0);
    bad.batch.paraxial_index(0, 0) = 0.0;
    s = EvalSettings{};
    EXPECT_THROW(mko::eval_cpu(bad.batch, s), std::invalid_argument);
    s.reference = EvalReference::Vertex; // braucht keine Paraxialrechnung
    EXPECT_NO_THROW(mko::eval_cpu(bad.batch, s));
}

// Speicherabschätzung: Batch + Ausgaben, jeder Bereich ausgerichtet (docs/architektur.md).
TEST(EvalMemory, DeviceBytesFollowLayout) {
    const SystemBatchHost b = mko::make_system_batch(
        mko::load_system(MKO_SOURCE_DIR "/data/systems/testsystem_mehrlinser.json"));
    const mko::BatchTopology& t = b.topology();
    const std::int64_t n = mko::make_eval_grid(64, false).samples();
    const int V = 7;
    const mko::EvalOutputLayout o = mko::make_eval_output_layout(t, V, n);
    const auto tuples = static_cast<std::size_t>(V * t.fields * t.wavelengths);
    const auto rays = tuples * static_cast<std::size_t>(n);
    EXPECT_EQ(o.opd.bytes, rays * 8);
    EXPECT_EQ(o.spot_x.bytes, rays * 8);
    EXPECT_EQ(o.spot_y.bytes, rays * 8);
    EXPECT_EQ(o.status.bytes, rays);
    EXPECT_EQ(o.field_rays.bytes, tuples * 4 * 5 * 8);
    EXPECT_EQ(o.field_status.bytes, tuples * 4);
    EXPECT_EQ(o.reference_curvature.bytes, static_cast<std::size_t>(V * t.fields) * 8);
    EXPECT_EQ(o.points.bytes, static_cast<std::size_t>(n) * sizeof(mko::EvalPoint));
    std::size_t sum = 0;
    for (const mko::BufferRegion* r : {&o.opd, &o.spot_x, &o.spot_y, &o.status, &o.field_rays,
                                       &o.field_status, &o.reference_curvature, &o.points}) {
        EXPECT_EQ(r->offset % mko::kBatchAlignment, 0u);
        EXPECT_EQ(r->offset, sum);
        sum += mko::align_bytes(r->bytes);
    }
    EXPECT_EQ(o.total_bytes, sum);
    EXPECT_EQ(mko::eval_device_bytes(t, V, n),
              mko::make_batch_layout(t, V).total_bytes + o.total_bytes);
    EXPECT_THROW(mko::make_eval_output_layout(t, -1, n), std::invalid_argument);
    EXPECT_THROW(mko::make_eval_output_layout(t, 1, 0), std::invalid_argument);

    // Beispiel für die Doku: Zielsystem-Topologie S = 30, L = 7, F = 5, G = 6, K = 8, N = 128
    mko::BatchTopology z;
    z.surfaces = 30;
    z.wavelengths = 7;
    z.fields = 5;
    z.ghost_plates = 6;
    z.ghost_terms = 8;
    z.type.assign(30, mko::SurfaceKind::Sphere);
    z.type[0] = mko::SurfaceKind::Stop;
    for (int g = 0; g < 6; ++g) {
        z.type[static_cast<std::size_t>(3 + 4 * g)] = mko::SurfaceKind::GhostPlate;
    }
    const std::int64_t n128 = mko::make_eval_grid(128, false).samples();
    const double per_variant = static_cast<double>(mko::eval_device_bytes(z, 100, n128)) / 100.0;
    std::printf("[ Info     ] Zielsystem-Topologie, N = 128 (%lld Punkte): ca. %.1f MB pro "
                "Variante, ca. %.0f Varianten in 16 GiB\n",
                static_cast<long long>(n128), per_variant / 1e6, 16.0 * (1 << 30) / per_variant);
}

// ---------------------------------------------------------------------------------------------
// Pro-Strahl-Teil
// ---------------------------------------------------------------------------------------------

// Referenzsphäre durch die Austrittspupille: Mit E im Scheitel (κ = σ/|M|) ist der Schritt bis
// auf den konstanten Anteil σ n R derselbe wie reference_sphere_step; für κ → 0 stetig.
TEST(EvalRay, ExitPupilStepMatchesVertexSphere) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double max_diff = 0.0;
    for (const double w : {80.0, 3.0, -40.0}) {
        for (const double h : {0.0, 0.7, -2.0}) {
            const Vec3<double> M{0.0, h, w};
            const double R = mko::norm(M);
            const double sigma = w >= 0.0 ? 1.0 : -1.0;
            const double n = 1.3;
            for (int i = 0; i < 200; ++i) {
                const Vec3<double> p{2.0 * u(rng), 2.0 * u(rng), 0.3 * u(rng)};
                Vec3<double> to = M + Vec3<double>{0.05 * u(rng), 0.05 * u(rng), 0.05 * u(rng)};
                Vec3<double> d = sigma > 0.0 ? to - p : p - to;
                d = (1.0 / mko::norm(d)) * d;
                mko::Ray<double> a{p, d, 10.0};
                mko::Ray<double> b = a;
                ASSERT_EQ(mko::reference_sphere_step(a, M, n), TraceStatus::Ok);
                ASSERT_EQ(mko::exit_pupil_sphere_step(b, M, sigma / R, n), TraceStatus::Ok);
                const double diff = std::abs((b.opl - sigma * n * R) - a.opl);
                EXPECT_LT(diff, mko::test::kReferenceSphereFp64);
                max_diff = std::max(max_diff, diff);
                // Grenzfall κ → 0 (telezentrisch): stetig, s + σR → b = −(p − M)·d, und für
                // kleines κ weicht der Schritt um höchstens n κ |t|/2 <= n κ |p − M|²/2 ab
                mko::Ray<double> c{p, d, 0.0};
                mko::Ray<double> e{p, d, 0.0};
                const double kappa = 1e-9;
                ASSERT_EQ(mko::exit_pupil_sphere_step(c, M, 0.0, n), TraceStatus::Ok);
                ASSERT_EQ(mko::exit_pupil_sphere_step(e, M, kappa, n), TraceStatus::Ok);
                EXPECT_EQ(c.opl, -n * mko::dot(p - M, d));
                const double q2 = mko::dot(p - M, p - M);
                EXPECT_LE(std::abs(e.opl - c.opl), 0.5 * n * kappa * q2 * (1.0 + kUnitFp64));
            }
        }
    }
    // Strahl verfehlt eine kleine Sphäre: Missed, OPL unverändert
    mko::Ray<double> miss{{5.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 2.0};
    EXPECT_EQ(mko::exit_pupil_sphere_step(miss, Vec3<double>{0.0, 0.0, 10.0}, 1.0, 1.0),
              TraceStatus::Missed);
    EXPECT_EQ(miss.opl, 2.0);
    std::printf("[ Info     ] Schritt durch die Austrittspupille gegen reference_sphere_step "
                "(E im Scheitel): max |ΔOPL| = %.3e mm\n",
                max_diff);
}

// Lage der Referenzsphäre: z_AP aus der Paraxialrechnung, κ = σ/|M − E| pro Feld.
TEST(EvalRay, ReferenceGeometryFromExitPupil) {
    const mko::LoadedSystem ls =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    const SystemBatchHost b = mko::make_system_batch(ls);
    const mko::ParaxialData p = mko::compute_paraxial(ls.system, ls.catalog);
    const mko::EvalReferenceGeometry geo =
        mko::eval_reference_geometry(b, EvalReference::ExitPupil);
    ASSERT_EQ(geo.exit_pupil_position.size(), 1u);
    EXPECT_NEAR(geo.exit_pupil_position[0], p.exit_pupil_position,
                mko::test::kParaxialRelative * std::abs(p.exit_pupil_position));
    const double w = b.thickness(0, b.surfaces() - 1);
    for (int f = 0; f < b.fields(); ++f) {
        const double R = std::hypot(b.field_target(0, f), w - p.exit_pupil_position);
        EXPECT_NEAR(geo.curvature[static_cast<std::size_t>(f)] * R, 1.0,
                    mko::test::kParaxialRelative);
    }
    const mko::EvalReferenceGeometry vtx = mko::eval_reference_geometry(b, EvalReference::Vertex);
    EXPECT_TRUE(std::isnan(vtx.exit_pupil_position[0]));
    EXPECT_DOUBLE_EQ(vtx.curvature[1], 1.0 / std::hypot(b.field_target(0, 1), w));
    std::printf("[ Info     ] Einzellinse: Austrittspupille bei z = %.6f mm (letzte Fläche), "
                "R_ref = %.6f mm statt %.6f mm durch den Scheitel\n",
                p.exit_pupil_position, 1.0 / geo.curvature[0], w);
}

// Lage Scheitel: dieselben Strahlen wie M1-T7 (Achsfächer der Gauß-Quadratur) ergeben
// denselben RMS wie evaluate_wavefront.
TEST(Eval, VertexReferenceReproducesWavefrontOfM1T7) {
    mko::LoadedSystem ls = mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    const SystemBatchHost b = mko::make_system_batch(ls);
    const mko::WavefrontResult wf = mko::evaluate_wavefront(ls);
    for (int l = 0; l < b.wavelengths(); ++l) {
        const mko::test::GaussWavefront g =
            mko::test::gauss_wavefront(b, 0, 0, l, EvalReference::Vertex, 10, 4);
        EXPECT_EQ(g.rms, wf.at(0, l).rms) << "l=" << l; // dieselben Strahlen und Summen
    }
}

// ---------------------------------------------------------------------------------------------
// Abnahme (analytische Orakel)
// ---------------------------------------------------------------------------------------------

TEST(Eval, AplanaticPointSourceHasVanishingOpd) {
    mko::test::check_aplanatic(kCpu, "CPU");
}

TEST(Eval, SingleLensGridMatchesSeidelAndGaussModel) {
    mko::test::check_lens_grid(kCpu, "CPU");
}

TEST(Eval, GridRmsConvergesToGaussQuadrature) {
    mko::test::check_grid_convergence(kCpu, "CPU", 256);
}

TEST(Eval, DistortionAndFieldCurvatureOfSingleSurface) {
    mko::test::check_single_surfaces(kCpu, "CPU");
}

// (4) Schwerpunkt des Spots bei kleinem Feld gegen die paraxiale Bildhöhe f_λ tan θ
// (Einzellinse, Bildebene im paraxialen Fokus jeder Wellenlänge, f_λ paraxiale Brennweite bei
// λ). Die Abweichung ist Koma (∝ NA²) und Verzeichnung (∝ θ²) der dritten Ordnung.
TEST(Eval, SpotCentroidAtSmallFieldMatchesParaxialImageHeight) {
    const mko::LoadedSystem ls =
        mko::load_system(MKO_SOURCE_DIR "/data/systems/einzellinse_bk7.json");
    std::vector<double> deviation; // [a][l]
    for (const double a : {2.5, 1.25}) {
        const mko::test::LensCase lc = mko::test::lens_case(a, {0.05});
        EvalSettings s;
        s.grid = 64;
        const EvalResult r = mko::eval_cpu(lc.batch, s);
        ASSERT_EQ(r.failed_count(), 0u);
        for (int l = 0; l < lc.batch.wavelengths(); ++l) {
            const EvalTupleStats st = mko::eval_statistics(r, l)[static_cast<std::size_t>(l)];
            const double f_l = mko::compute_paraxial(ls.system, ls.index, l).focal_length;
            const double h_par = f_l * std::tan(lc.batch.field_angle(l, 0));
            const double y = st.centroid_y + st.image_height;
            const double rel = std::abs(y - h_par) / h_par;
            EXPECT_LT(rel, mko::test::kEvalCentroidParaxial) << "a=" << a << " l=" << l;
            EXPECT_NEAR(st.centroid_x, 0.0, kLengthFp64);
            deviation.push_back((y - h_par) / h_par);
            std::printf("[ Info     ] Einzellinse a = %.2f mm, θ = 0,05°, l = %d: Schwerpunkt "
                        "%.9e mm, paraxial %.9e mm, rel. %.2e; Hauptstrahl rel. %.2e\n",
                        a, l, y, h_par, rel,
                        std::abs(st.chief_y + st.image_height - h_par) / h_par);
        }
    }
    // Koma der dritten Ordnung: halbe Apertur, ein Viertel der Abweichung
    for (std::size_t l = 0; l < 3; ++l) {
        EXPECT_NEAR(deviation[l] / deviation[l + 3], 4.0, 4.0 * mko::test::kEvalComaScaling);
    }
}

// ---------------------------------------------------------------------------------------------
// Verhalten
// ---------------------------------------------------------------------------------------------

// Halbe Pupille: dieselben Kennzahlen wie die volle (Symmetrie x → −x), gespiegelte Karte.
TEST(Eval, HalfPupilMatchesFullPupil) {
    const SystemBatchHost b = mko::make_system_batch(
        mko::load_system(MKO_SOURCE_DIR "/data/systems/testsystem_mehrlinser.json"));
    double max_diff = 0.0;
    for (const int N : {40, 41}) {
        EvalSettings s;
        s.grid = N;
        const EvalResult full = mko::eval_cpu(b, s);
        s.half_pupil = true;
        const EvalResult half = mko::eval_cpu(b, s);
        EXPECT_EQ(half.failed_count(), 0u);
        const auto a = mko::eval_statistics(full, 0);
        const auto h = mko::eval_statistics(half, 0);
        ASSERT_EQ(a.size(), h.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double tol = mko::test::kEvalHalfPupil;
            EXPECT_NEAR(h[i].rms_opd, a[i].rms_opd, tol);
            EXPECT_NEAR(h[i].rms_opd_tilt, a[i].rms_opd_tilt, tol);
            EXPECT_NEAR(h[i].rms_opd_defocus, a[i].rms_opd_defocus, tol);
            EXPECT_NEAR(h[i].pv_opd, a[i].pv_opd, tol);
            EXPECT_EQ(h[i].centroid_x, 0.0); // gespiegelte Paare heben sich exakt auf
            EXPECT_NEAR(a[i].centroid_x, 0.0, tol);
            EXPECT_NEAR(h[i].centroid_y, a[i].centroid_y, tol);
            EXPECT_NEAR(h[i].rms_spot, a[i].rms_spot, tol);
            EXPECT_NEAR(h[i].defocus, a[i].defocus, tol);
            EXPECT_NEAR(h[i].tilt_y, a[i].tilt_y, tol);
            max_diff = std::max({max_diff, std::abs(h[i].rms_opd - a[i].rms_opd),
                                 std::abs(h[i].rms_opd_defocus - a[i].rms_opd_defocus),
                                 std::abs(h[i].centroid_y - a[i].centroid_y),
                                 std::abs(h[i].rms_spot - a[i].rms_spot)});
            // Feldstrahlen hängen nicht vom Raster ab: bitgleich (NaN auf dem Achsfeld)
            EXPECT_EQ(h[i].focus_tangential, a[i].focus_tangential);
            EXPECT_EQ(h[i].focus_sagittal, a[i].focus_sagittal);
            EXPECT_TRUE(h[i].distortion_chief == a[i].distortion_chief ||
                        (std::isnan(h[i].distortion_chief) && std::isnan(a[i].distortion_chief)));
        }
        const mko::EvalMap mf = mko::eval_opd_map(full, 0, 1, 1);
        const mko::EvalMap mh = mko::eval_opd_map(half, 0, 1, 1);
        ASSERT_EQ(mf.values.size(), mh.values.size());
        for (std::size_t i = 0; i < mf.values.size(); ++i) {
            ASSERT_EQ(std::isnan(mf.values[i]), std::isnan(mh.values[i])) << i;
            if (!std::isnan(mf.values[i])) {
                EXPECT_NEAR(mh.values[i], mf.values[i], mko::test::kEvalHalfPupil);
                max_diff = std::max(max_diff, std::abs(mh.values[i] - mf.values[i]));
            }
        }
    }
    std::printf("[ Info     ] Halbe gegen volle Pupille (Mehrlinser, N = 40 und 41): max "
                "Abweichung der Kennzahlen und Karten %.3e mm\n",
                max_diff);
}

// OPD-Karte: NaN außerhalb der Pupille; nach Abzug von Kippung bzw. Defokus ist ihr RMS der der
// Kennzahlen, und eine verschobene Bildebene erscheint als Defokus.
TEST(Eval, OpdMapRemovesTiltAndDefocus) {
    // Variante 1 hat die Bildebene im paraxialen Fokus von λ_1; die Kopie mit um 0,2 mm
    // verschobener Bildebene ist defokussiert (Fokus vor der Bildebene).
    const mko::test::LensCase lc = mko::test::lens_case(5.0, {0.0, 1.0});
    SystemBatchHost shifted = lc.batch;
    const int S = shifted.surfaces();
    shifted.thickness(1, S - 1) += 0.2;
    EvalSettings s;
    s.grid = 48;
    const EvalResult r0 = mko::eval_cpu(lc.batch, s);
    const EvalResult r1 = mko::eval_cpu(shifted, s);
    const int v = 1;
    const int l = 1;
    const double f_ref =
        mko::compute_paraxial(lc_system().system, lc_system().index, l).focal_length;
    const auto st0 = mko::eval_statistics(r0, v);
    const auto st1 = mko::eval_statistics(r1, v);
    for (const EvalResult* r : {&r0, &r1}) {
        const auto& st = r == &r0 ? st0 : st1;
        for (int f = 0; f < 2; ++f) {
            const EvalTupleStats& t = st[static_cast<std::size_t>(f * 3 + l)];
            for (const mko::OpdRemoval removal :
                 {mko::OpdRemoval::Piston, mko::OpdRemoval::Tilt, mko::OpdRemoval::Defocus}) {
                const mko::EvalMap m = mko::eval_opd_map(*r, v, f, l, removal);
                double sq = 0.0;
                int count = 0;
                for (int j = 0; j < m.size; ++j) {
                    for (int i = 0; i < m.size; ++i) {
                        const double x = (2.0 * i + 1.0 - m.size) / m.size;
                        const double y = (2.0 * j + 1.0 - m.size) / m.size;
                        if (x * x + y * y > 1.0) {
                            EXPECT_TRUE(std::isnan(m.at(i, j)));
                            continue;
                        }
                        sq += m.at(i, j) * m.at(i, j);
                        ++count;
                    }
                }
                ASSERT_EQ(count, r->samples());
                const double rms = std::sqrt(sq / count);
                const double expected = removal == mko::OpdRemoval::Piston ? t.rms_opd
                                        : removal == mko::OpdRemoval::Tilt ? t.rms_opd_tilt
                                                                           : t.rms_opd_defocus;
                EXPECT_NEAR(rms, expected, kUnitFp64 * t.rms_opd);
            }
            EXPECT_LE(t.rms_opd_defocus, t.rms_opd_tilt);
            EXPECT_LE(t.rms_opd_tilt, t.rms_opd * (1.0 + kUnitFp64));
        }
    }
    for (int f = 0; f < 2; ++f) {
        const EvalTupleStats& t0 = st0[static_cast<std::size_t>(f * 3 + l)];
        const EvalTupleStats& t1 = st1[static_cast<std::size_t>(f * 3 + l)];
        // Defokus dominiert; nach seinem Abzug bleibt die sphärische Aberration (bis auf den
        // ρ⁴-Anteil des Defokus bei endlicher NA, relativ ~NA²/4 des Defokus).
        EXPECT_GT(t1.rms_opd, 5.0 * t1.rms_opd_defocus);
        EXPECT_NEAR(t1.rms_opd_defocus, t0.rms_opd_defocus, 0.05 * t0.rms_opd_defocus);
        EXPECT_LT(t1.defocus, 0.0); // Fokus vor der Bildebene: Rand eilt voraus
        // Zusätzlicher Defokus n δz (1 − cos U) mit sin U = ρ NA: ρ²-Koeffizient des Fits
        // n δz (NA²/2 + NA⁴/8) (der ρ⁴-Anteil projiziert mit 1 auf ρ²), NA = sin atan(a/f)
        const double na = std::sin(std::atan(5.0 / f_ref));
        const double expected = -0.2 * (0.5 * na * na + 0.125 * na * na * na * na);
        EXPECT_NEAR((t1.defocus - t0.defocus) / expected, 1.0, mko::test::kEvalDefocusModel);
        // Die Brennlinien rücken um die Verschiebung der Bildebene
        EXPECT_NEAR(t1.focus_sagittal - t0.focus_sagittal, -0.2, mko::test::kEvalFocusRelative);
        EXPECT_NEAR(t1.focus_tangential - t0.focus_tangential, -0.2, mko::test::kEvalFocusRelative);
        std::printf("[ Info     ] Einzellinse, Feld %d: RMS %.4e mm, ohne Defokus %.4e mm; mit "
                    "0,2 mm Defokus %.4e bzw. %.4e mm, Änderung des Defokus-Koeffizienten "
                    "%.6e mm (n δz (NA²/2 + NA⁴/8) = %.6e)\n",
                    f, t0.rms_opd, t0.rms_opd_defocus, t1.rms_opd, t1.rms_opd_defocus,
                    t1.defocus - t0.defocus, expected);
    }
}

// Hartes Versagen: Strahlen, die die Fläche verfehlen, haben Status Missed und NaN in OPD und
// Spot; Kolben und Kennzahlen nur über die gültigen Strahlen.
TEST(Eval, FailedRaysAreHardAndExcluded) {
    const mko::LoadedSystem ls = mko::test::load_eval_system(
        R"({"format": "mikroptim.system/1", "name": "Versagen", "units": "mm",
            "glass_catalogs": [],
            "spectrum": {"wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56},
            "source": {"type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0]},
            "target": {"focal_length": 18.0},
            "surfaces": [
              {"type": "stop", "semi_diameter": 8.0, "thickness": 2.0, "material": "AIR"},
              {"type": "sphere", "curvature": 0.16666666666666666, "thickness": 18.0,
               "material": {"model": "constant", "n": 1.5}, "semi_diameter": 9.0}]})");
    const SystemBatchHost b = mko::make_system_batch(ls);
    EvalSettings s;
    s.grid = 40;
    s.reference = EvalReference::Vertex;
    const EvalResult r = mko::eval_cpu(b, s);
    std::int64_t failed = 0;
    double mean = 0.0;
    double w_ok = 0.0;
    for (std::int64_t k = 0; k < r.samples(); ++k) {
        const auto& p = r.grid.points[static_cast<std::size_t>(k)];
        const std::size_t i = r.ray_index(0, 0, 0, k);
        const double h = 8.0 * std::hypot(p.x, p.y);
        if (h > 6.0) {
            EXPECT_EQ(static_cast<TraceStatus>(r.status[i]), TraceStatus::Missed);
        }
        if (static_cast<TraceStatus>(r.status[i]) != TraceStatus::Ok) {
            ++failed;
            EXPECT_TRUE(std::isnan(r.opd[i]));
            EXPECT_TRUE(std::isnan(r.spot_x[i]));
            EXPECT_TRUE(std::isnan(r.spot_y[i]));
        } else {
            mean += p.weight * r.opd[i];
            w_ok += p.weight;
        }
    }
    ASSERT_GT(failed, 0);
    ASSERT_LT(failed, r.samples());
    EXPECT_EQ(r.failed_count(), static_cast<std::size_t>(failed));
    const EvalTupleStats st = mko::eval_statistics(r, 0)[0];
    EXPECT_EQ(st.failed, failed);
    EXPECT_NEAR(st.valid_weight, w_ok, kUnitFp64);
    EXPECT_NEAR(mean / w_ok, 0.0, kLengthFp64);
    EXPECT_TRUE(std::isfinite(st.rms_opd));
    EXPECT_TRUE(std::isfinite(st.rms_spot));
    // Die Feldstrahlen liegen nahe der Mitte und versagen nicht
    EXPECT_EQ(r.field_ray_status(0, 0, 0, mko::kEvalChiefRay), TraceStatus::Ok);
    EXPECT_TRUE(std::isnan(st.distortion_chief)); // Achsfeld
}
