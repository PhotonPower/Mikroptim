// Weiches Strahlversagen (M4-T2, CPU; docs/numerik.md, Abschnitt 11): Fortsetzungsregeln der
// Flächenoperationen gegen geometrische Orakel, Strafgrößen und Status im Trace, Stetigkeit der
// Merit entlang einer fein abgetasteten Geraden durch eine Versagensgrenze (im harten Modus zum
// Vergleich ein Sprung), Abnahme-Optimierung eines Startsystems mit etwa 30 % versagten
// Strahlen und obere Schranke der Vorfilterung im weichen Modus.

#include "fd_test_systems.hpp"
#include "lm_test_systems.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/trace.hpp"
#include "residual_test_systems.hpp"
#include "soft_failure_test_tools.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using mko::FailureMode;
using mko::MeritResult;
using mko::Ray;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SoftFailure;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::Vec3;
using mko::test::ContinuityReport;
using mko::test::kContinuityNeighborRatio;
using mko::test::kContinuityShrink;
using mko::test::kLengthFp64;
using mko::test::kUnitFp64;

namespace {

constexpr double kDeg = std::numbers::pi / 180.0;

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

SoftFailure<double> fresh() {
    return SoftFailure<double>{0.0, TraceStatus::Ok};
}

ResidualSettings with_mode(FailureMode mode, ResidualSettings s = {}) {
    s.failure = mode;
    return s;
}

// Strahl in der Tangentialebene (z = 0) bei Höhe y, Richtung (0, sin θ, cos θ).
Ray<double> meridional(double y, double theta) {
    return Ray<double>{{0.0, y, 0.0}, {0.0, std::sin(theta), std::cos(theta)}, 0.0};
}

} // namespace

// ---- Flächenoperationen --------------------------------------------------------------------

// Fläche verfehlt (D = B² − cH < 0): Strafgröße −D, Fortsetzung im Punkt der Geraden mit dem
// kleinsten Abstand zum Krümmungsmittelpunkt C = (0, 0, 1/c), (p − C)·d = 0, |s| < Höhe; auch
// für B <= 0 endlich. Stetig an der Grenze (Berührpunkt der Tangente). Ohne Akkumulator wie
// bisher Missed ohne Änderung des Strahls.
TEST(SoftFailure, MissedSurfaceContinuation) {
    int cases = 0;
    int nonpositive_b = 0;
    for (const double c : {0.1, -0.1, 0.5, -0.02}) {
        const double R = 1.0 / std::abs(c);
        for (const double deg : {0.0, 20.0, -30.0, 70.0}) {
            for (const double frac : {0.3, 0.9, 1.01, 1.5, 3.0}) {
                const double y = frac * R;
                Ray<double> r = meridional(y, deg * kDeg);
                const double H = c * (r.p.x * r.p.x + r.p.y * r.p.y);
                const double B = r.d.z - c * (r.p.x * r.d.x + r.p.y * r.d.y);
                const double D = B * B - c * H;
                if (!(D < 0.0)) {
                    continue;
                }
                ++cases;
                nonpositive_b += B <= 0.0 ? 1 : 0;
                Ray<double> hard = r;
                double s_hard = -7.0;
                EXPECT_EQ(mko::intersect_sphere(hard, c, s_hard), TraceStatus::Missed);
                EXPECT_EQ(hard.p.y, r.p.y);
                EXPECT_EQ(s_hard, -7.0);

                SoftFailure<double> soft = fresh();
                double s = 0.0;
                ASSERT_EQ(mko::intersect_sphere(r, c, s, &soft), TraceStatus::SoftMissed);
                EXPECT_TRUE(same_bits(soft.penalty, -D));
                EXPECT_EQ(soft.first, TraceStatus::SoftMissed);
                EXPECT_TRUE(std::isfinite(s));
                EXPECT_LT(std::abs(s), y);
                const Vec3<double> q{r.p.x, r.p.y, r.p.z - 1.0 / c};
                EXPECT_NEAR(mko::dot(q, r.d), 0.0, kLengthFp64 * R);
                // Normale im fortgesetzten Punkt normiert, cos I = 0
                Ray<double> refr = r;
                SoftFailure<double> soft2 = fresh();
                ASSERT_EQ(mko::refract_sphere(refr, c, 1.0 / 1.5, &soft2, true), TraceStatus::Ok);
                EXPECT_EQ(soft2.penalty, 0.0); // μ < 1: keine Totalreflexion bei cos I = 0
                EXPECT_NEAR(mko::norm(refr.d), 1.0, kUnitFp64);
                const Vec3<double> nraw{-c * r.p.x, -c * r.p.y, 1.0 - c * r.p.z};
                const Vec3<double> n = (1.0 / mko::norm(nraw)) * nraw;
                EXPECT_NEAR(mko::dot(r.d, n), 0.0, kUnitFp64);
                // d' = μ d + √(1 − μ²) N
                const double mu = 1.0 / 1.5;
                const Vec3<double> expected = mu * r.d + std::sqrt(1.0 - mu * mu) * n;
                EXPECT_NEAR(refr.d.y, expected.y, kUnitFp64);
                EXPECT_NEAR(refr.d.z, expected.z, kUnitFp64);
            }
        }
    }
    EXPECT_GT(cases, 10);
    EXPECT_GT(nonpositive_b, 0) << "Fall B <= 0 nicht geprüft";

    // Stetigkeit an der Grenze: kollimierter Strahl, c = 0,1 (R = 10), Grenze bei y = R.
    const double c = 0.1;
    for (const double delta : {1e-6, 1e-9, 1e-12}) {
        Ray<double> inside = meridional(10.0 * (1.0 - delta), 0.0);
        Ray<double> outside = meridional(10.0 * (1.0 + delta), 0.0);
        double si = 0.0;
        double so = 0.0;
        SoftFailure<double> soft = fresh();
        ASSERT_EQ(mko::intersect_sphere(inside, c, si, &soft), TraceStatus::Ok);
        EXPECT_EQ(soft.penalty, 0.0);
        ASSERT_EQ(mko::intersect_sphere(outside, c, so, &soft), TraceStatus::SoftMissed);
        // Abstand der Punkte O(R √(2δ)) (Wurzel der Sphäre am Rand), Strafe 2δ + δ²
        EXPECT_LT(std::abs(so - si), 3.0 * 10.0 * std::sqrt(2.0 * delta));
        EXPECT_NEAR(soft.penalty, 2.0 * delta, 1e-15 + 4.0 * delta * delta);
    }
}

// Totalreflexion (k < 0): Strafgröße −k, Fortsetzung mit k := 0 und normierter Richtung, also
// tangential zur Fläche (d'·N = 0, |d'| = 1). Stetig an der Grenze k = 0. An einer Ebene ist
// danach d_z = 0 und die nächste Prüfung meldet Backward.
TEST(SoftFailure, TotalReflectionContinuation) {
    const double n = 1.5;
    const double mu = n / 1.0; // Glas → Luft
    for (const double c : {0.0, 0.05, -0.08}) {
        for (const double y : {0.0, 2.0, -3.0}) {
            for (const double deg : {45.0, 60.0, -70.0, 85.0}) {
                // Punkt auf der Sphäre in Höhe y
                Ray<double> r = meridional(y, deg * kDeg);
                r.p.z = c == 0.0 ? 0.0 : (1.0 - std::sqrt(1.0 - c * c * y * y)) / c;
                const Vec3<double> N{-c * r.p.x, -c * r.p.y, 1.0 - c * r.p.z};
                const double cos_i = mko::dot(r.d, N);
                const double k = 1.0 - mu * mu * (1.0 - cos_i * cos_i);
                if (!(k < 0.0)) {
                    continue;
                }
                Ray<double> hard = r;
                EXPECT_EQ(mko::refract_sphere(hard, c, mu), TraceStatus::TIR);
                SoftFailure<double> soft = fresh();
                ASSERT_EQ(mko::refract_sphere(r, c, mu, &soft), TraceStatus::SoftTIR);
                EXPECT_TRUE(same_bits(soft.penalty, -k));
                EXPECT_EQ(soft.first, TraceStatus::SoftTIR);
                EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
                EXPECT_NEAR(mko::dot(r.d, N), 0.0, kUnitFp64);
                // Richtung der Projektion von d auf die Tangentialebene
                EXPECT_GT(mko::dot(r.d, hard.d - cos_i * N), 0.0);
            }
        }
    }
    // Stetigkeit an der Grenze (Ebene): sin θ* = 1/n
    const double theta_c = std::asin(1.0 / n);
    for (const double delta : {1e-6, 1e-10}) {
        Ray<double> below = meridional(0.0, theta_c * (1.0 - delta));
        Ray<double> above = meridional(0.0, theta_c * (1.0 + delta));
        SoftFailure<double> soft = fresh();
        ASSERT_EQ(mko::refract_sphere(below, 0.0, mu, &soft), TraceStatus::Ok);
        EXPECT_EQ(soft.penalty, 0.0);
        ASSERT_EQ(mko::refract_sphere(above, 0.0, mu, &soft), TraceStatus::SoftTIR);
        EXPECT_GT(soft.penalty, 0.0);
        EXPECT_LT(soft.penalty, 10.0 * delta);
        // Richtungen nähern sich wie √|k|
        const double dy = std::abs(above.d.y - below.d.y);
        const double dz = std::abs(above.d.z - below.d.z);
        EXPECT_LT(dy + dz, 10.0 * std::sqrt(delta));
        EXPECT_EQ(above.d.z, 0.0);
        // danach versagt der Strahl an der nächsten Fläche hart
        EXPECT_EQ(mko::propagate_surface(above, 5.0, mko::SurfaceKind::Plane, 0.0, 1.0, 1.0,
                                         nullptr, &soft),
                  TraceStatus::Backward);
    }
}

// Geisterplatte: Radikand < 0 → Strafgröße −Radikand, d' = (d'_x, d'_y, 0)/|d'_⊥|, danach
// hartes Versagen (Backward) an der nächsten Prüfung von d_z.
TEST(SoftFailure, GhostPlateContinuation) {
    Ray<double> r{{0.3, 0.4, 0.0}, {0.0, 0.6, 0.8}, 1.0};
    const double gx = 0.2;
    const double gy = 0.5;
    const double dx = (r.d.x + gx);
    const double dy = (r.d.y + gy);
    const double radicand = 1.0 - dx * dx - dy * dy;
    ASSERT_LT(radicand, 0.0);
    Ray<double> hard = r;
    EXPECT_EQ(mko::phase_surface_step(hard, 0.01, gx, gy, 1.0, 1.0), TraceStatus::TIR);
    SoftFailure<double> soft = fresh();
    ASSERT_EQ(mko::phase_surface_step(r, 0.01, gx, gy, 1.0, 1.0, &soft), TraceStatus::SoftTIR);
    EXPECT_TRUE(same_bits(soft.penalty, -radicand));
    EXPECT_EQ(soft.first, TraceStatus::SoftTIR);
    EXPECT_EQ(r.d.z, 0.0);
    EXPECT_NEAR(mko::norm(r.d), 1.0, kUnitFp64);
    EXPECT_NEAR(r.d.x / r.d.y, dx / dy, kUnitFp64);
    EXPECT_EQ(r.opl, 1.01);
    EXPECT_EQ(
        mko::propagate_surface(r, 2.0, mko::SurfaceKind::Plane, 0.0, 1.0, 1.0, nullptr, &soft),
        TraceStatus::Backward);
}

// Referenzsphäre verfehlt (b² − g < 0): Strafgröße −(b² − g)/R_ref², Fortsetzung im Punkt
// kleinsten Abstands zu M (s = b, danach (p − M)·d = 0), OPL += n·b; stetig an der Grenze.
TEST(SoftFailure, ReferenceSphereContinuation) {
    const Vec3<double> center{0.0, 0.0, 2.0}; // R_ref = 2
    for (const double y : {2.5, 4.0, -3.0}) {
        Ray<double> r{{0.0, y, 0.0}, {0.0, 0.0, 1.0}, 10.0};
        const Vec3<double> q = r.p - center;
        const double b = -mko::dot(q, r.d);
        const double g = mko::dot(r.p, r.p - 2.0 * center);
        const double disc = b * b - g;
        ASSERT_LT(disc, 0.0);
        Ray<double> hard = r;
        EXPECT_EQ(mko::reference_sphere_step(hard, center, 1.0), TraceStatus::Missed);
        SoftFailure<double> soft = fresh();
        double s = 0.0;
        ASSERT_EQ(mko::reference_sphere_step(r, center, 1.0, &s, &soft), TraceStatus::SoftMissed);
        EXPECT_EQ(s, b);
        EXPECT_TRUE(same_bits(soft.penalty, -disc / 4.0));
        EXPECT_NEAR(soft.penalty, (y * y - 4.0) / 4.0, kUnitFp64);
        EXPECT_NEAR(mko::dot(r.p - center, r.d), 0.0, kLengthFp64);
        EXPECT_EQ(r.opl, 10.0 + b);
    }
    // Grenze y = R_ref: beide Seiten treffen denselben Punkt im Grenzwert
    for (const double delta : {1e-6, 1e-10}) {
        Ray<double> in{{0.0, 2.0 * (1.0 - delta), 0.0}, {0.0, 0.0, 1.0}, 0.0};
        Ray<double> out{{0.0, 2.0 * (1.0 + delta), 0.0}, {0.0, 0.0, 1.0}, 0.0};
        SoftFailure<double> soft = fresh();
        ASSERT_EQ(mko::reference_sphere_step(in, center, 1.0, nullptr, &soft), TraceStatus::Ok);
        ASSERT_EQ(mko::reference_sphere_step(out, center, 1.0, nullptr, &soft),
                  TraceStatus::SoftMissed);
        EXPECT_LT(std::abs(out.opl - in.opl), 3.0 * 2.0 * std::sqrt(2.0 * delta));
    }
    // R_ref = 0 bleibt hart
    Ray<double> r{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, 0.0};
    SoftFailure<double> soft = fresh();
    EXPECT_EQ(mko::reference_sphere_step(r, Vec3<double>{0.0, 0.0, 0.0}, 1.0, nullptr, &soft),
              TraceStatus::Missed);
}

// Trace aller Strahlen gestörter Varianten (Mehrlinser, Einzellinse mit Geisterplatten) im
// harten und im weichen Modus: Gültige Strahlen sind bitgleich (OPL) ohne Strafe; ein im harten
// Modus an Fläche s verfehlter oder totalreflektierter Strahl ist im weichen Modus weich (erstes
// Ereignis an derselben Fläche und von derselben Art) oder hart an einer späteren Prüfung;
// Backward und Unsupported bleiben unverändert und ohne Strafe.
TEST(SoftFailure, TraceMatchesHardModeAndReportsFirstEvent) {
    std::int64_t ok = 0;
    std::int64_t soft_count = 0;
    std::int64_t soft_then_hard = 0;
    std::int64_t backward = 0;
    for (const char* file : {"testsystem_mehrlinser.json", "einzellinse_geisterplatte.json"}) {
        SCOPED_TRACE(file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(file);
        const SystemBatchHost batch = mko::test::perturbed_batch(ls, 80, 4711);
        const ResidualLayout layout(batch.topology());
        const mko::SystemBatchView<double> view = batch.view();
        for (int v = 0; v < batch.variants(); ++v) {
            const mko::SurfaceTableView<double> table = view.surface_table(v);
            for (int f = 0; f < layout.fields(); ++f) {
                const Vec3<double> center = mko::batch_image_point(view, table, v, f);
                for (int l = 0; l < layout.wavelengths(); ++l) {
                    for (int k = 0; k < layout.samples(f); ++k) {
                        const mko::PupilSample<double> s = mko::sample(layout.quadrature(f), k);
                        Ray<double> rh = mko::batch_source_ray(view, v, f, l, s);
                        Ray<double> rs = rh;
                        int fh = -2;
                        int fs = -2;
                        const TraceStatus sh =
                            mko::trace_to_reference_sphere(table, l, rh, center, &fh);
                        SoftFailure<double> acc = fresh();
                        mko::NoSurfaceObserver* const none = nullptr;
                        const TraceStatus ss =
                            mko::trace_to_reference_sphere(table, l, rs, center, &fs, none, &acc);
                        switch (sh) {
                        case TraceStatus::Ok:
                            ++ok;
                            ASSERT_EQ(ss, TraceStatus::Ok);
                            ASSERT_TRUE(same_bits(rs.opl, rh.opl));
                            ASSERT_EQ(acc.penalty, 0.0);
                            ASSERT_EQ(fs, -1);
                            break;
                        case TraceStatus::Missed:
                        case TraceStatus::TIR: {
                            const TraceStatus kind = sh == TraceStatus::Missed
                                                         ? TraceStatus::SoftMissed
                                                         : TraceStatus::SoftTIR;
                            if (mko::is_soft_failure(ss)) {
                                ++soft_count;
                                ASSERT_EQ(ss, kind);
                                ASSERT_EQ(fs, fh);
                                ASSERT_GT(acc.penalty, 0.0);
                                ASSERT_TRUE(std::isfinite(rs.opl));
                            } else {
                                ++soft_then_hard;
                                ASSERT_TRUE(mko::is_hard_failure(ss));
                                ASSERT_GE(fs, fh);
                                // erstes Ereignis weich (Strafe > 0), außer der Fall D >= 0,
                                // B + √D <= 0 (nur durch Rundung) an derselben Fläche
                                ASSERT_TRUE(acc.first == kind || (fs == fh && ss == sh));
                            }
                            break;
                        }
                        default:
                            ++backward;
                            ASSERT_EQ(ss, sh);
                            ASSERT_EQ(fs, fh);
                            ASSERT_EQ(acc.penalty, 0.0);
                            break;
                        }
                    }
                }
            }
        }
    }
    std::printf("[ Info     ] gültig %lld, weich %lld, weich und danach hart %lld, Backward/"
                "Unsupported %lld\n",
                static_cast<long long>(ok), static_cast<long long>(soft_count),
                static_cast<long long>(soft_then_hard), static_cast<long long>(backward));
    EXPECT_GT(soft_count, 0);
    EXPECT_GT(soft_then_hard, 0);
    EXPECT_GT(backward, 0);
}

// Strafresiduen: p_k = √(w_fail ŵ_f ŵ_λ W_k) · π_k mit π_k = Strafe (weich) bzw. Strafe + 1
// (hart), 0 für gültige Strahlen; Wellenfrontresiduen weich versagter Strahlen sind Teil des
// Kolbens (≠ 0), hart versagter 0. w_fail skaliert die Strafresiduen linear in √w_fail.
TEST(SoftFailure, PenaltyResidualsFollowDefinition) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 40, 99);
    ResidualSettings settings;
    settings.fail_weight = 2.5e5;
    const ResidualLayout layout(batch.topology(), settings);
    ResidualSettings quarter = settings;
    quarter.fail_weight = settings.fail_weight / 4.0;
    const ResidualResult r = mko::residuals_cpu(batch, layout);
    const ResidualResult r4 = mko::residuals_cpu(batch, ResidualLayout(batch.topology(), quarter));
    const mko::SystemBatchView<double> view = batch.view();
    std::int64_t soft = 0;
    std::int64_t hard = 0;
    for (int v = 0; v < batch.variants(); ++v) {
        const mko::SurfaceTableView<double> table = view.surface_table(v);
        for (int f = 0; f < layout.fields(); ++f) {
            const Vec3<double> center = mko::batch_image_point(view, table, v, f);
            for (int l = 0; l < layout.wavelengths(); ++l) {
                const double tw = mko::tuple_weight(view, f, l);
                for (int k = 0; k < layout.samples(f); ++k) {
                    const std::int64_t i = layout.offset(f, l) + k;
                    const mko::PupilSample<double> s = mko::sample(layout.quadrature(f), k);
                    SoftFailure<double> acc = fresh();
                    double opl = 0.0;
                    mko::NoSurfaceObserver* const none = nullptr;
                    const TraceStatus st =
                        mko::batch_ray_opl(view, table, l, v, f, l, s, center, &opl, none, &acc);
                    ASSERT_EQ(st, r.status_at(v, i));
                    double pi = 0.0;
                    if (mko::is_soft_failure(st)) {
                        pi = acc.penalty;
                        ++soft;
                    } else if (mko::is_hard_failure(st)) {
                        pi = acc.penalty + mko::kHardFailurePenalty;
                        ++hard;
                        EXPECT_EQ(r.residual(v, i), 0.0);
                    }
                    const double expected =
                        std::sqrt(settings.fail_weight * tw) * std::sqrt(s.weight) * pi;
                    EXPECT_NEAR(r.penalty(v, i), expected, 1e-14 * std::abs(expected));
                    EXPECT_NEAR(r4.penalty(v, i), 0.5 * r.penalty(v, i),
                                1e-14 * std::abs(r.penalty(v, i)));
                    EXPECT_TRUE(same_bits(r4.residual(v, i), r.residual(v, i)));
                }
            }
        }
    }
    EXPECT_GT(soft, 0);
    EXPECT_GT(hard, 0);
}

// ---- Stetigkeit der Merit (Abnahme) --------------------------------------------------------

// Totalreflexion: Plankonvexlinse, der äußerste Ring tritt bei x₀ knapp unter dem Grenzwinkel
// aus (tir_problem, M3-T2). Gerade durch die Grenze in der hinteren Krümmung c₂: 10⁴ Punkte
// über 6·10⁻⁵ relativ, danach das Intervall des Übergangs noch einmal mit 10⁴ Punkten. Weich:
// Kein Nachbarsprung ragt heraus, und der Sprung am Übergang fällt mit der Intervallbreite
// (mindestens wie √Δt: auf der gültigen Seite geht die Richtung des gebrochenen Strahls mit √k).
// Eine weitere Verfeinerung erreicht die Auflösung von c₂ (ulp ≈ 1e-16 relativ), wo die Rundung
// von √k nahe k = 0 die Merit um ~1e-3 streut. Hart: Der Sprung bleibt (der Strahl verlässt
// Kolben und Residuen).
TEST(MeritContinuity, AcrossTotalReflectionCpu) {
    const mko::test::FailureLine line = mko::test::total_reflection_line();
    const ContinuityReport soft = mko::test::continuity_across_failure(
        line.base, line.set, with_mode(FailureMode::Soft), line.t0, line.t1, 10000, 1,
        mko::test::cpu_merit_evaluator);
    const ContinuityReport hard = mko::test::continuity_across_failure(
        line.base, line.set, with_mode(FailureMode::Hard), line.t0, line.t1, 10000, 1,
        mko::test::cpu_merit_evaluator);
    mko::test::print_report("Totalreflexion, weich", soft);
    mko::test::print_report("Totalreflexion, hart", hard);
    ASSERT_TRUE(soft.crossed);
    ASSERT_EQ(soft.jumps.size(), 2u);
    EXPECT_LT(soft.max_ratio, kContinuityNeighborRatio);
    EXPECT_LT(soft.jumps[1], kContinuityShrink * soft.jumps[0]);
    EXPECT_LT(soft.jumps[1], 1e-5 * soft.merit_scale);
    ASSERT_EQ(hard.jumps.size(), 2u);
    EXPECT_GT(hard.max_ratio, 10.0 * kContinuityNeighborRatio);
    EXPECT_GT(hard.jumps[1], 0.5 * hard.jumps[0]);
    EXPECT_GT(hard.jumps[1], 0.1 * hard.merit_scale);
}

// Fläche verfehlt: Meniskus aus dem Startsystem (c₂ = +0,06, damit streifend eintretende Strahlen
// die Rückfläche passieren), Gerade in c₁ durch die Grenze c₁ = 1/y des äußersten Strahls.
// Auswertung wie bei der Totalreflexion (Wurzelsingularität s = H/(B + √D) auf der gültigen
// Seite, glatte Fortsetzung s = B/c auf der anderen).
TEST(MeritContinuity, AcrossMissedSurfaceCpu) {
    const mko::test::FailureLine line = mko::test::missed_surface_line();
    const ContinuityReport soft = mko::test::continuity_across_failure(
        line.base, line.set, with_mode(FailureMode::Soft), line.t0, line.t1, 10000, 1,
        mko::test::cpu_merit_evaluator);
    const ContinuityReport hard = mko::test::continuity_across_failure(
        line.base, line.set, with_mode(FailureMode::Hard), line.t0, line.t1, 10000, 1,
        mko::test::cpu_merit_evaluator);
    mko::test::print_report("Fläche verfehlt, weich", soft);
    mko::test::print_report("Fläche verfehlt, hart", hard);
    // Auf der weichen Seite ist der Strahl weich versagt (kein hartes Versagen danach)
    SystemBatchHost probe = line.base;
    line.set(probe, 0, line.t1);
    const ResidualLayout layout(probe.topology());
    EXPECT_EQ(mko::merit_cpu(probe, layout, false).hard_failed[0], 0);
    EXPECT_GT(mko::merit_cpu(probe, layout, false).failed[0], 0);
    ASSERT_TRUE(soft.crossed);
    ASSERT_EQ(soft.jumps.size(), 2u);
    EXPECT_LT(soft.max_ratio, kContinuityNeighborRatio);
    EXPECT_LT(soft.jumps[1], kContinuityShrink * soft.jumps[0]);
    EXPECT_LT(soft.jumps[1], 1e-5 * soft.merit_scale);
    ASSERT_EQ(hard.jumps.size(), 2u);
    EXPECT_GT(hard.max_ratio, 10.0 * kContinuityNeighborRatio);
    EXPECT_GT(hard.jumps[1], 0.5 * hard.jumps[0]);
}

// ---- Abnahme-Optimierung -------------------------------------------------------------------

// Startsystem weiches_versagen_start.json: 30 % der Strahlen versagen (Vorderfläche verfehlt,
// Totalreflexion an der Rückfläche), alle weich. LM (CPU, Standardoptionen: weicher Modus,
// w_fail = 10⁶ mit äußerem Strafverfahren) ohne Eingriff: am Ende kein versagter Strahl (weich und
// hart), Merit gesunken, Strafen 0 (Merit im harten Modus bitgleich). Zum Vergleich der harte
// Modus aus demselben Start (nur Ausgabe).
TEST(SoftFailure, AcceptanceOptimizationCpu) {
    const mko::test::FdProblem p =
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json"));
    const ResidualLayout layout(p.base.topology());
    const MeritResult start = mko::merit_cpu(p.base, layout, false);
    const double fraction =
        static_cast<double>(start.failed[0]) / static_cast<double>(layout.rays_per_variant());
    EXPECT_NEAR(fraction, 0.3, 0.02);
    EXPECT_EQ(start.hard_failed[0], 0);

    mko::LmOptions o;
    std::vector<std::int64_t> failed_path;
    int last_iteration = 0;
    o.on_iteration = [&failed_path, &last_iteration](const mko::LmIterationInfo& info) {
        EXPECT_EQ(info.run, 0);
        EXPECT_EQ(info.iteration, last_iteration + 1); // fortlaufend über die Runden
        last_iteration = info.iteration;
        if (info.accepted) {
            failed_path.push_back(info.failed_rays);
        }
    };
    const mko::LmResult res = mko::optimize_lm_cpu(p.set, p.base, p.x0, o);
    const mko::LmRunResult& r = res.runs[0];
    std::size_t first_clean = failed_path.size();
    for (std::size_t i = 0; i < failed_path.size(); ++i) {
        if (failed_path[i] == 0) {
            first_clean = i + 1;
            break;
        }
    }
    std::printf("[ Info     ] weich: %lld von %lld Strahlen versagt (%.1f %%), F %.6e -> %.6e, "
                "%d Zyklen (%d angenommen, versagensfrei nach %zu), %d Runden bis w_fail = %.0e, "
                "Status %s, x = (%.6f, %.6f, %.4f)\n",
                static_cast<long long>(start.failed[0]),
                static_cast<long long>(layout.rays_per_variant()), 100.0 * fraction,
                r.history.front(), r.merit, r.iterations, r.accepted, first_clean, r.penalty_rounds,
                r.fail_weight, mko::to_string(r.status), r.x[0], r.x[1], r.x[2]);
    EXPECT_EQ(r.failed_rays, 0);
    EXPECT_EQ(r.hard_failed_rays, 0);
    EXPECT_LT(r.merit, r.history.front());
    EXPECT_EQ(last_iteration, r.iterations);
    EXPECT_TRUE(mko::lm_converged(r.status)) << mko::to_string(r.status);
    EXPECT_TRUE(p.set.feasible(r.x));
    SystemBatchHost end = p.base;
    p.set.apply(end, 0, r.x);
    const MeritResult soft_end = mko::merit_cpu(end, layout, false);
    const MeritResult hard_end =
        mko::merit_cpu(end, ResidualLayout(end.topology(), with_mode(FailureMode::Hard)), false);
    EXPECT_EQ(soft_end.failed[0], 0);
    EXPECT_TRUE(same_bits(hard_end.merit[0], soft_end.merit[0]));

    mko::LmOptions oh;
    oh.jacobian.quadrature.failure = FailureMode::Hard;
    const mko::LmRunResult h = mko::optimize_lm_cpu(p.set, p.base, p.x0, oh).runs[0];
    std::printf("[ Info     ] hart:  F %.6e -> %.6e, %d Zyklen, Status %s, am Ende %lld "
                "versagte Strahlen\n",
                h.history.front(), h.merit, h.iterations, mko::to_string(h.status),
                static_cast<long long>(h.failed_rays));
    EXPECT_EQ(h.penalty_rounds, 1);
}

// Robustheit (108 Starts um das Startsystem, 102 davon mit versagten Strahlen; Plankonvexlinse
// an der Totalreflexion mit 2 bis 8 von 10 versagten Strahlen, je 12 Starts): Mit den
// Standardoptionen enden (MSVC, x64) alle Läufe versagensfrei. Verlangt werden mindestens
// 106 von 108: Starts mit c₂ = 0 liegen an der Kante, an der die tangentiale Fortsetzung der
// Totalreflexion in hartes Versagen übergeht (numerik.md, Abschnitt 11); welcher Weg dort
// eingeschlagen wird, hängt von der Rundung ab (GPU: 107 von 108). Ohne das äußere
// Strafverfahren (ein fester Wert w_fail) gelingt das nur in einem schmalen, vom System
// abhängigen Bereich von w_fail (Ausgabe).
TEST(SoftFailure, AcceptanceManyStartsCpu) {
    const auto clean_runs = [](const mko::LmResult& res) {
        int n = 0;
        for (const mko::LmRunResult& r : res.runs) {
            n += r.failed_rays == 0 ? 1 : 0;
        }
        return n;
    };
    const mko::test::FdProblem lens =
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json"));
    const std::vector<double> starts = mko::test::soft_failure_start_grid();
    const int runs = static_cast<int>(starts.size() / 3);
    int failing_starts = 0;
    for (int i = 0; i < runs; ++i) {
        SystemBatchHost b = lens.base;
        lens.set.apply(b, 0,
                       std::span<const double>(starts).subspan(3 * static_cast<std::size_t>(i), 3));
        failing_starts += mko::merit_cpu(b, ResidualLayout(b.topology()), false).failed[0] > 0;
    }
    const mko::LmResult res = mko::optimize_lm_cpu(lens.set, lens.base, starts);
    int max_rounds = 0;
    for (const mko::LmRunResult& r : res.runs) {
        max_rounds = std::max(max_rounds, r.penalty_rounds);
        EXPECT_EQ(r.hard_failed_rays, 0);
    }
    std::printf("[ Info     ] Bikonvexlinse: %d von %d Läufen versagensfrei (%d Starts mit "
                "Versagen), höchstens %d Runden\n",
                clean_runs(res), runs, failing_starts, max_rounds);
    EXPECT_GE(clean_runs(res), runs - 2);
    EXPECT_GT(failing_starts, runs / 2);

    for (const double factor : {1.05, 1.5, 3.0}) {
        const mko::test::FdProblem p = mko::test::tir_start_problem(factor);
        const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, 12, 0.2, 9);
        const mko::LmResult r = mko::optimize_lm_cpu(p.set, p.base, x0);
        const MeritResult m0 = mko::merit_cpu(p.base, ResidualLayout(p.base.topology()), false);
        std::printf("[ Info     ] Plankonvexlinse c₂ x %.2f (%lld von 10 versagt): %d von 12 "
                    "versagensfrei\n",
                    factor, static_cast<long long>(m0.failed[0]), clean_runs(r));
        EXPECT_EQ(clean_runs(r), 12);
    }

    // Vergleich: festes w_fail ohne Strafverfahren (nur Ausgabe)
    for (const double w : {1e6, 1e8, 1e9, 1e10, 1e12}) {
        mko::LmOptions o;
        o.fail_weight_factor = 1.0;
        o.jacobian.quadrature.fail_weight = w;
        const int lens_clean = clean_runs(mko::optimize_lm_cpu(lens.set, lens.base, starts, o));
        const mko::test::FdProblem p = mko::test::tir_start_problem(1.5);
        const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, 12, 0.2, 9);
        const int tir_clean = clean_runs(mko::optimize_lm_cpu(p.set, p.base, x0, o));
        std::printf("[ Info     ] festes w_fail = %.0e: Bikonvexlinse %d von %d, Plankonvexlinse "
                    "%d von 12 versagensfrei\n",
                    w, lens_clean, runs, tir_clean);
    }
}

// ---- Vorfilterung --------------------------------------------------------------------------

// Im weichen Modus hat eine Variante mit sicher versagten Strahlen keine obere Schranke der
// Merit (Straf- und Wellenfrontresiduen weich versagter Strahlen fehlen im FP32-Pfad); ohne
// Versagen und im harten Modus wie bisher. Die untere Schranke ist in beiden Modi gleich.
TEST(SoftFailure, PrefilterUpperBoundNeedsFailureFreeVariants) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 3);
    const ResidualLayout soft(batch.topology());
    const ResidualLayout hard(batch.topology(), with_mode(FailureMode::Hard));
    mko::PrefilterStats s;
    s.variants = 3;
    s.rays_per_variant = soft.rays_per_variant();
    s.merit = {4.0, 1.0, 9.0};
    s.optical_length = {0.0, 0.0, 0.0};
    s.failed = {0, 3, 0};
    s.uncertain_ok = {0, 0, 0};
    s.uncertain_failed = {0, 0, 0};
    mko::PrefilterOptions o;
    o.criteria.max_merit_factor = 5.0;
    const mko::PrefilterResult rs = mko::prefilter_decide(s, batch, soft, o);
    const mko::PrefilterResult rh = mko::prefilter_decide(s, batch, hard, o);
    constexpr double kInf = std::numeric_limits<double>::infinity();
    EXPECT_LT(rs.merit_upper[0], kInf);
    EXPECT_EQ(rs.merit_upper[1], kInf);
    EXPECT_LT(rh.merit_upper[1], kInf);
    EXPECT_EQ(rs.merit_lower, rh.merit_lower);
    // Referenz für (c): im harten Modus die Variante mit der kleinsten oberen Schranke (1),
    // im weichen die kleinste ohne Versagen (0); 9 > 5 · U_1, aber nicht > 5 · U_0
    EXPECT_EQ(rh.reference_variant, 1);
    EXPECT_EQ(rs.reference_variant, 0);
    EXPECT_EQ(rh.reject[2], mko::kPrefilterMeritRelative);
    EXPECT_EQ(rs.reject[2], mko::kPrefilterKeep);
}
