// FP32-Vorfilterung (M7-T2) auf der CPU (auch in der CI): Grenzabstände an einer
// Versagensgrenze, Fehlermodell der OPD pro Strahl gegen FP64, Entscheidungsregel mit
// synthetischen Rohdaten und die Abnahme "keine falsch aussortierten Varianten" mit der
// CPU-Referenz in float (docs/numerik.md, Abschnitt 12).

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/merit/prefilter_ray.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "prefilter_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using mko::MeritResult;
using mko::PrefilterOptions;
using mko::PrefilterRayData;
using mko::PrefilterResult;
using mko::PrefilterStats;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::test::kPrefilterMarginFp32VsFp64;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Strahl der Vorfilterung in FP64 über dieselbe Funktion (R = double, Tabelle des Batches).
mko::PrefilterRay<double> prefilter_ray_fp64(const SystemBatchHost& batch, int v, int f, int l,
                                             int k, const ResidualLayout& layout) {
    const mko::SystemBatchView<double> view = batch.view();
    const mko::SurfaceTableView<double> table = view.surface_table(v);
    const mko::PupilSample<double> s = mko::sample(layout.quadrature(f), k);
    const mko::Vec3<double> center = mko::batch_image_point(view, table, v, f);
    return mko::prefilter_ray(view, table, l, v, f, l, s, center, 0.0);
}

// Rohdaten mit Nullwerten für V Varianten.
PrefilterStats synthetic_stats(int variants, std::int64_t rays) {
    PrefilterStats s;
    const auto n = static_cast<std::size_t>(variants);
    s.variants = variants;
    s.rays_per_variant = rays;
    s.merit.assign(n, 0.0);
    s.optical_length.assign(n, 0.0);
    s.failed.assign(n, 0);
    s.uncertain_ok.assign(n, 0);
    s.uncertain_failed.assign(n, 0);
    return s;
}

} // namespace

// Grenzabstand an der Totalreflexion: äußerster Strahl des Achsfeldes von einzellinse_bk7.json
// mit Pupillenradius a*·(1 + δ) um den Grenzradius a*, an dem der Strahl in FP64 an der zweiten
// Fläche totalreflektiert. Der Grenzabstand fällt gegen 0 (wie √|δ|), FP32 und FP64 liefern
// denselben Status oder FP32 meldet den Strahl als unsicher; FP32 und FP64 messen denselben
// Grenzabstand (bei Totalreflexion √(−k)).
TEST(Prefilter, MarginAtTirBoundary) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost batch = mko::make_system_batch(ls, 1);
    const ResidualLayout layout(batch.topology());
    const int k = layout.samples(0) - 1; // äußerster Ring, Achsfeld
    const auto status_at = [&](double a) {
        batch.pupil_radius(0) = a;
        return prefilter_ray_fp64(batch, 0, 0, 0, k, layout).status;
    };
    double lo = 1.0;  // gültig
    double hi = 80.0; // versagt
    ASSERT_EQ(status_at(lo), TraceStatus::Ok);
    ASSERT_NE(status_at(hi), TraceStatus::Ok);
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        (status_at(mid) == TraceStatus::Ok ? lo : hi) = mid;
    }
    ASSERT_EQ(status_at(hi), TraceStatus::TIR);
    const double a_star = lo;
    std::printf("[ Info     ] Grenzradius a* = %.6f mm\n", a_star);
    const double eta = PrefilterOptions{}.margins.boundary;
    double previous = kInf;
    for (double delta : {-1e-1, -1e-2, -1e-4, -1e-6, 1e-6, 1e-4, 1e-2, 1e-1}) {
        batch.pupil_radius(0) = a_star * (1.0 + delta);
        const mko::PrefilterRay<double> r64 = prefilter_ray_fp64(batch, 0, 0, 0, k, layout);
        const PrefilterRayData d = mko::prefilter_rays_cpu(batch, 0, layout, eta);
        const auto i = static_cast<std::size_t>(layout.offset(0, 0) + k);
        const bool same = (d.status[i] == 0) == (r64.status == TraceStatus::Ok);
        EXPECT_TRUE(same || d.certain[i] == 0) << "δ = " << delta;
        EXPECT_EQ(r64.status, delta < 0 ? TraceStatus::Ok : TraceStatus::TIR) << "δ = " << delta;
        // FP32 und FP64 messen denselben Grenzabstand (bis auf die Rundung von float)
        EXPECT_NEAR(d.margin[i], r64.margin, kPrefilterMarginFp32VsFp64) << "δ = " << delta;
        if (std::abs(delta) <= 1e-4) {
            EXPECT_EQ(d.certain[i], 0) << "δ = " << delta;
            EXPECT_LT(r64.margin, 0.02) << "δ = " << delta;
        }
        if (delta > 0.0) {
            EXPECT_GT(r64.margin, previous) << "δ = " << delta;
        }
        previous = delta < 0.0 ? 0.0 : r64.margin;
        std::printf("[ Info     ] δ = %+.0e: Status FP64 %s, Grenzabstand FP64 %.3e, FP32 %.3e\n",
                    delta, mko::trace_status_name(r64.status), r64.margin, d.margin[i]);
    }
}

// Fehlermodell der FP32-Vorfilterung gegen FP64 (CPU-Referenz in float gegen residuals_cpu)
// für Multistart-Varianten der drei Testsysteme: Kein sicher entschiedener Strahl hat in FP64
// einen anderen Status, und in Tupeln ohne unsichere Strahlen gilt pro Strahl
// |OPD_FP32 − OPD_FP64| <= κ u Λ_v + m |OPD_FP64| mit den Standardwerten κ, m.
TEST(Prefilter, RayClassesAndErrorModel) {
    const PrefilterOptions options;
    const double eta = options.margins.boundary;
    const double kappa = options.margins.opl_error_factor;
    const double m = options.margins.merit_relative;
    for (const auto& set : mko::test::prefilter_test_sets(600, 77)) {
        SCOPED_TRACE(set.name);
        const ResidualLayout layout(set.batch.topology());
        std::size_t rays = 0;
        std::size_t failed64 = 0;
        std::size_t uncertain = 0;
        std::size_t flips = 0;
        std::size_t certain_flips = 0;
        std::size_t over_bound = 0;
        double max_kappa = 0.0;      // nötiges κ bei m = 0
        double max_kappa_m = 0.0;    // nötiges κ bei m = Standard
        double max_margin_err = 0.0; // Grenzabstände bis 1 (Kosinus), FP32 gegen FP64
        for (int v = 0; v < set.batch.variants(); ++v) {
            const PrefilterRayData d = mko::prefilter_rays_cpu(set.batch, v, layout, eta);
            const mko::ResidualResult ref = mko::residuals_cpu(set.batch, v, layout);
            double lambda = 0.0;
            for (double x : d.optical_length) {
                lambda = std::max(lambda, x);
            }
            const double unit = mko::kFloatUnitRoundoff * lambda;
            for (int f = 0; f < layout.fields(); ++f) {
                for (int l = 0; l < layout.wavelengths(); ++l) {
                    const std::int64_t o = layout.offset(f, l);
                    bool clean = true;
                    for (int k = 0; k < layout.samples(f); ++k) {
                        const auto i = static_cast<std::size_t>(o + k);
                        const bool ok32 = d.status[i] == 0;
                        const bool ok64 = ref.status[i] == 0;
                        ++rays;
                        failed64 += ok64 ? 0 : 1;
                        uncertain += d.certain[i] ? 0 : 1;
                        flips += ok32 != ok64 ? 1 : 0;
                        certain_flips += ok32 != ok64 && d.certain[i] ? 1 : 0;
                        clean = clean && d.certain[i] && ok32 == ok64;
                        if (d.certain[i] && ok32 == ok64 && v % 10 == 0) {
                            const mko::PrefilterRay<double> r64 =
                                prefilter_ray_fp64(set.batch, v, f, l, k, layout);
                            if (r64.margin <= 1.0) { // Kosinus-Bereich; große Beträge relativ
                                max_margin_err =
                                    std::max(max_margin_err, std::abs(r64.margin - d.margin[i]));
                            }
                        }
                    }
                    if (!clean) {
                        continue;
                    }
                    for (int k = 0; k < layout.samples(f); ++k) {
                        const auto i = static_cast<std::size_t>(o + k);
                        if (d.status[i] != 0) {
                            continue;
                        }
                        const double err = std::abs(d.opd[i] - ref.opd[i]);
                        const double rel = m * std::abs(ref.opd[i]);
                        max_kappa = std::max(max_kappa, err / unit);
                        max_kappa_m = std::max(max_kappa_m, std::max(0.0, err - rel) / unit);
                        over_bound += err > kappa * unit + rel ? 1 : 0;
                    }
                }
            }
        }
        EXPECT_EQ(certain_flips, 0u);
        EXPECT_EQ(over_bound, 0u);
        std::printf("[ Info     ] %-26s %zu Strahlen, FP64 versagt %zu, unsicher %zu, Status "
                    "verschieden %zu (davon sicher %zu); nötiges κ: %.1f (m = 0), %.1f (m = %.2f); "
                    "max |Δ Grenzabstand| %.1e\n",
                    set.name.c_str(), rays, failed64, uncertain, flips, certain_flips, max_kappa,
                    max_kappa_m, m, max_margin_err);
    }
}

// Entscheidungsregel mit synthetischen Rohdaten: Schranken, Kriterien (a), (b), (c), Referenz
// nur aus Varianten ohne unsichere Strahlen, Anteil weiterer Residuen, NaN, Prüfung der
// Optionen und die Regel ohne Margen.
TEST(Prefilter, DecisionRuleSynthetic) {
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 6);
    const ResidualLayout layout(batch.topology());
    const std::int64_t R = layout.rays_per_variant();
    // √(Σ_(f,l) s²) mit s = residual_scale (M4-T1: √(ŵ_f ŵ_λ)/λ_ref, hier 1/λ_ref)
    const double scale = mko::test::residual_scale_norm(batch, layout);
    EXPECT_NEAR(scale * ls.system.spectrum.reference_nm * 1e-6, 1.0, mko::test::kUnitFp64);
    PrefilterStats s = synthetic_stats(6, R);
    s.merit = {4.0, 1.0, 100.0, 0.25, 9.0, std::nan("")};
    // Λ_1 so, dass Δ_1 = κ u Λ_1 · scale ≈ 3·10⁻⁴ klein gegen √F_1 = 1 bleibt (Merit in Wellen²)
    s.optical_length = {0.0, 0.1, 0.0, 0.0, 0.0, 0.0};
    s.failed = {0, 0, 0, R / 2 + 1, 0, 0};
    s.uncertain_ok = {0, 0, 0, 0, 1, 0};
    PrefilterOptions o;
    o.criteria.max_failed_fraction = 0.5;
    o.criteria.max_merit = 50.0;
    o.criteria.max_merit_factor = 5.0;
    const std::vector<double> extra = {0.5, 0.0, 0.0, 0.0, 0.0, 0.0};
    const PrefilterResult r = mko::prefilter_decide(s, batch, layout, o, extra);
    const double m = o.margins.merit_relative;
    const double delta1 = o.margins.opl_error_factor * mko::kFloatUnitRoundoff * 0.1 * scale;
    // Schranken
    EXPECT_DOUBLE_EQ(r.merit_lower[0], std::pow(2.0 / (1.0 + m), 2) + 0.5);
    EXPECT_DOUBLE_EQ(r.merit_upper[0], std::pow(2.0 / (1.0 - m), 2) + 0.5);
    EXPECT_DOUBLE_EQ(r.merit_lower[1], std::pow((1.0 - delta1) / (1.0 + m), 2));
    EXPECT_DOUBLE_EQ(r.merit_upper[1], std::pow((1.0 + delta1) / (1.0 - m), 2));
    EXPECT_EQ(r.merit_upper[4], kInf); // unsichere Strahlen
    EXPECT_EQ(r.merit_lower[5], 0.0);  // NaN: keine Aussage
    EXPECT_EQ(r.merit_upper[5], kInf);
    // Referenz: kleinste obere Schranke ohne unsichere Strahlen und ohne (a), also Variante 1
    // (Variante 3 hat zu viel Versagen, Variante 4 unsichere Strahlen)
    EXPECT_EQ(r.reference_variant, 1);
    EXPECT_DOUBLE_EQ(r.reference_merit, r.merit_upper[1]);
    // Kriterien
    EXPECT_EQ(r.reject[0], mko::kPrefilterKeep);
    EXPECT_EQ(r.reject[1], mko::kPrefilterKeep);
    EXPECT_EQ(r.reject[2], mko::kPrefilterMeritAbsolute | mko::kPrefilterMeritRelative);
    EXPECT_EQ(r.reject[3], mko::kPrefilterFailedRays);
    EXPECT_EQ(r.reject[4], mko::kPrefilterMeritRelative); // 9/1,01² > 5 · 1,02
    EXPECT_EQ(r.reject[5], mko::kPrefilterKeep);
    EXPECT_EQ(r.kept, 3);
    EXPECT_TRUE(r.keep(0));
    EXPECT_FALSE(r.keep(2));

    // Abgeschaltete Kriterien behalten alles
    PrefilterOptions off;
    off.criteria.max_failed_fraction = 1.0;
    const PrefilterResult all = mko::prefilter_decide(s, batch, layout, off);
    EXPECT_EQ(all.kept, 6);
    EXPECT_EQ(all.reference_variant, -1);

    // Prüfungen
    PrefilterOptions bad = o;
    bad.margins.merit_relative = 1.0;
    EXPECT_THROW((void)mko::prefilter_decide(s, batch, layout, bad), std::invalid_argument);
    bad = o;
    bad.margins.boundary = kInf;
    EXPECT_THROW((void)mko::prefilter_decide(s, batch, layout, bad), std::invalid_argument);
    bad = o;
    bad.criteria.max_merit = -1.0;
    EXPECT_THROW((void)mko::prefilter_decide(s, batch, layout, bad), std::invalid_argument);
    const std::vector<double> short_extra(2, 0.0);
    EXPECT_THROW((void)mko::prefilter_decide(s, batch, layout, o, short_extra),
                 std::invalid_argument);
    PrefilterStats wrong = s;
    wrong.failed.pop_back();
    EXPECT_THROW((void)mko::prefilter_decide(wrong, batch, layout, o), std::invalid_argument);
    EXPECT_THROW((void)mko::prefilter_stats_cpu(batch, layout, -1.0), std::invalid_argument);

    // Regel ohne Margen (FP64-Wahrheit): beste Merit nur aus Varianten ohne (a) und mit
    // mindestens einem gültigen Strahl
    MeritResult exact;
    exact.variants = 4;
    exact.rays_per_variant = R;
    exact.merit = {1.0, 0.0, 6.0, 0.1};
    exact.failed = {0, R, 0, R / 2 + 1};
    mko::PrefilterCriteria c;
    c.max_failed_fraction = 0.5;
    c.max_merit_factor = 5.0;
    const std::vector<std::uint8_t> e = mko::prefilter_decide_exact(exact, c);
    EXPECT_EQ(e[0], mko::kPrefilterKeep);
    EXPECT_EQ(e[1], mko::kPrefilterFailedRays);
    EXPECT_EQ(e[2], mko::kPrefilterMeritRelative);
    EXPECT_EQ(e[3], mko::kPrefilterFailedRays);
}

// Abnahme M7-T2 auf der CPU (auch in der CI): je 1.000 Multistart-Varianten der drei
// Testsysteme, CPU-Referenz in float gegen merit_cpu in FP64 mit denselben Kriterien wie der
// GPU-Test: keine falsch aussortierte Variante, kein falscher Grund, Schranken und Klammer der
// versagten Strahlen halten.
TEST(Prefilter, CpuAcceptanceNoFalseRejections) {
    int false_positive = 0;
    int false_negative = 0;
    int rejected64 = 0;
    for (const auto& set : mko::test::prefilter_test_sets(1000, 31415)) {
        SCOPED_TRACE(set.name);
        const ResidualLayout layout(set.batch.topology());
        const MeritResult truth = mko::merit_cpu(set.batch, layout, false);
        const PrefilterStats stats =
            mko::prefilter_stats_cpu(set.batch, layout, PrefilterOptions{}.margins.boundary);
        for (const mko::test::NamedCriteria& c : mko::test::acceptance_criteria(truth)) {
            PrefilterOptions o;
            o.criteria = c.criteria;
            const PrefilterResult r = mko::prefilter_decide(stats, set.batch, layout, o);
            const mko::test::DecisionComparison cmp = mko::test::compare_decisions(
                r.reject, mko::prefilter_decide_exact(truth, c.criteria));
            EXPECT_EQ(cmp.false_positive, 0) << c.name;
            EXPECT_EQ(cmp.reason_mismatch, 0) << c.name;
            false_positive += cmp.false_positive;
            false_negative += cmp.false_negative;
            rejected64 += cmp.rejected64;
        }
        const PrefilterResult r =
            mko::prefilter_decide(stats, set.batch, layout, PrefilterOptions{});
        for (int v = 0; v < set.batch.variants(); ++v) {
            const auto i = static_cast<std::size_t>(v);
            EXPECT_LE(r.merit_lower[i], truth.merit[i]) << "Variante " << v;
            EXPECT_GE(r.merit_upper[i], truth.merit[i]) << "Variante " << v;
            EXPECT_LE(stats.failed[i], truth.failed[i]) << "Variante " << v;
            EXPECT_GE(stats.failed[i] + stats.uncertain(v), truth.failed[i]) << "Variante " << v;
        }
    }
    std::printf("[ Info     ] CPU: falsch aussortiert %d, zusätzlich behalten %d von %d "
                "FP64-Ablehnungen\n",
                false_positive, false_negative, rejected64);
}
