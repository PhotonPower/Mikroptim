// FP32-Vorfilterung auf der GPU (M7-T2, nur mit MIKROPTIM_ENABLE_CUDA): Abnahme "keine falsch
// aussortierten Varianten" gegen die FP64-Wahrheit über die TraceEngine, Abgleich mit der
// CPU-Referenz in float, Determinismus und Tranchen, Grenzen und Durchsatz
// (docs/numerik.md, Abschnitt 12; docs/benchmarks.md, "FP32-Vorfilterung").

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/prefilter.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "prefilter_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using mko::MeritResult;
using mko::PrefilterOptions;
using mko::PrefilterResult;
using mko::PrefilterStats;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::cuda::EngineOptions;
using mko::cuda::Precision;
using mko::cuda::PrefilterSettings;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
using mko::test::kPrefilterGpuCpuLengthRelative;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool identical(const PrefilterResult& a, const PrefilterResult& b) {
    return a.stats.variants == b.stats.variants && same_bytes(a.stats.merit, b.stats.merit) &&
           same_bytes(a.stats.optical_length, b.stats.optical_length) &&
           same_bytes(a.stats.failed, b.stats.failed) &&
           same_bytes(a.stats.uncertain_ok, b.stats.uncertain_ok) &&
           same_bytes(a.stats.uncertain_failed, b.stats.uncertain_failed) &&
           same_bytes(a.merit_lower, b.merit_lower) && same_bytes(a.merit_upper, b.merit_upper) &&
           same_bytes(a.reject, b.reject) && a.kept == b.kept &&
           a.reference_variant == b.reference_variant;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Vergleich FP32-Entscheidung gegen FP64-Wahrheit für einen Testsatz und alle Kriterien der
// Abnahme (Schwellen auch genau an Quantilen der FP64-Werte, eine Konfiguration mit Anteil
// weiterer Residuen). Prüft: keine falsch aussortierte Variante, kein Grund, den FP64 nicht
// hat, Schranken L_v <= F_FP64 <= U_v, sicher versagte und unsichere Strahlen klammern die
// FP64-Zahl ein. Rückgabe: Summen über alle Kriterien.
struct AcceptanceTotals {
    int decisions = 0;
    int false_positive = 0;
    int false_negative = 0;
    int rejected64 = 0;
    std::int64_t uncertain = 0;
};

AcceptanceTotals check_acceptance(TraceEngine& engine, const mko::test::PrefilterTestSet& set,
                                  bool verbose) {
    AcceptanceTotals totals;
    const SystemBatchHost& batch = set.batch;
    TraceSettings fp64;
    const MeritResult truth = engine.run(batch, TraceMode::Merit, fp64).merit;
    std::int64_t failed64 = 0;
    int massive = 0;
    std::vector<double> merit_ok;
    for (int v = 0; v < batch.variants(); ++v) {
        const auto i = static_cast<std::size_t>(v);
        failed64 += truth.failed[i];
        massive += truth.failed[i] * 2 > truth.rays_per_variant ? 1 : 0;
        if (truth.failed[i] < truth.rays_per_variant) {
            merit_ok.push_back(truth.merit[i]);
        }
    }
    if (verbose) {
        std::printf("[ Info     ] %s: %d Varianten, %lld Strahlen pro Variante, versagt %.2f %%, "
                    "%d mit > 50 %% Versagen; Merit (FP64) Quantile 0/0,5/1: %.3e / %.3e / "
                    "%.3e\n",
                    set.name.c_str(), batch.variants(),
                    static_cast<long long>(truth.rays_per_variant),
                    100.0 * static_cast<double>(failed64) /
                        (static_cast<double>(truth.rays_per_variant) * batch.variants()),
                    massive, mko::test::quantile(merit_ok, 0.0), mko::test::quantile(merit_ok, 0.5),
                    mko::test::quantile(merit_ok, 1.0));
    }

    // Anteil weiterer Residuen für eine Konfiguration (wie paraxiale Nebenbedingungen)
    std::vector<double> extra(static_cast<std::size_t>(batch.variants()));
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double scale = mko::test::quantile(merit_ok, 0.5);
    for (double& e : extra) {
        e = scale * unit(rng);
    }

    const std::vector<mko::test::NamedCriteria> configs = mko::test::acceptance_criteria(truth);
    for (std::size_t c = 0; c <= configs.size(); ++c) {
        const bool with_extra = c == configs.size();
        PrefilterSettings settings;
        settings.options.criteria = with_extra ? configs.back().criteria : configs[c].criteria;
        const std::string name =
            with_extra ? configs.back().name + " + weitere Residuen" : configs[c].name;
        if (with_extra) {
            settings.extra_merit = extra;
        }
        const PrefilterResult pre = engine.prefilter_variants(batch, settings);
        const std::vector<std::uint8_t> exact = mko::prefilter_decide_exact(
            truth, settings.options.criteria,
            with_extra ? std::span<const double>(extra) : std::span<const double>());
        const mko::test::DecisionComparison cmp = mko::test::compare_decisions(pre.reject, exact);
        EXPECT_EQ(cmp.false_positive, 0) << set.name << ", " << name;
        EXPECT_EQ(cmp.reason_mismatch, 0) << set.name << ", " << name;
        ++totals.decisions;
        totals.false_positive += cmp.false_positive;
        totals.false_negative += cmp.false_negative;
        totals.rejected64 += cmp.rejected64;
        if (verbose) {
            std::printf("[ Info     ]   %-36s FP64 sortiert aus %5d, FP32 %5d, falsch aussortiert "
                        "%d, zusätzlich behalten %4d\n",
                        name.c_str(), cmp.rejected64, cmp.rejected32, cmp.false_positive,
                        cmp.false_negative);
        }
    }
    // Schranken der Merit und Klammer der versagten Strahlen
    const PrefilterResult pre = engine.prefilter_variants(batch, PrefilterSettings{});
    int below = 0;
    int above = 0;
    int bracket = 0;
    for (int v = 0; v < batch.variants(); ++v) {
        const auto i = static_cast<std::size_t>(v);
        below += truth.merit[i] < pre.merit_lower[i] ? 1 : 0;
        above += truth.merit[i] > pre.merit_upper[i] ? 1 : 0;
        totals.uncertain += pre.stats.uncertain(v);
        bracket += pre.stats.failed[i] <= truth.failed[i] &&
                           pre.stats.failed[i] + pre.stats.uncertain(v) >= truth.failed[i]
                       ? 0
                       : 1;
    }
    EXPECT_EQ(below, 0) << set.name;
    EXPECT_EQ(above, 0) << set.name;
    EXPECT_EQ(bracket, 0) << set.name;
    if (verbose) {
        std::printf("[ Info     ]   unsichere Strahlen insgesamt %lld\n",
                    static_cast<long long>(totals.uncertain));
    }
    return totals;
}

} // namespace

// Abnahme M7-T2: 10.000 Multistart-Varianten je System (Einzellinse, Mehrlinser, Einzellinse
// mit Geisterplatten; Krümmungen ±50 %, Dicken ±30 %, extreme Varianten mit massivem
// Versagen). Für jedes Kriterium wird die FP32-Entscheidung mit derselben Regel ohne Margen auf
// den FP64-Werten (Modus MERIT in FP64) verglichen: keine falsch aussortierte Variante.
// Berichtet werden außerdem die Varianten, die FP32 behält, FP64 aber aussortiert (erlaubt).
TEST(CudaPrefilter, AcceptanceNoFalseRejections) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    AcceptanceTotals all;
    for (const auto& set : mko::test::prefilter_test_sets(10000, 20261018)) {
        const AcceptanceTotals t = check_acceptance(engine, set, true);
        all.false_positive += t.false_positive;
        all.false_negative += t.false_negative;
        all.rejected64 += t.rejected64;
    }
    EXPECT_EQ(all.false_positive, 0);
    std::printf("[ Info     ] gesamt: falsch aussortiert %d, zusätzlich behalten %d von %d "
                "FP64-Ablehnungen\n",
                all.false_positive, all.false_negative, all.rejected64);
}

// Stresstest (abgeschaltet, lokal mit --gtest_also_run_disabled_tests): je 10.000 Varianten
// der drei Systeme für 10 weitere Seeds.
TEST(CudaPrefilter, DISABLED_StressManySeeds) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    AcceptanceTotals all;
    int variants = 0;
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
        for (const auto& set : mko::test::prefilter_test_sets(10000, 1000 * seed)) {
            const AcceptanceTotals t = check_acceptance(engine, set, false);
            all.decisions += t.decisions;
            all.false_positive += t.false_positive;
            all.false_negative += t.false_negative;
            all.rejected64 += t.rejected64;
            all.uncertain += t.uncertain;
            variants += set.batch.variants();
        }
    }
    EXPECT_EQ(all.false_positive, 0);
    std::printf("[ Info     ] %d Varianten, %d Entscheidungsläufe: falsch aussortiert %d, "
                "zusätzlich behalten %d von %d FP64-Ablehnungen, unsichere Strahlen %lld\n",
                variants, all.decisions, all.false_positive, all.false_negative, all.rejected64,
                static_cast<long long>(all.uncertain));
}

// GPU gegen CPU-Referenz in float: Klassen der Strahlen gleich (bis auf Strahlen genau an der
// Schwelle η, FMA-Kontraktion auf dem Gerät), Merit innerhalb der Fehlerschranke Δ_v.
TEST(CudaPrefilter, MatchesCpuReference) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    for (const auto& set : mko::test::prefilter_test_sets(300, 4242)) {
        SCOPED_TRACE(set.name);
        const SystemBatchHost& batch = set.batch;
        const ResidualLayout layout(batch.topology());
        const PrefilterOptions options;
        const PrefilterStats cpu =
            mko::prefilter_stats_cpu(batch, layout, options.margins.boundary);
        const PrefilterResult gpu = engine.prefilter_variants(batch, PrefilterSettings{});
        int count_mismatch = 0;
        double max_rel = 0.0;
        for (int v = 0; v < batch.variants(); ++v) {
            const auto i = static_cast<std::size_t>(v);
            if (cpu.failed[i] != gpu.stats.failed[i] ||
                cpu.uncertain_ok[i] != gpu.stats.uncertain_ok[i] ||
                cpu.uncertain_failed[i] != gpu.stats.uncertain_failed[i]) {
                ++count_mismatch;
                continue;
            }
            // Beide Seiten weichen von FP64 höchstens um Δ_v ab (numerik.md, Abschnitt 12), mit
            // √(Σ_(f,l) s²) der Residuen in Wellen (M4-T1)
            const double bound = 2.0 * options.margins.opl_error_factor * mko::kFloatUnitRoundoff *
                                     cpu.optical_length[i] *
                                     mko::test::residual_scale_norm(batch, layout) +
                                 2.0 * options.margins.merit_relative * std::sqrt(cpu.merit[i]);
            const double diff = std::abs(std::sqrt(cpu.merit[i]) - std::sqrt(gpu.stats.merit[i]));
            EXPECT_LE(diff, bound) << "Variante " << v;
            if (cpu.merit[i] > 0.0) {
                max_rel = std::max(max_rel, std::abs(gpu.stats.merit[i] / cpu.merit[i] - 1.0));
            }
            EXPECT_NEAR(gpu.stats.optical_length[i], cpu.optical_length[i],
                        kPrefilterGpuCpuLengthRelative * cpu.optical_length[i]);
        }
        EXPECT_LE(count_mismatch, 1);
        std::printf("[ Info     ] %s: Klassen verschieden in %d von %d Varianten, max |F_GPU/F_CPU "
                    "- 1| = %.2e\n",
                    set.name.c_str(), count_mismatch, batch.variants(), max_rel);
    }
}

// Determinismus (ADR-0007): wiederholte Läufe und Läufe in mehreren Tranchen sind bitgleich.
TEST(CudaPrefilter, RepeatedRunsAndTranchesAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost batch = mko::test::multistart_batch(
        mko::test::load_residual_system("testsystem_mehrlinser.json"), 700, 5005);
    PrefilterSettings settings;
    settings.options.criteria.max_merit_factor = 100.0;
    TraceEngine engine(0);
    PrefilterResult first;
    const mko::cuda::TraceRunInfo info = engine.prefilter_variants(batch, settings, first);
    EXPECT_EQ(info.tranches, 1);
    EXPECT_GT(first.kept, 0);
    EXPECT_LT(first.kept, batch.variants());
    PrefilterResult out;
    for (int run = 1; run < 10; ++run) {
        engine.prefilter_variants(batch, settings, out);
        EXPECT_TRUE(identical(out, first)) << "Lauf " << run;
    }
    // Vier Tranchen (Grenze für genau 200 Varianten)
    EngineOptions limited;
    limited.memory_limit_bytes = mko::cuda::prefilter_device_bytes(batch.topology(), 200);
    limited.reserve = mko::MemoryReserve{0, 0.0};
    TraceEngine small(0, limited);
    const mko::cuda::TraceRunInfo split = small.prefilter_variants(batch, settings, out);
    EXPECT_EQ(split.tranches, 4);
    EXPECT_EQ(split.tranche_variants, 200);
    EXPECT_TRUE(identical(out, first));
    // Wechsel zwischen Vorfilterung und FP64-Lauf auf derselben Engine
    const TraceOutput m1 = engine.run(batch, TraceMode::Merit);
    engine.prefilter_variants(batch, settings, out);
    EXPECT_TRUE(identical(out, first));
    const TraceOutput m2 = engine.run(batch, TraceMode::Merit);
    EXPECT_TRUE(same_bytes(m1.merit.merit, m2.merit.merit));
}

// Grenzen und Fehler: Shared Memory, ungültige Optionen.
TEST(CudaPrefilter, LimitsAndInvalidUse) {
    MKO_REQUIRE_GPU();
    const int plain = mko::cuda::prefilter_max_samples();
    const int ghosts = mko::cuda::prefilter_max_samples(64, 32);
    EXPECT_GT(plain, 7000);  // OPL in float: 5 Byte pro Stützstelle
    EXPECT_GT(ghosts, 4000); // ungünstigster Fall mit Geisterplatten
    EXPECT_LT(ghosts, plain);
    EXPECT_THROW((void)mko::cuda::prefilter_max_samples(-1, 0), std::invalid_argument);

    const SystemBatchHost batch =
        mko::test::multistart_batch(mko::test::load_residual_system("einzellinse_bk7.json"), 10, 1);
    TraceEngine engine(0);
    PrefilterSettings bad;
    bad.options.margins.boundary = -1.0;
    EXPECT_THROW((void)engine.prefilter_variants(batch, bad), std::invalid_argument);
    bad = PrefilterSettings{};
    bad.options.criteria.max_failed_fraction = std::nan("");
    EXPECT_THROW((void)engine.prefilter_variants(batch, bad), std::invalid_argument);
    bad = PrefilterSettings{};
    const std::vector<double> extra(3, 0.0);
    bad.extra_merit = extra;
    EXPECT_THROW((void)engine.prefilter_variants(batch, bad), std::invalid_argument);
    bad = PrefilterSettings{};
    bad.quadrature = {16, plain / 16 + 1};
    EXPECT_THROW((void)engine.prefilter_variants(batch, bad), std::invalid_argument);
}

// Durchsatz (Info, kein Benchmark; Messungen: bench/trace_throughput --type fp32prefilter):
// 10.000 Multistart-Varianten des Mehrlinsers, Vorfilterung gegen MERIT in FP64 und df64.
TEST(CudaPrefilter, Throughput) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost batch = mko::test::multistart_batch(
        mko::test::load_residual_system("testsystem_mehrlinser.json"), 10000, 777);
    TraceEngine engine(0);
    PrefilterResult pre;
    TraceOutput out;
    TraceSettings s64;
    TraceSettings sdf;
    sdf.precision = Precision::Df64;
    engine.prefilter_variants(batch, PrefilterSettings{}, pre); // Aufwärmen
    engine.run(batch, TraceMode::Merit, s64, out);
    engine.run(batch, TraceMode::Merit, sdf, out);
    const auto best = [](auto&& fn) {
        double t = 1e30;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            t = std::min(t, seconds_since(t0));
        }
        return t;
    };
    const double t32 = best([&] { engine.prefilter_variants(batch, PrefilterSettings{}, pre); });
    const double t64 = best([&] { engine.run(batch, TraceMode::Merit, s64, out); });
    const double tdf = best([&] { engine.run(batch, TraceMode::Merit, sdf, out); });
    const double rays = static_cast<double>(pre.stats.rays_per_variant) * batch.variants();
    std::printf("[ Info     ] %s, 10.000 Varianten (%.2e Strahlen) über die Engine: "
                "Vorfilterung FP32 %.1f ms (%.2e Strahlen/s), MERIT FP64 %.1f ms, df64 %.1f ms\n",
                engine.device_name().c_str(), rays, 1e3 * t32, rays / t32, 1e3 * t64, 1e3 * tdf);
    EXPECT_GT(pre.kept, 0);
}
