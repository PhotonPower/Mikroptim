// Weiches Strahlversagen auf der GPU (M4-T2): Stetigkeit der Merit entlang fein abgetasteter
// Geraden durch Versagensgrenzen (Modus MERIT der TraceEngine), Determinismus mit Strafresiduen
// (wiederholte Läufe, Tranchen), df64 gegen FP64 im weichen Modus und die Abnahme-Optimierung
// mit dem GPU-Backend des LM. CPU-Teil: test_soft_failure.cpp.

#include "lm_test_systems.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/multi_levenberg_marquardt.hpp"
#include "mikroptim/cuda/multi_trace_engine.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "residual_test_systems.hpp"
#include "soft_failure_test_tools.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

using mko::FailureMode;
using mko::MeritResult;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::EngineOptions;
using mko::cuda::Precision;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
using mko::test::ContinuityReport;
using mko::test::kContinuityNeighborRatio;
using mko::test::kContinuityShrink;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

ResidualSettings with_mode(FailureMode mode) {
    ResidualSettings s;
    s.failure = mode;
    return s;
}

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Merit über die TraceEngine (Modus MERIT, FP64 oder df64).
mko::test::MeritEvaluator gpu_evaluator(TraceEngine& engine, Precision precision) {
    return [&engine, precision](const SystemBatchHost& batch, const ResidualSettings& settings) {
        TraceSettings s;
        s.quadrature = settings;
        s.precision = precision;
        return engine.run(batch, TraceMode::Merit, s).merit;
    };
}

void expect_continuous(const ContinuityReport& soft, const ContinuityReport& hard) {
    ASSERT_TRUE(soft.crossed);
    ASSERT_EQ(soft.jumps.size(), 2u);
    EXPECT_LT(soft.max_ratio, kContinuityNeighborRatio);
    EXPECT_LT(soft.jumps[1], kContinuityShrink * soft.jumps[0]);
    EXPECT_LT(soft.jumps[1], 1e-5 * soft.merit_scale);
    ASSERT_EQ(hard.jumps.size(), 2u);
    EXPECT_GT(hard.max_ratio, 10.0 * kContinuityNeighborRatio);
    EXPECT_GT(hard.jumps[1], 0.5 * hard.jumps[0]);
}

} // namespace

// Abnahme (GPU): Merit entlang der Geraden durch die Totalreflexion und durch das Verfehlen einer
// Fläche, 10⁴ Punkte und eine Verfeinerung um 10⁴ (wie MeritContinuity.*Cpu), im Modus MERIT in
// FP64 und df64. Im weichen Modus stetig, im harten mit Sprung.
TEST(CudaSoftFailure, MeritContinuity) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    struct Line {
        const char* name;
        mko::test::FailureLine line;
    };
    const Line lines[] = {{"Totalreflexion", mko::test::total_reflection_line()},
                          {"Fläche verfehlt", mko::test::missed_surface_line()}};
    for (const Line& l : lines) {
        for (const Precision p : {Precision::Fp64, Precision::Df64}) {
            SCOPED_TRACE(l.name);
            SCOPED_TRACE(p == Precision::Df64 ? "df64" : "FP64");
            const mko::test::MeritEvaluator eval = gpu_evaluator(engine, p);
            const ContinuityReport soft = mko::test::continuity_across_failure(
                l.line.base, l.line.set, with_mode(FailureMode::Soft), l.line.t0, l.line.t1, 10000,
                1, eval);
            const ContinuityReport hard = mko::test::continuity_across_failure(
                l.line.base, l.line.set, with_mode(FailureMode::Hard), l.line.t0, l.line.t1, 10000,
                1, eval);
            std::printf("[ Info     ] GPU %s:\n", p == Precision::Df64 ? "df64" : "FP64");
            mko::test::print_report(l.name, soft);
            mko::test::print_report(l.name, hard);
            if (p == Precision::Df64) {
                // df64 rechnet die Prüfgrößen mit ~48 Bit; nahe der Grenze streut die Merit
                // stärker, die Stetigkeit (Nachbarsprünge) bleibt
                ASSERT_TRUE(soft.crossed);
                EXPECT_LT(soft.max_ratio, kContinuityNeighborRatio);
                EXPECT_LT(soft.jumps.back(), 1e-3 * soft.merit_scale);
                ASSERT_FALSE(hard.jumps.empty());
                EXPECT_GT(hard.max_ratio, 10.0 * kContinuityNeighborRatio);
            } else {
                expect_continuous(soft, hard);
            }
        }
    }
}

// ADR-0007 mit Strafresiduen: zehn Läufe und vier Tranchen bitgleich (RESIDUALS mit
// Strafresiduen, MERIT mit versagten und hart versagten Strahlen), FP64 und df64.
TEST(CudaSoftFailure, RepeatedRunsAndTranchesAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 800, 12012);
    for (const Precision p : {Precision::Fp64, Precision::Df64}) {
        TraceEngine engine(0);
        TraceSettings s;
        s.precision = p;
        s.opd = true;
        const TraceOutput r0 = engine.run(batch, TraceMode::Residuals, s);
        const TraceOutput m0 = engine.run(batch, TraceMode::Merit, s);
        ASSERT_GT(r0.residuals.penalty_rows, 0);
        std::size_t soft = 0;
        for (const std::uint8_t st : r0.residuals.status) {
            soft += mko::is_soft_failure(static_cast<TraceStatus>(st)) ? 1 : 0;
        }
        EXPECT_GT(soft, 0u);
        for (int run = 1; run < 10; ++run) {
            const TraceOutput r = engine.run(batch, TraceMode::Residuals, s);
            const TraceOutput m = engine.run(batch, TraceMode::Merit, s);
            EXPECT_TRUE(same_bytes(r.residuals.residuals, r0.residuals.residuals)) << run;
            EXPECT_TRUE(same_bytes(r.residuals.status, r0.residuals.status)) << run;
            EXPECT_TRUE(same_bytes(m.merit.merit, m0.merit.merit)) << run;
            EXPECT_TRUE(same_bytes(m.merit.hard_failed, m0.merit.hard_failed)) << run;
        }
        EngineOptions limited;
        limited.memory_limit_bytes = engine.device_bytes() / 3;
        limited.reserve = mko::MemoryReserve{0, 0.0};
        TraceEngine small(0, limited);
        const TraceOutput rt = small.run(batch, TraceMode::Residuals, s);
        EXPECT_GT(rt.info.tranches, 1);
        EXPECT_TRUE(same_bytes(rt.residuals.residuals, r0.residuals.residuals));
        const TraceOutput mt = small.run(batch, TraceMode::Merit, s);
        EXPECT_TRUE(same_bytes(mt.merit.merit, m0.merit.merit));
        std::printf("[ Info     ] %s: %zu weich versagte Strahlen, %d Tranchen bitgleich\n",
                    p == Precision::Df64 ? "df64" : "FP64", soft, rt.info.tranches);
    }
}

// df64 gegen FP64 im weichen Modus (1.000 gestörte Varianten des Mehrlinsers): gleicher Status,
// Strafgrößen innerhalb max(kDf64VsFp64, kCpuGpuPenaltyRelative · |π|) bis auf Strahlen tief im
// Versagensbereich (Ausgabe), Wellenfrontresiduen gültiger Strahlen wie in M7-T1.
TEST(CudaSoftFailure, Df64MatchesFp64) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, 2002);
    const ResidualLayout layout(batch.topology());
    const std::vector<double> units = mko::test::residual_units(batch, layout);
    const std::vector<double> punits = mko::test::penalty_units(batch, layout);
    TraceEngine engine(0);
    TraceSettings s;
    s.opd = true;
    const ResidualResult a = engine.run(batch, TraceMode::Residuals, s).residuals;
    s.precision = Precision::Df64;
    const ResidualResult b = engine.run(batch, TraceMode::Residuals, s).residuals;
    std::size_t flips = 0;
    std::size_t soft = 0;
    std::size_t penalty_over = 0;
    double max_dpi_rel = 0.0;
    double max_dr_ok = 0.0;
    double max_dr_soft = 0.0;
    for (int v = 0; v < batch.variants(); ++v) {
        for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
            const TraceStatus sa = a.status_at(v, i);
            if (sa != b.status_at(v, i)) {
                ++flips;
                continue;
            }
            const double u = units[static_cast<std::size_t>(i)];
            const double pu = punits[static_cast<std::size_t>(i)];
            if (pu > 0.0) {
                const double pa = a.penalty(v, i) / pu;
                const double pb = b.penalty(v, i) / pu;
                const double d = std::abs(pa - pb);
                if (d > mko::test::kDf64VsFp64) {
                    max_dpi_rel = std::max(max_dpi_rel, d / std::abs(pa));
                }
                penalty_over += d > std::max(mko::test::kDf64VsFp64,
                                             mko::test::kCpuGpuPenaltyRelative * std::abs(pa))
                                    ? 1
                                    : 0;
            }
            if (u > 0.0) {
                const double dr = std::abs(a.residual(v, i) - b.residual(v, i)) / u;
                if (sa == TraceStatus::Ok) {
                    max_dr_ok = std::max(max_dr_ok, dr);
                    EXPECT_LE(dr, std::max(mko::test::kDf64VsFp64, mko::test::kDf64VsFp64Relative *
                                                                       std::abs(a.opd_at(v, i))))
                        << "v = " << v << ", i = " << i;
                } else if (mko::is_soft_failure(sa)) {
                    ++soft;
                    max_dr_soft = std::max(max_dr_soft, dr);
                }
            }
        }
    }
    std::printf("[ Info     ] df64 gegen FP64 (weich): %zu Statuswechsel, %zu weich versagt, max "
                "|dr| gültig %.3e mm, weich versagt %.3e mm, max |dπ|/π %.2e, %zu Strafen über der "
                "Schranke\n",
                flips, soft, max_dr_ok, max_dr_soft, max_dpi_rel, penalty_over);
    EXPECT_EQ(flips, 0u);
    EXPECT_GT(soft, 0u);
    EXPECT_EQ(penalty_over, 0u);
}

// Abnahme auf der GPU: dasselbe Startsystem wie SoftFailure.AcceptanceOptimizationCpu (30 %
// weich versagte Strahlen) mit optimize_lm_gpu und Standardoptionen: versagensfrei, Merit
// gesunken, Status konvergiert; dazu die 108 Startpunkte der Robustheitsprüfung gemeinsam in
// einem Aufruf (alle versagensfrei, gemeinsam bitgleich zu einzeln für eine Stichprobe).
TEST(CudaSoftFailure, AcceptanceOptimizationGpu) {
    MKO_REQUIRE_GPU();
    const mko::test::FdProblem p =
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json"));
    TraceEngine engine(0);
    mko::cuda::LmGpuStats stats;
    const mko::LmResult res =
        mko::cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, mko::LmOptions{}, &stats);
    const mko::LmRunResult& r = res.runs[0];
    const mko::LmRunResult cpu = mko::optimize_lm_cpu(p.set, p.base, p.x0).runs[0];
    std::printf("[ Info     ] GPU: F %.6e -> %.6e, %d Zyklen, %d Runden bis w_fail = %.0e, Status "
                "%s, x = (%.6f, %.6f, %.4f); CPU: F -> %.6e, %d Zyklen, %d Runden\n",
                r.history.front(), r.merit, r.iterations, r.penalty_rounds, r.fail_weight,
                mko::to_string(r.status), r.x[0], r.x[1], r.x[2], cpu.merit, cpu.iterations,
                cpu.penalty_rounds);
    EXPECT_EQ(r.failed_rays, 0);
    EXPECT_EQ(r.hard_failed_rays, 0);
    EXPECT_LT(r.merit, r.history.front());
    EXPECT_TRUE(mko::lm_converged(r.status)) << mko::to_string(r.status);
    EXPECT_GE(stats.merit_calls, r.iterations);
    EXPECT_NEAR(r.x[0], cpu.x[0], 1e-3 * std::abs(cpu.x[0]));

    const std::vector<double> starts = mko::test::soft_failure_start_grid();
    const mko::LmResult group = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, starts);
    int clean = 0;
    for (std::size_t i = 0; i < group.runs.size(); ++i) {
        const mko::LmRunResult& g = group.runs[i];
        clean += g.failed_rays == 0 ? 1 : 0;
        if (g.failed_rays > 0) {
            std::printf(
                "[ Info     ] Start %zu (%.4f, %.3f, %.1f): %lld versagt (%lld hart), %s nach "
                "%d Zyklen, %d Runden, w_fail = %.0e, F = %.6e, x = (%.6f, %.6f, %.4f)\n",
                i, starts[3 * i], starts[3 * i + 1], starts[3 * i + 2],
                static_cast<long long>(g.failed_rays), static_cast<long long>(g.hard_failed_rays),
                mko::to_string(g.status), g.iterations, g.penalty_rounds, g.fail_weight, g.merit,
                g.x[0], g.x[1], g.x[2]);
        }
    }
    std::printf("[ Info     ] GPU: %d von %zu Startpunkten versagensfrei\n", clean,
                group.runs.size());
    EXPECT_GE(clean, static_cast<int>(group.runs.size()) - 2); // siehe AcceptanceManyStartsCpu
    for (const int i : {0, 41, 107}) {
        const mko::LmRunResult single =
            mko::cuda::optimize_lm_gpu(
                engine, p.set, p.base,
                std::span<const double>(starts).subspan(3 * static_cast<std::size_t>(i), 3))
                .runs[0];
        const mko::LmRunResult& g = group.runs[static_cast<std::size_t>(i)];
        EXPECT_EQ(std::memcmp(single.x.data(), g.x.data(), 3 * sizeof(double)), 0) << i;
        EXPECT_EQ(single.history, g.history) << i;
        EXPECT_EQ(single.penalty_rounds, g.penalty_rounds) << i;
    }
}

// Mehrere GPUs (M7-T3) mit dem Strafverfahren: optimize_lm_multi_gpu auf zwei logischen Slots
// (feste ungleiche Anteile) und mit gemessenen Anteilen auf drei Slots ist pro Lauf bitgleich
// zu optimize_lm_gpu auf einer Engine, auch über mehrere Runden (w_fail, Runden, hart versagte
// Strahlen); die Statistik pro Slot summiert alle Runden.
TEST(CudaSoftFailure, MultiGpuRoundsBitIdenticalToSingleGpu) {
    MKO_REQUIRE_GPU();
    const mko::test::FdProblem p =
        mko::test::make_fd_problem(mko::test::load_residual_system("weiches_versagen_start.json"));
    const std::vector<double> grid = mko::test::soft_failure_start_grid();
    std::vector<double> starts; // jeder dritte Startpunkt (36 Läufe)
    for (std::size_t i = 0; 3 * i + 2 < grid.size(); i += 3) {
        starts.insert(starts.end(), grid.begin() + static_cast<std::ptrdiff_t>(3 * i),
                      grid.begin() + static_cast<std::ptrdiff_t>(3 * i + 3));
    }
    TraceEngine engine(0);
    mko::cuda::LmGpuStats single_stats;
    const mko::LmResult reference =
        mko::cuda::optimize_lm_gpu(engine, p.set, p.base, starts, {}, &single_stats);
    int multi_round = 0;
    for (const mko::LmRunResult& r : reference.runs) {
        multi_round += r.penalty_rounds > 1 ? 1 : 0;
    }
    EXPECT_GT(multi_round, 0); // das Strafverfahren wird wirklich geprüft
    for (const auto& w : std::vector<std::vector<double>>{{1.0, 3.0}, {}}) {
        mko::cuda::MultiEngineOptions o;
        o.devices.assign(w.empty() ? 3 : 2, 0);
        o.weights = w;
        mko::cuda::MultiTraceEngine multi(o);
        std::vector<mko::cuda::LmGpuStats> stats;
        const mko::LmResult result =
            mko::cuda::optimize_lm_multi_gpu(multi, p.set, p.base, starts, {}, &stats);
        ASSERT_EQ(result.runs.size(), reference.runs.size());
        EXPECT_EQ(result.cycles, reference.cycles);
        for (std::size_t i = 0; i < reference.runs.size(); ++i) {
            EXPECT_TRUE(mko::test::same_run(result.runs[i], reference.runs[i])) << "Lauf " << i;
        }
        ASSERT_EQ(static_cast<int>(stats.size()), multi.size());
        std::int64_t merit_points = 0;
        for (const mko::cuda::LmGpuStats& st : stats) {
            merit_points += st.merit_points;
        }
        EXPECT_EQ(merit_points, single_stats.merit_points);
    }
    std::printf("[ Info     ] Mehrere GPUs: %zu Läufe bitgleich, davon %d mit mehreren Runden\n",
                reference.runs.size(), multi_round);
}
