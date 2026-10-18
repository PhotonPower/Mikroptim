// Gebündelter Levenberg-Marquardt auf der GPU (M3-T4, nur mit MIKROPTIM_ENABLE_CUDA):
// Normalgleichungen bitgleich zum Host, Abnahme (Einzellinse, Rückkehr aus 1 % Störung,
// 1.000 Läufe gleichzeitig gegen einzeln), Tranchen und Teilstücke, GPU gegen CPU.

#include "lm_test_systems.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/normal_equations.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/normal_equations.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using mko::LmOptions;
using mko::LmResult;
using mko::LmRunResult;
using mko::cuda::DeviceFdJacobian;
using mko::cuda::DeviceNormalEquations;
using mko::cuda::EngineOptions;
using mko::cuda::LmGpuStats;
using mko::cuda::TraceEngine;
using mko::test::FdProblem;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

LmOptions paraxial_options(const FdProblem& p) {
    LmOptions o;
    o.jacobian.extra = mko::test::paraxial_extra(p);
    return o;
}

// Gram-Matrizen vom Gerät gegen normal_equations_host auf dem heruntergeladenen J: bitgleich.
void expect_gram_matches_host(const FdProblem& p, int runs, const mko::ExtraResiduals& extra) {
    const int P = p.set.size();
    const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, runs, 0.02, 7);
    TraceEngine engine(0);
    DeviceFdJacobian jac;
    mko::FdJacobianOptions fo;
    fo.extra = extra;
    jac.compute(engine, p.set, p.base, x0, fo);
    DeviceNormalEquations normal;
    normal.compute(jac);
    std::vector<double> a;
    std::vector<double> g;
    normal.download(a, g);
    const mko::FdJacobian host = jac.download();
    const auto PP = static_cast<std::size_t>(P);
    const auto R = static_cast<std::size_t>(host.shape.rows());
    std::vector<double> ha(PP * PP);
    std::vector<double> hg(PP);
    for (int i = 0; i < runs; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        mko::normal_equations_host(
            std::span<const double>(host.jacobian).subspan(ii * PP * R, PP * R), host.residual(i),
            P, ha, hg);
        for (std::size_t k = 0; k < PP * PP; ++k) {
            ASSERT_TRUE(mko::test::same_bits(a[ii * PP * PP + k], ha[k]))
                << "Lauf " << i << ", A " << k;
        }
        for (std::size_t k = 0; k < PP; ++k) {
            ASSERT_TRUE(mko::test::same_bits(g[ii * PP + k], hg[k])) << "Lauf " << i << ", g " << k;
        }
    }
    // r₀ᵀr₀ in der letzten Ecke der Gram-Matrix
    const std::vector<double> gram = normal.download_gram();
    const std::size_t n = PP + 1;
    for (int i = 0; i < runs; ++i) {
        const auto r0 = host.residual(i);
        EXPECT_TRUE(mko::test::same_bits(
            gram[static_cast<std::size_t>(i) * n * n + n * n - 1],
            mko::gram_dot(r0.data(), r0.data(), static_cast<std::int64_t>(r0.size()))));
    }
}

// Läufe gemeinsam gegen einzeln, bitgleich in allen Feldern.
void expect_same_runs(const LmResult& group, const std::vector<LmResult>& singles) {
    ASSERT_EQ(group.runs.size(), singles.size());
    for (std::size_t i = 0; i < singles.size(); ++i) {
        ASSERT_EQ(singles[i].runs.size(), 1u);
        EXPECT_TRUE(mko::test::same_run(group.runs[i], singles[i].runs[0])) << "Lauf " << i;
    }
}

} // namespace

// Ein Kachelpaar (Mehrlinser, P + 1 = 14) und mehrere Kacheln (Linsenstapel, P + 1 = 30, drei
// Kachelpaare): Gerät bitgleich zur FMA-Kette auf dem Host.
TEST(CudaNormalEquations, BitwiseEqualToHost) {
    MKO_REQUIRE_GPU();
    const FdProblem m = mko::test::mehrlinser_problem();
    expect_gram_matches_host(m, 5, mko::test::mehrlinser_extra(m.set));
    const FdProblem s = mko::test::lens_stack_problem(7);
    ASSERT_EQ(s.set.size(), 29);
    expect_gram_matches_host(s, 3, {});
}

// Abnahme 1 auf der GPU: Formfaktor im Bereich 0,74 ± 0,05; dasselbe Optimum wie auf der CPU.
TEST(CudaLm, LensBendingMinimalSphericalAberration) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::lens_bending_problem(mko::test::kLensFocalWeight);
    const LmOptions o = paraxial_options(p);
    TraceEngine engine(0);
    const LmResult gpu = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, o);
    const LmResult cpu = mko::optimize_lm_cpu(p.set, p.base, p.x0, o);
    const LmRunResult& r = gpu.runs[0];
    const double q = mko::test::shape_factor(r.x);
    const double qc = mko::test::shape_factor(cpu.runs[0].x);
    std::printf("[GPU] Einzellinse: q = %.6f (CPU %.6f, Dünnlinse %.6f), F = %.10e (CPU %.10e), "
                "%d Zyklen (CPU %d), Status %s\n",
                q, qc, mko::test::thin_lens_best_shape(p.base.index(0, 1, 0)), r.merit,
                cpu.runs[0].merit, r.iterations, cpu.runs[0].iterations, mko::to_string(r.status));
    EXPECT_TRUE(mko::lm_converged(r.status)) << mko::to_string(r.status);
    EXPECT_NEAR(q, 0.74, 0.05);
    EXPECT_EQ(r.failed_rays, 0);
    EXPECT_TRUE(p.set.feasible(r.x));
    // GPU und CPU am selben Optimum (Rundungsunterschiede ändern den Weg, nicht das Ziel)
    EXPECT_NEAR(q, qc, mko::test::kLmShapeAgreement);
    EXPECT_LE(std::abs(r.merit / cpu.runs[0].merit - 1.0), mko::test::kLmReturnRelative);
}

// Abnahme 2 auf der GPU: Mehrlinser (13 Variablen, mehrere an Schranken) optimiert, dann jede
// Variable um 1 % gestört (vier Vorzeichenmuster, gemeinsam): F_end ≤ F_opt·(1 + 1·10⁻⁶).
TEST(CudaLm, ReturnsFromOnePercentPerturbation) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_lm_problem(mko::test::kMehrlinserFocalWeight);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 500;
    TraceEngine engine(0);
    const LmResult opt = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, o);
    const LmRunResult& best = opt.runs[0];
    ASSERT_TRUE(mko::lm_converged(best.status)) << mko::to_string(best.status);
    const std::vector<double> starts = mko::test::one_percent_starts(p.set, best.x);
    const LmResult back = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, starts, o);
    std::printf("[GPU] Mehrlinser: F %.6e -> F_opt %.12e (%d Zyklen, %s)\n", best.history.front(),
                best.merit, best.iterations, mko::to_string(best.status));
    for (const LmRunResult& r : back.runs) {
        std::printf("[GPU]   aus 1 %%: F_start %.6e, F_end %.12e, relativ %+.3e, %d Zyklen, %s\n",
                    r.history.front(), r.merit, r.merit / best.merit - 1.0, r.iterations,
                    mko::to_string(r.status));
        EXPECT_TRUE(mko::lm_converged(r.status)) << mko::to_string(r.status);
        EXPECT_LE(r.merit, best.merit * (1.0 + mko::test::kLmReturnRelative));
        EXPECT_TRUE(p.set.feasible(r.x));
        EXPECT_EQ(r.failed_rays, 0);
    }
}

// Abnahme 3: 1.000 Läufe der Einzellinse (Startpunkte ±5 % gestört) gleichzeitig liefern
// bitgleich dieselben Ergebnisse (x, F, Verlauf, Zähler, Status) wie jeder Lauf einzeln; auch
// mit erzwungenen Tranchen der Engine und Teilstücken der Jacobi-Matrizen.
TEST(CudaLm, ThousandRunsBitwiseEqualToSingleRuns) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::lens_bending_problem(mko::test::kLensFocalWeight);
    const int P = p.set.size();
    const int runs = 1000;
    const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, runs, 0.05, 2026);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 200;
    TraceEngine engine(0);
    LmGpuStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    const LmResult group = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o, &stats);
    const double t_group = seconds_since(t0);

    std::vector<LmResult> singles;
    singles.reserve(static_cast<std::size_t>(runs));
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) {
        singles.push_back(
            mko::cuda::optimize_lm_gpu(engine, p.set, p.base, mko::test::run_start(x0, P, i), o));
    }
    const double t_single = seconds_since(t1);
    expect_same_runs(group, singles);

    int converged = 0;
    long long cycles = 0;
    int min_it = 1 << 30;
    int max_it = 0;
    for (const LmRunResult& r : group.runs) {
        converged += mko::lm_converged(r.status) ? 1 : 0;
        cycles += r.iterations;
        min_it = std::min(min_it, r.iterations);
        max_it = std::max(max_it, r.iterations);
        EXPECT_TRUE(p.set.feasible(r.x));
    }
    std::printf("[GPU] %d Läufe gemeinsam: %.2f s (%d Zyklen der Schleife, %.0f Lauf-Zyklen/s, "
                "%d J-Aufrufe), einzeln: %.2f s; konvergiert %d, Zyklen pro Lauf %d … %d\n",
                runs, t_group, group.cycles, static_cast<double>(cycles) / t_group,
                stats.jacobian_calls, t_single, converged, min_it, max_it);
    EXPECT_EQ(converged, runs);
    EXPECT_LT(min_it, max_it); // verschieden schnell konvergierende Läufe im selben Batch

    // Erzwungene Aufteilung: Speichergrenze so, dass die Jacobi-Matrizen in viele Teilstücke
    // (je höchstens 8 Läufe) und die Merit-Batches in mehrere Tranchen der Engine zerfallen.
    // (Tranchen der Engine innerhalb einer FD-Rechnung sind in M3-T2 getestet.)
    EngineOptions limited;
    limited.memory_limit_bytes = std::size_t{64} << 10; // 64 KiB
    limited.reserve = mko::MemoryReserve{0, 0.0};
    TraceEngine small(0, limited);
    LmGpuStats small_stats;
    const LmResult tranched = mko::cuda::optimize_lm_gpu(small, p.set, p.base, x0, o, &small_stats);
    std::printf("[GPU]   mit Grenze 64 KiB: %d Teilstücke in %d J-Aufrufen (höchstens %d Läufe), "
                "%lld Tranchen der Engine für J, %lld für %d Merit-Aufrufe\n",
                small_stats.jacobian_chunks, small_stats.jacobian_calls,
                small_stats.max_runs_per_chunk,
                static_cast<long long>(small_stats.jacobian_tranches),
                static_cast<long long>(small_stats.merit_tranches), small_stats.merit_calls);
    EXPECT_GT(small_stats.jacobian_chunks, small_stats.jacobian_calls);
    EXPECT_GE(small_stats.jacobian_tranches, small_stats.jacobian_chunks);
    EXPECT_GT(small_stats.merit_tranches, small_stats.merit_calls);
    expect_same_runs(tranched, singles);
}

// Mehrere Variablen an Schranken: 40 Läufe des Mehrlinsers gemeinsam gegen einzeln bitgleich;
// alle Endpunkte zulässig, mindestens eine Variable auf einer Schranke.
TEST(CudaLm, MehrlinserRunsBitwiseEqualToSingleRuns) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_lm_problem(mko::test::kMehrlinserFocalWeight);
    const int P = p.set.size();
    const int runs = 40;
    const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, runs, 0.03, 11);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 60;
    TraceEngine engine(0);
    const LmResult group = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o);
    std::vector<LmResult> singles;
    for (int i = 0; i < runs; ++i) {
        singles.push_back(
            mko::cuda::optimize_lm_gpu(engine, p.set, p.base, mko::test::run_start(x0, P, i), o));
    }
    expect_same_runs(group, singles);
    int on_bound = 0;
    for (const LmRunResult& r : group.runs) {
        EXPECT_TRUE(p.set.feasible(r.x));
        EXPECT_LT(r.merit, r.history.front());
        for (int k = 0; k < P; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            on_bound += (r.x[kk] == p.set.lower()[kk] || r.x[kk] == p.set.upper()[kk]) ? 1 : 0;
        }
    }
    EXPECT_GT(on_bound, 0);
}

// Versagte Strahlen: Start, an dem die äußeren Ringe totalreflektieren. Der Lauf bricht nicht
// ab, die Zahl versagter Strahlen steigt nie, F bleibt endlich; Ergebnis wie auf der CPU.
TEST(CudaLm, RunWithFailedRays) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::tir_start_problem(1.05);
    TraceEngine engine(0);
    LmOptions o;
    o.max_iterations = 50;
    std::vector<std::int64_t> failed;
    o.on_iteration = [&failed](const mko::LmIterationInfo& info) {
        if (info.accepted) {
            failed.push_back(info.failed_rays);
        }
    };
    const LmResult gpu = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, o);
    const LmRunResult& r = gpu.runs[0];
    std::printf("[GPU] Totalreflexion: versagt %lld -> %lld, F %.6e -> %.6e, %d Zyklen, %s\n",
                static_cast<long long>(failed.empty() ? -1 : failed.front()),
                static_cast<long long>(r.failed_rays), r.history.front(), r.merit, r.iterations,
                mko::to_string(r.status));
    ASSERT_FALSE(failed.empty());
    EXPECT_GT(failed.front(), 0);
    EXPECT_NE(r.status, mko::LmStatus::NumericalError);
    EXPECT_TRUE(std::isfinite(r.merit));
    EXPECT_LT(r.merit, r.history.front());
    EXPECT_TRUE(p.set.feasible(r.x));
    for (std::size_t k = 1; k < failed.size(); ++k) {
        EXPECT_LE(failed[k], failed[k - 1]);
    }
    const LmResult cpu = mko::optimize_lm_cpu(p.set, p.base, p.x0, o);
    EXPECT_EQ(cpu.runs[0].failed_rays, r.failed_rays);
}

// Durchsatz (kein Benchmark, nur Größenordnung für die Doku): 1.000 Läufe des Mehrlinsers
// (13 Variablen, 680 Strahlen + 1 Zusatzresiduum, zentrale Differenzen mit 27 Varianten),
// 20 Zyklen. Prüft nur, dass alle Läufe die Merit senken.
TEST(CudaLm, ThroughputThousandMehrlinserRuns) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_lm_problem(mko::test::kMehrlinserFocalWeight);
    const int runs = 1000;
    const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, runs, 0.03, 5);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 20;
    TraceEngine engine(0);
    LmGpuStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    const LmResult res = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o, &stats);
    const double t = seconds_since(t0);
    long long cycles = 0;
    for (const LmRunResult& r : res.runs) {
        cycles += r.iterations;
        EXPECT_LT(r.merit, r.history.front());
    }
    std::printf("[GPU] Durchsatz Mehrlinser: %d Läufe, %lld Lauf-Zyklen in %.2f s (%.0f "
                "Lauf-Zyklen/s, %.1f Zyklen der Schleife/s), %lld Jacobi-Läufe in %d Teilstücken, "
                "%lld Merit-Punkte\n",
                runs, cycles, t, static_cast<double>(cycles) / t,
                static_cast<double>(res.cycles) / t, static_cast<long long>(stats.jacobian_runs),
                stats.jacobian_chunks, static_cast<long long>(stats.merit_points));
}
