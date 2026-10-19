// Adjungierte Jacobi im gebündelten Levenberg-Marquardt auf der GPU (M5-T3, nur mit
// MIKROPTIM_ENABLE_CUDA): Normalgleichungen aus DeviceAdjointJacobian bitgleich zum Host, gleiche
// Optimierungsergebnisse wie mit finiten Differenzen (Abnahmen aus M3-T4, M4-T2, M4-T6, M6-T1,
// M4-T3), Bitgleichheit Gruppe gegen Einzellauf, Teilstücke und mehrere (logische) GPUs,
// Laufzeit der Jacobi-Matrix bei P ≥ 60 (docs/optimierung.md, Abschnitt 4).

#include "ghost_test_systems.hpp"
#include "glass_test_systems.hpp"
#include "lm_test_systems.hpp"
#include "mechanical_test_systems.hpp"
#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/multi_levenberg_marquardt.hpp"
#include "mikroptim/cuda/multi_trace_engine.hpp"
#include "mikroptim/cuda/normal_equations.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/normal_equations.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (cuda::device_count() == 0) {                                                           \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

LmOptions with_method(LmOptions o, JacobianMethod method) {
    o.jacobian_method = method;
    return o;
}

LmOptions adjoint_paraxial_options(const FdProblem& p) {
    LmOptions o;
    o.jacobian.extra = paraxial_extra(p);
    o.jacobian_method = JacobianMethod::Adjoint;
    return o;
}

double relative(double a, double b) {
    return std::abs(a / b - 1.0);
}

void expect_same_runs(const LmResult& group, const std::vector<LmResult>& singles) {
    ASSERT_EQ(group.runs.size(), singles.size());
    for (std::size_t i = 0; i < singles.size(); ++i) {
        ASSERT_EQ(singles[i].runs.size(), 1u);
        EXPECT_TRUE(same_run(group.runs[i], singles[i].runs[0])) << "Lauf " << i;
    }
}

// Normalgleichungen aus der adjungierten Jacobi auf dem Gerät: bitgleich zur FMA-Kette auf dem
// Host über das heruntergeladene J (ein Kachelpaar beim Mehrlinser, drei beim Linsenstapel mit
// P + 1 = 30); falsche Sichten und nicht berechnete Jacobi-Matrizen werden abgelehnt.
TEST(CudaAdjointLm, NormalEquationsBitwiseEqualToHost) {
    MKO_REQUIRE_GPU();
    cuda::TraceEngine engine(0);
    for (const FdProblem& p :
         {mehrlinser_lm_problem(kMehrlinserFocalWeight), lens_stack_problem(7)}) {
        const int P = p.set.size();
        const auto n = static_cast<std::size_t>(P);
        const int runs = 5;
        const std::vector<double> x0 = perturbed_starts(p.set, p.x0, runs, 0.02, 7);
        AdjointJacobianOptions ao;
        ao.extra = paraxial_extra(p);
        cuda::DeviceAdjointJacobian jac;
        jac.compute(engine, p.set, p.base, x0, ao);
        cuda::DeviceNormalEquations normal;
        normal.compute(jac);
        std::vector<double> a;
        std::vector<double> g;
        normal.download(a, g);
        const AdjointJacobian host = jac.download();
        const auto R = static_cast<std::size_t>(host.shape.rows());
        for (int i = 0; i < runs; ++i) {
            const auto ii = static_cast<std::size_t>(i);
            std::vector<double> ah(n * n);
            std::vector<double> gh(n);
            normal_equations_host(std::span<const double>(host.jacobian).subspan(ii * n * R, n * R),
                                  host.residual(i), P, ah, gh);
            for (std::size_t k = 0; k < n * n; ++k) {
                ASSERT_TRUE(same_bits(ah[k], a[ii * n * n + k])) << P << " " << i << " " << k;
            }
            for (std::size_t k = 0; k < n; ++k) {
                ASSERT_TRUE(same_bits(gh[k], g[ii * n + k])) << P << " " << i << " " << k;
            }
        }
        // Dieselbe Sicht über DeviceJacobianRef: dieselben Bits
        cuda::DeviceNormalEquations by_ref;
        by_ref.compute(cuda::device_jacobian_ref(jac));
        EXPECT_EQ(by_ref.download_gram(), normal.download_gram());
    }
    cuda::DeviceAdjointJacobian empty;
    cuda::DeviceNormalEquations normal;
    EXPECT_THROW(normal.compute(empty), std::logic_error);
    cuda::DeviceJacobianRef bad;
    bad.device = 0;
    bad.shape = JacobianShape{1, 3, 10, 0};
    EXPECT_THROW(normal.compute(bad), std::invalid_argument);
}

// GPU-Backend: A und g mit adjungierter Jacobi gegen FD (Fehler der FD-Jacobi) und gegen das
// CPU-Backend mit adjungierter Jacobi (Rundung der Kernel), Mehrlinser und mechanische
// Bedingungen mit trace-abhängigen Zeilen (Extrema der FD-Varianten aus einem MERIT-Lauf).
TEST(CudaAdjointLm, BackendMatchesFdAndCpu) {
    MKO_REQUIRE_GPU();
    cuda::TraceEngine engine(0);
    struct Case {
        const char* name;
        FdProblem p;
        LmOptions o;
    };
    std::vector<Case> cases;
    {
        FdProblem p = mehrlinser_lm_problem(kMehrlinserFocalWeight);
        LmOptions o;
        o.jacobian.extra = paraxial_extra(p);
        cases.push_back({"Mehrlinser", std::move(p), std::move(o)});
    }
    {
        FdProblem p = mechanical_problem();
        LmOptions o = mechanical_options(p.loaded.system, 10);
        cases.push_back({"Zweilinser mechanisch", std::move(p), std::move(o)});
    }
    for (const Case& c : cases) {
        const auto n = static_cast<std::size_t>(c.p.set.size());
        const int run0 = 0;
        const std::span<const int> runs(&run0, 1);
        std::vector<double> a_fd(n * n), g_fd(n), a_ad(n * n), g_ad(n), a_cpu(n * n), g_cpu(n);
        const LmOptions oa = with_method(c.o, JacobianMethod::Adjoint);
        cuda::GpuLmBackend fd(engine, c.p.set, c.p.base, c.o);
        fd.normal_equations(runs, c.p.x0, a_fd, g_fd);
        cuda::GpuLmBackend ad(engine, c.p.set, c.p.base, oa);
        ad.normal_equations(runs, c.p.x0, a_ad, g_ad);
        CpuLmBackend cpu(c.p.set, c.p.base, oa);
        cpu.normal_equations(runs, c.p.x0, a_cpu, g_cpu);
        std::vector<double> s(n);
        for (std::size_t k = 0; k < n; ++k) {
            s[k] = std::max(std::abs(c.p.x0[k]), c.p.set.typical()[k]);
        }
        double a_max = 0.0, d_fd = 0.0, d_cpu = 0.0;
        for (std::size_t q = 0; q < n; ++q) {
            for (std::size_t k = 0; k < n; ++k) {
                const double f = s[k] * s[q];
                a_max = std::max(a_max, std::abs(a_ad[q * n + k]) * f);
                d_fd = std::max(d_fd, std::abs(a_ad[q * n + k] - a_fd[q * n + k]) * f);
                d_cpu = std::max(d_cpu, std::abs(a_ad[q * n + k] - a_cpu[q * n + k]) * f);
            }
        }
        if (!(a_max > 0.0)) {
            ADD_FAILURE() << c.name << ": A = 0";
            continue;
        }
        std::printf("[GPU] %-22s P = %2d: |A_ad − A_fd| %.2e, |A_ad − A_cpu| %.2e (skaliert, "
                    "relativ), %d Teilstück(e)\n",
                    c.name, c.p.set.size(), d_fd / a_max, d_cpu / a_max,
                    ad.stats().jacobian_chunks);
        EXPECT_LE(d_fd / a_max, kAdjointLmNormalEquations) << c.name;
        EXPECT_LE(d_cpu / a_max, kAdjointCpuGpu) << c.name;
        EXPECT_EQ(ad.jacobian_method(), JacobianMethod::Adjoint);
    }
}

// Abnahme 1 aus M3-T4 auf der GPU mit der adjungierten Jacobi: Formfaktor 0,74 ± 0,05, dasselbe
// Optimum wie mit FD auf der GPU.
TEST(CudaAdjointLm, LensBendingSameOptimumAsFd) {
    MKO_REQUIRE_GPU();
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const LmOptions oa = adjoint_paraxial_options(p);
    cuda::TraceEngine engine(0);
    const LmRunResult ad = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, oa).runs[0];
    const LmRunResult fd = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0,
                                                 with_method(oa, JacobianMethod::FiniteDifferences))
                               .runs[0];
    std::printf("[GPU] Einzellinse: adjungiert q = %.6f, F = %.12e, %d Zyklen, %s; FD q = %.6f, "
                "F = %.12e, %d Zyklen; F relativ %.2e\n",
                shape_factor(ad.x), ad.merit, ad.iterations, to_string(ad.status),
                shape_factor(fd.x), fd.merit, fd.iterations, relative(ad.merit, fd.merit));
    EXPECT_TRUE(lm_converged(ad.status)) << to_string(ad.status);
    EXPECT_NEAR(shape_factor(ad.x), 0.74, kLmShapeFactor);
    EXPECT_NEAR(shape_factor(ad.x), shape_factor(fd.x), kLmShapeAgreement);
    EXPECT_LE(relative(ad.merit, fd.merit), kLmReturnRelative);
    EXPECT_EQ(ad.failed_rays, 0);
}

// Abnahme 2 aus M3-T4 auf der GPU: F_opt wie mit FD, Rückkehr aus 1 % Störung.
TEST(CudaAdjointLm, ReturnsFromOnePercentPerturbation) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mehrlinser_lm_problem(kMehrlinserFocalWeight);
    LmOptions oa = adjoint_paraxial_options(p);
    oa.max_iterations = 500;
    cuda::TraceEngine engine(0);
    const LmRunResult best = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, oa).runs[0];
    const LmRunResult fd = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0,
                                                 with_method(oa, JacobianMethod::FiniteDifferences))
                               .runs[0];
    ASSERT_TRUE(lm_converged(best.status)) << to_string(best.status);
    double dx = 0.0;
    for (int k = 0; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        dx = std::max(dx, std::abs(best.x[kk] - fd.x[kk]) /
                              std::max(std::abs(fd.x[kk]), p.set.typical()[kk]));
    }
    std::printf("[GPU] Mehrlinser: adjungiert F_opt %.12e (%d Zyklen), FD %.12e (%d Zyklen), "
                "relativ %.2e, max|Δx|/s %.2e\n",
                best.merit, best.iterations, fd.merit, fd.iterations,
                relative(best.merit, fd.merit), dx);
    EXPECT_LE(relative(best.merit, fd.merit), kLmReturnRelative);
    EXPECT_LE(dx, kAdjointLmEndpoint);
    const LmResult back =
        cuda::optimize_lm_gpu(engine, p.set, p.base, one_percent_starts(p.set, best.x), oa);
    for (const LmRunResult& r : back.runs) {
        std::printf("[GPU]   aus 1 %%: F_end %.12e, relativ %+.3e, %d Zyklen, %s\n", r.merit,
                    r.merit / best.merit - 1.0, r.iterations, to_string(r.status));
        EXPECT_TRUE(lm_converged(r.status)) << to_string(r.status);
        EXPECT_LE(r.merit, best.merit * (1.0 + kLmReturnRelative));
        EXPECT_TRUE(p.set.feasible(r.x));
    }
}

// Abnahme 3 aus M3-T4 mit der adjungierten Jacobi: 120 Läufe gemeinsam bitgleich zu den
// Einzelläufen, auch mit erzwungenen Teilstücken (64 KiB Speichergrenze: ein Lauf pro
// Teilstück) und auf mehreren logischen GPUs (feste ungleiche Anteile).
TEST(CudaAdjointLm, RunsBitwiseEqualToSingleRunsChunksAndSlots) {
    MKO_REQUIRE_GPU();
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const int P = p.set.size();
    const int runs = 120;
    const std::vector<double> x0 = perturbed_starts(p.set, p.x0, runs, 0.05, 2026);
    LmOptions o = adjoint_paraxial_options(p);
    o.max_iterations = 200;
    cuda::TraceEngine engine(0);
    cuda::LmGpuStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    const LmResult group = cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o, &stats);
    const double t_group =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<LmResult> singles;
    for (int i = 0; i < runs; ++i) {
        singles.push_back(cuda::optimize_lm_gpu(engine, p.set, p.base, run_start(x0, P, i), o));
    }
    expect_same_runs(group, singles);
    int converged = 0;
    for (const LmRunResult& r : group.runs) {
        converged += lm_converged(r.status) ? 1 : 0;
    }
    std::printf("[GPU] %d Läufe adjungiert gemeinsam: %.2f s, %d J-Aufrufe, %d Teilstücke, "
                "konvergiert %d\n",
                runs, t_group, stats.jacobian_calls, stats.jacobian_chunks, converged);
    EXPECT_EQ(converged, runs);

    cuda::EngineOptions limited;
    limited.memory_limit_bytes = std::size_t{64} << 10;
    limited.reserve = MemoryReserve{0, 0.0};
    cuda::TraceEngine small(0, limited);
    cuda::LmGpuStats small_stats;
    const LmResult chunked = cuda::optimize_lm_gpu(small, p.set, p.base, x0, o, &small_stats);
    std::printf("[GPU]   mit Grenze 64 KiB: %d Teilstücke in %d J-Aufrufen (höchstens %d Läufe)\n",
                small_stats.jacobian_chunks, small_stats.jacobian_calls,
                small_stats.max_runs_per_chunk);
    EXPECT_GT(small_stats.jacobian_chunks, small_stats.jacobian_calls);
    EXPECT_EQ(small_stats.max_runs_per_chunk, 1);
    expect_same_runs(chunked, singles);

    cuda::MultiEngineOptions mo;
    mo.devices = {0, 0};
    mo.weights = {1.0, 3.0};
    cuda::MultiTraceEngine multi(mo);
    std::vector<cuda::LmGpuStats> slot_stats;
    const LmResult distributed =
        cuda::optimize_lm_multi_gpu(multi, p.set, p.base, x0, o, &slot_stats);
    expect_same_runs(distributed, singles);
    ASSERT_EQ(slot_stats.size(), 2u);
    EXPECT_GT(slot_stats[0].jacobian_runs, 0);
    EXPECT_GT(slot_stats[1].jacobian_runs, 0);
}

// Weiches Versagen (Abnahme M4-T2) auf der GPU mit der adjungierten Jacobi: versagensfrei,
// Endmerit wie mit FD.
TEST(CudaAdjointLm, SoftFailureAcceptance) {
    MKO_REQUIRE_GPU();
    const FdProblem p = make_fd_problem(load_residual_system("weiches_versagen_start.json"));
    cuda::TraceEngine engine(0);
    const LmRunResult ad = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0,
                                                 with_method(LmOptions{}, JacobianMethod::Adjoint))
                               .runs[0];
    const LmRunResult fd = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0).runs[0];
    std::printf("[GPU] weiches Versagen: adjungiert F -> %.12e, %d Zyklen, %d Runden, %s; FD "
                "%.12e, %d Zyklen, %d Runden; relativ %.2e\n",
                ad.merit, ad.iterations, ad.penalty_rounds, to_string(ad.status), fd.merit,
                fd.iterations, fd.penalty_rounds, relative(ad.merit, fd.merit));
    EXPECT_EQ(ad.failed_rays, 0);
    EXPECT_EQ(ad.hard_failed_rays, 0);
    EXPECT_TRUE(lm_converged(ad.status)) << to_string(ad.status);
    EXPECT_LE(relative(ad.merit, fd.merit), kLmReturnRelative);
}

// Geisterplatten (M4-T6) auf der GPU mit der adjungierten Jacobi: Exaktheit und Bedarf wie mit
// FD, Endmerit wie mit FD.
TEST(CudaAdjointLm, GhostExactnessAndNeed) {
    MKO_REQUIRE_GPU();
    cuda::TraceEngine engine(0);
    {
        const FdProblem p = ghost_exact_problem();
        const GhostPenalty pen = ghost_penalty_of(p.loaded.system);
        LmRunResult fd;
        const ExactnessResult e = ghost_exactness(p, pen, [&](const LmOptions& o) {
            fd = cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0, o).runs.at(0);
            return cuda::optimize_lm_gpu(engine, p.set, p.base, p.x0,
                                         with_method(o, JacobianMethod::Adjoint))
                .runs.at(0);
        });
        print_exactness("GPU adjungiert", p, e);
        std::printf("[GPU] FD: F = %.12e, %d Zyklen; relativ %.2e\n", fd.merit, fd.iterations,
                    relative(e.run.merit, fd.merit));
        EXPECT_TRUE(lm_converged(e.run.status)) << to_string(e.run.status);
        EXPECT_LT(max_ghost(p.set, e.run.x), kGhostExactCoefficient);
        EXPECT_LT(e.threshold_ratio, 1.0);
        EXPECT_LE(e.max_stationarity_error, kGhostStationarity);
        EXPECT_LE(relative(e.run.merit, fd.merit), kLmReturnRelative);
    }
    {
        const FdProblem p = ghost_need_problem();
        LmRunResult fd_run;
        const NeedResult n = ghost_need(p, [&](const FdProblem& q, const LmOptions& o) {
            fd_run = cuda::optimize_lm_gpu(engine, q.set, q.base, q.x0, o).runs.at(0);
            return cuda::optimize_lm_gpu(engine, q.set, q.base, q.x0,
                                         with_method(o, JacobianMethod::Adjoint))
                .runs.at(0);
        });
        print_need("GPU adjungiert", p, n);
        std::printf("[GPU] FD mit Geisterplatten: F = %.12e, %d Zyklen; relativ %.2e\n",
                    fd_run.merit, fd_run.iterations, relative(n.run.merit, fd_run.merit));
        check_need(p, n);
        EXPECT_LE(relative(n.run.merit, fd_run.merit), kLmReturnRelative);
    }
}

// Glas als Variable (M6-T1) auf der GPU mit der adjungierten Jacobi (Spalten (1, ω, ω²)):
// Penalty-Fortsetzung endet auf dem Rand der Hülle, Endmerit wie mit FD.
TEST(CudaAdjointLm, GlassStaysInsideHull) {
    MKO_REQUIRE_GPU();
    const GlassProblem g = glass_problem();
    cuda::TraceEngine engine(0);
    const auto optimize_with = [&](JacobianMethod method) {
        return [&engine, &g, method](const LmOptions& o, std::span<const double> x0) {
            return cuda::optimize_lm_gpu(engine, g.p.set, g.p.base, x0, with_method(o, method));
        };
    };
    const std::vector<GlassStage> ad =
        glass_continuation(g, optimize_with(JacobianMethod::Adjoint));
    const std::vector<GlassStage> fd =
        glass_continuation(g, optimize_with(JacobianMethod::FiniteDifferences));
    ASSERT_EQ(ad.size(), fd.size());
    for (std::size_t i = 0; i < ad.size(); ++i) {
        std::printf("[GPU]   w = %8.1e: adjungiert F -> %.10e, %4d Zyklen, %-16s Abstand %.3e; "
                    "FD %.10e, %4d Zyklen\n",
                    ad[i].weight, ad[i].run.merit, ad[i].run.iterations,
                    to_string(ad[i].run.status), ad[i].distance, fd[i].run.merit,
                    fd[i].run.iterations);
        EXPECT_TRUE(lm_converged(ad[i].run.status)) << ad[i].weight;
        EXPECT_EQ(ad[i].run.failed_rays, 0);
    }
    EXPECT_LE(ad.back().distance, kGlassHullPenaltyDistance);
    EXPECT_LE(relative(ad.back().run.merit, fd.back().run.merit), kLmReturnRelative);
}

// Mechanische Bedingungen (M4-T3) auf der GPU mit der adjungierten Jacobi: die trace-abhängigen
// Zeilen bleiben FD (hybrid). Fortsetzung hält alle Grenzen ein; die Endpunkte beider Verfahren
// sind Minima für beide Verfahren.
TEST(CudaAdjointLm, MechanicalKeepsAllLimits) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mechanical_problem();
    cuda::TraceEngine engine(0);
    const auto optimize_with = [&](JacobianMethod method) {
        return [&engine, &p, method](const LmOptions& o, std::span<const double> x0) {
            return cuda::optimize_lm_gpu(engine, p.set, p.base, x0, with_method(o, method));
        };
    };
    const std::vector<MechanicalStage> ad =
        mechanical_continuation(p, optimize_with(JacobianMethod::Adjoint));
    const std::vector<MechanicalStage> fd =
        mechanical_continuation(p, optimize_with(JacobianMethod::FiniteDifferences));
    ASSERT_EQ(ad.size(), fd.size());
    for (std::size_t i = 0; i < ad.size(); ++i) {
        std::printf("[GPU]   w = %8.1e: adjungiert F -> %.10e, %3d Zyklen, %-16s; FD %.10e, %3d "
                    "Zyklen\n",
                    ad[i].weight, ad[i].run.merit, ad[i].run.iterations,
                    to_string(ad[i].run.status), fd[i].run.merit, fd[i].run.iterations);
        EXPECT_TRUE(lm_converged(ad[i].run.status)) << ad[i].weight;
        EXPECT_EQ(ad[i].run.failed_rays, 0);
    }
    const MechanicalStage& last = ad.back();
    for (std::size_t k = 0; k < last.state.violation.size(); ++k) {
        const ConstraintType t = last.state.constraints.items[k].type;
        EXPECT_LE(last.state.violation[k],
                  mechanical_tolerance(t, kMechanicalPenaltyLength, kMechanicalPenaltyCos))
            << to_string(t);
    }
    EXPECT_LT(std::abs(focal_length_at(p, last.run.x) - 10.0), kMechanicalFocalLength);
    // Die Strafen √w·max(0, g) aus Extrema sind nicht glatt; die Fortsetzung kann daher je nach
    // Rundung der Jacobi-Matrix in verschiedenen lokalen Minima enden (beobachtet: ab w = 10⁵
    // adjungiert F = 18,23, FD auf CPU und GPU F = 18,53). Gleich sind die Minima selbst: Jedes
    // Verfahren bleibt am Endpunkt des anderen (Merit relativ ≤ kLmReturnRelative).
    const LmOptions o_last = mechanical_options(
        with_mechanical_weight(p.loaded.system, last.weight), kMechanicalStageIterations);
    const LmRunResult fd_at_ad =
        optimize_with(JacobianMethod::FiniteDifferences)(o_last, last.run.x).runs.at(0);
    const LmRunResult ad_at_fd =
        optimize_with(JacobianMethod::Adjoint)(o_last, fd.back().run.x).runs.at(0);
    std::printf("[GPU] Endpunkt adjungiert F = %.10e, FD von dort %.10e (%d Zyklen); Endpunkt FD "
                "F = %.10e, adjungiert von dort %.10e (%d Zyklen)\n",
                last.run.merit, fd_at_ad.merit, fd_at_ad.iterations, fd.back().run.merit,
                ad_at_fd.merit, ad_at_fd.iterations);
    EXPECT_LE(relative(fd_at_ad.merit, last.run.merit), kLmReturnRelative);
    EXPECT_LE(relative(ad_at_fd.merit, fd.back().run.merit), kLmReturnRelative);
}

} // namespace
} // namespace mko::test
