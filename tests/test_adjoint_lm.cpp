// Adjungierte Jacobi im gebündelten Levenberg-Marquardt (M5-T3) auf der CPU: Verfahrenswahl
// (LmOptions::jacobian_method), Normalgleichungen gegen die FD-Jacobi, gleiche
// Optimierungsergebnisse wie mit finiten Differenzen (Abnahmen aus M3-T4, M4-T2, M4-T6, M4-T3),
// Bitgleichheit Gruppe gegen Einzellauf und Threadzahl (docs/optimierung.md, Abschnitt 4,
// „Jacobi-Verfahren im LM“).

#include "ghost_test_systems.hpp"
#include "lm_test_systems.hpp"
#include "mechanical_test_systems.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/normal_equations.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

namespace mko::test {
namespace {

LmOptions with_method(LmOptions o, JacobianMethod method) {
    o.jacobian_method = method;
    return o;
}

LmOptions paraxial_options(const FdProblem& p) {
    LmOptions o;
    o.jacobian.extra = paraxial_extra(p);
    return o;
}

// Größte Abweichung |A_ad − A_fd| bzw. |g_ad − g_fd| relativ zur größten Komponente, in der
// Skalierung S = diag(max(|x_k|, x_typ,k)) (Einheit des Residuums pro relativer Änderung).
struct NormalDeviation {
    double a = 0.0;
    double g = 0.0;
};

NormalDeviation normal_deviation(const FdProblem& p, const LmOptions& o,
                                 std::span<const double> x) {
    const int P = p.set.size();
    const auto n = static_cast<std::size_t>(P);
    std::vector<double> a_fd(n * n);
    std::vector<double> g_fd(n);
    std::vector<double> a_ad(n * n);
    std::vector<double> g_ad(n);
    const int run0 = 0;
    const std::span<const int> runs(&run0, 1);
    CpuLmBackend fd(p.set, p.base, o);
    fd.normal_equations(runs, x, a_fd, g_fd);
    const LmOptions oa = with_method(o, JacobianMethod::Adjoint);
    CpuLmBackend ad(p.set, p.base, oa);
    ad.normal_equations(runs, x, a_ad, g_ad);
    std::vector<double> s(n);
    for (std::size_t k = 0; k < n; ++k) {
        s[k] = std::max(std::abs(x[k]), p.set.typical()[k]);
    }
    double a_max = 0.0;
    double g_max = 0.0;
    double da = 0.0;
    double dg = 0.0;
    for (std::size_t q = 0; q < n; ++q) {
        for (std::size_t k = 0; k < n; ++k) {
            const double f = s[k] * s[q];
            a_max = std::max(a_max, std::abs(a_fd[q * n + k]) * f);
            da = std::max(da, std::abs(a_ad[q * n + k] - a_fd[q * n + k]) * f);
        }
        g_max = std::max(g_max, std::abs(g_fd[q]) * s[q]);
        dg = std::max(dg, std::abs(g_ad[q] - g_fd[q]) * s[q]);
    }
    if (!(a_max > 0.0) || !(g_max > 0.0)) {
        return NormalDeviation{da, dg}; // nicht erwartet (A = 0): absolute Abweichung
    }
    return NormalDeviation{da / a_max, dg / g_max};
}

double relative(double a, double b) {
    return std::abs(a / b - 1.0);
}

// ---- Verfahrenswahl -------------------------------------------------------------------------

TEST(AdjointLm, MethodNamesAndInvalidMethod) {
    EXPECT_STREQ(to_string(JacobianMethod::FiniteDifferences), "fd");
    EXPECT_STREQ(to_string(JacobianMethod::Adjoint), "adjoint");
    EXPECT_EQ(jacobian_method_from_string("fd"), JacobianMethod::FiniteDifferences);
    EXPECT_EQ(jacobian_method_from_string("adjoint"), JacobianMethod::Adjoint);
    EXPECT_FALSE(jacobian_method_from_string("Adjoint").has_value());
    EXPECT_FALSE(jacobian_method_from_string("").has_value());
    EXPECT_EQ(LmOptions{}.jacobian_method, JacobianMethod::FiniteDifferences); // Standard
    // Unbekannter Wert: klare Meldung statt stillem Rückfall
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    LmOptions o = paraxial_options(p);
    o.jacobian_method = static_cast<JacobianMethod>(7);
    try {
        optimize_lm_cpu(p.set, p.base, p.x0, o);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Verfahren der Jacobi-Matrix"), std::string::npos)
            << e.what();
    }
}

// Normalgleichungen des CPU-Backends: adjungiert gegen FD (Mehrlinser mit Brennweite, Einzellinse
// an Schranken, mechanische Bedingungen mit trace-abhängigen Zeilen). Die Abweichung ist der
// Fehler der FD-Jacobi (M5-T2: spaltenskaliert ≤ 2,7e-7). Das adjungierte A ist bitgleich zu
// normal_equations_host auf adjoint_jacobian_cpu, für jede Threadzahl.
TEST(AdjointLm, CpuNormalEquationsMatchFd) {
    struct Case {
        const char* name;
        FdProblem p;
        LmOptions o;
    };
    std::vector<Case> cases;
    {
        FdProblem p = mehrlinser_lm_problem(kMehrlinserFocalWeight);
        LmOptions o = paraxial_options(p);
        cases.push_back({"Mehrlinser", std::move(p), std::move(o)});
    }
    {
        FdProblem p = einzellinse_bound_problem();
        cases.push_back({"Einzellinse an Schranken", std::move(p), LmOptions{}});
    }
    {
        FdProblem p = mechanical_problem();
        LmOptions o = mechanical_options(p.loaded.system, 10);
        cases.push_back({"Zweilinser mechanisch", std::move(p), std::move(o)});
    }
    for (const Case& c : cases) {
        const NormalDeviation d = normal_deviation(c.p, c.o, c.p.x0);
        std::printf("[CPU] %-26s P = %2d: |ΔA| %.2e, |Δg| %.2e (skaliert, relativ)\n", c.name,
                    c.p.set.size(), d.a, d.g);
        EXPECT_LE(d.a, kAdjointLmNormalEquations) << c.name;
        EXPECT_LE(d.g, kAdjointLmNormalEquations) << c.name;
    }
    // Bitgleich zur FMA-Kette auf der adjungierten Jacobi, unabhängig von der Threadzahl
    const Case& c = cases.front();
    const auto n = static_cast<std::size_t>(c.p.set.size());
    const std::vector<double> x0 = perturbed_starts(c.p.set, c.p.x0, 3, 0.01, 5);
    AdjointJacobianOptions ao = adjoint_options_of(c.o.jacobian);
    ao.threads = 1;
    const AdjointJacobian j1 = adjoint_jacobian_cpu(c.p.set, c.p.base, x0, ao);
    ao.threads = 5;
    const AdjointJacobian j5 = adjoint_jacobian_cpu(c.p.set, c.p.base, x0, ao);
    ASSERT_EQ(j1.jacobian.size(), j5.jacobian.size());
    for (std::size_t i = 0; i < j1.jacobian.size(); ++i) {
        ASSERT_TRUE(same_bits(j1.jacobian[i], j5.jacobian[i])) << i;
    }
    EXPECT_EQ(j1.residuals, j5.residuals);
    EXPECT_EQ(j1.status, j5.status);
    LmOptions o = with_method(c.o, JacobianMethod::Adjoint);
    o.threads = 3;
    CpuLmBackend backend(c.p.set, c.p.base, o);
    const std::vector<int> runs{0, 1, 2};
    std::vector<double> a(3 * n * n);
    std::vector<double> g(3 * n);
    backend.normal_equations(runs, x0, a, g);
    const auto R = static_cast<std::size_t>(j1.shape.rows());
    for (std::size_t i = 0; i < 3; ++i) {
        std::vector<double> ai(n * n);
        std::vector<double> gi(n);
        normal_equations_host(std::span<const double>(j1.jacobian).subspan(i * n * R, n * R),
                              j1.residual(static_cast<int>(i)), c.p.set.size(), ai, gi);
        for (std::size_t k = 0; k < n * n; ++k) {
            ASSERT_TRUE(same_bits(ai[k], a[i * n * n + k])) << i << " " << k;
        }
        for (std::size_t k = 0; k < n; ++k) {
            ASSERT_TRUE(same_bits(gi[k], g[i * n + k])) << i << " " << k;
        }
    }
}

// ---- Gleiche Optimierungsergebnisse wie mit finiten Differenzen -----------------------------

// Abnahme 1 aus M3-T4 mit der adjungierten Jacobi: Formfaktor 0,74 ± 0,05, dasselbe Optimum wie
// mit FD (Formfaktor auf kLmShapeAgreement, Merit relativ auf kLmReturnRelative).
TEST(AdjointLm, LensBendingSameOptimumAsFd) {
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const LmOptions o = paraxial_options(p);
    const LmRunResult fd = optimize_lm_cpu(p.set, p.base, p.x0, o).runs[0];
    const LmRunResult ad =
        optimize_lm_cpu(p.set, p.base, p.x0, with_method(o, JacobianMethod::Adjoint)).runs[0];
    std::printf("[CPU] Einzellinse: adjungiert q = %.6f, F = %.12e, %d Zyklen, %s; FD q = %.6f, "
                "F = %.12e, %d Zyklen; F relativ %.2e\n",
                shape_factor(ad.x), ad.merit, ad.iterations, to_string(ad.status),
                shape_factor(fd.x), fd.merit, fd.iterations, relative(ad.merit, fd.merit));
    EXPECT_TRUE(lm_converged(ad.status)) << to_string(ad.status);
    EXPECT_NEAR(shape_factor(ad.x), 0.74, kLmShapeFactor);
    EXPECT_NEAR(shape_factor(ad.x), shape_factor(fd.x), kLmShapeAgreement);
    EXPECT_LE(relative(ad.merit, fd.merit), kLmReturnRelative);
    EXPECT_EQ(ad.failed_rays, 0);
    EXPECT_TRUE(p.set.feasible(ad.x));
}

// Abnahme 2 aus M3-T4: Mehrlinser (13 Variablen, mehrere an Schranken) mit der adjungierten
// Jacobi optimiert, F_opt wie mit FD (relativ ≤ kLmReturnRelative), Rückkehr aus 1 % Störung
// (vier Vorzeichenmuster): F_end ≤ F_opt (1 + kLmReturnRelative).
TEST(AdjointLm, ReturnsFromOnePercentPerturbation) {
    const FdProblem p = mehrlinser_lm_problem(kMehrlinserFocalWeight);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 500;
    const LmOptions oa = with_method(o, JacobianMethod::Adjoint);
    const LmRunResult fd = optimize_lm_cpu(p.set, p.base, p.x0, o).runs[0];
    const LmRunResult best = optimize_lm_cpu(p.set, p.base, p.x0, oa).runs[0];
    ASSERT_TRUE(lm_converged(best.status)) << to_string(best.status);
    double dx = 0.0;
    for (int k = 0; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        dx = std::max(dx, std::abs(best.x[kk] - fd.x[kk]) /
                              std::max(std::abs(fd.x[kk]), p.set.typical()[kk]));
    }
    std::printf("[CPU] Mehrlinser: adjungiert F_opt %.12e (%d Zyklen, %s), FD %.12e (%d Zyklen), "
                "relativ %.2e, max|Δx|/s %.2e\n",
                best.merit, best.iterations, to_string(best.status), fd.merit, fd.iterations,
                relative(best.merit, fd.merit), dx);
    EXPECT_LE(relative(best.merit, fd.merit), kLmReturnRelative);
    EXPECT_LE(dx, kAdjointLmEndpoint);
    const std::vector<double> starts = one_percent_starts(p.set, best.x);
    const LmResult back = optimize_lm_cpu(p.set, p.base, starts, oa);
    for (const LmRunResult& r : back.runs) {
        std::printf("[CPU]   aus 1 %%: F_end %.12e, relativ %+.3e, %d Zyklen, %s\n", r.merit,
                    r.merit / best.merit - 1.0, r.iterations, to_string(r.status));
        EXPECT_TRUE(lm_converged(r.status)) << to_string(r.status);
        EXPECT_LE(r.merit, best.merit * (1.0 + kLmReturnRelative));
        EXPECT_TRUE(p.set.feasible(r.x));
    }
}

// Abnahme 3 aus M3-T4 mit der adjungierten Jacobi: Läufe gemeinsam bitgleich zu den
// Einzelläufen, auch bei anderer Threadzahl.
TEST(AdjointLm, RunsTogetherBitwiseEqualToSingleRuns) {
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const int P = p.set.size();
    const int runs = 8;
    const std::vector<double> x0 = perturbed_starts(p.set, p.x0, runs, 0.05, 2026);
    LmOptions o = with_method(paraxial_options(p), JacobianMethod::Adjoint);
    o.max_iterations = 200;
    o.threads = 4;
    const LmResult group = optimize_lm_cpu(p.set, p.base, x0, o);
    o.threads = 1;
    for (int i = 0; i < runs; ++i) {
        const LmResult single = optimize_lm_cpu(p.set, p.base, run_start(x0, P, i), o);
        ASSERT_TRUE(same_run(group.runs[static_cast<std::size_t>(i)], single.runs[0])) << i;
        EXPECT_TRUE(lm_converged(single.runs[0].status));
    }
}

// Weiches Versagen (Abnahme M4-T2): Start mit 30 % weich versagten Strahlen, Standardoptionen
// mit äußerem Strafverfahren; mit der adjungierten Jacobi ebenso versagensfrei, Endmerit wie mit
// FD.
TEST(AdjointLm, SoftFailureAcceptance) {
    const FdProblem p = make_fd_problem(load_residual_system("weiches_versagen_start.json"));
    const LmRunResult fd = optimize_lm_cpu(p.set, p.base, p.x0).runs[0];
    const LmRunResult ad =
        optimize_lm_cpu(p.set, p.base, p.x0, with_method(LmOptions{}, JacobianMethod::Adjoint))
            .runs[0];
    std::printf("[CPU] weiches Versagen: adjungiert F %.6e -> %.12e, %d Zyklen, %d Runden, %s; "
                "FD F -> %.12e, %d Zyklen, %d Runden; relativ %.2e\n",
                ad.history.front(), ad.merit, ad.iterations, ad.penalty_rounds,
                to_string(ad.status), fd.merit, fd.iterations, fd.penalty_rounds,
                relative(ad.merit, fd.merit));
    EXPECT_EQ(ad.failed_rays, 0);
    EXPECT_EQ(ad.hard_failed_rays, 0);
    EXPECT_TRUE(lm_converged(ad.status)) << to_string(ad.status);
    EXPECT_TRUE(p.set.feasible(ad.x));
    EXPECT_LE(relative(ad.merit, fd.merit), kLmReturnRelative);
}

// Geisterplatten (Abnahme „Exaktheit“ aus M4-T6): Trace-Zeilen der Geisterkoeffizienten
// adjungiert, L1-Strafe als FD-Zusatzzeilen. Alle |a| < 1e-8, Stationarität wie vorhergesagt,
// Endmerit wie mit FD.
TEST(AdjointLm, GhostExactness) {
    const FdProblem p = ghost_exact_problem();
    const GhostPenalty pen = ghost_penalty_of(p.loaded.system);
    LmRunResult fd;
    const ExactnessResult e = ghost_exactness(p, pen, [&](const LmOptions& o) {
        fd = optimize_lm_cpu(p.set, p.base, p.x0, o).runs.at(0);
        return optimize_lm_cpu(p.set, p.base, p.x0, with_method(o, JacobianMethod::Adjoint))
            .runs.at(0);
    });
    print_exactness("CPU adjungiert", p, e);
    std::printf("[CPU] FD: F = %.12e, %d Zyklen; relativ %.2e\n", fd.merit, fd.iterations,
                relative(e.run.merit, fd.merit));
    EXPECT_TRUE(lm_converged(e.run.status)) << to_string(e.run.status);
    EXPECT_EQ(e.run.failed_rays, 0);
    EXPECT_LT(max_ghost(p.set, e.run.x), kGhostExactCoefficient);
    EXPECT_LT(e.threshold_ratio, 1.0);
    EXPECT_LE(e.max_stationarity_error, kGhostStationarity);
    EXPECT_LE(relative(e.run.merit, fd.merit), kLmReturnRelative);
}

} // namespace
} // namespace mko::test
