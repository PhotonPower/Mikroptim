// Tests des gebündelten Levenberg-Marquardt (M3-T4) auf der CPU: Steuerlogik an analytischen
// Problemen und die Abnahme mit der Einzellinse (docs/optimierung.md, Abschnitt 4).

#include "lm_test_systems.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/normal_equations.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mko::test {
namespace {

LmResult run_dense(const DenseProblem& problem, const DenseBounds& bounds,
                   std::span<const double> x0, const LmOptions& options = {}) {
    DenseBackend backend(problem);
    return levenberg_marquardt(backend, problem.variables, bounds.view(), x0, options);
}

// ---- Normalgleichungen ---------------------------------------------------------------------

TEST(NormalEquations, FmaChainAndSymmetry) {
    // J (3 × 2) spaltenweise, r (3)
    const std::vector<double> j = {1.0, 2.0, 3.0, -1.0, 0.5, 4.0};
    const std::vector<double> r = {0.25, -1.0, 2.0};
    std::vector<double> a(4);
    std::vector<double> g(2);
    normal_equations_host(j, r, 2, a, g);
    EXPECT_EQ(a[0], 14.0);              // 1 + 4 + 9
    EXPECT_EQ(a[3], 1.0 + 0.25 + 16.0); // Spalte 2
    EXPECT_EQ(a[1], -1.0 + 1.0 + 12.0); // A(1, 0)
    EXPECT_EQ(a[2], a[1]);              // symmetrisch
    EXPECT_EQ(g[0], 0.25 - 2.0 + 6.0);  // Jᵀr
    EXPECT_EQ(g[1], -0.25 - 0.5 + 8.0);
    // Reihenfolge: FMA-Kette in aufsteigender Zeilenfolge
    const double u[3] = {0.1, 0.2, 0.3};
    const double v[3] = {0.7, 0.11, 0.13};
    const double expected = std::fma(0.3, 0.13, std::fma(0.2, 0.11, std::fma(0.1, 0.7, 0.0)));
    EXPECT_TRUE(same_bits(gram_dot(u, v, 3), expected));
    EXPECT_THROW(normal_equations_host(j, r, 3, a, g), std::invalid_argument);
}

// ---- Analytische Probleme --------------------------------------------------------------------

// Lineares Problem: Mit verschwindender Dämpfung ist der erste Schritt der exakte
// Gauß-Newton-Schritt; mit Standarddämpfung konvergiert LM in wenigen Schritten gegen die
// Lösung der Normalgleichungen.
TEST(LevenbergMarquardt, LinearProblemExactInOneStep) {
    // M (4 × 3), b (4); Lösung der Kleinste-Quadrate-Aufgabe per Hand: MᵀM x = Mᵀb
    const std::vector<double> m = {1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0, -1.0, 1.0, 2.0};
    const std::vector<double> b = {1.0, 2.0, 2.5, 5.0};
    const DenseProblem problem = linear_problem(m, b, 3);
    // Referenz über die Normalgleichungen (Cramer in long double)
    std::vector<double> a(9);
    std::vector<double> g(3);
    std::vector<double> zero(3, 0.0);
    std::vector<double> r0(4);
    std::vector<double> jac(12);
    problem.evaluate(zero, r0, jac);
    normal_equations_host(jac, r0, 3, a, g); // g = −Mᵀb
    const auto det3 = [](const long double c[9]) {
        return c[0] * (c[4] * c[8] - c[7] * c[5]) - c[3] * (c[1] * c[8] - c[7] * c[2]) +
               c[6] * (c[1] * c[5] - c[4] * c[2]);
    };
    long double am[9];
    for (int i = 0; i < 9; ++i) {
        am[i] = a[static_cast<std::size_t>(i)];
    }
    const long double det = det3(am);
    std::vector<double> xs(3);
    for (int k = 0; k < 3; ++k) {
        long double c[9];
        for (int i = 0; i < 9; ++i) {
            c[i] = am[i];
        }
        for (int i = 0; i < 3; ++i) {
            c[k * 3 + i] = -g[static_cast<std::size_t>(i)];
        }
        xs[static_cast<std::size_t>(k)] = static_cast<double>(det3(c) / det);
    }

    const DenseBounds bounds = open_bounds(3);
    LmOptions o;
    o.initial_damping = 1e-300;
    o.max_iterations = 1;
    const LmResult one = run_dense(problem, bounds, zero, o);
    ASSERT_EQ(one.runs.size(), 1u);
    const LmRunResult& r1 = one.runs[0];
    EXPECT_EQ(r1.accepted, 1);
    EXPECT_EQ(r1.status, LmStatus::MaxIterations);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(r1.x[k], xs[k], 1e-13 * std::max(1.0, std::abs(xs[k]))) << k;
    }

    // Standard: Abbruch, wenn das Modell weniger als ftol relative Abnahme vorhersagt; F liegt
    // dann höchstens etwa ftol·F über dem Minimum.
    std::vector<double> rs(4);
    problem.evaluate(xs, rs, {});
    double fmin = 0.0;
    for (const double v : rs) {
        fmin += v * v;
    }
    const LmResult full = run_dense(problem, bounds, zero);
    const LmRunResult& rf = full.runs[0];
    EXPECT_TRUE(lm_converged(rf.status)) << to_string(rf.status);
    EXPECT_LE(rf.merit, fmin * (1.0 + 2e-10));
    // ftol = 0: Abbruch erst über Gradient oder Schrittnorm des Gauß-Newton-Schritts
    LmOptions exact;
    exact.ftol = 0.0;
    const LmRunResult re = run_dense(problem, bounds, zero, exact).runs[0];
    EXPECT_TRUE(lm_converged(re.status)) << to_string(re.status);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(re.x[k], xs[k], 1e-9 * std::max(1.0, std::abs(xs[k]))) << k;
    }
    EXPECT_EQ(rf.history.size(), static_cast<std::size_t>(rf.iterations) + 1);
    for (std::size_t k = 1; k < rf.history.size(); ++k) {
        EXPECT_LE(rf.history[k], rf.history[k - 1]); // monoton
    }
}

TEST(LevenbergMarquardt, Rosenbrock) {
    const DenseProblem problem = rosenbrock_problem();
    const DenseBounds bounds = open_bounds(2);
    const std::vector<double> x0 = {-1.2, 1.0};
    const LmResult res = run_dense(problem, bounds, x0);
    const LmRunResult& r = res.runs[0];
    EXPECT_TRUE(lm_converged(r.status)) << to_string(r.status);
    EXPECT_NEAR(r.x[0], 1.0, 1e-8);
    EXPECT_NEAR(r.x[1], 1.0, 1e-8);
    EXPECT_LT(r.merit, 1e-20);
    EXPECT_GT(r.iterations, 3);
    EXPECT_LT(r.iterations, 100);
}

// Schranken: inaktiv (Lösung innen, unverändert) und aktiv (Projektion; getrennte Variablen).
TEST(LevenbergMarquardt, SeparableBounds) {
    // r = x − c
    const std::vector<double> m = {1.0, 0.0, 0.0, 1.0};
    const DenseProblem problem = linear_problem(m, {2.0, -3.0}, 2);
    DenseBounds bounds = open_bounds(2);
    const std::vector<double> x0 = {0.5, 0.0};
    // inaktiv
    bounds.lower = {-10.0, -10.0};
    bounds.upper = {10.0, 10.0};
    const LmRunResult inner = run_dense(problem, bounds, x0).runs[0];
    EXPECT_TRUE(lm_converged(inner.status));
    EXPECT_NEAR(inner.x[0], 2.0, 1e-9);
    EXPECT_NEAR(inner.x[1], -3.0, 1e-9);
    // aktiv: x ∈ [0, 1] × [−1, 1] → (1, −1) exakt auf den Schranken
    bounds.lower = {0.0, -1.0};
    bounds.upper = {1.0, 1.0};
    const LmRunResult outer = run_dense(problem, bounds, x0).runs[0];
    EXPECT_TRUE(lm_converged(outer.status)) << to_string(outer.status);
    EXPECT_EQ(outer.x[0], 1.0);
    EXPECT_EQ(outer.x[1], -1.0);
    EXPECT_EQ(outer.gradient_norm, 0.0); // beide Variablen aus dem projizierten Gradienten
}

// Gekoppelte Variablen mit aktiver Schranke: f = (x₁ + x₂ − 2)² + 0,01 (x₁ − x₂)², x₁ ≤ 0,5.
// KKT: x₁ = 0,5, x₂ = 3,01/2,02 (der freie Optimierer würde (1, 1) wählen).
TEST(LevenbergMarquardt, CoupledActiveBound) {
    const std::vector<double> m = {1.0, 0.1, 1.0, -0.1};
    const DenseProblem problem = linear_problem(m, {2.0, 0.0}, 2);
    DenseBounds bounds = open_bounds(2);
    bounds.upper[0] = 0.5;
    const std::vector<double> x0 = {0.0, 0.0};
    LmOptions exact;
    exact.ftol = 0.0;
    const LmRunResult r = run_dense(problem, bounds, x0, exact).runs[0];
    EXPECT_TRUE(lm_converged(r.status)) << to_string(r.status);
    EXPECT_EQ(r.x[0], 0.5);
    EXPECT_NEAR(r.x[1], 3.01 / 2.02, 1e-9);
    // Start auf der Schranke mit nach innen zeigendem Gradienten: verlässt die Schranke
    bounds.upper[0] = 2.0;
    bounds.lower[0] = 0.0;
    const std::vector<double> x1 = {0.0, 3.0};
    const LmRunResult r2 = run_dense(problem, bounds, x1, exact).runs[0];
    EXPECT_TRUE(lm_converged(r2.status));
    EXPECT_NEAR(r2.x[0], 1.0, 1e-9);
    EXPECT_NEAR(r2.x[1], 1.0, 1e-9);
}

// Mehrere Läufe gleichzeitig: verschieden schnell konvergierende Läufe, jedes Ergebnis bitgleich
// zum Einzellauf, unabhängig von der Reihenfolge und der Threadzahl.
TEST(LevenbergMarquardt, RunsAreIndependent) {
    const DenseProblem problem = rosenbrock_problem();
    const DenseBounds bounds = open_bounds(2);
    const std::vector<double> x0 = {-1.2, 1.0, 1.0, 1.0, 0.9, 0.8, 3.0, -2.0, 1.0001, 1.0};
    LmOptions o;
    o.threads = 3;
    const LmResult group = run_dense(problem, bounds, x0, o);
    ASSERT_EQ(group.runs.size(), 5u);
    EXPECT_EQ(group.runs[1].status, LmStatus::ConvergedMerit); // Start im Optimum, F = 0
    EXPECT_EQ(group.runs[1].iterations, 0);
    int min_it = 1000;
    int max_it = 0;
    for (int i = 0; i < 5; ++i) {
        o.threads = 1;
        const LmResult single = run_dense(problem, bounds, run_start(x0, 2, i), o);
        EXPECT_TRUE(same_run(group.runs[static_cast<std::size_t>(i)], single.runs[0])) << i;
        if (i != 1) {
            min_it = std::min(min_it, single.runs[0].iterations);
            max_it = std::max(max_it, single.runs[0].iterations);
        }
    }
    EXPECT_LT(min_it, max_it); // tatsächlich verschieden schnelle Läufe
    // umgekehrte Reihenfolge
    std::vector<double> rev;
    for (int i = 4; i >= 0; --i) {
        const auto s = run_start(x0, 2, i);
        rev.insert(rev.end(), s.begin(), s.end());
    }
    const LmResult reversed = run_dense(problem, bounds, rev);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(same_run(group.runs[static_cast<std::size_t>(i)],
                             reversed.runs[static_cast<std::size_t>(4 - i)]))
            << i;
    }
}

TEST(LevenbergMarquardt, IterationLimitHistoryAndCallback) {
    const DenseProblem problem = rosenbrock_problem();
    const DenseBounds bounds = open_bounds(2);
    const std::vector<double> x0 = {-1.2, 1.0, 3.0, -2.0};
    LmOptions o;
    o.max_iterations = 3;
    std::vector<LmIterationInfo> log;
    o.on_iteration = [&log](const LmIterationInfo& info) { log.push_back(info); };
    const LmResult res = run_dense(problem, bounds, x0, o);
    for (const LmRunResult& r : res.runs) {
        EXPECT_EQ(r.status, LmStatus::MaxIterations);
        EXPECT_EQ(r.iterations, 3);
        EXPECT_EQ(r.history.size(), 4u);
    }
    ASSERT_EQ(log.size(), 6u);
    for (std::size_t k = 0; k < log.size(); ++k) {
        EXPECT_EQ(log[k].run, static_cast<int>(k % 2)); // Laufreihenfolge pro Zyklus
        EXPECT_EQ(log[k].iteration, static_cast<int>(k / 2) + 1);
        EXPECT_EQ(log[k].merit, res.runs[k % 2].history[k / 2 + 1]);
    }
    o.max_iterations = 0;
    const LmResult none = run_dense(problem, bounds, x0, o);
    EXPECT_EQ(none.runs[0].status, LmStatus::MaxIterations);
    EXPECT_EQ(none.runs[0].x, std::vector<double>({-1.2, 1.0}));
    EXPECT_EQ(none.runs[0].jacobians, 0);
}

// Versagen: Punkte mit „versagten Strahlen“ werden nicht angenommen, nicht endliche Merit am
// Versuchspunkt verwirft den Schritt, am Startpunkt beendet sie den Lauf (NumericalError).
TEST(LevenbergMarquardt, FailuresAndNonFiniteMerit) {
    DenseProblem problem = rosenbrock_problem();
    // versagt für x₁ > 0,5: das Optimum (1, 1) ist nicht erreichbar
    problem.failed = [](std::span<const double> x) -> std::int64_t { return x[0] > 0.5 ? 1 : 0; };
    const DenseBounds bounds = open_bounds(2);
    const std::vector<double> x0 = {-1.2, 1.0};
    const LmRunResult r = run_dense(problem, bounds, x0).runs[0];
    EXPECT_LE(r.x[0], 0.5);
    EXPECT_EQ(r.failed_rays, 0);
    EXPECT_NE(r.status, LmStatus::NumericalError);
    EXPECT_LT(r.merit, r.history.front());
    // ohne Ablehnung läuft es ins „versagte“ Optimum
    LmOptions o;
    o.reject_new_failures = false;
    const LmRunResult r2 = run_dense(problem, bounds, x0, o).runs[0];
    EXPECT_NEAR(r2.x[0], 1.0, 1e-8);
    EXPECT_EQ(r2.failed_rays, 1);

    DenseProblem nan_problem = rosenbrock_problem();
    const auto base_eval = nan_problem.evaluate;
    nan_problem.evaluate = [base_eval](std::span<const double> x, std::span<double> res,
                                       std::span<double> jac) {
        base_eval(x, res, jac);
        if (x[1] > 1.5) {
            res[0] = std::numeric_limits<double>::quiet_NaN();
        }
    };
    const std::vector<double> starts = {-1.2, 1.0, 0.0, 2.0};
    const LmResult res = run_dense(nan_problem, bounds, starts);
    EXPECT_TRUE(lm_converged(res.runs[0].status)) << to_string(res.runs[0].status);
    EXPECT_LE(res.runs[0].x[1], 1.5);
    EXPECT_EQ(res.runs[1].status, LmStatus::NumericalError);
    EXPECT_EQ(res.runs[1].iterations, 0);
}

TEST(LevenbergMarquardt, InvalidArguments) {
    const DenseProblem problem = rosenbrock_problem();
    DenseBounds bounds = open_bounds(2);
    DenseBackend backend(problem);
    const std::vector<double> x0 = {-1.2, 1.0};
    EXPECT_THROW(levenberg_marquardt(backend, 2, bounds.view(), std::vector<double>{1.0}),
                 std::invalid_argument);
    EXPECT_THROW(levenberg_marquardt(backend, 0, bounds.view(), x0), std::invalid_argument);
    LmOptions o;
    o.initial_damping = 0.0;
    EXPECT_THROW(levenberg_marquardt(backend, 2, bounds.view(), x0, o), std::invalid_argument);
    o = {};
    o.ftol = -1.0;
    EXPECT_THROW(levenberg_marquardt(backend, 2, bounds.view(), x0, o), std::invalid_argument);
    bounds.upper[0] = -2.0; // Start außerhalb
    EXPECT_THROW(levenberg_marquardt(backend, 2, bounds.view(), x0), std::invalid_argument);
    bounds = open_bounds(2);
    bounds.typical[1] = 0.0;
    EXPECT_THROW(levenberg_marquardt(backend, 2, bounds.view(), x0), std::invalid_argument);
}

// ---- Optische Probleme (CPU-Backend) ---------------------------------------------------------

LmOptions paraxial_options(const FdProblem& p) {
    LmOptions o;
    o.jacobian.extra = paraxial_extra(p);
    return o;
}

// Abnahme 1: Einzellinse N-BK7 mit fester Brennweite und kleiner Apertur (NA ≈ 0,025) auf
// minimale sphärische Aberration, Formfaktor q im Bereich 0,74 ± 0,05.
TEST(LmCpu, LensBendingMinimalSphericalAberration) {
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const LmOptions o = paraxial_options(p);
    const LmResult res = optimize_lm_cpu(p.set, p.base, p.x0, o);
    const LmRunResult& r = res.runs[0];
    const double n = p.base.index(0, 1, 0);
    const double q = shape_factor(r.x);
    std::vector<double> focal(1);
    SystemBatchHost b = p.base;
    p.set.apply(b, 0, r.x);
    paraxial_extra(p).evaluate(b, 0, focal);
    std::printf("[CPU] Einzellinse: q = %.6f (Dünnlinse %.6f, n = %.6f), F %.6e -> %.10e, "
                "%d Zyklen, %d angenommen, Status %s, f − 100 mm = %.2e mm\n",
                q, thin_lens_best_shape(n), n, r.history.front(), r.merit, r.iterations, r.accepted,
                to_string(r.status), focal[0] / std::sqrt(kLensFocalWeight));
    EXPECT_TRUE(lm_converged(r.status)) << to_string(r.status);
    EXPECT_NEAR(q, 0.74, kLmShapeFactor);
    EXPECT_NEAR(thin_lens_best_shape(n), 0.7397, 1e-4);
    EXPECT_EQ(r.failed_rays, 0);
    EXPECT_TRUE(p.set.feasible(r.x));
    // Die Brennweite bleibt fest: |f − f_s| < 1e-4 mm (Residuum √w·(f − f_s) bis auf O(ε²))
    EXPECT_LT(std::abs(focal[0]) / std::sqrt(kLensFocalWeight), 1e-4);
}

// Abnahme 2: optimierter Mehrlinser (13 Variablen, mehrere an Schranken), jede Variable um 1 %
// gestört (vier Vorzeichenmuster, ein gemeinsamer Aufruf): F_end ≤ F_opt·(1 + 1·10⁻⁶).
TEST(LmCpu, ReturnsFromOnePercentPerturbation) {
    const FdProblem p = mehrlinser_lm_problem(kMehrlinserFocalWeight);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 500;
    const LmResult opt = optimize_lm_cpu(p.set, p.base, p.x0, o);
    const LmRunResult& best = opt.runs[0];
    ASSERT_TRUE(lm_converged(best.status)) << to_string(best.status);
    int on_bound = 0;
    for (int k = 0; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        on_bound += (best.x[kk] == p.set.lower()[kk] || best.x[kk] == p.set.upper()[kk]) ? 1 : 0;
    }
    std::printf("[CPU] Mehrlinser: F %.6e -> F_opt %.12e (%d Zyklen, %s), %d Variablen auf "
                "Schranken\n",
                best.history.front(), best.merit, best.iterations, to_string(best.status),
                on_bound);
    EXPECT_GT(on_bound, 0);
    const std::vector<double> starts = one_percent_starts(p.set, best.x);
    const LmResult back = optimize_lm_cpu(p.set, p.base, starts, o);
    for (const LmRunResult& r : back.runs) {
        std::printf("[CPU]   aus 1 %%: F_start %.6e, F_end %.12e, relativ %+.3e, %d Zyklen, %s\n",
                    r.history.front(), r.merit, r.merit / best.merit - 1.0, r.iterations,
                    to_string(r.status));
        EXPECT_TRUE(lm_converged(r.status)) << to_string(r.status);
        EXPECT_LE(r.merit, best.merit * (1.0 + kLmReturnRelative));
        EXPECT_TRUE(p.set.feasible(r.x));
    }
}

// Abnahme 3 auf der CPU: Läufe gemeinsam bitgleich zu den Einzelläufen, auch bei anderer
// Threadzahl. (1.000 Läufe auf der GPU: CudaLm.ThousandRunsBitwiseEqualToSingleRuns.)
TEST(LmCpu, RunsTogetherBitwiseEqualToSingleRuns) {
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const int P = p.set.size();
    const int runs = 100;
    const std::vector<double> x0 = perturbed_starts(p.set, p.x0, runs, 0.05, 2026);
    LmOptions o = paraxial_options(p);
    o.max_iterations = 200;
    o.threads = 4;
    const LmResult group = optimize_lm_cpu(p.set, p.base, x0, o);
    o.threads = 1;
    int min_it = 1 << 30;
    int max_it = 0;
    for (int i = 0; i < runs; ++i) {
        const LmResult single = optimize_lm_cpu(p.set, p.base, run_start(x0, P, i), o);
        ASSERT_TRUE(same_run(group.runs[static_cast<std::size_t>(i)], single.runs[0])) << i;
        EXPECT_TRUE(lm_converged(single.runs[0].status));
        min_it = std::min(min_it, single.runs[0].iterations);
        max_it = std::max(max_it, single.runs[0].iterations);
    }
    EXPECT_LT(min_it, max_it);
}

// Vorlage pro Lauf: Läufe mit verschiedenen Vorlagen (hier verschiedene Pupillenradien) im
// selben Aufruf wie einzeln mit ihrer eigenen Vorlage.
TEST(LmCpu, TemplatePerRun) {
    const FdProblem p = lens_bending_problem(kLensFocalWeight);
    const int P = p.set.size();
    SystemBatchHost bases = p.base;
    bases.resize(3);
    for (int v = 0; v < 3; ++v) {
        bases.copy_variant_from(p.base, 0, v);
        bases.pupil_radius(v) = 1.5 + 0.5 * v;
    }
    std::vector<double> x0;
    for (int v = 0; v < 3; ++v) {
        x0.insert(x0.end(), p.x0.begin(), p.x0.end());
    }
    LmOptions o = paraxial_options(p);
    o.max_iterations = 30;
    const LmResult group = optimize_lm_cpu(p.set, bases, x0, o);
    for (int v = 0; v < 3; ++v) {
        SystemBatchHost one = p.base;
        one.copy_variant_from(bases, v, 0);
        const LmResult single = optimize_lm_cpu(p.set, one, run_start(x0, P, v), o);
        EXPECT_TRUE(same_run(group.runs[static_cast<std::size_t>(v)], single.runs[0])) << v;
    }
    EXPECT_NE(group.runs[0].merit, group.runs[2].merit);
    SystemBatchHost two = bases;
    two.resize(2);
    EXPECT_THROW(optimize_lm_cpu(p.set, two, x0, o), std::invalid_argument);
}

// Versagte Strahlen: Start, an dem die äußeren Ringe der Plankonvexlinse totalreflektieren. Der
// Lauf bricht nicht ab, versagte Strahlen nehmen nie zu, F sinkt und bleibt endlich.
TEST(LmCpu, RunWithFailedRays) {
    const FdProblem p = tir_start_problem(1.05);
    LmOptions o;
    o.max_iterations = 50;
    std::vector<std::int64_t> failed;
    o.on_iteration = [&failed](const LmIterationInfo& info) {
        if (info.accepted) {
            failed.push_back(info.failed_rays);
        }
    };
    const LmRunResult r = optimize_lm_cpu(p.set, p.base, p.x0, o).runs[0];
    ASSERT_FALSE(failed.empty());
    EXPECT_GT(failed.front(), 0);
    for (std::size_t k = 1; k < failed.size(); ++k) {
        EXPECT_LE(failed[k], failed[k - 1]);
    }
    EXPECT_NE(r.status, LmStatus::NumericalError);
    EXPECT_TRUE(std::isfinite(r.merit));
    EXPECT_LT(r.merit, r.history.front());
    EXPECT_TRUE(p.set.feasible(r.x));
}

} // namespace
} // namespace mko::test
