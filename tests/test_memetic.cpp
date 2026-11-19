// Tests der memetischen Suche (M11-T7): CMA-ES-Kern (exp/log, Eigenzerlegung, Parameter,
// Konvergenz auf analytischen Funktionen, Determinismus) und die Abnahme am Mehrlinser.

#include "lm_test_systems.hpp"
#include "memetic_test_cases.hpp"
#include "mikroptim/optim/cma_es.hpp"
#include "mikroptim/optim/memetic.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace mko {
namespace {

using test::FdProblem;
using test::same_bits;
using test::same_run;

// ---- exp und log ------------------------------------------------------------------------

TEST(PortableMath, ExpAndLogMatchStandardLibrary) {
    std::mt19937_64 gen(7);
    double worst_exp = 0.0;
    double worst_log = 0.0;
    for (int i = 0; i < 20000; ++i) {
        const double u = unit_uniform(gen());
        const double x = -700.0 + 1400.0 * u;
        worst_exp = std::max(worst_exp, std::abs(portable_exp(x) / std::exp(x) - 1.0));
        const double y = std::ldexp(0.5 + u, static_cast<int>(gen() % 2000) - 1000);
        worst_log = std::max(worst_log, std::abs(portable_log(y) - std::log(y)) /
                                            std::max(1.0, std::abs(std::log(y))));
    }
    // nahe 1 (kleine Logarithmen) und kleine Argumente der Exponentialfunktion
    for (const double x : {1e-300, 1e-17, -1e-12, 0.3, -0.3466, 0.3466, 1.0, -1.0, 20.0}) {
        worst_exp = std::max(worst_exp, std::abs(portable_exp(x) / std::exp(x) - 1.0));
    }
    for (const double y : {1.0, 1.0 + 1e-15, 1.0 - 1e-15, 0.7071, 1.4142, 2.0, 0.5, 1e300}) {
        worst_log = std::max(worst_log, std::abs(portable_log(y) - std::log(y)) /
                                            std::max(1e-300, std::abs(std::log(y))));
    }
    EXPECT_LE(worst_exp, test::kPortableMathRelative);
    EXPECT_LE(worst_log, test::kPortableMathRelative);
    std::cout << "portable_exp: " << worst_exp << ", portable_log: " << worst_log << '\n';
}

TEST(PortableMath, SpecialValues) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(portable_exp(0.0), 1.0);
    EXPECT_EQ(portable_exp(710.0), inf);
    EXPECT_EQ(portable_exp(-746.0), 0.0);
    EXPECT_TRUE(std::isnan(portable_exp(std::nan(""))));
    EXPECT_EQ(portable_log(1.0), 0.0);
    EXPECT_EQ(portable_log(0.0), -inf);
    EXPECT_EQ(portable_log(inf), inf);
    EXPECT_TRUE(std::isnan(portable_log(-1.0)));
    EXPECT_EQ(portable_log(2.0), 0.6931471805599453); // ln 2, korrekt gerundet
}

// Bitmuster, unabhängig vom Compiler (nur korrekt gerundete Operationen); e liegt 1 ulp über
// dem korrekt gerundeten Wert.
TEST(PortableMath, PlatformIndependentValues) {
    EXPECT_TRUE(same_bits(portable_exp(1.0), 2.7182818284590455)) << portable_exp(1.0);
    EXPECT_TRUE(same_bits(portable_log(10.0), 2.3025850929940459)) << portable_log(10.0);
}

// ---- Eigenzerlegung ----------------------------------------------------------------------

TEST(CmaEs, SymmetricEigenDecomposes) {
    const int n = 7;
    std::mt19937_64 gen(3);
    std::vector<double> a(n * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            const double v = 2.0 * unit_uniform(gen()) - 1.0 + (i == j ? 3.0 : 0.0);
            a[j * n + i] = v;
            a[i * n + j] = v;
        }
    }
    std::vector<double> work = a;
    std::vector<double> values(n);
    std::vector<double> vectors(n * n);
    symmetric_eigen(n, work, values, vectors);
    double worst = 0.0;
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            double av = 0.0;
            for (int j = 0; j < n; ++j) {
                av += a[j * n + i] * vectors[k * n + j];
            }
            worst = std::max(worst, std::abs(av - values[k] * vectors[k * n + i]));
        }
        for (int l = 0; l < n; ++l) {
            double dot = 0.0;
            for (int i = 0; i < n; ++i) {
                dot += vectors[k * n + i] * vectors[l * n + i];
            }
            worst = std::max(worst, std::abs(dot - (k == l ? 1.0 : 0.0)));
        }
    }
    EXPECT_LE(worst, test::kUnitFp64);
}

// ---- Parameter ---------------------------------------------------------------------------

TEST(CmaEs, ParametersFollowHansen) {
    const CmaParameters p = cma_parameters(10);
    EXPECT_EQ(p.lambda, 10); // 4 + ⌊3 ln 10⌋ = 4 + 6
    EXPECT_EQ(p.mu, 5);
    EXPECT_EQ(cma_default_lambda(13), 11);
    EXPECT_EQ(cma_default_lambda(1), 4);
    double sum = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < p.mu; ++i) {
        const double w = (std::log(5.5) - std::log(i + 1.0));
        sum += w;
        sum2 += w * w;
    }
    for (int i = 0; i < p.mu; ++i) {
        EXPECT_NEAR(p.weights[i], (std::log(5.5) - std::log(i + 1.0)) / sum, test::kUnitFp64);
    }
    const double me = sum * sum / sum2;
    EXPECT_NEAR(p.mu_eff, me, 1e-12);
    EXPECT_NEAR(p.c_sigma, (me + 2.0) / (10.0 + me + 5.0), test::kUnitFp64);
    EXPECT_NEAR(p.c_1, 2.0 / (11.3 * 11.3 + me), test::kUnitFp64);
    EXPECT_NEAR(p.chi_n, std::sqrt(10.0) * (1.0 - 1.0 / 40.0 + 1.0 / 2100.0), test::kUnitFp64);
    EXPECT_THROW(cma_parameters(0), std::invalid_argument);
    EXPECT_THROW(cma_parameters(3, 1), std::invalid_argument);
}

// ---- Konvergenz auf analytischen Funktionen ----------------------------------------------

// Reine CMA-ES (ohne lokale Verfeinerung) auf f(x), λ Proben pro Generation.
template <class F>
std::pair<double, CmaEs> run_cma(const F& f, std::vector<double> mean, double sigma, int gens,
                                 std::uint64_t seed) {
    CmaEs es(mean, sigma);
    const int n = es.dimension();
    const int lambda = es.lambda();
    double best = std::numeric_limits<double>::infinity();
    std::vector<double> pts(static_cast<std::size_t>(lambda * n));
    std::vector<double> fit(static_cast<std::size_t>(lambda));
    for (int g = 0; g < gens; ++g) {
        for (int i = 0; i < lambda; ++i) {
            es.sample(seed, static_cast<std::uint64_t>(i),
                      std::span<double>(pts.data() + i * n, static_cast<std::size_t>(n)));
            fit[i] = f(std::span<const double>(pts.data() + i * n, static_cast<std::size_t>(n)));
            best = std::min(best, fit[i]);
        }
        std::vector<int> rank(static_cast<std::size_t>(lambda));
        std::iota(rank.begin(), rank.end(), 0);
        std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) { return fit[a] < fit[b]; });
        es.update(pts, rank);
    }
    return {best, es};
}

TEST(CmaEs, ConvergesOnRotatedEllipsoid) {
    // f = Σ 10^{6(i/(n−1))} z_i², z = R x mit fester Drehung (Konditionszahl 10⁶): ohne
    // Kovarianzanpassung kaum lösbar in 400 Generationen.
    const int n = 6;
    std::vector<double> rot(n * n);
    {
        std::vector<double> a(n * n);
        std::mt19937_64 gen(11);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                const double v = 2.0 * unit_uniform(gen()) - 1.0;
                a[j * n + i] = v;
                a[i * n + j] = v;
            }
        }
        std::vector<double> vals(n);
        symmetric_eigen(n, a, vals, rot);
    }
    const auto f = [&](std::span<const double> x) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            double z = 0.0;
            for (int j = 0; j < n; ++j) {
                z += rot[i * n + j] * x[j];
            }
            s += std::pow(10.0, 6.0 * i / (n - 1.0)) * z * z;
        }
        return s;
    };
    const auto [best, es] = run_cma(f, std::vector<double>(n, 1.0), 0.5, 400, 5);
    EXPECT_LT(best, 1e-10);
    // Determinismus: zweiter Lauf bitgleich
    const auto [best2, es2] = run_cma(f, std::vector<double>(n, 1.0), 0.5, 400, 5);
    EXPECT_TRUE(same_bits(best, best2));
    EXPECT_TRUE(same_bits(es.sigma(), es2.sigma()));
    for (std::size_t k = 0; k < es.covariance().size(); ++k) {
        EXPECT_TRUE(same_bits(es.covariance()[k], es2.covariance()[k]));
    }
}

TEST(CmaEs, SampleDependsOnlyOnSeedGenerationAndIndex) {
    CmaEs es(std::vector<double>{1.0, 2.0, 3.0}, 0.1, 6);
    std::vector<double> a(3), b(3), c(3);
    es.sample(9, 4, a);
    es.sample(9, 4, b);
    es.sample(9, 5, c);
    for (int k = 0; k < 3; ++k) {
        EXPECT_TRUE(same_bits(a[k], b[k]));
    }
    EXPECT_FALSE(same_bits(a[0], c[0]));
    // Irwin-Hall mit C = I: |x − m| < 6σ
    for (int k = 0; k < 3; ++k) {
        EXPECT_LT(std::abs(a[k] - (k + 1.0)), 0.6);
    }
}

TEST(CmaEs, UpdateClipsInjectedPoints) {
    // Ein weit entfernter eingespeister Punkt verschiebt den Mittelwert höchstens um
    // w₁ σ clip_norm (Mahalanobis-Länge bei C = I).
    CmaEs es(std::vector<double>{0.0, 0.0}, 1.0, 4);
    const double clip = es.parameters().clip_norm;
    const double w1 = es.parameters().weights[0];
    std::vector<double> pts{1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<int> rank{0, 1, 2, 3};
    es.update(pts, rank);
    EXPECT_NEAR(es.mean()[0], w1 * clip, 1e-12);
    EXPECT_THROW(es.update(pts, std::vector<int>{0}), std::invalid_argument);
    EXPECT_THROW(es.update(pts, std::vector<int>{0, 0}), std::invalid_argument);
}

// ---- Memetische Suche --------------------------------------------------------------------

TEST(Memetic, OptionsAreChecked) {
    MemeticOptions o;
    EXPECT_NO_THROW(check_memetic_options(o));
    o.population = 1;
    EXPECT_THROW(check_memetic_options(o), std::invalid_argument);
    o = {};
    o.generations = 0;
    EXPECT_THROW(check_memetic_options(o), std::invalid_argument);
    o = {};
    o.lm_iterations = 0;
    EXPECT_THROW(check_memetic_options(o), std::invalid_argument);
    o = {};
    o.sigma = 0.0;
    EXPECT_THROW(check_memetic_options(o), std::invalid_argument);
    o = {};
    o.prefilter_oversampling = 0;
    EXPECT_THROW(check_memetic_options(o), std::invalid_argument);
    EXPECT_EQ(global_search_method_from_string("cma"), GlobalSearchMethod::Cma);
    EXPECT_EQ(global_search_method_from_string("starts"), GlobalSearchMethod::PerturbedStarts);
    EXPECT_FALSE(global_search_method_from_string("ga").has_value());
}

// Abnahme (M11-T7): Mehrlinser mit Brennweitenbedingung, CMA-ES mit Lamarck und Elitismus gegen
// perturbed_starts bei mindestens derselben Zahl von LM-Zyklen: (a) gleiche Zyklen pro Lauf
// (L = 20, mehr Starts), (b) längere Läufe (L = 100, weniger Starts).
TEST(Memetic, CmaBeatsPerturbedStartsOnMehrlinser) {
    const test::MemeticCase c = test::mehrlinser_memetic_case();
    const MemeticResult cma = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    ASSERT_GE(cma.best, 0);
    const MemeticIndividual& best = cma.best_individual();
    EXPECT_EQ(cma.lambda, 11);
    EXPECT_EQ(cma.budget, 1760);
    EXPECT_LE(cma.cycles, cma.budget + cma.budget / 10); // Strafrunden des weichen Versagens
    const auto cpu = [](const VariableSet& set, const SystemBatchHost& base,
                        std::span<const double> x,
                        const LmOptions& o) { return optimize_lm_cpu(set, base, x, o); };
    const test::BaselineResult same_l = test::perturbed_starts_baseline(c, cma.cycles, cpu);
    test::MemeticCase longer = c;
    longer.options.lm_iterations = 100;
    const test::BaselineResult long_l = test::perturbed_starts_baseline(longer, cma.cycles, cpu);
    std::cout << "Mehrlinser (CPU): CMA F = " << best.run.merit << " (" << cma.cycles
              << " Zyklen, Startmerit " << cma.individuals.front().run.history.front()
              << "); perturbed_starts L = 20: F = " << same_l.best_merit << " (" << same_l.runs
              << " Läufe, " << same_l.cycles << " Zyklen); L = 100: F = " << long_l.best_merit
              << " (" << long_l.runs << " Läufe, " << long_l.cycles << " Zyklen)\n";
    EXPECT_EQ(best.run.failed_rays, 0);
    EXPECT_GE(same_l.cycles, cma.cycles);
    EXPECT_GE(long_l.cycles, cma.cycles);
    EXPECT_LT(best.run.merit, same_l.best_merit);
    EXPECT_LT(best.run.merit, long_l.best_merit);
    // Zwischenergebnisse der Generationen passen zur Liste der Individuen
    std::int64_t cycles = 0;
    for (const MemeticGeneration& g : cma.generations) {
        cycles += g.cycles;
        EXPECT_TRUE(same_bits(g.best_merit, cma.individuals[g.best].run.merit));
    }
    EXPECT_EQ(cycles, cma.cycles);
}

// Zwei Läufe bitgleich, auch mit anderer Threadzahl des LM; anderer Startwert, anderer Lauf.
TEST(Memetic, TwoRunsAreBitIdentical) {
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.options.generations = 2;
    c.options.lm_iterations = 5;
    c.lm.threads = 1;
    const MemeticResult a = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    c.lm.threads = 3;
    const MemeticResult b = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    EXPECT_TRUE(test::same_memetic(a, b));
    c.options.seed = 2;
    const MemeticResult other = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    EXPECT_FALSE(test::same_memetic(a, other));
}

// Elitismus und Startpunkt: Generation 0 beginnt bei x₀, Kandidat 0 jeder späteren Generation
// beim verfeinerten Punkt des bis dahin besten Individuums.
TEST(Memetic, StartAndEliteCandidates) {
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.options.generations = 3;
    c.options.lm_iterations = 3;
    const MemeticResult r = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    ASSERT_EQ(r.generations.size(), 3u);
    const int lambda = r.lambda;
    for (std::size_t k = 0; k < c.p.x0.size(); ++k) {
        EXPECT_TRUE(same_bits(r.individuals[0].start[k], c.p.x0[k]));
    }
    int best = -1;
    for (int g = 0; g < 3; ++g) {
        if (g > 0) {
            const std::vector<double>& elite = r.individuals[best].run.x;
            const std::vector<double>& start = r.individuals[g * lambda].start;
            for (std::size_t k = 0; k < elite.size(); ++k) {
                EXPECT_TRUE(same_bits(start[k], elite[k])) << g << " " << k;
            }
        }
        for (int i = 0; i < lambda; ++i) {
            const int idx = g * lambda + i;
            EXPECT_TRUE(c.p.set.feasible(r.individuals[idx].start));
            if (best < 0 || memetic_better(r.individuals[idx].run, r.individuals[best].run)) {
                best = idx;
            }
        }
    }
    EXPECT_EQ(r.best, best);
}

// Vergleichsverfahren: die Starts sind die von perturbed_starts (perturbed_vector mit
// Startwert seed, Variante j = g·λ + i), ohne x₀ bei include_start = false.
TEST(Memetic, PerturbedStartsMethodUsesExistingGenerator) {
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.options.method = GlobalSearchMethod::PerturbedStarts;
    c.options.generations = 2;
    c.options.lm_iterations = 1;
    c.options.include_start = false;
    const MemeticResult r = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    const int n = 2 * r.lambda;
    ASSERT_EQ(static_cast<int>(r.individuals.size()), n);
    const std::vector<double> expected =
        test::perturbed_starts(c.p.set, c.p.x0, n, c.options.sigma, c.options.seed);
    const std::size_t P = c.p.x0.size();
    for (int i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < P; ++k) {
            EXPECT_TRUE(same_bits(r.individuals[i].start[k], expected[i * P + k])) << i;
        }
    }
    EXPECT_DOUBLE_EQ(r.generations[1].sigma, c.options.sigma);
}

// Budget: Zyklen pro Individuum ⌊Rest/λ⌋, keine Generation ohne einen Zyklus pro Individuum;
// hartes Versagen (kein Strafverfahren), damit die Zählung exakt ist.
TEST(Memetic, BudgetLimitsCycles) {
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.lm.jacobian.quadrature.failure = FailureMode::Hard;
    c.options.generations = 5;
    c.options.lm_iterations = 4;
    c.options.cycle_budget = 70; // λ = 11: zuerst 4 Zyklen pro Individuum, dann ⌊Rest/11⌋
    const MemeticResult r = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    EXPECT_EQ(r.budget, 70);
    EXPECT_LE(r.cycles, 70);
    ASSERT_GE(r.generations.size(), 1u);
    EXPECT_EQ(r.generations[0].lm_iterations, 4);
    for (const MemeticIndividual& ind : r.individuals) {
        EXPECT_LE(ind.run.iterations, r.generations[ind.generation].lm_iterations);
    }
    std::int64_t used = 0;
    for (const MemeticGeneration& g : r.generations) {
        EXPECT_LE(static_cast<std::int64_t>(g.lm_iterations) * r.lambda, 70 - used);
        used += g.cycles;
    }
    EXPECT_LT(70 - used, r.lambda); // danach passt kein Zyklus pro Individuum mehr
}

// Rückruf: fortlaufender Index des Individuums, ein Aufruf pro Zyklus.
TEST(Memetic, IterationCallbackUsesIndividualIndex) {
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.options.generations = 2;
    c.options.lm_iterations = 2;
    int max_run = -1;
    std::int64_t calls = 0;
    c.lm.on_iteration = [&](const LmIterationInfo& info) {
        max_run = std::max(max_run, info.run);
        ++calls;
    };
    const MemeticResult r = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    EXPECT_EQ(max_run, static_cast<int>(r.individuals.size()) - 1);
    EXPECT_EQ(calls, r.cycles);
}

// FP32-Vorfilter (nur Vorfilterung): K·λ Kandidaten, behaltene zuerst in Kandidatenreihenfolge;
// deterministisch.
TEST(Memetic, PrefilterSelectsKeptCandidates) {
    test::MemeticCase c = test::mehrlinser_memetic_case();
    c.options.generations = 2;
    c.options.lm_iterations = 2;
    c.options.sigma = 0.3;
    c.options.prefilter = true;
    c.options.prefilter_oversampling = 3;
    const MemeticResult a = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    const MemeticResult b = memetic_search(c.p.set, c.p.base, c.p.x0, c.lm, c.options);
    EXPECT_TRUE(test::same_memetic(a, b));
    for (const MemeticGeneration& g : a.generations) {
        EXPECT_EQ(g.candidates, 3 * a.lambda);
        EXPECT_GE(g.prefilter_kept, 1);
        EXPECT_LE(g.prefilter_kept, g.candidates);
        // Die ersten min(kept, λ) Individuen sind behaltene Kandidaten in aufsteigender Reihenfolge
        int last = -1;
        for (int i = 0; i < std::min(g.prefilter_kept, a.lambda); ++i) {
            const int cand = a.individuals[g.generation * a.lambda + i].candidate;
            EXPECT_GT(cand, last);
            last = cand;
        }
    }
    EXPECT_THROW(memetic_search(c.p.set, c.p.base, std::vector<double>(3, 0.0), c.lm, c.options),
                 std::invalid_argument);
}

} // namespace
} // namespace mko
