// Memetische globale Suche (M11-T7, docs/optimierung.md, Abschnitt 15).

#include "mikroptim/optim/memetic.hpp"

#include "mikroptim/optim/cma_es.hpp"
#include "mikroptim/optim/variant_generator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("memetic_search: " + message);
}

// Auswahl der λ Kandidaten aus dem Ergebnis des Vorfilters: die behaltenen in
// Kandidatenreihenfolge, danach (falls zu wenige) die aussortierten nach merit_lower aufsteigend,
// bei Gleichstand nach Index.
std::vector<int> select_candidates(const PrefilterResult& res, int candidates, int lambda) {
    std::vector<int> chosen;
    std::vector<int> rejected;
    for (int i = 0; i < candidates; ++i) {
        if (res.keep(i)) {
            if (static_cast<int>(chosen.size()) < lambda) {
                chosen.push_back(i);
            }
        } else {
            rejected.push_back(i);
        }
    }
    if (static_cast<int>(chosen.size()) < lambda) {
        std::stable_sort(rejected.begin(), rejected.end(), [&](int a, int b) {
            const double la = res.merit_lower[static_cast<std::size_t>(a)];
            const double lb = res.merit_lower[static_cast<std::size_t>(b)];
            // NaN zuletzt
            if (std::isnan(la) != std::isnan(lb)) {
                return std::isnan(lb);
            }
            return la < lb;
        });
        for (std::size_t k = 0; k < rejected.size() && static_cast<int>(chosen.size()) < lambda;
             ++k) {
            chosen.push_back(rejected[k]);
        }
    }
    return chosen;
}

} // namespace

const char* to_string(GlobalSearchMethod method) {
    switch (method) {
    case GlobalSearchMethod::Cma:
        return "cma";
    case GlobalSearchMethod::PerturbedStarts:
        return "starts";
    }
    return "?";
}

std::optional<GlobalSearchMethod> global_search_method_from_string(std::string_view name) {
    if (name == "cma") {
        return GlobalSearchMethod::Cma;
    }
    if (name == "starts") {
        return GlobalSearchMethod::PerturbedStarts;
    }
    return std::nullopt;
}

void check_memetic_options(const MemeticOptions& o) {
    if (o.population != 0 && o.population < 2) {
        invalid("population muss 0 (Standard) oder >= 2 sein");
    }
    if (o.generations < 1) {
        invalid("generations muss >= 1 sein");
    }
    if (o.lm_iterations < 1) {
        invalid("lm_iterations muss >= 1 sein");
    }
    if (o.cycle_budget < 0) {
        invalid("cycle_budget muss >= 0 sein");
    }
    if (!(std::isfinite(o.sigma) && o.sigma > 0.0)) {
        invalid("sigma muss endlich und > 0 sein");
    }
    if (o.prefilter_oversampling < 1) {
        invalid("prefilter_oversampling muss >= 1 sein");
    }
    check_prefilter_options(o.prefilter_options);
}

bool memetic_better(const LmRunResult& a, const LmRunResult& b) {
    const bool fa = std::isfinite(a.merit);
    const bool fb = std::isfinite(b.merit);
    if (fa != fb) {
        return fa;
    }
    if (!fa) {
        return false;
    }
    if (a.failed_rays != b.failed_rays) {
        return a.failed_rays < b.failed_rays;
    }
    return a.merit < b.merit;
}

MemeticResult memetic_search(const VariableSet& set, const SystemBatchHost& base,
                             std::span<const double> x0, const LmOptions& lm,
                             const MemeticOptions& options, const MemeticOptimizer& optimizer,
                             const MemeticPrefilterFn& prefilter) {
    check_memetic_options(options);
    const int P = set.size();
    if (P < 1) {
        invalid("keine Variablen");
    }
    const auto uP = static_cast<std::size_t>(P);
    if (x0.size() != uP) {
        invalid("x0 hat Länge " + std::to_string(x0.size()) + ", erwartet " + std::to_string(P));
    }
    for (const double v : x0) {
        if (!std::isfinite(v)) {
            invalid("x0 nicht endlich");
        }
    }
    if (!set.feasible(x0)) {
        invalid("x0 liegt außerhalb der Schranken");
    }
    if (base.variants() != 1) {
        invalid("base muss genau eine Variante haben");
    }

    MemeticResult result;
    const int lambda = options.population > 0 ? options.population : cma_default_lambda(P);
    result.lambda = lambda;
    const std::int64_t budget =
        options.cycle_budget > 0
            ? options.cycle_budget
            : static_cast<std::int64_t>(lambda) * options.generations * options.lm_iterations;
    result.budget = budget;

    // Normierung u_k = x_k/d_k, d_k = max(|x₀_k|, x_typ,k)
    std::vector<double> scale(uP);
    std::vector<double> u0(uP);
    for (std::size_t k = 0; k < uP; ++k) {
        scale[k] = std::max(std::abs(x0[k]), set.typical()[k]);
        u0[k] = x0[k] / scale[k];
    }
    std::optional<CmaEs> cma;
    if (options.method == GlobalSearchMethod::Cma) {
        cma.emplace(u0, options.sigma, lambda);
    }
    PerturbationOptions perturb;
    perturb.curvature = {options.sigma, 0.0};
    perturb.thickness = {options.sigma, 0.0};
    perturb.working_distance = {options.sigma, 0.0};
    perturb.ghost_coefficient = {options.sigma, 0.0};
    perturb.glass = {options.sigma, 0.0};
    perturb.distribution = Distribution::Uniform;
    perturb.seed = options.seed;
    perturb.first_is_base = false;

    const int candidates = options.prefilter ? options.prefilter_oversampling * lambda : lambda;
    std::vector<double> cand(static_cast<std::size_t>(candidates) * uP);
    std::vector<double> u(uP);
    for (int g = 0; g < options.generations; ++g) {
        const std::int64_t rest = budget - result.cycles;
        const std::int64_t per = std::min<std::int64_t>(options.lm_iterations, rest / lambda);
        if (per < 1) {
            break;
        }
        MemeticGeneration gen;
        gen.generation = g;
        gen.individuals = lambda;
        gen.candidates = candidates;
        gen.lm_iterations = static_cast<int>(per);
        gen.sigma = cma ? cma->sigma() : options.sigma;

        // Kandidaten ziehen
        for (int i = 0; i < candidates; ++i) {
            const std::span<double> x(cand.data() + static_cast<std::size_t>(i) * uP, uP);
            if (options.include_start && g == 0 && i == 0) {
                std::copy(x0.begin(), x0.end(), x.begin());
                continue;
            }
            // Elitismus (nur CMA-ES): Kandidat 0 setzt das bisher beste Individuum am
            // verfeinerten Punkt fort
            if (cma && options.elitist && g > 0 && i == 0 && result.best >= 0) {
                const MemeticIndividual& e =
                    result.individuals[static_cast<std::size_t>(result.best)];
                const bool usable =
                    e.run.x.size() == uP && std::all_of(e.run.x.begin(), e.run.x.end(),
                                                        [](double v) { return std::isfinite(v); });
                const std::vector<double>& src = usable ? e.run.x : e.start;
                std::copy(src.begin(), src.end(), x.begin());
                continue;
            }
            if (cma) {
                cma->sample(options.seed, static_cast<std::uint64_t>(i), u);
                for (std::size_t k = 0; k < uP; ++k) {
                    x[k] = u[k] * scale[k];
                }
                set.clamp_to_bounds(x);
            } else {
                const std::uint64_t j =
                    static_cast<std::uint64_t>(g) * static_cast<std::uint64_t>(candidates) +
                    static_cast<std::uint64_t>(i);
                const std::vector<double> y = perturbed_vector(set, x0, perturb, j);
                std::copy(y.begin(), y.end(), x.begin());
            }
        }

        // Vorfilter (FP32, nur Vorfilterung): Auswahl der λ Starts
        std::vector<int> chosen(static_cast<std::size_t>(lambda));
        std::iota(chosen.begin(), chosen.end(), 0);
        gen.prefilter_kept = candidates;
        if (options.prefilter) {
            std::vector<int> runs(static_cast<std::size_t>(candidates));
            std::iota(runs.begin(), runs.end(), 0);
            const SystemBatchHost batch = lm_point_batch(set, base, runs, cand);
            // Zusatzresiduen in FP64 ohne die trace-abhängigen: L_v bleibt eine untere Schranke
            const std::vector<double> extra =
                extra_merit_values(lm.jacobian.extra, batch, lm.jacobian.extra_threads);
            const PrefilterResult res =
                prefilter
                    ? prefilter(batch, lm.jacobian.quadrature, options.prefilter_options, extra)
                    : prefilter_cpu(batch, ResidualLayout(batch.topology(), lm.jacobian.quadrature),
                                    options.prefilter_options, extra);
            gen.prefilter_kept = res.kept;
            chosen = select_candidates(res, candidates, lambda);
        }
        std::vector<double> starts(static_cast<std::size_t>(lambda) * uP);
        for (int i = 0; i < lambda; ++i) {
            const auto src = static_cast<std::size_t>(chosen[static_cast<std::size_t>(i)]) * uP;
            std::copy(cand.begin() + static_cast<std::ptrdiff_t>(src),
                      cand.begin() + static_cast<std::ptrdiff_t>(src + uP),
                      starts.begin() +
                          static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i) * uP));
        }

        // Gebündelter LM aller Individuen der Generation
        LmOptions lo = lm;
        lo.max_iterations = static_cast<int>(per);
        const int offset = static_cast<int>(result.individuals.size());
        if (lm.on_iteration) {
            lo.on_iteration = [&lm, offset](const LmIterationInfo& info) {
                LmIterationInfo shifted = info;
                shifted.run = info.run + offset;
                lm.on_iteration(shifted);
            };
        }
        LmResult res =
            optimizer ? optimizer(set, base, starts, lo) : optimize_lm_cpu(set, base, starts, lo);
        if (res.runs.size() != static_cast<std::size_t>(lambda)) {
            invalid("Optimierer lieferte " + std::to_string(res.runs.size()) + " statt " +
                    std::to_string(lambda) + " Läufe");
        }
        for (int i = 0; i < lambda; ++i) {
            MemeticIndividual ind;
            ind.generation = g;
            ind.candidate = chosen[static_cast<std::size_t>(i)];
            ind.start.assign(
                starts.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i) * uP),
                starts.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i + 1) * uP));
            ind.run = std::move(res.runs[static_cast<std::size_t>(i)]);
            gen.cycles += ind.run.iterations;
            const int index = static_cast<int>(result.individuals.size());
            if (gen.best < 0 ||
                memetic_better(ind.run,
                               result.individuals[static_cast<std::size_t>(gen.best)].run)) {
                gen.best = index;
            }
            result.individuals.push_back(std::move(ind));
        }
        result.cycles += gen.cycles;
        gen.best_merit = result.individuals[static_cast<std::size_t>(gen.best)].run.merit;
        if (result.best < 0 ||
            memetic_better(result.individuals[static_cast<std::size_t>(gen.best)].run,
                           result.individuals[static_cast<std::size_t>(result.best)].run)) {
            result.best = gen.best;
        }

        // CMA-ES fortschreiben: Rang nach dem verfeinerten Ergebnis, Punkte verfeinert (Lamarck)
        // oder Starts (Baldwin), normiert
        if (cma) {
            std::vector<int> ranking(static_cast<std::size_t>(lambda));
            std::iota(ranking.begin(), ranking.end(), 0);
            std::stable_sort(ranking.begin(), ranking.end(), [&](int a, int b) {
                return memetic_better(result.individuals[static_cast<std::size_t>(offset + a)].run,
                                      result.individuals[static_cast<std::size_t>(offset + b)].run);
            });
            std::vector<double> points(static_cast<std::size_t>(lambda) * uP);
            std::vector<double> sampled(static_cast<std::size_t>(lambda) * uP);
            for (int i = 0; i < lambda; ++i) {
                const MemeticIndividual& ind =
                    result.individuals[static_cast<std::size_t>(offset + i)];
                bool use_x = options.lamarckian && ind.run.x.size() == uP;
                if (use_x) {
                    for (const double v : ind.run.x) {
                        use_x = use_x && std::isfinite(v);
                    }
                }
                const std::vector<double>& p = use_x ? ind.run.x : ind.start;
                for (std::size_t k = 0; k < uP; ++k) {
                    points[static_cast<std::size_t>(i) * uP + k] = p[k] / scale[k];
                    sampled[static_cast<std::size_t>(i) * uP + k] = ind.start[k] / scale[k];
                }
            }
            if (options.lamarckian && options.step_size_from_starts) {
                cma->update(points, ranking, sampled);
            } else {
                cma->update(points, ranking);
            }
        }
        result.generations.push_back(gen);
    }
    return result;
}

} // namespace mko
