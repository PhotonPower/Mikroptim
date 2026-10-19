// Glas-Einrasten (docs/optimierung.md, Abschnitt 9).

#include "mikroptim/optim/glass_snap.hpp"

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/glass_variables.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Glas-Einrasten: " + message);
}

void check_weights(const GlassDistanceWeights& weights) {
    for (const double w : weights) {
        if (!(std::isfinite(w) && w >= 0.0)) {
            invalid("Gewichte des Abstands müssen endlich und mindestens 0 sein");
        }
    }
}

// (Kosten, choice lexikographisch): strikte Ordnung der Kombinationen.
bool combination_less(const GlassCombination& a, const GlassCombination& b) {
    if (a.cost != b.cost) {
        return a.cost < b.cost;
    }
    return a.choice < b.choice;
}

// Gilt das Glas bei allen Wellenlängen des Spektrums und bei λ_ref?
bool covers_spectrum(const Glass& glass, const SystemDescription& system) {
    for (const double l : system.spectrum.wavelengths_nm) {
        if (!glass.in_range(l)) {
            return false;
        }
    }
    return glass.in_range(system.spectrum.reference_nm);
}

} // namespace

double glass_distance(const GlassHull& hull, const Point3& a, const Point3& b,
                      const GlassDistanceWeights& weights) {
    check_weights(weights);
    const Point3 u = hull.normalize(a);
    const Point3 v = hull.normalize(b);
    double s = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
        const double d = u[j] - v[j];
        s += weights[j] * (d * d);
    }
    return std::sqrt(s);
}

std::vector<GlassCandidate>
nearest_glasses(const GlassHull& hull, const Point3& x, int k, const GlassDistanceWeights& weights,
                const std::function<bool(const GlassHullPoint&)>& usable) {
    if (k < 1) {
        invalid("Zahl der Nachbarn muss mindestens 1 sein");
    }
    check_weights(weights);
    const std::vector<GlassHullPoint>& points = hull.points();
    std::vector<GlassCandidate> all;
    all.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const GlassHullPoint& p = points[i];
        if (usable && !usable(p)) {
            continue;
        }
        GlassCandidate c;
        c.name = p.name;
        c.hull_index = static_cast<int>(i);
        c.coordinates = p.coordinates;
        c.distance = glass_distance(hull, x, p.coordinates, weights);
        all.push_back(std::move(c));
    }
    std::sort(all.begin(), all.end(), [](const GlassCandidate& a, const GlassCandidate& b) {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }
        return a.hull_index < b.hull_index;
    });
    if (all.size() > static_cast<std::size_t>(k)) {
        all.resize(static_cast<std::size_t>(k));
    }
    return all;
}

std::uint64_t glass_combination_count(const std::vector<std::vector<double>>& costs) {
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t n = 1;
    for (const std::vector<double>& c : costs) {
        const auto m = static_cast<std::uint64_t>(c.size());
        if (m == 0) {
            return 0;
        }
        n = n > kMax / m ? kMax : n * m;
    }
    return n;
}

std::vector<GlassCombination> glass_combinations(const std::vector<std::vector<double>>& costs,
                                                 int max_combinations) {
    if (max_combinations < 1) {
        invalid("Höchstzahl der Kombinationen muss mindestens 1 sein");
    }
    for (std::size_t e = 0; e < costs.size(); ++e) {
        if (costs[e].empty()) {
            invalid("Element " + std::to_string(e) + " hat keine Kandidaten");
        }
        for (const double c : costs[e]) {
            if (!(std::isfinite(c) && c >= 0.0)) {
                invalid("Kosten müssen endlich und mindestens 0 sein");
            }
        }
    }
    const auto limit = static_cast<std::size_t>(max_combinations);
    std::vector<GlassCombination> beam(1); // leere Teilkombination, Kosten 0
    for (const std::vector<double>& element : costs) {
        std::vector<GlassCombination> next;
        next.reserve(beam.size() * element.size());
        for (const GlassCombination& partial : beam) {
            for (std::size_t j = 0; j < element.size(); ++j) {
                GlassCombination c;
                c.choice = partial.choice;
                c.choice.push_back(static_cast<int>(j));
                c.cost = partial.cost + element[j];
                next.push_back(std::move(c));
            }
        }
        std::sort(next.begin(), next.end(), combination_less);
        if (next.size() > limit) {
            next.resize(limit);
        }
        beam = std::move(next);
    }
    return beam;
}

int select_best_run(const LmResult& result) {
    int best = -1;
    for (std::size_t i = 0; i < result.runs.size(); ++i) {
        const LmRunResult& r = result.runs[i];
        if (!std::isfinite(r.merit)) {
            continue;
        }
        if (best < 0) {
            best = static_cast<int>(i);
            continue;
        }
        const LmRunResult& b = result.runs[static_cast<std::size_t>(best)];
        if (r.failed_rays < b.failed_rays ||
            (r.failed_rays == b.failed_rays && r.merit < b.merit)) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

double GlassSnapResult::degradation() const {
    return best_run().run.merit - continuous_merit;
}

double GlassSnapResult::relative_degradation() const {
    if (continuous_merit == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return degradation() / continuous_merit;
}

SystemDescription snapped_description(const SystemDescription& system,
                                      std::span<const int> file_surfaces,
                                      std::span<const std::string> glasses) {
    if (file_surfaces.size() != glasses.size()) {
        invalid("ungleich viele Flächen und Gläser");
    }
    SystemDescription out = system;
    for (std::size_t e = 0; e < file_surfaces.size(); ++e) {
        const int s = file_surfaces[e];
        if (s < 0 || static_cast<std::size_t>(s) >= out.surfaces.size()) {
            invalid("Fläche " + std::to_string(s) + " existiert nicht");
        }
        MaterialSpec m;
        m.kind = MaterialKind::Catalog;
        m.name = glasses[e];
        out.surfaces[static_cast<std::size_t>(s)].material = std::move(m);
    }
    if (out.variables) {
        std::erase_if(out.variables->items,
                      [](const VariableSpec& v) { return v.type == VariableType::Glass; });
        if (out.variables->items.empty()) {
            out.variables.reset();
        }
    }
    return out;
}

LoadedSystem rebuild_loaded_system(const LoadedSystem& like, SystemDescription system) {
    validate_system(system);
    validate_materials(system,
                       [&like](std::string_view name) { return like.catalog.contains(name); });
    LoadedSystem out;
    out.catalog_files = like.catalog_files;
    out.catalog = like.catalog;
    out.index = make_system_index_table(system, out.catalog);
    out.table = make_surface_table(system, out.index);
    out.system = std::move(system);
    return out;
}

GlassSnapResult snap_glasses(const LoadedSystem& system, const GlassHull& hull,
                             const GlassSnapOptions& options, const GlassSnapOptimizer& optimize) {
    if (options.neighbors < 1) {
        invalid("Zahl der Nachbarn muss mindestens 1 sein");
    }
    if (options.max_combinations < 1) {
        invalid("Höchstzahl der Kombinationen muss mindestens 1 sein");
    }
    check_weights(options.distance_weights);
    if (hull.empty()) {
        invalid("keine Gläser in der Hülle");
    }
    const GlassSnapOptimizer run_lm =
        optimize ? optimize
                 : GlassSnapOptimizer([](const VariableSet& set, const SystemBatchHost& base,
                                         std::span<const double> x0, const LmOptions& o) {
                       return optimize_lm_cpu(set, base, x0, o);
                   });
    // Nur auswerten: kein Zyklus, kein Neustart des Strafverfahrens, kein Protokoll.
    LmOptions eval = options.lm;
    eval.max_iterations = 0;
    eval.fail_weight_factor = 1.0;
    eval.on_iteration = {};

    GlassSnapResult r;

    // 1. Kontinuierliches System (Startgläser → Buchdahl, wie in der Optimierung)
    SystemDescription cont = system.system;
    if (convert_glass_variables(cont, system.catalog).empty()) {
        invalid("das System hat keine Glasvariablen");
    }
    r.continuous = rebuild_loaded_system(system, std::move(cont));
    const SystemBatchHost cont_base = make_system_batch(r.continuous, 1);
    r.continuous_set = VariableSet(r.continuous.system, cont_base.topology());
    r.continuous_x = r.continuous_set.extract(cont_base, 0);
    {
        const LmResult res = run_lm(r.continuous_set, cont_base, r.continuous_x, eval);
        r.continuous_merit = res.runs.at(0).history.at(0);
        r.continuous_failed_rays = res.runs.at(0).failed_rays;
    }

    // 2. Kandidaten pro Glaselement
    const SystemDescription& cs = r.continuous.system;
    const auto usable = [&](const GlassHullPoint& p) {
        const Glass* g = system.catalog.find(p.name);
        return g != nullptr && covers_spectrum(*g, cs);
    };
    std::vector<std::vector<double>> costs;
    std::vector<int> file_surfaces;
    for (const Variable& v : r.continuous_set.variables()) {
        if (v.type != VariableType::Glass || v.glass_component != 0) {
            continue;
        }
        GlassSnapElement e;
        e.file_surface = v.file_surface;
        e.batch_surface = v.batch_surface;
        e.coordinates = batch_glass_coordinates(cont_base, 0, v.batch_surface);
        e.hull_distance = hull.distance(e.coordinates);
        e.candidates = nearest_glasses(hull, e.coordinates, options.neighbors,
                                       options.distance_weights, usable);
        if (e.candidates.empty()) {
            invalid("kein Katalogglas für Fläche " + std::to_string(v.file_surface) +
                    " (keines deckt das Spektrum ab)");
        }
        std::vector<double> c;
        for (const GlassCandidate& g : e.candidates) {
            c.push_back(g.distance * g.distance);
        }
        costs.push_back(std::move(c));
        file_surfaces.push_back(v.file_surface);
        r.elements.push_back(std::move(e));
    }

    // 3. Kombinationen
    r.total_combinations = glass_combination_count(costs);
    const std::vector<GlassCombination> combinations =
        glass_combinations(costs, options.max_combinations);
    r.truncated = combinations.size() < r.total_combinations;

    // 4. Vorlage mit einer Variante pro Kombination: Brechzahlen der Katalog-Gläser wie beim
    // Laden der Ergebnisdatei (dieselbe Beschreibung, dieselbe Brechzahltabelle).
    SystemBatchHost& base = r.base;
    for (std::size_t c = 0; c < combinations.size(); ++c) {
        GlassSnapRun run;
        run.combination = combinations[c];
        for (std::size_t e = 0; e < r.elements.size(); ++e) {
            run.glasses.push_back(
                r.elements[e].candidates[static_cast<std::size_t>(combinations[c].choice[e])].name);
        }
        const LoadedSystem loaded = rebuild_loaded_system(
            r.continuous, snapped_description(cs, file_surfaces, run.glasses));
        const SystemBatchHost b = make_system_batch(loaded, 1);
        if (c == 0) {
            base = b;
            base.resize(static_cast<int>(combinations.size()));
            r.set = VariableSet(loaded.system, base.topology());
            if (r.set.empty()) {
                invalid("das System hat außer Glas keine Variablen (nichts zu reoptimieren)");
            }
            r.x0 = r.set.extract(b, 0);
        } else {
            base.copy_variant_from(b, 0, static_cast<int>(c));
        }
        r.runs.push_back(std::move(run));
    }

    // 5. Alle Kombinationen gemeinsam: eingerastet ohne Reoptimierung, dann reoptimiert
    std::vector<double> starts;
    starts.reserve(r.x0.size() * combinations.size());
    for (std::size_t c = 0; c < combinations.size(); ++c) {
        starts.insert(starts.end(), r.x0.begin(), r.x0.end());
    }
    const LmResult snapped = run_lm(r.set, base, starts, eval);
    const LmResult result = run_lm(r.set, base, starts, options.lm);
    for (std::size_t c = 0; c < r.runs.size(); ++c) {
        r.runs[c].snapped_merit = snapped.runs.at(c).history.at(0);
        r.runs[c].snapped_failed_rays = snapped.runs.at(c).failed_rays;
        r.runs[c].run = result.runs.at(c);
    }
    r.best = select_best_run(result);
    if (r.best < 0) {
        throw std::runtime_error("Glas-Einrasten: Merit in keinem Lauf endlich");
    }

    // 6. Ergebnis als Systembeschreibung
    const GlassSnapRun& best = r.best_run();
    SystemDescription out = snapped_description(cs, file_surfaces, best.glasses);
    r.set.apply_to_description(out, best.run.x);
    r.result = rebuild_loaded_system(r.continuous, std::move(out));
    return r;
}

} // namespace mko
