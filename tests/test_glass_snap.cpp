// Glas-Einrasten (M6-T5) auf der CPU: Nachbarsuche und Kombinationen gegen Bruteforce,
// Abnahme am Achromaten mit zwei Glasvariablen (nach dem Einrasten nur Katalog-Gläser,
// Merit-Verschlechterung berichtet, bester Lauf, Ergebnisdatei, Determinismus;
// docs/optimierung.md, Abschnitt 9).

#include "glass_snap_test_systems.hpp"
#include "mikroptim/io/system_json.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {
namespace {

// Gleichverteilt in [0, 1) aus 53 Bits (plattformunabhängig).
double unit_random(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

LmResult cpu_optimize(const VariableSet& set, const SystemBatchHost& base,
                      std::span<const double> x0, const LmOptions& o) {
    return optimize_lm_cpu(set, base, x0, o);
}

GlassHull test_hull() {
    return make_glass_hull(load_residual_system(kSnapSystem).catalog);
}

// Unabhängige Bruteforce-Suche: Abstand aus Mittelwert und Spannweite der Hülle, alle Gläser
// sortiert nach (Abstand, Index).
std::vector<std::pair<double, int>> brute_force_nearest(const GlassHull& hull, const Point3& x,
                                                        const GlassDistanceWeights& w) {
    std::vector<std::pair<double, int>> all;
    for (std::size_t i = 0; i < hull.points().size(); ++i) {
        const Point3& p = hull.points()[i].coordinates;
        double s = 0.0;
        for (int j = 0; j < 3; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            const double du = (x[jj] - p[jj]) / hull.scale()[jj];
            s += w[jj] * du * du;
        }
        all.emplace_back(std::sqrt(s), static_cast<int>(i));
    }
    std::sort(all.begin(), all.end());
    return all;
}

TEST(GlassSnap, NearestGlassesMatchBruteForce) {
    const GlassHull hull = test_hull();
    ASSERT_GE(hull.points().size(), 30u);
    std::mt19937_64 rng(20261019);
    const std::vector<GlassDistanceWeights> weights = {{1.0, 1.0, 1.0}, {4.0, 1.0, 0.25}};
    int checked = 0;
    for (const GlassDistanceWeights& w : weights) {
        for (int trial = 0; trial < 200; ++trial) {
            // Punkte in und um die Hüllbox (normiert −1 … 1)
            Point3 u{};
            for (double& c : u) {
                c = 2.0 * unit_random(rng) - 1.0;
            }
            const Point3 x = hull.denormalize(u);
            const int k = 1 + trial % 6;
            const std::vector<GlassCandidate> got = nearest_glasses(hull, x, k, w);
            const std::vector<std::pair<double, int>> ref = brute_force_nearest(hull, x, w);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(k));
            for (std::size_t i = 0; i < got.size(); ++i) {
                EXPECT_EQ(got[i].hull_index, ref[i].second) << trial << " " << i;
                EXPECT_EQ(got[i].name, hull.points()[static_cast<std::size_t>(ref[i].second)].name);
                EXPECT_NEAR(got[i].distance, ref[i].first, kGlassSnapDistance);
                EXPECT_EQ(got[i].coordinates,
                          hull.points()[static_cast<std::size_t>(ref[i].second)].coordinates);
                EXPECT_EQ(got[i].distance,
                          glass_distance(hull, x, got[i].coordinates, w)); // dieselbe Funktion
                ++checked;
            }
        }
    }
    std::printf("Nachbarsuche: %d Kandidaten gegen Bruteforce geprüft (%zu Gläser)\n", checked,
                hull.points().size());
}

TEST(GlassSnap, NearestGlassesSelfFilterAndErrors) {
    const GlassHull hull = test_hull();
    // Ein Glas der Hülle ist sein eigener nächster Nachbar (Abstand exakt 0).
    for (std::size_t i = 0; i < hull.points().size(); ++i) {
        const std::vector<GlassCandidate> c =
            nearest_glasses(hull, hull.points()[i].coordinates, 1);
        ASSERT_EQ(c.size(), 1u);
        EXPECT_EQ(c[0].distance, 0.0);
        EXPECT_EQ(c[0].name, hull.points()[i].name);
    }
    // Filter: ohne N-BK7 kommt der zweitnächste zuerst
    const Point3 bk7 = hull.points().at(0).coordinates;
    ASSERT_EQ(hull.points()[0].name, "N-BK7");
    const std::vector<GlassCandidate> all = nearest_glasses(hull, bk7, 3);
    const std::vector<GlassCandidate> filtered = nearest_glasses(
        hull, bk7, 2, {1.0, 1.0, 1.0}, [](const GlassHullPoint& p) { return p.name != "N-BK7"; });
    ASSERT_EQ(filtered.size(), 2u);
    EXPECT_EQ(filtered[0].name, all[1].name);
    EXPECT_EQ(filtered[1].name, all[2].name);
    // mehr Nachbarn als Gläser: alle
    EXPECT_EQ(nearest_glasses(hull, bk7, 1000).size(), hull.points().size());
    // Fehler
    EXPECT_THROW(nearest_glasses(hull, bk7, 0), std::invalid_argument);
    EXPECT_THROW(nearest_glasses(hull, bk7, 1, {1.0, -1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(glass_distance(hull, bk7, bk7, {1.0, 1.0, std::nan("")}), std::invalid_argument);
}

// Bruteforce: alle Kombinationen (Odometer), Kosten in Elementreihenfolge summiert, sortiert
// nach (Kosten, choice), die ersten n.
std::vector<GlassCombination> brute_force_combinations(const std::vector<std::vector<double>>& c,
                                                       int n) {
    std::vector<GlassCombination> all;
    std::vector<int> idx(c.size(), 0);
    for (;;) {
        GlassCombination g;
        g.choice = idx;
        for (std::size_t e = 0; e < c.size(); ++e) {
            g.cost += c[e][static_cast<std::size_t>(idx[e])];
        }
        all.push_back(g);
        std::size_t e = c.size();
        while (e > 0) {
            --e;
            if (++idx[e] < static_cast<int>(c[e].size())) {
                break;
            }
            idx[e] = 0;
            if (e == 0) {
                e = c.size() + 1; // fertig
                break;
            }
        }
        if (e == c.size() + 1 || c.empty()) {
            break;
        }
    }
    std::sort(all.begin(), all.end(), [](const GlassCombination& a, const GlassCombination& b) {
        return a.cost != b.cost ? a.cost < b.cost : a.choice < b.choice;
    });
    if (all.size() > static_cast<std::size_t>(n)) {
        all.resize(static_cast<std::size_t>(n));
    }
    return all;
}

TEST(GlassSnap, CombinationsMatchBruteForce) {
    std::mt19937_64 rng(42);
    int cases = 0;
    for (int elements = 1; elements <= 4; ++elements) {
        for (int k = 1; k <= 4; ++k) {
            for (const bool ties : {false, true}) {
                std::vector<std::vector<double>> costs(static_cast<std::size_t>(elements));
                for (std::vector<double>& c : costs) {
                    for (int j = 0; j < k; ++j) {
                        // mit Gleichständen: Vielfache von 1/4 (exakt summierbar)
                        c.push_back(ties ? 0.25 * static_cast<double>(rng() % 4)
                                         : unit_random(rng));
                    }
                    std::sort(c.begin(), c.end()); // Kandidaten aufsteigend wie nearest_glasses
                }
                const std::uint64_t total = glass_combination_count(costs);
                EXPECT_EQ(total, static_cast<std::uint64_t>(std::pow(k, elements)));
                for (const int n : {1, 2, 3, 5, 8, 17, 64, 300}) {
                    const std::vector<GlassCombination> got = glass_combinations(costs, n);
                    const std::vector<GlassCombination> ref = brute_force_combinations(costs, n);
                    ASSERT_EQ(got.size(),
                              std::min<std::size_t>(static_cast<std::size_t>(n), total));
                    EXPECT_EQ(got, ref) << elements << " Elemente, k = " << k << ", n = " << n
                                        << (ties ? ", mit Gleichständen" : "");
                    // deterministisch
                    EXPECT_EQ(glass_combinations(costs, n), got);
                    ++cases;
                }
            }
        }
    }
    std::printf("Kombinationen: %d Fälle gegen Bruteforce geprüft\n", cases);
    // Sättigung und Fehler
    const std::vector<std::vector<double>> many(40, std::vector<double>(8, 0.0));
    EXPECT_EQ(glass_combination_count(many), UINT64_MAX);
    EXPECT_EQ(glass_combinations(many, 5).size(), 5u);
    EXPECT_EQ(glass_combination_count({{0.0}, {}}), 0u);
    EXPECT_EQ(glass_combination_count({}), 1u);
    EXPECT_THROW(glass_combinations({{0.0}}, 0), std::invalid_argument);
    EXPECT_THROW(glass_combinations({{0.0}, {}}, 3), std::invalid_argument);
    EXPECT_THROW(glass_combinations({{-1.0}}, 3), std::invalid_argument);
}

TEST(GlassSnap, SnappedDescription) {
    const LoadedSystem raw = load_residual_system(kSnapSystem);
    const std::vector<int> surfaces = {1, 2};
    const std::vector<std::string> glasses = {"CAF2", "SYN-20"};
    const SystemDescription d = snapped_description(raw.system, surfaces, glasses);
    EXPECT_EQ(d.surfaces[1].material.kind, MaterialKind::Catalog);
    EXPECT_EQ(d.surfaces[1].material.name, "CAF2");
    EXPECT_EQ(d.surfaces[2].material.name, "SYN-20");
    ASSERT_TRUE(d.variables.has_value());
    for (const VariableSpec& v : d.variables->items) {
        EXPECT_NE(v.type, VariableType::Glass);
    }
    EXPECT_EQ(d.variables->items.size(), 2u); // Krümmungen, Arbeitsabstand
    // nur Glasvariablen: Abschnitt entfällt
    SystemDescription only_glass = raw.system;
    std::erase_if(only_glass.variables->items,
                  [](const VariableSpec& v) { return v.type != VariableType::Glass; });
    EXPECT_FALSE(snapped_description(only_glass, surfaces, glasses).variables.has_value());
    const LoadedSystem l = rebuild_loaded_system(raw, d);
    EXPECT_EQ(l.index.at(1, 0), raw.catalog.at("CAF2").index(486.13));
    EXPECT_THROW(snapped_description(raw.system, surfaces, std::vector<std::string>{"CAF2"}),
                 std::invalid_argument);
    EXPECT_THROW(
        snapped_description(raw.system, std::vector<int>{9}, std::vector<std::string>{"CAF2"}),
        std::invalid_argument);
}

TEST(GlassSnap, Errors) {
    const GlassHull hull = test_hull();
    // ohne Glasvariablen
    const LoadedSystem plain = load_residual_system("einzellinse_bk7_biegung.json");
    EXPECT_THROW(snap_glasses(plain, hull), std::invalid_argument);
    // nur Glasvariablen: nichts zu reoptimieren
    const LoadedSystem raw = load_residual_system(kSnapSystem);
    SystemDescription only_glass = raw.system;
    std::erase_if(only_glass.variables->items,
                  [](const VariableSpec& v) { return v.type != VariableType::Glass; });
    EXPECT_THROW(snap_glasses(rebuild_loaded_system(raw, only_glass), hull), std::invalid_argument);
    GlassSnapOptions o;
    o.neighbors = 0;
    EXPECT_THROW(snap_glasses(raw, hull, o), std::invalid_argument);
    o = {};
    o.max_combinations = 0;
    EXPECT_THROW(snap_glasses(raw, hull, o), std::invalid_argument);
    EXPECT_THROW(snap_glasses(raw, GlassHull{}), std::invalid_argument);
}

void print_snap(const char* tag, const GlassSnapResult& r) {
    for (const GlassSnapElement& e : r.elements) {
        std::printf("[%s] Fläche %d: n0 = %.7f nu1 = %.6e nu2 = %.6e, Abstand zur Hülle %.3e\n",
                    tag, e.file_surface, e.coordinates[0], e.coordinates[1], e.coordinates[2],
                    e.hull_distance);
        for (const GlassCandidate& c : e.candidates) {
            std::printf("[%s]     %-8s d = %.6e\n", tag, c.name.c_str(), c.distance);
        }
    }
    for (std::size_t i = 0; i < r.runs.size(); ++i) {
        const GlassSnapRun& run = r.runs[i];
        std::string names;
        for (const std::string& g : run.glasses) {
            names += (names.empty() ? "" : ", ") + g;
        }
        std::printf("[%s]   %2zu %-18s Σd² = %.4e  eingerastet %.6e  reoptimiert %.10e  %4d Zyklen "
                    "%-18s versagt %lld%s\n",
                    tag, i, names.c_str(), run.combination.cost, run.snapped_merit, run.run.merit,
                    run.run.iterations, to_string(run.run.status),
                    static_cast<long long>(run.run.failed_rays),
                    static_cast<int>(i) == r.best ? "  bester" : "");
    }
    std::printf("[%s] Merit kontinuierlich %.10e, eingerastet %.10e, reoptimiert %.10e, "
                "Verschlechterung %.4e (relativ %.4e)\n",
                tag, r.continuous_merit, r.best_run().snapped_merit, r.best_run().run.merit,
                r.degradation(), r.relative_degradation());
}

// Merit (ohne Zyklus) eines geladenen Systems mit den Einstellungen des Einrastens.
double loaded_merit(const LoadedSystem& loaded, const GlassSnapOptions& o) {
    const SystemBatchHost b = make_system_batch(loaded, 1);
    const VariableSet set(loaded.system, b.topology());
    LmOptions eval = o.lm;
    eval.max_iterations = 0;
    eval.jacobian.extra = snap_extra(loaded.system);
    return optimize_lm_cpu(set, b, set.extract(b, 0), eval).runs.at(0).history.at(0);
}

TEST(GlassSnap, AcceptanceAchromatCpu) {
    const ContinuousOptimum c = continuous_optimum(cpu_optimize);
    std::printf("[CPU] kontinuierlich: F %.6e -> %.10e, %d Zyklen, %s\n", c.run.history.front(),
                c.run.merit, c.run.iterations, to_string(c.run.status));
    ASSERT_TRUE(lm_converged(c.run.status));
    ASSERT_EQ(c.run.failed_rays, 0);
    const GlassSnapOptions o = snap_options(c.loaded.system, 3, 64);
    const GlassSnapResult r = snap_glasses(c.loaded, c.hull, o);
    print_snap("CPU", r);

    ASSERT_EQ(r.elements.size(), 2u);
    EXPECT_EQ(r.total_combinations, 9u);
    EXPECT_FALSE(r.truncated);
    ASSERT_EQ(r.runs.size(), 9u);
    ASSERT_GE(r.best, 0);

    // Nur Katalog-Gläser: im Ergebnis keine Buchdahl-Materialien und keine Glasvariablen
    const SystemDescription& res = r.result.system;
    for (const Surface& s : res.surfaces) {
        EXPECT_NE(s.material.kind, MaterialKind::Buchdahl);
        EXPECT_NE(s.material.kind, MaterialKind::Constant);
    }
    for (std::size_t e = 0; e < r.elements.size(); ++e) {
        const MaterialSpec& m =
            res.surfaces[static_cast<std::size_t>(r.elements[e].file_surface)].material;
        EXPECT_EQ(m.kind, MaterialKind::Catalog);
        EXPECT_EQ(m.name, r.best_run().glasses[e]);
        EXPECT_TRUE(r.result.catalog.contains(m.name));
    }
    ASSERT_TRUE(res.variables.has_value());
    for (const VariableSpec& v : res.variables->items) {
        EXPECT_NE(v.type, VariableType::Glass);
    }
    for (const Variable& v : r.set.variables()) {
        EXPECT_NE(v.type, VariableType::Glass);
    }
    // ... und im Batch: jede Variante hat die Sellmeier-Brechzahlen ihrer Katalog-Gläser
    for (std::size_t run = 0; run < r.runs.size(); ++run) {
        for (std::size_t e = 0; e < r.elements.size(); ++e) {
            const Glass& g = r.result.catalog.at(r.runs[run].glasses[e]);
            const int s = r.elements[e].batch_surface;
            for (int l = 0; l < r.base.wavelengths(); ++l) {
                EXPECT_EQ(r.base.index(static_cast<int>(run), s, l),
                          g.index(r.base.wavelength_nm(l)));
            }
            EXPECT_EQ(r.base.paraxial_index(static_cast<int>(run), s),
                      g.index(r.base.reference_wavelength_nm));
        }
        EXPECT_EQ(r.runs[run].snapped_merit, r.runs[run].run.history.front());
    }

    // Bester Lauf: kleinste Merit der Liste (alle versagensfrei)
    for (std::size_t i = 0; i < r.runs.size(); ++i) {
        EXPECT_EQ(r.runs[i].run.failed_rays, 0);
        EXPECT_GE(r.runs[i].run.merit, r.best_run().run.merit) << i;
    }
    // Verschlechterung berichtet und nicht negativ (bis auf Rundung)
    EXPECT_TRUE(std::isfinite(r.degradation()));
    EXPECT_GE(r.degradation(), -kGlassSnapMeritSlack * r.continuous_merit);
    EXPECT_GE(r.best_run().snapped_merit, r.best_run().run.merit);

    // Ergebnisdatei: lädt und liefert die berichtete Merit (bitgleich)
    SystemDescription file = res;
    file.glass_catalogs.clear();
    for (const std::filesystem::path& p : r.result.catalog_files) {
        const std::u8string u = p.generic_u8string();
        file.glass_catalogs.emplace_back(u.begin(), u.end());
    }
    const std::filesystem::path out =
        std::filesystem::temp_directory_path() / "mikroptim_glass_snap_test.json";
    write_system_file(file, out);
    const LoadedSystem reloaded = load_system(out);
    std::filesystem::remove(out);
    EXPECT_EQ(loaded_merit(reloaded, o), r.best_run().run.merit);
    EXPECT_EQ(loaded_merit(r.result, o), r.best_run().run.merit);

    // Determinismus: zweiter Lauf bitgleich
    const GlassSnapResult again = snap_glasses(c.loaded, c.hull, o);
    ASSERT_EQ(again.runs.size(), r.runs.size());
    EXPECT_EQ(again.best, r.best);
    EXPECT_EQ(again.continuous_merit, r.continuous_merit);
    for (std::size_t i = 0; i < r.runs.size(); ++i) {
        EXPECT_EQ(again.runs[i].glasses, r.runs[i].glasses);
        EXPECT_EQ(again.runs[i].run.x, r.runs[i].run.x);
        EXPECT_EQ(again.runs[i].run.merit, r.runs[i].run.merit);
        EXPECT_EQ(again.runs[i].run.history, r.runs[i].run.history);
    }
    EXPECT_TRUE(again.result.system == r.result.system);

    // Begrenzung: 4 Kombinationen = die 4 mit der kleinsten Summe d² aus der vollen Liste
    const GlassSnapResult limited =
        snap_glasses(c.loaded, c.hull, snap_options(c.loaded.system, 3, 4));
    EXPECT_TRUE(limited.truncated);
    EXPECT_EQ(limited.total_combinations, 9u);
    ASSERT_EQ(limited.runs.size(), 4u);
    for (std::size_t i = 0; i < limited.runs.size(); ++i) {
        EXPECT_EQ(limited.runs[i].combination, r.runs[i].combination);
        EXPECT_EQ(limited.runs[i].run.x, r.runs[i].run.x); // Läufe unabhängig voneinander
    }
}

} // namespace
} // namespace mko::test
