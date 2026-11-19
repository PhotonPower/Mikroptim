// Abnahme der algebraischen Glasvorauswahl gegen Bruteforce-Raytrace (M11-T5g,
// docs/optimierung-erweiterungen.md, Abschnitt 5): Jedes Glaspaar der Testkataloge
// (synthetisch.json + basis.json), das die Vorauswahl löst, wird als Kittglied (frei erfunden:
// f = 100 mm, NA 0,05, Blende am Glied, Linien g, F, e, d, C) mit den Startradien des ganzen
// Glieds aufgebaut – beide Lösungen von S_I = 0, damit jedes Paar in beiden Becken startet – und
// mit dem gebündelten LM optimiert (drei Krümmungen und Arbeitsabstand, Brennweite als Gleichung;
// Merit eines Paars = kleinere der beiden). Der beste Vorschlag der Vorauswahl (Rang nach dem
// spektralen Rest über dieselben Linien) muss unter den besten fünf nach Merit liegen.

#include "fd_test_systems.hpp"
#include "tolerances.hpp"

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace mko::test {
namespace {

constexpr double kFocal = 100.0;   // f [mm]
constexpr double kStop = 5.0;      // Blendenradius [mm] (NA 0,05)
constexpr double kAperture = 5.5;  // Halbdurchmesser der Linsen [mm]
constexpr double kMinEdge = 1.0;   // Randdicke beim Aufbau [mm]
constexpr double kMinCenter = 1.5; // Mindest-Mittendicke [mm]
// Linien g, F, e, d, C [nm]; λ_ref = e (Bezug der Buchdahl-Koordinate)
constexpr std::array<double, 5> kLines{435.835, 486.133, 546.074, 587.562, 656.273};
constexpr int kBruteForceCycles = 150; // mit 400 Zyklen und 10 Ringen dieselbe Rangfolge

double sag(double c, double h) {
    return c * h * h / (1.0 + std::sqrt(1.0 - c * c * h * h));
}

struct PairSystem {
    bool buildable = false;
    SystemDescription system;
};

// Kittglied aus zwei Gläsern mit den Startkrümmungen c (3 Flächen); Mittendicken so, dass die
// Randdicke bei kAperture kMinEdge beträgt (mindestens kMinCenter); Arbeitsabstand = paraxiale
// hintere Schnittweite bei λ_ref.
PairSystem pair_system(const GlassCatalog& cat, const std::string& g1, const std::string& g2,
                       const std::vector<double>& c) {
    PairSystem r;
    for (double x : c) {
        if (std::abs(x) * kAperture >= 0.95) {
            return r;
        }
    }
    const double t1 = std::max(kMinCenter, kMinEdge + sag(c[0], kAperture) - sag(c[1], kAperture));
    const double t2 = std::max(kMinCenter, kMinEdge + sag(c[1], kAperture) - sag(c[2], kAperture));
    const auto json = [&](double wd) {
        std::ostringstream s;
        s.precision(17);
        s << R"({"format": "mikroptim.system/1", "name": "Kittglied", "units": "mm",)"
          << R"("glass_catalogs": [")" << MKO_SOURCE_DIR << R"(/data/glass/synthetisch.json", ")"
          << MKO_SOURCE_DIR << R"(/data/glass/basis.json"],)"
          << R"("spectrum": {"wavelengths_nm": [)" << kLines[0] << ", " << kLines[1] << ", "
          << kLines[2] << ", " << kLines[3] << ", " << kLines[4] << "],"
          << R"("weights": [1, 1, 1, 1, 1], "reference_nm": 546.074},)"
          << R"("source": {"type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0]},)"
          << R"("target": {"focal_length": )" << kFocal << "},"
          << R"("surfaces": [{"type": "stop", "semi_diameter": )" << kStop
          << R"(, "thickness": 0.0, "material": "AIR"},)"
          << R"({"type": "sphere", "curvature": )" << c[0] << R"(, "thickness": )" << t1
          << R"(, "material": ")" << g1 << R"(", "semi_diameter": )" << kAperture << "},"
          << R"({"type": "sphere", "curvature": )" << c[1] << R"(, "thickness": )" << t2
          << R"(, "material": ")" << g2 << R"(", "semi_diameter": )" << kAperture << "},"
          << R"({"type": "sphere", "curvature": )" << c[2] << R"(, "thickness": )" << wd
          << R"(, "material": "AIR", "semi_diameter": )" << kAperture << "}],"
          << R"("variables": {"items": [{"type": "curvature", "surfaces": [1, 2, 3]},)"
          << R"({"type": "working_distance"}]},)"
          << R"("constraints": {"items": [{"type": "focal_length", "weight": 100.0}]}})";
        return read_system_json(s.str());
    };
    SystemDescription d = json(kFocal);
    const ParaxialData p = compute_paraxial(d, cat, 546.074);
    if (!(p.back_focal_distance > 1.0) || !std::isfinite(p.back_focal_distance)) {
        return r;
    }
    r.system = json(p.back_focal_distance);
    r.buildable = true;
    return r;
}

struct Candidate {
    std::vector<std::string> glasses;
    int algebraic_rank = 0;         // 1 = bester Vorschlag
    double residual = 0.0;          // Koeffizient der Ordnung 2 (sekundäres Spektrum)
    double spectral_residual = 0.0; // Rang der Vorauswahl (Spektrum der Abnahme)
    double power_sum = 0.0;
    std::vector<int> runs;   // Varianten: Startformen (beide Lösungen von S_I = 0)
    std::vector<int> chosen; // 1, wenn die Startform die von thin_member_shape ist
    std::vector<int> flat;   // 1, wenn die Startform die flachere ist (kleinere Σc²)
    double merit = 0.0;      // kleinste Merit über die Startformen
    int best_run = -1;
    bool best_is_chosen = false; // gewann die Startform von thin_member_shape?
    bool best_is_flat = false;
    int status = 0;
    std::int64_t failed = 0;
};

TEST(GlassPreselectionRaytrace, BestPairAmongTopFiveAfterRaytrace) {
    GlassCatalog cat = GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/synthetisch.json");
    cat.merge(GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json"));
    const std::vector<GlassDispersion> g = catalog_dispersions(cat);
    GlassPreselectionOptions o;
    o.power = 1.0 / kFocal;
    o.max_results = 1000000; // alle lösbaren Paare mit Σ|φ|/|φ| ≤ 10 (Standard)
    o.wavelengths_nm = std::vector<double>(kLines.begin(), kLines.end());
    const std::vector<PreselectedGlasses> all = preselect_glasses(g, o);
    const std::size_t pairs_total = g.size() * (g.size() - 1) / 2;

    // Ein Batch mit einer Variante pro Paar (gleiche Topologie, eigene Brechzahlen)
    std::vector<Candidate> cand;
    std::vector<SystemBatchHost> batches;
    std::vector<double> x0;
    std::optional<FdProblem> first;
    int not_buildable = 0; // Startformen
    int variants = 0;
    for (std::size_t k = 0; k < all.size(); ++k) {
        const PreselectedGlasses& c = all[k];
        const std::string& n1 = g[static_cast<std::size_t>(c.glasses[0])].name;
        const std::string& n2 = g[static_cast<std::size_t>(c.glasses[1])].name;
        const std::vector<double> n{g[static_cast<std::size_t>(c.glasses[0])].n0,
                                    g[static_cast<std::size_t>(c.glasses[1])].n0};
        const ThinMemberShape shape = thin_member_shape(n, c.powers);
        // Startformen: die gewählte Biegung (flachere Lösung von S_I = 0) und, wenn es zwei reelle
        // Lösungen gibt, die andere (Summe der Nullstellen −b/a), damit jedes Paar in beiden
        // Becken startet.
        std::vector<std::vector<double>> starts{shape.curvatures};
        if (shape.spherical_zero && shape.spherical_coefficients[0] != 0.0) {
            const double other =
                -shape.spherical_coefficients[1] / shape.spherical_coefficients[0] -
                shape.curvatures[0];
            std::vector<double> alt = shape.curvatures;
            for (double& x : alt) {
                x += other - shape.curvatures[0];
            }
            starts.push_back(alt);
        }
        Candidate cd;
        cd.glasses = {n1, n2};
        cd.algebraic_rank = static_cast<int>(k) + 1;
        cd.residual = c.residual;
        cd.spectral_residual = c.spectral_residual;
        cd.power_sum = c.power_sum;
        for (std::size_t s = 0; s < starts.size(); ++s) {
            const PairSystem ps = pair_system(cat, n1, n2, starts[s]);
            if (!ps.buildable) {
                ++not_buildable;
                continue;
            }
            FdProblem p = make_fd_problem(
                prepare_system(ps.system, std::filesystem::path(MKO_SOURCE_DIR "/data/systems")));
            cd.runs.push_back(static_cast<int>(batches.size()));
            cd.chosen.push_back(s == 0 ? 1 : 0);
            const auto sq = [](const std::vector<double>& v) {
                return std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
            };
            cd.flat.push_back(sq(starts[s]) <= sq(starts[starts.size() - 1 - s]) ? 1 : 0);
            ++variants;
            x0.insert(x0.end(), p.x0.begin(), p.x0.end());
            batches.push_back(p.base);
            if (!first) {
                first = std::move(p);
            }
        }
        if (!cd.runs.empty()) {
            cand.push_back(cd);
        }
    }
    ASSERT_TRUE(first.has_value());
    ASSERT_FALSE(cand.empty());
    ASSERT_EQ(cand.front().algebraic_rank, 1) << "bester Vorschlag nicht aufbaubar";
    SystemBatchHost base = first->base;
    base.resize(static_cast<int>(batches.size()));
    for (std::size_t v = 0; v < batches.size(); ++v) {
        base.copy_variant_from(batches[v], 0, static_cast<int>(v));
    }

    LmOptions lo;
    lo.max_iterations = kBruteForceCycles;
    lo.jacobian.extra = paraxial_extra(*first);
    lo.jacobian.quadrature.rings = 6;
    const auto t0 = std::chrono::steady_clock::now();
    const LmResult res = optimize_lm_cpu(first->set, base, x0, lo);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ASSERT_EQ(res.runs.size(), batches.size());
    int chosen_wins = 0; // unter den Paaren mit zwei Startformen
    int flat_wins = 0;
    int two_starts = 0;
    for (Candidate& c : cand) {
        for (std::size_t s = 0; s < c.runs.size(); ++s) {
            const LmRunResult& run = res.runs[static_cast<std::size_t>(c.runs[s])];
            if (c.best_run < 0 || run.merit < c.merit) {
                c.best_run = c.runs[s];
                c.merit = run.merit;
                c.best_is_chosen = c.chosen[s] == 1;
                c.best_is_flat = c.flat[s] == 1;
                c.status = static_cast<int>(run.status);
                c.failed = run.failed_rays;
            }
        }
        if (c.runs.size() == 2) {
            ++two_starts;
            chosen_wins += c.best_is_chosen ? 1 : 0;
            flat_wins += c.best_is_flat ? 1 : 0;
        }
    }

    std::vector<std::size_t> order(cand.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return cand[a].merit < cand[b].merit; });
    int raytrace_rank_of_best = 0;
    for (std::size_t r = 0; r < order.size(); ++r) {
        if (cand[order[r]].algebraic_rank == 1) {
            raytrace_rank_of_best = static_cast<int>(r) + 1;
        }
    }
    std::printf("Bruteforce: %zu Gläser, %zu Paare, %zu lösbar (Σ|φ|/φ ≤ 10), %zu optimiert in %d "
                "Startformen (%d nicht aufbaubar), %d Zyklen, %.1f s\n",
                g.size(), pairs_total, all.size(), cand.size(), variants, not_buildable,
                kBruteForceCycles, seconds);
    std::printf("Paare mit zwei Startformen: %d; besser war die von thin_member_shape bei %d, die "
                "flachere bei %d\n",
                two_starts, chosen_wins, flat_wins);
    std::printf("Rang Raytrace | Rang Vorauswahl | Gläser | Merit [Wellen²] | spektraler Rest | "
                "Koeffizient | Σ|φ|/φ\n");
    for (std::size_t r = 0; r < std::min<std::size_t>(10, order.size()); ++r) {
        const Candidate& c = cand[order[r]];
        std::printf("%4zu | %4d | %s + %s | %.4e | %.3e | %.3e | %.2f | %s, Start %s%s\n", r + 1,
                    c.algebraic_rank, c.glasses[0].c_str(), c.glasses[1].c_str(), c.merit,
                    c.spectral_residual, c.residual, c.power_sum,
                    to_string(static_cast<LmStatus>(c.status)),
                    c.best_is_chosen ? "gewählt" : "andere Lösung",
                    c.failed ? " (versagte Strahlen)" : "");
    }
    // Zum Vergleich: Rang nur nach dem Koeffizienten des sekundären Spektrums (ohne Spektrum)
    const auto coef_best =
        std::min_element(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
            return a.residual != b.residual ? a.residual < b.residual : a.power_sum < b.power_sum;
        });
    const auto coef_pos =
        std::find(order.begin(), order.end(), static_cast<std::size_t>(coef_best - cand.begin()));
    std::printf("Rang nach dem Koeffizienten (ohne Spektrum): bestes Paar %s + %s, Raytrace-Rang "
                "%td\n",
                coef_best->glasses[0].c_str(), coef_best->glasses[1].c_str(),
                coef_pos - order.begin() + 1);
    for (std::size_t k = 0; k < std::min<std::size_t>(5, cand.size()); ++k) {
        const auto it = std::find(order.begin(), order.end(), k);
        std::printf("Vorauswahl %zu: %s + %s, Raytrace-Rang %td, Merit %.4e\n", k + 1,
                    cand[k].glasses[0].c_str(), cand[k].glasses[1].c_str(), it - order.begin() + 1,
                    cand[k].merit);
    }
    // Die Startform von thin_member_shape (flachere Lösung) ist meist die bessere
    EXPECT_GT(2 * chosen_wins, two_starts);
    EXPECT_EQ(cand.front().failed, 0);
    EXPECT_GE(raytrace_rank_of_best, 1);
    EXPECT_LE(raytrace_rank_of_best, kGlassPreselectionRaytraceTopRank);
}

} // namespace
} // namespace mko::test
