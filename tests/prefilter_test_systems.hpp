#pragma once

// Testsatz der FP32-Vorfilterung (M7-T2, CPU und GPU): stark gestörte Multistart-Varianten,
// sodass ein Teil massiv versagt und die Merit über Größenordnungen streut.

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "residual_test_systems.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {

// Batch mit `variants` Multistart-Varianten des Systems; Variante 0 bleibt ungestört, alle
// anderen werden mit festem Seed gestört: Krümmungen der Sphären ±50 %, Dicken einschließlich
// Arbeitsabstand ±30 %, Pupillenradius Faktor 0,5 … 2, Brechzahlen der Gläser ±0,05,
// Feldwinkel ±50 %, Feldziele ±10 %, Koeffizienten der Geisterplatten ×(1 ± 0,5) plus
// ±2e-4/(k + 1) und deren Gläser ±0,05. Etwa jede siebte Variante ist "extrem" (Pupille
// ×2 … ×5, Krümmungen ×1 … ×4), damit Strahlen massiv versagen. Achsfelder bleiben auf der
// Achse.
inline SystemBatchHost multistart_batch(const LoadedSystem& system, int variants,
                                        std::uint64_t seed) {
    SystemBatchHost b = make_system_batch(system, variants);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const int S = b.surfaces();
    for (int v = 1; v < variants; ++v) {
        const bool extreme = unit(rng) < 0.15;
        for (int s = 1; s < S; ++s) {
            if (b.type(s) == SurfaceKind::Sphere) {
                b.curvature(v, s) *= 1.0 + 0.5 * u(rng);
                if (extreme) {
                    b.curvature(v, s) *= 1.0 + 1.5 * (1.0 + u(rng));
                }
            }
        }
        for (int s = 0; s < S; ++s) {
            b.thickness(v, s) *= 1.0 + 0.3 * u(rng);
        }
        b.pupil_radius(v) *= std::exp(0.693 * u(rng));
        if (extreme) {
            b.pupil_radius(v) *= 3.5 + 1.5 * u(rng);
        }
        for (int s = 0; s < S; ++s) {
            if (b.index(v, s, 0) > 1.0) {
                const double dn = 0.05 * u(rng);
                for (int l = 0; l < b.wavelengths(); ++l) {
                    b.index(v, s, l) += dn;
                }
            }
        }
        for (int f = 0; f < b.fields(); ++f) {
            if (b.field_angle(v, f) != 0.0) {
                b.field_angle(v, f) *= 1.0 + 0.5 * u(rng);
                b.field_target(v, f) *= 1.0 + 0.1 * u(rng);
            }
        }
        for (int g = 0; g < b.ghost_plates(); ++g) {
            for (int k = 0; k < b.ghost_terms(); ++k) {
                b.ghost_coeff(v, g, k) =
                    b.ghost_coeff(v, g, k) * (1.0 + 0.5 * u(rng)) + 2e-4 * u(rng) / (k + 1);
            }
            const double dn = 0.05 * u(rng);
            for (int l = 0; l < b.wavelengths(); ++l) {
                b.ghost_index(v, g, l) += dn;
            }
        }
    }
    b.validate_values();
    return b;
}

struct PrefilterTestSet {
    std::string name;
    SystemBatchHost batch;
};

// Testsatz der Abnahme: Einzellinse, Mehrlinser und Einzellinse mit Geisterplatten, je
// `variants` Multistart-Varianten (verschiedene Seeds).
inline std::vector<PrefilterTestSet> prefilter_test_sets(int variants, std::uint64_t seed) {
    std::vector<PrefilterTestSet> out;
    out.push_back({"einzellinse_bk7", multistart_batch(load_residual_system("einzellinse_bk7.json"),
                                                       variants, seed + 1)});
    out.push_back(
        {"testsystem_mehrlinser",
         multistart_batch(load_residual_system("testsystem_mehrlinser.json"), variants, seed + 2)});
    out.push_back({"einzellinse_geisterplatte",
                   multistart_batch(load_residual_system("einzellinse_geisterplatte.json"),
                                    variants, seed + 3)});
    return out;
}

// Kriterien der Abnahme, abgeleitet aus der FP64-Wahrheit eines Testsatzes: Schwellen an
// Quantilen der FP64-Werte, damit viele Varianten nahe an einer Schwelle liegen.
struct NamedCriteria {
    std::string name;
    PrefilterCriteria criteria;
};

// Quantil q der Werte (sortiert, Index floor(q · (n − 1))).
inline double quantile(std::vector<double> x, double q) {
    if (x.empty()) {
        return 0.0;
    }
    std::sort(x.begin(), x.end());
    return x[static_cast<std::size_t>(q * static_cast<double>(x.size() - 1))];
}

inline std::vector<NamedCriteria> acceptance_criteria(const MeritResult& fp64) {
    std::vector<double> merit;
    std::vector<double> failed;
    for (int v = 0; v < fp64.variants; ++v) {
        const auto i = static_cast<std::size_t>(v);
        failed.push_back(static_cast<double>(fp64.failed[i]) /
                         static_cast<double>(fp64.rays_per_variant));
        if (fp64.failed[i] < fp64.rays_per_variant) {
            merit.push_back(fp64.merit[i]);
        }
    }
    std::vector<NamedCriteria> out;
    const auto add = [&out](std::string name, double fraction, double max_merit, double factor) {
        PrefilterCriteria c;
        c.max_failed_fraction = fraction;
        c.max_merit = max_merit;
        c.max_merit_factor = factor;
        out.push_back({std::move(name), c});
    };
    const double inf = std::numeric_limits<double>::infinity();
    add("Versagen > 1 %", 0.01, inf, inf);
    add("Versagen > 10 %", 0.1, inf, inf);
    add("Versagen > 50 %", 0.5, inf, inf);
    // Schwelle genau an einem Quantil des FP64-Versagensanteils
    add("Versagen > Quantil 0,9", quantile(failed, 0.9), inf, inf);
    add("Merit > Quantil 0,25", 1.0, quantile(merit, 0.25), inf);
    add("Merit > Quantil 0,5", 1.0, quantile(merit, 0.5), inf);
    add("Merit > Quantil 0,9", 1.0, quantile(merit, 0.9), inf);
    add("Merit > 2 x beste", 0.5, inf, 2.0);
    add("Merit > 10 x beste", 0.5, inf, 10.0);
    add("Merit > 100 x beste", 0.5, inf, 100.0);
    add("Merit > 10^4 x beste", 0.5, inf, 1e4);
    add("kombiniert", 0.2, quantile(merit, 0.75), 100.0);
    return out;
}

// Vergleich FP32-Entscheidung gegen FP64-Wahrheit.
struct DecisionComparison {
    int variants = 0;
    int rejected64 = 0;      // von FP64 aussortiert
    int rejected32 = 0;      // von FP32 aussortiert
    int false_positive = 0;  // FP32 sortiert aus, FP64 behält (verboten)
    int reason_mismatch = 0; // FP32 sortiert aus einem Grund aus, den FP64 nicht hat (verboten)
    int false_negative = 0;  // FP32 behält, FP64 sortiert aus (erlaubt)
};

inline DecisionComparison compare_decisions(const std::vector<std::uint8_t>& reject32,
                                            const std::vector<std::uint8_t>& reject64) {
    DecisionComparison c;
    c.variants = static_cast<int>(reject32.size());
    for (std::size_t i = 0; i < reject32.size(); ++i) {
        c.rejected32 += reject32[i] != 0 ? 1 : 0;
        c.rejected64 += reject64[i] != 0 ? 1 : 0;
        c.false_positive += reject32[i] != 0 && reject64[i] == 0 ? 1 : 0;
        c.reason_mismatch += (reject32[i] & ~reject64[i]) != 0 ? 1 : 0;
        c.false_negative += reject32[i] == 0 && reject64[i] != 0 ? 1 : 0;
    }
    return c;
}

} // namespace mko::test
