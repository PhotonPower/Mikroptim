#pragma once

// Testsysteme und zufällig gestörte Varianten für die Residuen-Tests (CPU und GPU, M2-T2).

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <string>

namespace mko::test {

inline const std::string kResidualSystems = MKO_SOURCE_DIR "/data/systems/";

inline LoadedSystem load_residual_system(const std::string& file) {
    return load_system(kResidualSystems + file);
}

// Batch mit `variants` Varianten des Systems; Variante 0 bleibt ungestört, alle anderen werden
// mit festem Seed gestört: Krümmungen der Sphären (±30 %), Dicken (±30 %, Arbeitsabstand
// ±10 %), Pupillenradius (Faktor 0,5 … 2), Brechzahlen der Gläser (±0,05, gleich für alle
// Wellenlängen), Feldwinkel und Feldziele der Feldpunkte (±50 % bzw. ±10 %). Etwa jede
// siebte Variante ist "extrem" (Pupille ×2 … ×5, Krümmungen ×1 … ×4), damit Strahlen versagen
// (Fläche verfehlt, Totalreflexion, rückwärts). Achsfelder bleiben auf der Achse.
inline SystemBatchHost perturbed_batch(const LoadedSystem& system, int variants,
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
                b.curvature(v, s) *= 1.0 + 0.3 * u(rng);
                if (extreme) {
                    b.curvature(v, s) *= 1.0 + 1.5 * (1.0 + u(rng));
                }
            }
        }
        for (int s = 0; s < S - 1; ++s) {
            b.thickness(v, s) *= 1.0 + 0.3 * u(rng);
        }
        b.thickness(v, S - 1) *= 1.0 + 0.1 * u(rng);
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
    }
    return b;
}

} // namespace mko::test
