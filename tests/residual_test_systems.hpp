#pragma once

// Testsysteme und zufällig gestörte Varianten für die Residuen- und Merit-Tests (CPU und GPU,
// M2-T2, M2-T3).

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

// Stapel aus S = kMaxSurfaces Flächen: Blende, 31 schwach gekrümmte Linsen (Gläser n = 1,5
// und 1,7 abwechselnd), letzte Fläche (Luft auf beiden Seiten) mit Arbeitsabstand; zwei
// Felder (Achse, Feldpunkt).
inline SystemBatchHost many_surface_batch(int variants, std::uint64_t seed) {
    BatchTopology t;
    t.surfaces = kMaxSurfaces;
    t.wavelengths = 2;
    t.fields = 2;
    t.type.assign(static_cast<std::size_t>(t.surfaces), SurfaceKind::Sphere);
    t.type[0] = SurfaceKind::Stop;
    t.pupil_mode = {PupilMode::Axis, PupilMode::Half};
    SystemBatchHost b(t, variants);
    b.wavelength_nm(0) = 486.13;
    b.wavelength_nm(1) = 656.27;
    b.wavelength_weight(0) = b.wavelength_weight(1) = 1.0;
    b.field_weight(0) = b.field_weight(1) = 1.0;
    b.reference_wavelength_nm = 587.56;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int S = t.surfaces;
    for (int v = 0; v < variants; ++v) {
        const double p = v == 0 ? 0.0 : 1.0; // Variante 0 ungestört
        for (int s = 1; s < S; ++s) {
            const bool glass = s % 2 == 1 && s < S - 1; // Medium nach Fläche s
            b.curvature(v, s) = (glass ? 0.0008 : -0.0006) * (1.0 + 0.2 * p * u(rng));
            b.thickness(v, s) = (glass ? 0.8 : 0.4) * (1.0 + 0.2 * p * u(rng));
            const double n = glass ? (s % 4 == 1 ? 1.5 : 1.7) : 1.0;
            for (int l = 0; l < 2; ++l) {
                b.index(v, s, l) = glass ? n + 0.01 * (1 - l) : 1.0;
            }
        }
        for (int l = 0; l < 2; ++l) {
            b.index(v, 0, l) = 1.0;
        }
        b.thickness(v, 0) = 2.0;
        b.thickness(v, S - 1) = 40.0 * (1.0 + 0.1 * p * u(rng));
        b.pupil_radius(v) = 3.0 * (1.0 + 0.3 * p * u(rng));
        b.field_angle(v, 1) = 0.02 * (1.0 + 0.3 * p * u(rng));
        b.field_target(v, 1) = 0.8 * (1.0 + 0.1 * p * u(rng));
    }
    b.validate_values();
    return b;
}

} // namespace mko::test
