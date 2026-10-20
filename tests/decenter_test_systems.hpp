#pragma once

// Testsysteme für Dezentrierung und Verkippung (M8-T3, CPU und GPU): Planplatte als
// Flächentabelle, Batches mit voller Pupille, gestörte Varianten mit zufälligen Transformationen
// und dezentrierte Varianten der Testfälle der adjungierten Jacobi-Matrix (M5-T2).

#include "adjoint_test_systems.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "residual_test_systems.hpp"

#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace mko::test {

inline constexpr double kDecenterDeg = std::numbers::pi / 180.0;

// Planparallele Platte: Blende (Abstand d1), Glasfläche (Brechzahl n, Abstand gap), Austritt in
// Luft (Abstand w zur Bildebene); eine Wellenlänge, Pupillenradius 10 mm.
inline HostSurfaceTable plate_table(double n, double d1, double gap, double w) {
    HostSurfaceTable t;
    t.surfaces = 3;
    t.wavelengths = 1;
    t.type = {SurfaceKind::Stop, SurfaceKind::Plane, SurfaceKind::Plane};
    t.curvature = {0.0, 0.0, 0.0};
    t.thickness = {d1, gap, w};
    t.index = {1.0, n, 1.0};
    t.semi_diameter = {10.0, 10.0, 10.0};
    return t;
}

// Kopie mit derselben Topologie, aber voller Pupille für alle Felder (koaxial): Vergleichsbasis
// für Batches mit Transformation 0 (make_non_coaxial).
inline SystemBatchHost with_full_pupil(const SystemBatchHost& batch) {
    BatchTopology t = batch.topology();
    t.pupil_mode.assign(static_cast<std::size_t>(t.fields), PupilMode::Full);
    SystemBatchHost out(t, batch.variants());
    out.reference_wavelength_nm = batch.reference_wavelength_nm;
    out.source_point = batch.source_point;
    out.source_na = batch.source_na;
    const auto copy = [](std::span<const double> from, std::span<double> to) {
        std::copy(from.begin(), from.end(), to.begin());
    };
    copy(batch.ghost_norm_radius_data(), out.ghost_norm_radius_data());
    copy(batch.wavelength_nm_data(), out.wavelength_nm_data());
    copy(batch.wavelength_weight_data(), out.wavelength_weight_data());
    copy(batch.field_weight_data(), out.field_weight_data());
    copy(batch.curvature_data(), out.curvature_data());
    copy(batch.thickness_data(), out.thickness_data());
    copy(batch.index_data(), out.index_data());
    copy(batch.pupil_radius_data(), out.pupil_radius_data());
    copy(batch.field_angle_data(), out.field_angle_data());
    copy(batch.field_target_data(), out.field_target_data());
    copy(batch.ghost_coeff_data(), out.ghost_coeff_data());
    copy(batch.ghost_index_data(), out.ghost_index_data());
    copy(batch.paraxial_index_data(), out.paraxial_index_data());
    copy(batch.paraxial_ghost_index_data(), out.paraxial_ghost_index_data());
    copy(batch.buchdahl_coeff_data(), out.buchdahl_coeff_data());
    return out;
}

// Zufällige Dezentrierungen (|Δ| <= max_decenter mm) und Verkippungen (|α|, |β| <= max_tilt rad)
// aller Flächen außer der Blende und den Geisterplatten in den Varianten [first, V) eines nicht
// koaxialen Batches; etwa jede fünfte Fläche bleibt unbewegt.
inline void randomize_transforms(SystemBatchHost& batch, std::uint64_t seed,
                                 double max_decenter = 0.05, double max_tilt = 0.5 * kDecenterDeg,
                                 int first = 1) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int v = first; v < batch.variants(); ++v) {
        for (int s = 1; s < batch.surfaces(); ++s) {
            if (batch.type(s) == SurfaceKind::GhostPlate || unit(rng) < 0.2) {
                continue;
            }
            batch.set_surface_transform(v, s, max_decenter * u(rng), max_decenter * u(rng),
                                        max_tilt * u(rng), max_tilt * u(rng));
        }
    }
}

// Gestörte Varianten (perturbed_batch) eines Systems als nicht koaxialer Batch; Variante 0 bleibt
// unbewegt, alle übrigen erhalten zufällige Transformationen.
inline SystemBatchHost decentered_batch(const LoadedSystem& system, int variants,
                                        std::uint64_t seed, double max_decenter = 0.05,
                                        double max_tilt = 0.5 * kDecenterDeg) {
    SystemBatchHost b = make_non_coaxial(perturbed_batch(system, variants, seed));
    randomize_transforms(b, seed + 1, max_decenter, max_tilt);
    return b;
}

// Die Testfälle der adjungierten Jacobi-Matrix (adjoint_cases) mit dezentrierten und verkippten
// Flächen (nicht koaxialer Basisbatch, Variablen für dessen Topologie, x₀ unverändert).
inline std::vector<std::unique_ptr<AdjointCase>> decentered_adjoint_cases() {
    auto cases = adjoint_cases();
    std::uint64_t seed = 83100;
    for (auto& c : cases) {
        SystemBatchHost b = make_non_coaxial(c->p.base);
        randomize_transforms(b, ++seed, 0.05, 0.5 * kDecenterDeg, 0);
        c->name += " (dezentriert)";
        c->p.set = VariableSet(c->p.loaded.system, b.topology());
        c->p.base = std::move(b);
    }
    return cases;
}

} // namespace mko::test
