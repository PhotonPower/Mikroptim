#pragma once

// Art der Referenzsphäre (ADR-0017, docs/numerik.md, Abschnitt 7). Eigener, leichter Header,
// damit Datenmodell (SystemDescription), Batch und Kernel dieselbe Aufzählung verwenden.

#include "mikroptim/config.hpp"

#include <cstdint>

namespace mko {

// Lage der Referenzsphäre der Merit (1 Byte, Werte stabil). Mittelpunkt ist immer der ideale
// Bildpunkt M_f = (0, h_f, w) in lokalen Koordinaten der letzten Fläche.
enum class ReferenceSphereKind : std::uint8_t {
    // Sphäre durch den Scheitel der letzten Fläche, Radius |M_f| (M1-T7). Standard für
    // Systemdateien ohne Feld `reference_sphere` (bitgleich zu den Ergebnissen vor ADR-0017).
    LastSurface = 0,
    // Radius bis zur Mitte der paraxialen Austrittspupille E = (0, 0, z_AP) bei λ_ref,
    // κ = σ/|M_f − E| beim Aufbau des Batches eingefroren (update_reference_curvature);
    // bildseitig telezentrisch κ = 0.
    ImageExitPupil = 1,
    // Unendlicher Radius (κ = 0): OPD als Funktion der Strahlrichtung (Debye-Näherung).
    ImageInfinite = 2,
};

// Hängt die Sphäre am Bildpunkt (ADR-0017), nicht an der letzten Fläche?
MKO_HD constexpr bool is_image_reference(ReferenceSphereKind kind) {
    return kind != ReferenceSphereKind::LastSurface;
}

// Name in Systemdatei und CLI: "last_surface", "image_exit_pupil", "image_infinite".
constexpr const char* reference_sphere_name(ReferenceSphereKind kind) {
    switch (kind) {
    case ReferenceSphereKind::LastSurface:
        return "last_surface";
    case ReferenceSphereKind::ImageExitPupil:
        return "image_exit_pupil";
    case ReferenceSphereKind::ImageInfinite:
        return "image_infinite";
    }
    return "?";
}

} // namespace mko
