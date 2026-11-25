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

// Höhe des Mittelpunkts der Referenzsphäre (ADR-0019, 1 Byte, Werte stabil). Nur bei den Arten
// am Bildpunkt (is_image_reference) wählbar.
enum class ReferencePointKind : std::uint8_t {
    // Idealer Bildpunkt M_f = (0, h_f, w) mit h_f = f_soll · tan θ_f (ADR-0017, Punkt 3):
    // Verzeichnung und Querfarbe erscheinen als Kippung der OPD. Standard (bitgleich).
    Ideal = 0,
    // Durchstoßpunkt des realen Hauptstrahls bei λ_ref in der Bildebene, M_f = (0, ȳ_f, w),
    // gemeinsam für alle Wellenlängen des Feldes: Die Verzeichnung verlässt die Wellenfront
    // (eigene Nebenbedingung `distortion`), die Querfarbe bleibt als Kippung der Wellenlängen
    // relativ zu λ_ref. ȳ_f folgt den Variablen (update_reference_height nach jedem apply).
    ChiefRay = 1,
    // Schwerpunkt der Strahlen bei λ_ref in der Bildebene (Pupillenquadratur mit
    // kReferenceCentroidRings × kReferenceCentroidSpokes, Gewichte der Quadratur), gemeinsam für
    // alle Wellenlängen: entfernt bei λ_ref die mittlere Kippung (Lage des Bildes, wie die
    // Bewertung „ohne Kippung“), die Querfarbe bleibt als Kippung relativ zu λ_ref. Folgt den
    // Variablen wie ChiefRay.
    Centroid = 2,
};

// Wird die Höhe des Mittelpunkts aus realen Strahlen gerechnet (ChiefRay, Centroid)?
MKO_HD constexpr bool is_traced_reference_point(ReferencePointKind kind) {
    return kind != ReferencePointKind::Ideal;
}

// Pupillenquadratur des Schwerpunkts (ReferencePointKind::Centroid): Ringe und Speichen der
// halben Pupille (Gauß-Legendre radial, wie die Merit).
inline constexpr int kReferenceCentroidRings = 8;
inline constexpr int kReferenceCentroidSpokes = 8;

// Name in Systemdatei und CLI: "ideal", "chief_ray", "centroid".
constexpr const char* reference_point_name(ReferencePointKind kind) {
    switch (kind) {
    case ReferencePointKind::Ideal:
        return "ideal";
    case ReferencePointKind::ChiefRay:
        return "chief_ray";
    case ReferencePointKind::Centroid:
        return "centroid";
    }
    return "?";
}

} // namespace mko
