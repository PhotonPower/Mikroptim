#pragma once

// Rückkehr vom realen System zu Geisterplatten (M6-T10): Aus einem System ohne Geisterplatten
// (z. B. nach Feinoptimierung und Rückbau am Ende einer Stufe) wird wieder ein Startsystem der
// Synthese mit Geisterplatten-Wörterbüchern (Koeffizienten 0, place_ghost_dictionary). Die Merit
// bleibt bis auf Rundung gleich. Verfahren: docs/optimierung.md, „Rückkehr zu Geisterplatten“.

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/model/system.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace mko {

// Lagen der Wörterbücher (Luftspalte des realen Systems).
enum class GhostReturnLocation {
    Stop,     // Blendenspalt (nach Fläche 0): kollimiertes Bündel vor der ersten Linse
    Aperture, // innerer Luftspalt mit der größten paraxialen Randstrahlhöhe |y|
    Field,    // innerer Luftspalt mit dem größten Verhältnis |ȳ/y| (Feldlinsenlage, M6-T8)
    Front,    // letzter innerer Luftspalt (vor der Frontlinse am Objekt)
};

std::string_view to_string(GhostReturnLocation location);

struct GhostReturnOptions {
    // Gläser des Wörterbuchs, eine Platte pro Glas und Lage (Pflicht, nicht Luft).
    std::vector<MaterialSpec> glasses;
    // Lagen in dieser Reihenfolge; fallen zwei auf denselben Spalt, gilt die erste.
    std::vector<GhostReturnLocation> locations{GhostReturnLocation::Aperture,
                                               GhostReturnLocation::Field};
    int order = 10; // K der Platten (wie data/systems/zielspezifikation_start.json)
    // Neuer Eintrag ghost_penalty, wenn das System keinen hat (Werte wie im Startsystem der
    // Zielspezifikation: μ = 47,2 Wellen²/mm, w₀ = 1, κ = 1, ε = 10⁻⁶ mm). Mit Eintrag decken
    // dessen Einstellungen die neuen Platten ab (place_ghost_dictionary).
    double penalty_weight = 47.2;
    double order_weight = 1.0;
    double order_factor = 1.0;
    double smoothing = 1e-6;
    // Innere Luftspalte (Field, Aperture, Front) brauchen mindestens diese Dicke [mm].
    double min_gap = 0.2;
    // Spalte mit Platten werden fest (Dickenvariablen entfernt): Operator A teilt genau diese
    // Spalte, und eine geerbte untere Schranke wäre dort ein Fehler (docs/optimierung.md,
    // Abschnitt 6); die Lage der Platten (Abstand vom Scheitel) bliebe sonst nicht im Spalt.
    bool fix_plate_gaps = true;
    // Normierungsradius r_g = norm_margin × größte reale Strahlhöhe an der Platte über alle
    // Felder (Feldwinkel × norm_field_scale) und Wellenlängen, Quadratur `quadrature`.
    double norm_margin = 1.02;
    double norm_field_scale = 1.0;
    int rings = 16;
    int spokes = 8;
};

// Eine gesetzte Lage.
struct GhostReturnPosition {
    GhostReturnLocation location = GhostReturnLocation::Aperture;
    int after_surface = 0; // Dateifläche vor dem Spalt
    double distance = 0.0; // Abstand der Platten vom Scheitel (Spaltmitte)
    double norm_radius = 0.0;
    double marginal_height = 0.0; // paraxial |y| (Mittel der Spaltflächen)
    double ratio = 0.0;           // |ȳ/y| (Mittel der Spaltflächen)
    std::vector<int> plates;      // Indizes der Platten im Ergebnis
    bool gap_fixed = false;       // Dickenvariable des Spalts entfernt
};

struct GhostReturnResult {
    SystemDescription system;
    std::vector<GhostReturnPosition> positions;
    std::vector<std::string> notes; // übersprungene Lagen, Doppelungen
    // F wie im LM ohne Glas-Hülle (Strahlresiduen, paraxial, L1-Strafe, mechanisch; CPU,
    // Quadratur rings × spokes), vor und nach der Rückkehr.
    double merit_before = 0.0;
    double merit_after = 0.0;
};

// Setzt die Wörterbücher an die Lagen der Optionen (Spaltmitte), legt bei Bedarf den Eintrag
// ghost_penalty an (vor dem Einsetzen, damit er alle Platten mit "all" abdeckt), entfernt die
// Dickenvariablen der Plattenspalte und setzt die Normierungsradien aus den realen
// Strahlhöhen. Die Spaltwahl rechnet paraxial (compute_aberrations, λ_ref, größter Feldwinkel)
// am realen System. Wirft std::invalid_argument, wenn das System schon Geisterplatten hat, ohne
// Glas, bei ungültigen Optionen oder wenn keine Lage passt; FormatError/GlassError beim Laden.
GhostReturnResult return_to_ghost_plates(const LoadedSystem& real,
                                         const GhostReturnOptions& options);

// Lagen aus einer Liste wie "aperture,field" (stop, aperture, field, front). Wirft
// std::invalid_argument bei unbekannten oder doppelten Namen.
std::vector<GhostReturnLocation> parse_ghost_return_locations(std::string_view text);

// Bericht als JSON-Text (ein Objekt): positions (location, after_surface, distance,
// norm_radius, marginal_height, ratio, plates, gap_fixed), notes, merit_before, merit_after,
// relative_change. Nicht endliche Zahlen als null.
std::string ghost_return_json_text(const GhostReturnResult& result);

} // namespace mko
