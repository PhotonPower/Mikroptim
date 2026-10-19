#pragma once

// Nebenbedingungen der Systembeschreibung (Abschnitt `constraints` von `mikroptim.system/1`,
// docs/dateiformat.md, ab M3-T3). Das Modell hält die Einträge so, wie sie in der Datei stehen
// (fehlende Felder bleiben leer, damit das Schreiben die Form erhält). Die Sollwerte stehen
// nicht hier, sondern in `target` (eine Quelle der Wahrheit); nur reine Grenzwerte ohne
// Gegenstück in `target` (κ_max der Petzval-Bedingung) stehen im Eintrag. Residuen der paraxialen
// Bedingungen: mikroptim/optim/paraxial_residuals.hpp, Definitionen in docs/optimierung.md,
// Abschnitte 2.2 und 2.3. Die L1-Strafe der Geisterkoeffizienten (ghost_penalty, M4-T6) steht
// als weiterer Eintrag hier; ihre Residuen: mikroptim/optim/ghost_penalty.hpp. Die Strafe
// außerhalb der Glas-Hülle (glass_hull, M6-T1): mikroptim/optim/glass_variables.hpp.

#include "mikroptim/model/variables.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

struct SystemDescription;

// Art einer Nebenbedingung. Erweiterbar: mechanische Bedingungen (Durchmesser, Mitten- und
// Randdicke, Einfallswinkel) folgen in M4-T3. Unbekannte Typen lehnt der Leser mit FormatError
// ab.
enum class ConstraintType {
    FocalLength,     // Brennweite bei λ_ref, Soll target.focal_length (nur Gleichung)
    ParfocalLength,  // Abgleichlänge, Soll target.parfocal_length
    WorkingDistance, // Arbeitsabstand = Dicke der letzten Fläche, Soll target.working_distance
    ParaxialFocus,   // Bildebene im paraxialen Brennpunkt bei λ_ref, Soll 0 (nur Gleichung)
    Petzval,         // |κ_P| ≤ κ_max bei λ_ref (M4-T4), Grenze im Eintrag: max_curvature (nur
                     // Obergrenze)
    GhostPenalty,    // geglättete L1-Strafe der Geisterkoeffizienten (M4-T6): Gewicht μ,
                     // Ordnungsgewichte w_k = w₀ κ^k, Glättung ε; keine Sollgröße, kein Modus
    GlassHull,       // Strafe √w · Abstand außerhalb der Hülle der Katalog-Gläser für alle
                     // Glasvariablen (M6-T1): Gewicht w, Sicherheitsabstand nach innen; keine
                     // Sollgröße, kein Modus
};

// Art der Bedingung: Gleichung Ist = Soll oder einseitig Ist ≥ Soll (Untergrenze) bzw.
// Ist ≤ Soll (Obergrenze).
enum class ConstraintMode {
    Equal,
    Min,
    Max,
};

// Standardwerte fehlender Felder
inline constexpr double kDefaultConstraintWeight = 1.0;
inline constexpr int kDefaultParfocalReferenceSurface = 0; // Blende
inline constexpr double kDefaultParfocalOffset = 0.0;      // mm
// L1-Strafe der Geisterkoeffizienten (docs/optimierung.md, Abschnitte 2 und 3)
inline constexpr double kDefaultGhostOrderWeight = 1.0; // w₀
inline constexpr double kDefaultGhostOrderFactor = 2.0; // κ
inline constexpr double kDefaultGhostSmoothing = 1e-6;  // ε [mm]
// Glas-Hülle (docs/optimierung.md, Abschnitt 2.4): Gewicht w [Wellen²] für den Abstand d in
// normierten Glaskoordinaten (eine Einheit = Spannweite des Katalogs): d = 0,01 kostet bei
// w = 10⁴ eine Welle². Sicherheitsabstand nach innen (normiert), Standard 0.
inline constexpr double kDefaultGlassHullWeight = 1e4;
inline constexpr double kDefaultGlassHullMargin = 0.0;

// Ein Eintrag von `constraints.items`, wie er in der Datei steht.
struct ConstraintSpec {
    ConstraintType type = ConstraintType::FocalLength;
    std::optional<ConstraintMode> mode; // fehlt = Equal
    std::optional<double> weight;       // Gewicht w >= 0, fehlt = 1
    // Nur ParfocalLength: Dateifläche, ab deren Scheitel die Abgleichlänge zählt (fehlt = 0),
    // und Abstand der Anlagefläche vor diesem Scheitel [mm] (fehlt = 0).
    std::optional<int> reference_surface;
    std::optional<double> offset;
    std::optional<std::string> comment;
    // Nur Petzval (Pflicht): größter zulässiger Betrag der Petzvalkrümmung κ_max [1/mm], ≥ 0.
    // Einzige Quelle der Grenze (nicht in `target`); ein Mindest-Petzvalradius R_min entspricht
    // κ_max = 1/R_min.
    std::optional<double> max_curvature;
    // Nur GhostPenalty: Geisterplatten, für die der Eintrag gilt ("plate"/"plates", Index in
    // ghost_plates; fehlt = alle), Ordnungsgewichte w_k = order_weight · order_factor^k (w₀,
    // κ, beide > 0) und Glättung ε > 0 [mm, Einheit des Dickenprofils]. Das Strafgewicht μ ist
    // `weight`.
    SurfaceSelection plate_selection = SurfaceSelection::None;
    std::vector<int> plates{};
    std::optional<double> order_weight{};
    std::optional<double> order_factor{};
    std::optional<double> smoothing{};
    // Nur GlassHull: Sicherheitsabstand δ ≥ 0 nach innen in normierten Glaskoordinaten (die
    // Strafe beginnt δ vor dem Rand; nur bei einer dreidimensionalen Hülle). Das Gewicht ist
    // `weight` (Standard kDefaultGlassHullWeight).
    std::optional<double> margin{};

    friend bool operator==(const ConstraintSpec&, const ConstraintSpec&) = default;

    // Fehlender Modus: Equal, bei Petzval Max (einzig erlaubter Modus).
    ConstraintMode mode_or_default() const {
        return mode.value_or(type == ConstraintType::Petzval ? ConstraintMode::Max
                                                             : ConstraintMode::Equal);
    }
    // Fehlendes Gewicht: kDefaultGlassHullWeight bei GlassHull, sonst kDefaultConstraintWeight.
    double weight_or_default() const {
        return weight.value_or(type == ConstraintType::GlassHull ? kDefaultGlassHullWeight
                                                                 : kDefaultConstraintWeight);
    }
    double margin_or_default() const { return margin.value_or(kDefaultGlassHullMargin); }
    int reference_surface_or_default() const {
        return reference_surface.value_or(kDefaultParfocalReferenceSurface);
    }
    double offset_or_default() const { return offset.value_or(kDefaultParfocalOffset); }
    double order_weight_or_default() const {
        return order_weight.value_or(kDefaultGhostOrderWeight);
    }
    double order_factor_or_default() const {
        return order_factor.value_or(kDefaultGhostOrderFactor);
    }
    double smoothing_or_default() const { return smoothing.value_or(kDefaultGhostSmoothing); }
};

// Abschnitt `constraints` der Systemdatei.
struct Constraints {
    std::vector<ConstraintSpec> items;

    friend bool operator==(const Constraints&, const Constraints&) = default;
};

// Prüft `system.constraints` nach den Regeln aus docs/dateiformat.md (Sollwert in `target`
// vorhanden, Modus und Felder zum Typ passend, Gewicht endlich und >= 0, Referenzfläche
// existiert, max_curvature bei petzval vorhanden, endlich und >= 0, jeder Typ höchstens einmal
// außer ghost_penalty (dort jede Geisterplatte höchstens einmal), paraxial_focus nur bei
// kollimierter Quelle, glass_hull ohne Modus mit margin endlich und ≥ 0). Wirft
// FormatError mit JSON-Pfad, z. B. `constraints.items[1].mode`. Wird von validate_system
// aufgerufen; ohne Abschnitt `constraints` passiert nichts.
void check_constraints(const SystemDescription& system);

// Sollwert eines Typs aus `system.target` (Soll 0 bei ParaxialFocus); leer, wenn er fehlt, und
// immer leer bei Petzval (die Grenze steht im Eintrag, siehe constraint_bound), GhostPenalty und
// GlassHull.
std::optional<double> constraint_target(const SystemDescription& system, ConstraintType type);

// Soll- bzw. Grenzwert eines Eintrags: max_curvature bei Petzval, sonst
// constraint_target(system, spec.type); leer, wenn er fehlt (immer bei GhostPenalty und
// GlassHull).
std::optional<double> constraint_bound(const SystemDescription& system, const ConstraintSpec& spec);

// Wahr für die paraxialen Typen (auch Petzval), falsch für GhostPenalty und GlassHull.
bool is_paraxial(ConstraintType type);

// Namen für Meldungen und Dateiformat: "focal_length", "parfocal_length", "working_distance",
// "paraxial_focus", "petzval", "ghost_penalty", "glass_hull" bzw. "equal", "min", "max".
std::string_view to_string(ConstraintType type);
std::string_view to_string(ConstraintMode mode);
// Umkehrungen von to_string; leer bei unbekanntem Namen.
std::optional<ConstraintType> constraint_type_from_string(std::string_view name);
std::optional<ConstraintMode> constraint_mode_from_string(std::string_view name);

} // namespace mko
