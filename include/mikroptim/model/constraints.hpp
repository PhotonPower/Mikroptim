#pragma once

// Nebenbedingungen der Systembeschreibung (Abschnitt `constraints` von `mikroptim.system/1`,
// docs/dateiformat.md, ab M3-T3). Das Modell hält die Einträge so, wie sie in der Datei stehen
// (fehlende Felder bleiben leer, damit das Schreiben die Form erhält). Die Sollwerte stehen
// nicht hier, sondern in `target` (eine Quelle der Wahrheit). Residuen der paraxialen
// Bedingungen: mikroptim/optim/paraxial_residuals.hpp, Definitionen in docs/optimierung.md,
// Abschnitt 2.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

struct SystemDescription;

// Art einer Nebenbedingung. Erweiterbar: mechanische Bedingungen (Durchmesser, Mitten- und
// Randdicke, Einfallswinkel) folgen in M4-T3, die Petzval-Bedingung in M4-T4. Unbekannte Typen
// lehnt der Leser mit FormatError ab.
enum class ConstraintType {
    FocalLength,     // Brennweite bei λ_ref, Soll target.focal_length (nur Gleichung)
    ParfocalLength,  // Abgleichlänge, Soll target.parfocal_length
    WorkingDistance, // Arbeitsabstand = Dicke der letzten Fläche, Soll target.working_distance
    ParaxialFocus,   // Bildebene im paraxialen Brennpunkt bei λ_ref, Soll 0 (nur Gleichung)
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

    friend bool operator==(const ConstraintSpec&, const ConstraintSpec&) = default;

    ConstraintMode mode_or_default() const { return mode.value_or(ConstraintMode::Equal); }
    double weight_or_default() const { return weight.value_or(kDefaultConstraintWeight); }
    int reference_surface_or_default() const {
        return reference_surface.value_or(kDefaultParfocalReferenceSurface);
    }
    double offset_or_default() const { return offset.value_or(kDefaultParfocalOffset); }
};

// Abschnitt `constraints` der Systemdatei.
struct Constraints {
    std::vector<ConstraintSpec> items;

    friend bool operator==(const Constraints&, const Constraints&) = default;
};

// Prüft `system.constraints` nach den Regeln aus docs/dateiformat.md (Sollwert in `target`
// vorhanden, Modus und Felder zum Typ passend, Gewicht endlich und >= 0, Referenzfläche
// existiert, jeder Typ höchstens einmal, paraxial_focus nur bei kollimierter Quelle). Wirft
// FormatError mit JSON-Pfad, z. B. `constraints.items[1].mode`. Wird von validate_system
// aufgerufen; ohne Abschnitt `constraints` passiert nichts.
void check_constraints(const SystemDescription& system);

// Sollwert eines Typs aus `system.target` (Soll 0 bei ParaxialFocus); leer, wenn er fehlt.
std::optional<double> constraint_target(const SystemDescription& system, ConstraintType type);

// Wahr für die paraxialen Typen (alle bisher definierten; M4-T3/M4-T4 kommen hinzu).
bool is_paraxial(ConstraintType type);

// Namen für Meldungen und Dateiformat: "focal_length", "parfocal_length", "working_distance",
// "paraxial_focus" bzw. "equal", "min", "max".
std::string_view to_string(ConstraintType type);
std::string_view to_string(ConstraintMode mode);
// Umkehrungen von to_string; leer bei unbekanntem Namen.
std::optional<ConstraintType> constraint_type_from_string(std::string_view name);
std::optional<ConstraintMode> constraint_mode_from_string(std::string_view name);

} // namespace mko
