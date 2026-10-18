#pragma once

// Optimierungsvariablen der Systembeschreibung (Abschnitt `variables` von `mikroptim.system/1`,
// docs/dateiformat.md, ab M3-T1). Das Modell hält die Einträge so, wie sie in der Datei stehen
// (inklusive Kurzformen); resolve_variables löst sie in eine Liste einzelner Variablen auf.
// Die Abbildung auf einen SystemBatch steht in mikroptim/optim/variable_set.hpp.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

struct SystemDescription;

// Art einer Variablen. Erweiterbar: Geisterkoeffizienten folgen in M4-T6, Glasparameter
// (n₀, ν) in M6-T1. Unbekannte Typen lehnt der Leser mit FormatError ab.
enum class VariableType {
    Curvature,       // Krümmung c einer Fläche vom Typ sphere [1/mm]
    Thickness,       // Dicke bzw. Luftabstand t einer Fläche (nicht der letzten) [mm]
    WorkingDistance, // Arbeitsabstand w = Dicke der letzten Fläche [mm]
};

// Form der Flächenauswahl eines Eintrags (bestimmt, wie er geschrieben wird).
enum class SurfaceSelection {
    None, // kein Flächenbezug (working_distance)
    One,  // "surface": i
    List, // "surfaces": [i, j, …]
    All,  // "surfaces": "all"
};

// Ein Eintrag von `variables.items`, wie er in der Datei steht.
struct VariableSpec {
    VariableType type = VariableType::Curvature;
    SurfaceSelection selection = SurfaceSelection::One;
    // Dateiflächen (Index in SystemDescription::surfaces): bei One genau ein Eintrag, bei List
    // mindestens einer, bei All und None leer.
    std::vector<int> surfaces;
    std::optional<double> min;     // untere Schranke, fehlt = offen
    std::optional<double> max;     // obere Schranke, fehlt = offen
    std::optional<double> typical; // typische Größe x_typ > 0, fehlt = Standardwert des Typs
    std::optional<std::string> comment;

    friend bool operator==(const VariableSpec&, const VariableSpec&) = default;
};

// Abschnitt `variables` der Systemdatei.
struct Variables {
    std::vector<VariableSpec> items;

    friend bool operator==(const Variables&, const Variables&) = default;
};

// Standardwerte der typischen Größe x_typ (FD-Schrittweite h = ε^{1/3}·max(|x|, x_typ),
// docs/optimierung.md, Abschnitt 4), wenn ein Eintrag kein `typical` hat.
inline constexpr double kDefaultTypicalCurvature = 0.01;      // 1/mm (R = 100 mm)
inline constexpr double kDefaultTypicalThickness = 1.0;       // mm
inline constexpr double kDefaultTypicalWorkingDistance = 1.0; // mm
double default_typical(VariableType type);

// Eine einzelne, aufgelöste Variable.
struct ResolvedVariable {
    VariableType type = VariableType::Curvature;
    // Dateifläche, deren Wert die Variable ist: Krümmung bzw. Dicke von surfaces[surface];
    // bei WorkingDistance die letzte Fläche.
    int surface = 0;
    double lower = 0.0;   // untere Schranke, -∞ wenn offen
    double upper = 0.0;   // obere Schranke, +∞ wenn offen
    double typical = 0.0; // x_typ > 0
    int item = 0;         // Eintrag in variables.items (für Meldungen)

    friend bool operator==(const ResolvedVariable&, const ResolvedVariable&) = default;
};

// Löst `system.variables` in einzelne Variablen auf und prüft die Regeln aus
// docs/dateiformat.md (Flächen existieren, Typ passt, keine Dopplungen, Schranken geordnet,
// Startwert innerhalb der Schranken). Reihenfolge: Einträge in Dateireihenfolge, innerhalb
// eines Eintrags die Flächen in der angegebenen (bei "all" aufsteigender) Reihenfolge.
// Ohne Abschnitt `variables` ist das Ergebnis leer. Wirft FormatError mit JSON-Pfad, z. B.
// `variables.items[2].surface`. Wird von validate_system aufgerufen.
std::vector<ResolvedVariable> resolve_variables(const SystemDescription& system);

// Wert der Variable in der Systembeschreibung (Krümmung bzw. Dicke der Dateifläche).
double variable_value(const SystemDescription& system, const ResolvedVariable& variable);

// Namen für Meldungen und Dateiformat: "curvature", "thickness", "working_distance".
std::string_view to_string(VariableType type);
// Umkehrung von to_string; leer bei unbekanntem Namen.
std::optional<VariableType> variable_type_from_string(std::string_view name);

} // namespace mko
