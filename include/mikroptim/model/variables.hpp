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

// Art einer Variablen. Unbekannte Typen lehnt der Leser mit FormatError ab.
enum class VariableType {
    Curvature,        // Krümmung c einer Fläche vom Typ sphere [1/mm]
    Thickness,        // Dicke bzw. Luftabstand t einer Fläche (nicht der letzten) [mm]
    WorkingDistance,  // Arbeitsabstand w = Dicke der letzten Fläche [mm]
    GhostCoefficient, // Koeffizient a_k des Dickenprofils einer Geisterplatte [mm] (M4-T6)
    Glass,            // Glas des Mediums nach einer Fläche: Buchdahl-Koordinaten (n₀, ν₁, ν₂),
                      // drei Komponenten pro Fläche (M6-T1)
};

// Form einer Auswahl in einem Eintrag (bestimmt, wie er geschrieben wird). Gilt für Flächen
// ("surface"/"surfaces"), Geisterplatten ("plate"/"plates") und Ordnungen ("order"/"orders").
enum class SurfaceSelection {
    None, // Feld fehlt (working_distance ohne Fläche; bei Ordnungen: alle Ordnungen)
    One,  // "surface": i bzw. "plate": g, "order": k
    List, // "surfaces": [i, j, …] bzw. "plates", "orders"
    All,  // "surfaces": "all" bzw. "plates": "all", "orders": "all"
};

// Ein Eintrag von `variables.items`, wie er in der Datei steht.
struct VariableSpec {
    VariableType type = VariableType::Curvature;
    SurfaceSelection selection = SurfaceSelection::One;
    // Dateiflächen (Index in SystemDescription::surfaces): bei One genau ein Eintrag, bei List
    // mindestens einer, bei All und None leer. Bei GhostCoefficient immer None.
    std::vector<int> surfaces;
    // Nur GhostCoefficient: Geisterplatten (Index in SystemDescription::ghost_plates) und
    // Ordnungen k (1 … K der Platte, wie a_k). Auswahl wie bei den Flächen; fehlen die
    // Ordnungen (None), gelten alle Ordnungen jeder gewählten Platte.
    SurfaceSelection plate_selection = SurfaceSelection::None;
    std::vector<int> plates{};
    SurfaceSelection order_selection = SurfaceSelection::None;
    std::vector<int> orders{};
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
// Geisterkoeffizienten: klein gegen die Glättung ε der L1-Strafe (Standard 10⁻⁶ mm), damit die
// FD-Schrittweite bei a = 0 (h = 6·10⁻¹⁰ mm) den glatten Bereich der Strafe auflöst
// (docs/optimierung.md, Abschnitt 3).
inline constexpr double kDefaultTypicalGhostCoefficient = 1e-4; // mm
// Glas (M6-T1): typische Größen der Komponenten n₀, ν₁, ν₂ (dimensionslos). Katalog-Gläser
// haben n₀ ≈ 1,4 … 2,0, ν₁ ≈ −0,02 … −0,25, ν₂ ≈ 0,001 … 0,05; x_typ zählt nur, wenn |x| kleiner
// ist (z. B. ν₂ nahe 0), und hält die FD-Schrittweite dann über dem Rundungsrauschen.
inline constexpr double kDefaultTypicalGlassIndex = 1.0; // n₀
inline constexpr double kDefaultTypicalGlassNu1 = 0.01;  // ν₁
inline constexpr double kDefaultTypicalGlassNu2 = 0.001; // ν₂
// Komponenten einer Glasvariable
inline constexpr int kGlassComponents = 3;
// Standardwert von x_typ; bei Glass für die Komponente `component` (0 = n₀, 1 = ν₁, 2 = ν₂).
double default_typical(VariableType type, int component = 0);

// Eine einzelne, aufgelöste Variable.
struct ResolvedVariable {
    VariableType type = VariableType::Curvature;
    // Dateifläche, deren Wert die Variable ist: Krümmung bzw. Dicke von surfaces[surface];
    // bei WorkingDistance die letzte Fläche; bei GhostCoefficient after_surface der Platte; bei
    // Glass die Fläche, nach der das Medium liegt (surfaces[surface].material).
    int surface = 0;
    double lower = 0.0;   // untere Schranke, -∞ wenn offen
    double upper = 0.0;   // obere Schranke, +∞ wenn offen
    double typical = 0.0; // x_typ > 0
    int item = 0;         // Eintrag in variables.items (für Meldungen)
    // Nur GhostCoefficient: Geisterplatte g und Ordnung k (1 … K); Wert
    // ghost_plates[plate].coefficients[order − 1]. Sonst −1 bzw. 0.
    int plate = -1;
    int order = 0;
    // Nur Glass: Komponente 0 = n₀, 1 = ν₁, 2 = ν₂ der Buchdahl-Koordinaten des Mediums nach
    // `surface`; sonst 0. Jede Glasfläche ergibt drei aufeinanderfolgende Variablen.
    int component = 0;

    friend bool operator==(const ResolvedVariable&, const ResolvedVariable&) = default;
};

// Löst `system.variables` in einzelne Variablen auf und prüft die Regeln aus
// docs/dateiformat.md (Flächen existieren, Typ passt, keine Dopplungen, Schranken geordnet,
// Startwert innerhalb der Schranken). Reihenfolge: Einträge in Dateireihenfolge, innerhalb
// eines Eintrags die Flächen in der angegebenen (bei "all" aufsteigender) Reihenfolge.
// Ohne Abschnitt `variables` ist das Ergebnis leer. Wirft FormatError mit JSON-Pfad, z. B.
// `variables.items[2].surface`. Wird von validate_system aufgerufen.
std::vector<ResolvedVariable> resolve_variables(const SystemDescription& system);

// Wert der Variable in der Systembeschreibung (Krümmung bzw. Dicke der Dateifläche,
// Koeffizient der Geisterplatte; Glas: n₀ bzw. ν₁, ν₂ eines Buchdahl-Materials, n bzw. 0 bei
// einem konstanten Material). Wirft std::invalid_argument bei einer Glasvariable auf einem
// Katalogglas (erst umwandeln, convert_glass_variables in mikroptim/optim/glass_variables.hpp).
double variable_value(const SystemDescription& system, const ResolvedVariable& variable);

// Namen für Meldungen und Dateiformat: "curvature", "thickness", "working_distance",
// "ghost_coefficient", "glass".
std::string_view to_string(VariableType type);
// Umkehrung von to_string; leer bei unbekanntem Namen.
std::optional<VariableType> variable_type_from_string(std::string_view name);

} // namespace mko
