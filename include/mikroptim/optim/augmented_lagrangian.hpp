#pragma once

// Augmented Lagrangian für einseitige Bedingungen (M11-T2, Host): Verschiebung θ ≥ 0 pro
// Residuum r = √w · max(0, g + θ) und Fortschreibung θ ← min(max(0, θ + g*), θ_max) nach jedem
// LM-Lauf. Am Fixpunkt ist g* = 0 bei aktiver Bedingung, also exakte Einhaltung bei festem,
// moderatem Gewicht; der Lagrange-Multiplikator ist λ = 2wθ. Formeln, Schranken, Abbruch und
// Determinismus: docs/optimierung.md, Abschnitt 2.6. Die Verschiebungen stehen in
// ConstraintSpec::shift (Zustand des Optimierers, nicht Teil des Dateiformats).

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/constraints.hpp"

#include <vector>

namespace mko {

struct SystemDescription;

// Standardschranke der Verschiebung in Residueneinheiten: √w · θ ≤ kDefaultMaxShift, also
// höchstens kDefaultMaxShift² Wellen² Beitrag der Verschiebung zu F.
inline constexpr double kDefaultMaxShift = 1e3;
// Standard des Abbruchs der äußeren Schleife: max √w |Δθ| ≤ kDefaultShiftTolerance
// (Residueneinheiten, also Wellen bei λ_ref).
inline constexpr double kDefaultShiftTolerance = 1e-6;

// Einstellungen der äußeren Schleife.
struct AugmentedLagrangianOptions {
    int rounds = 0;                            // höchstens so viele LM-Läufe; 0 = aus (Standard)
    double tolerance = kDefaultShiftTolerance; // Abbruch bei max √w |Δθ| ≤ tolerance
    double max_shift = kDefaultMaxShift;       // Schranke √w · θ ≤ max_shift
};

// Fortschreibung einer Verschiebung: θ' = min(max(0, θ + g), max_shift/√w) bei w > 0, sonst 0.
// Nach Nocedal/Wright (17.4) mit λ = 2wθ. Elementweise und ohne Reduktion, also deterministisch.
double update_shift(double theta, double violation, double weight, double max_shift);

// Eine einseitige Bedingung am Punkt: Eintrag `item` von constraints.items, Residuum `index`
// innerhalb des Eintrags (paraxial 0, mechanisch die k-te Fläche aus constraint_surfaces).
struct ShiftEntry {
    int item = -1;
    int index = 0;
    ConstraintType type = ConstraintType::FocalLength;
    int file_surface = -1;  // Dateifläche (mechanisch), sonst −1
    double weight = 0.0;    // w des Residuums
    double violation = 0.0; // g am Punkt (> 0: verletzt; Einheit der Bedingung)
    double shift = 0.0;     // θ vor der Fortschreibung
    double new_shift = 0.0; // θ danach
};

struct ShiftUpdate {
    std::vector<ShiftEntry> entries; // in der Reihenfolge der Einträge, dann der Residuen
    double max_change = 0.0;         // max √w |θ' − θ| (Residueneinheiten; 0 ohne Einträge)
    double max_violation = 0.0;      // max √w · max(0, g) (Restverletzung in Residueneinheiten)
};

// Hat die Beschreibung Einträge, die eine Verschiebung tragen können (supports_shift)?
bool has_shiftable_constraints(const SystemDescription& system);

// Verletzungen aller einseitigen paraxialen und mechanischen Bedingungen der Beschreibung in
// Variante v des Batches (aus `system` gebaut, Werte am Punkt) und die fortgeschriebenen
// Verschiebungen (new_shift), ohne `system` zu ändern. Die Extrema der mechanischen Bedingungen
// kommen aus surface_extrema_cpu mit `quadrature` und der Glättung des Abschnitts (dieselben wie
// im LM). Gleichungen, ghost_penalty und glass_hull fehlen. Wirft wie make_paraxial_constraints,
// make_mechanical_constraints und die Prüfungen gegen die Topologie.
ShiftUpdate shift_update(const SystemDescription& system, const SystemBatchHost& batch, int v,
                         const ResidualSettings& quadrature, double max_shift = kDefaultMaxShift);

// Schreibt die Verschiebungen new_shift von `update` nach system.constraints->items[*].shift
// (Länge 1 bzw. Zahl der Flächen des Eintrags). Wirft std::invalid_argument, wenn ein Eintrag
// nicht passt.
void apply_shift_update(SystemDescription& system, const ShiftUpdate& update);

// Setzt alle Verschiebungen zurück (leere shift-Felder).
void clear_shifts(SystemDescription& system);

} // namespace mko
