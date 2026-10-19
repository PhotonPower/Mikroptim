#pragma once

// Abbildung Variablenvektor x ↔ SystemBatch (Host, docs/optimierung.md, Abschnitt
// „Variablenabbildung“). Die Variablen kommen aus dem Abschnitt `variables` der Systemdatei
// (resolve_variables); ihre Flächenindizes beziehen sich auf die Dateiflächen und werden hier
// einmal über die SurfaceMap (make_surface_map) auf Tabellenflächen des Batches abgebildet.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace mko {

// Eine Variable mit ihrer Lage im Batch.
struct Variable {
    VariableType type = VariableType::Curvature;
    // Dateifläche (SystemDescription::surfaces[i]); bei Geisterkoeffizienten after_surface der
    // Platte (nur zur Information).
    int file_surface = 0;
    // Tabellenfläche im Batch: Krümmung → die Dateifläche selbst; Dicke → das letzte Teilstück
    // des Spalts (vor der nächsten Dateifläche, also nach der letzten Geisterplatte im Spalt);
    // Arbeitsabstand → S − 1; Geisterkoeffizient → Fläche der Geisterplatte (ghost_surface);
    // Glas → die Dateifläche selbst (Medium danach; Geisterplatten liegen nur in Luftspalten).
    int batch_surface = 0;
    // Nur Dicke eines Spalts mit Geisterplatten (split): Abstand `distance` der letzten Platte
    // vom Scheitel der Dateifläche. Batch-Element = x − offset, x = offset + Batch-Element.
    bool split = false;
    double offset = 0.0;
    double lower = 0.0;   // untere Schranke, -∞ wenn offen (bei split mindestens offset)
    double upper = 0.0;   // obere Schranke, +∞ wenn offen
    double typical = 0.0; // x_typ > 0
    std::string name;     // für Berichte: "c[3]", "t[2]", "w", "a[1,2]" (Platte 1, Ordnung 2)
    // Nur Geisterkoeffizient: Platte g und Ordnung k (1 … K), Batch-Element ghost_coeff[v][g][k−1].
    int ghost_plate = -1;
    int ghost_order = 0;
    // Nur Glas (M6-T1): Komponente 0 = n₀, 1 = ν₁, 2 = ν₂, Batch-Element
    // buchdahl_coeff[v][batch_surface][glass_component]; sonst −1.
    int glass_component = -1;

    friend bool operator==(const Variable&, const Variable&) = default;
};

// Abbildung Dateiflächen ↔ Tabellenflächen (make_surface_map) für Batches mit `topology`.
// Einzige Stelle, an der VariableSet den Indexversatz durch Geisterplatten berücksichtigt.
// Wirft std::invalid_argument, wenn Flächenzahl, Zahl der Geisterplatten oder ein Flächentyp
// der Topologie nicht zur Systembeschreibung passen.
SurfaceMap checked_surface_map(const SystemDescription& system, const BatchTopology& topology);

// Name einer Variable für Berichte, z. B. "c[3]" (Krümmung der Dateifläche 3), "t[2]", "w".
std::string variable_name(VariableType type, int file_surface);
// Name eines Geisterkoeffizienten: "a[g,k]" (Platte g, Ordnung k).
std::string ghost_variable_name(int plate, int order);
// Name einer Glaskomponente: "n0[i]", "nu1[i]", "nu2[i]" (Medium nach Dateifläche i).
std::string glass_variable_name(int file_surface, int component);

// Feste Liste von P Variablen für Batches einer Topologie. Komponente k des Vektors x ist die
// Variable k in der Reihenfolge von resolve_variables.
class VariableSet {
public:
    VariableSet() = default;
    // Aus der Systembeschreibung (Abschnitt `variables`, geprüft mit resolve_variables) für
    // Batches mit `topology` (z. B. topology_of(system, index)). Glasvariablen verlangen ein
    // Buchdahl-Material (vorher convert_glass_variables, mikroptim/optim/glass_variables.hpp).
    // Wirft FormatError bei ungültigem Abschnitt, std::invalid_argument, wenn Datei und Topologie
    // nicht zusammenpassen oder eine Glasvariable kein Buchdahl-Material hat.
    VariableSet(const SystemDescription& system, const BatchTopology& topology);

    int size() const { return static_cast<int>(variables_.size()); }
    bool empty() const { return variables_.empty(); }
    const Variable& operator[](int k) const;
    std::span<const Variable> variables() const { return variables_; }
    const BatchTopology& topology() const { return topology_; }

    // Schranken und typische Größen als Vektoren der Länge P (±∞ für offene Schranken).
    std::span<const double> lower() const { return lower_; }
    std::span<const double> upper() const { return upper_; }
    std::span<const double> typical() const { return typical_; }

    // x aus Variante v des Batches: Krümmung bzw. Dicke, bei einem Spalt mit Geisterplatten
    // fl(offset + letztes Teilstück) (gleich der Dateidicke, wenn der Batch aus der Datei stammt
    // und x − offset exakt ist); Geisterkoeffizient ghost_coeff[v][g][k − 1]; Glas
    // buchdahl_coeff[v][s][c] (nicht aus den Brechzahlen, die allein nicht eindeutig sind).
    std::vector<double> extract(const SystemBatchHost& batch, int v) const;
    void extract(const SystemBatchHost& batch, int v, std::span<double> x) const;
    // Schreibt x in Variante v; alle übrigen Felder bleiben unverändert. Bei einem Spalt mit
    // Geisterplatten nur das letzte Teilstück: fl(x − offset), derselbe Ausdruck wie in
    // make_surface_table, die Plattenabstände bleiben fest. Glas: buchdahl_coeff[v][s][c] und
    // danach die Brechzahlen des Mediums neu, index[v][s][l] bei allen Wellenlängen und
    // paraxial_index[v][s] bei λ_ref, mit buchdahl_index(n₀, ν₁, ν₂, ν₃, λ) (derselbe Ausdruck
    // wie material_index; ν₃ = buchdahl_coeff[v][s][3] bleibt). Wirft std::invalid_argument bei
    // falscher Länge, nicht endlichen Werten, anderer Topologie oder (mit Glasvariablen) einer
    // Wellenlänge, bei der die Buchdahl-Koordinate nicht definiert ist (≤ 146,074 nm),
    // std::out_of_range bei ungültigem v.
    void apply(SystemBatchHost& batch, int v, std::span<const double> x) const;

    // Dasselbe für die Systembeschreibung (Dateiflächen, Koeffizienten der Geisterplatten,
    // Glas: n0 und nu[0], nu[1] des Buchdahl-Materials, nu[2] = ν₃ bleibt), z. B. zum Schreiben
    // eines optimierten Systems (M3-T5). Die Beschreibung muss zu der passen, aus der die Menge
    // gebaut wurde (gleiche Flächenzahl und -typen, gleiche Zahl und Ordnungen der
    // Geisterplatten, Buchdahl-Material bei jeder Glasvariable), sonst std::invalid_argument.
    std::vector<double> extract(const SystemDescription& system) const;
    void apply_to_description(SystemDescription& system, std::span<const double> x) const;

    // true, wenn lower_k ≤ x_k ≤ upper_k für alle k (verlangt Länge P).
    bool feasible(std::span<const double> x) const;
    // Projektion auf die Schranken: x_k ← min(max(x_k, lower_k), upper_k).
    void clamp_to_bounds(std::span<double> x) const;

private:
    void check_size(std::size_t n, const char* what) const;
    void check_batch(const SystemBatchHost& batch) const;
    void check_description(const SystemDescription& system) const;

    BatchTopology topology_;
    int file_surfaces_ = 0;
    std::vector<SurfaceType> file_types_;
    std::vector<int> ghost_orders_; // K jeder Geisterplatte der Datei
    // Glaselemente (je einmal, Reihenfolge des ersten Auftretens): Dateifläche und Tabellenfläche
    std::vector<int> glass_file_surfaces_;
    std::vector<int> glass_batch_surfaces_;
    std::vector<Variable> variables_;
    std::vector<double> lower_;
    std::vector<double> upper_;
    std::vector<double> typical_;
};

} // namespace mko
