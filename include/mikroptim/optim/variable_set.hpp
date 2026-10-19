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
    // Arbeitsabstand → S − 1; Geisterkoeffizient → Fläche der Geisterplatte (ghost_surface).
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

// Feste Liste von P Variablen für Batches einer Topologie. Komponente k des Vektors x ist die
// Variable k in der Reihenfolge von resolve_variables.
class VariableSet {
public:
    VariableSet() = default;
    // Aus der Systembeschreibung (Abschnitt `variables`, geprüft mit resolve_variables) für
    // Batches mit `topology` (z. B. topology_of(system, index)). Wirft FormatError bei
    // ungültigem Abschnitt, std::invalid_argument, wenn Datei und Topologie nicht zusammenpassen.
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
    // und x − offset exakt ist); Geisterkoeffizient ghost_coeff[v][g][k − 1].
    std::vector<double> extract(const SystemBatchHost& batch, int v) const;
    void extract(const SystemBatchHost& batch, int v, std::span<double> x) const;
    // Schreibt x in Variante v; alle übrigen Felder bleiben unverändert. Bei einem Spalt mit
    // Geisterplatten nur das letzte Teilstück: fl(x − offset), derselbe Ausdruck wie in
    // make_surface_table, die Plattenabstände bleiben fest. Brechzahlen ändern sich nicht
    // (Glasvariablen erst M6). Wirft std::invalid_argument bei falscher Länge, nicht
    // endlichen Werten oder anderer Topologie, std::out_of_range bei ungültigem v.
    void apply(SystemBatchHost& batch, int v, std::span<const double> x) const;

    // Dasselbe für die Systembeschreibung (Dateiflächen, Koeffizienten der Geisterplatten), z. B.
    // zum Schreiben eines optimierten Systems (M3-T5). Die Beschreibung muss zu der passen, aus
    // der die Menge gebaut wurde (gleiche Flächenzahl und -typen, gleiche Zahl und Ordnungen
    // der Geisterplatten), sonst std::invalid_argument.
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
    std::vector<Variable> variables_;
    std::vector<double> lower_;
    std::vector<double> upper_;
    std::vector<double> typical_;
};

} // namespace mko
