#pragma once

// Glas-Einrasten (M6-T5, Host): ersetzt die kontinuierlichen Glasvariablen (n₀, ν₁, ν₂) eines
// optimierten Systems durch Katalog-Gläser (docs/optimierung.md, Abschnitt 9).
//
// Ablauf (snap_glasses):
//  1. Glaselemente = Flächen mit Variable `glass` (Reihenfolge der Variablen); Startgläser
//     werden wie in der Optimierung in Buchdahl-Materialien umgewandelt (convert_glass_variables;
//     ein bereits optimiertes System hat sie schon).
//  2. Pro Element die k nächsten Katalog-Gläser im normierten (n₀, ν₁, ν₂)-Raum der Glas-Hülle
//     (nearest_glasses, Normierung GlassHull::normalize, Abstand euklidisch mit optionalen
//     Gewichten pro Koordinate).
//  3. Kombinationen: kartesisches Produkt der Kandidaten, bei mehr als max_combinations die
//     max_combinations Kombinationen mit der kleinsten Summe der quadrierten Abstände
//     (glass_combinations, Strahlsuche über die Elemente).
//  4. Jede Kombination wird ein Lauf des gebündelten LM: Glas fest auf das Katalogglas
//     (Material = Katalogname, Brechzahlen aus der Sellmeier-Formel des Katalogs), Glasvariablen
//     entfernt, übrige Variablen ab dem kontinuierlichen Ergebnis reoptimiert. Alle Läufe haben
//     dieselbe Topologie, die Vorlage hat eine Variante pro Lauf (eigene Brechzahlen).
//  5. Bester Lauf nach select_best_run (dieselbe Regel wie der Multistart von
//     `mikroptim optimize`); Ergebnis als Systembeschreibung mit Katalognamen.

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace mko {

// Ein Kandidat für ein Glaselement.
struct GlassCandidate {
    std::string name;      // Katalogname
    int hull_index = -1;   // Index in GlassHull::points()
    Point3 coordinates{};  // (n₀, ν₁, ν₂) des Katalogglases (Buchdahl-Fit der Hülle)
    double distance = 0.0; // Abstand zum Element im normierten Raum (siehe glass_distance)
};

// Gewichte der normierten Koordinaten (u_n₀, u_ν₁, u_ν₂) im Abstand; Standard alle 1.
using GlassDistanceWeights = std::array<double, 3>;

// Abstand zweier Gläser (Rohkoordinaten a, b) im normierten Raum der Hülle:
//     d = √(Σ_j w_j (u_j(a) − u_j(b))²),   u = hull.normalize(·).
// Wirft std::invalid_argument bei Gewichten, die nicht endlich und ≥ 0 sind.
double glass_distance(const GlassHull& hull, const Point3& a, const Point3& b,
                      const GlassDistanceWeights& weights = {1.0, 1.0, 1.0});

// Die k nächsten Gläser der Hülle (GlassHull::points(), nur die mit usable(point) == true, leer:
// alle) zum Punkt x (Rohkoordinaten), aufsteigend nach Abstand, bei gleichem Abstand nach dem
// Index in points(). Weniger als k, wenn nicht genug Gläser in Frage kommen. Wirft
// std::invalid_argument bei k < 1 oder ungültigen Gewichten.
std::vector<GlassCandidate>
nearest_glasses(const GlassHull& hull, const Point3& x, int k,
                const GlassDistanceWeights& weights = {1.0, 1.0, 1.0},
                const std::function<bool(const GlassHullPoint&)>& usable = {});

// Eine Kombination: Kandidat choice[e] für Element e, Kosten Σ_e costs[e][choice[e]]
// (aufsteigend in e summiert).
struct GlassCombination {
    std::vector<int> choice;
    double cost = 0.0;

    friend bool operator==(const GlassCombination&, const GlassCombination&) = default;
};

// Zahl aller Kombinationen Π_e costs[e].size(), gesättigt bei UINT64_MAX (0 bei einem Element
// ohne Kandidaten, 1 ohne Elemente).
std::uint64_t glass_combination_count(const std::vector<std::vector<double>>& costs);

// Die höchstens max_combinations Kombinationen mit den kleinsten Kosten, aufsteigend nach
// (cost, choice lexikographisch). Ist das kartesische Produkt nicht größer, sind es alle.
// Verfahren: Strahlsuche über die Elemente e = 0 … E − 1; nach jedem Element bleiben die
// max_combinations besten Teilkombinationen nach (Teilkosten, choice). Weil die Kosten eine Summe
// unabhängiger Beiträge sind, ist das Ergebnis gleich der Auswahl aus dem vollen Produkt (bis auf
// Gleichstände, die erst durch die Rundung der Summe entstehen). Kosten müssen endlich und ≥ 0
// sein. Wirft std::invalid_argument bei max_combinations < 1, ungültigen Kosten oder einem
// Element ohne Kandidaten.
std::vector<GlassCombination> glass_combinations(const std::vector<std::vector<double>>& costs,
                                                 int max_combinations);

// Bester Lauf: endliche Merit, dann die wenigsten versagten Strahlen, dann die kleinste Merit,
// bei Gleichstand der kleinste Index (Regel des Multistarts von `mikroptim optimize`). −1, wenn
// kein Lauf eine endliche Merit hat.
int select_best_run(const LmResult& result);

// Optimierer für die Reoptimierung: wie optimize_lm_cpu bzw. cuda::optimize_lm_gpu (base mit
// einer Variante pro Lauf).
using GlassSnapOptimizer =
    std::function<LmResult(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const LmOptions& options)>;

struct GlassSnapOptions {
    int neighbors = 3;         // k nächste Katalog-Gläser pro Element (≥ 1)
    int max_combinations = 64; // höchstens so viele Kombinationen = Läufe (≥ 1)
    GlassDistanceWeights distance_weights{1.0, 1.0, 1.0};
    // Reoptimierung (Zyklen, Quadratur, Strahlversagen, Zusatzresiduen in lm.jacobian.extra).
    // Die Zusatzresiduen gelten für das kontinuierliche und das eingerastete System; die
    // Glas-Hülle gehört nicht dazu (ohne Glasvariablen gibt es sie nicht). max_iterations = 0:
    // nur auswerten (eingerastet ohne Reoptimierung).
    LmOptions lm{};
};

// Ein Glaselement.
struct GlassSnapElement {
    int file_surface = 0;                   // Dateifläche (Medium danach)
    int batch_surface = 0;                  // Tabellenfläche
    Point3 coordinates{};                   // (n₀, ν₁, ν₂) des kontinuierlichen Ergebnisses
    double hull_distance = 0.0;             // Abstand zur Hülle (normiert, 0 innen)
    std::vector<GlassCandidate> candidates; // aufsteigend nach Abstand
};

// Ein Lauf = eine Kombination.
struct GlassSnapRun {
    GlassCombination combination;         // Kandidaten pro Element, Kosten Σ d²
    std::vector<std::string> glasses;     // Katalognamen pro Element
    double snapped_merit = 0.0;           // F eingerastet, ohne Reoptimierung (run.history.front())
    std::int64_t snapped_failed_rays = 0; // versagte Strahlen dort
    LmRunResult run;                      // Reoptimierung (run.merit: F nach Reoptimierung)
};

struct GlassSnapResult {
    std::vector<GlassSnapElement> elements;
    std::vector<GlassSnapRun> runs;       // Reihenfolge von glass_combinations
    std::uint64_t total_combinations = 0; // Größe des kartesischen Produkts (gesättigt)
    bool truncated = false;               // weniger Läufe als total_combinations
    int best = -1;                        // select_best_run über runs
    // Kontinuierliches System (Glas als Buchdahl-Material, mit Glasvariablen) und seine Merit
    // mit denselben Einstellungen und Zusatzresiduen wie die Läufe (ohne Glas-Hülle).
    LoadedSystem continuous;
    VariableSet continuous_set;
    std::vector<double> continuous_x;
    double continuous_merit = 0.0;
    std::int64_t continuous_failed_rays = 0;
    // Eingerastetes System: Variablen ohne Glas, Startpunkt (kontinuierliches Ergebnis) und das
    // Ergebnis des besten Laufs als Systembeschreibung (Katalognamen, ohne Glasvariablen, x des
    // besten Laufs) samt Brechzahl- und Flächentabelle.
    VariableSet set;
    std::vector<double> x0;
    SystemBatchHost base; // Vorlage der Läufe: Variante c = Kombination c am Startpunkt x0
    LoadedSystem result;

    const GlassSnapRun& best_run() const { return runs.at(static_cast<std::size_t>(best)); }
    // Verschlechterung F_best − F_kontinuierlich und relativ dazu (NaN bei F_kontinuierlich = 0).
    double degradation() const;
    double relative_degradation() const;
};

// Glas-Einrasten für ein geladenes System mit Glasvariablen (Materialien: Katalogglas, konstant
// oder Buchdahl; die Werte sind das kontinuierliche Ergebnis). Die Kandidaten stammen aus der
// Hülle `hull` (make_glass_hull(continuous.catalog)); in Frage kommen nur Gläser, die im Katalog
// des Systems stehen und deren range_nm alle Wellenlängen des Spektrums und λ_ref abdeckt.
// `optimize` leer: optimize_lm_cpu. Die übrigen Daten (Geisterplatten, Nebenbedingungen,
// Variablen außer Glas) bleiben; der Eintrag glass_hull bleibt stehen (ohne Glasvariablen
// wirkungslos).
// Wirft FormatError/GlassError aus dem Aufbau der Systeme, std::invalid_argument ohne
// Glasvariablen, ohne weitere Variablen, bei ungültigen Optionen, leerer Hülle oder einem
// Element ohne Kandidaten, und std::runtime_error, wenn kein Lauf eine endliche Merit hat.
GlassSnapResult snap_glasses(const LoadedSystem& system, const GlassHull& hull,
                             const GlassSnapOptions& options = {},
                             const GlassSnapOptimizer& optimize = {});

// Systembeschreibung `system` mit eingerasteten Gläsern: Material der Dateifläche
// file_surfaces[e] = Katalogglas glasses[e], alle Einträge vom Typ glass aus `variables`
// entfernt (der Abschnitt entfällt, wenn er dadurch leer wird). Wirft std::invalid_argument bei
// ungleichen Längen oder ungültigen Flächen.
SystemDescription snapped_description(const SystemDescription& system,
                                      std::span<const int> file_surfaces,
                                      std::span<const std::string> glasses);

// Geladenes System aus einer Beschreibung mit den Katalogen von `like` (ohne Dateizugriff):
// validate_system, validate_materials, Brechzahl- und Flächentabelle. Wirft FormatError,
// GlassError.
LoadedSystem rebuild_loaded_system(const LoadedSystem& like, SystemDescription system);

} // namespace mko
