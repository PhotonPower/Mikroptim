#pragma once

// Diagnosebericht der Geisterplatten nach einer Konvergenz (M4-T6, Host; docs/optimierung.md,
// Abschnitt 5 „Diagnose“): pro Geisterplatte Σ|a_k|, dominante Ordnungen, Glas (Brechzahl,
// Abbezahl, anomale Teildispersion), paraxiale Brechkraft φ_G, Lage und Beitrag zur Merit
// (Strafe und Merit-Gewinn gegenüber a = 0); pro Lage die Klassifikation nach der Tabelle in
// Abschnitt 5 mit Empfehlung eines Einfüge-Operators (A: Meniskus ohne Brechkraft,
// B: Kittglied, B mit Fluorit-artigem Glas). Ausgabe als Struktur, Text und JSON.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace mko {

struct SystemDescription;
class GlassCatalog;

// Empfohlener Einfüge-Operator (docs/optimierung.md, Abschnitte 5 und 6).
enum class GhostOperator {
    None,       // keine aktive Geisterplatte
    A,          // Meniskus ohne Brechkraft (monochromatische Korrektur)
    B,          // Aufspalten in ein Kittglied aus normalen Gläsern (Farbfehler 1. Ordnung)
    BAnomalous, // Kittglied mit Fluorit- oder FK51-artigem Glas (sekundäres Spektrum)
};

// "none", "A", "B", "B_anomalous"
const char* to_string(GhostOperator op);

// Linien für Abbezahl und Teildispersion [nm] (g, F, d, C).
inline constexpr double kLineG = 435.8343;
inline constexpr double kLineF = 486.1327;
inline constexpr double kLineD = 587.5618;
inline constexpr double kLineC = 656.2725;

// Glasdaten einer Geisterplatte. Werte, die nicht auswertbar sind (Wellenlänge außerhalb des
// Katalogbereichs, dispersionsfreies Glas), sind NaN.
struct GhostGlassData {
    std::string name;           // Katalogname oder Modell, z. B. "buchdahl(n0=1.72, …)"
    double n_ref = 0.0;         // n_G bei λ_ref (paraxial_ghost_index)
    double dispersion = 0.0;    // n_G(λ_kurz) − n_G(λ_lang) der Merit-Wellenlängen (0 bei einer)
    double n_d = 0.0;           // n_d
    double abbe = 0.0;          // ν_d = (n_d − 1)/(n_F − n_C)
    double partial_gf = 0.0;    // P_g,F = (n_g − n_F)/(n_F − n_C)
    double delta_partial = 0.0; // ΔP_g,F = P_g,F − (0,6438 − 0,001682 ν_d) (Normalgerade)
};

struct GhostPlateReport {
    int plate = 0;
    int after_surface = 0; // Dateifläche, nach der der Luftspalt liegt
    double distance = 0.0; // Abstand vom Scheitel von after_surface [mm]
    int table_surface = 0; // Tabellenfläche im Batch
    double norm_radius = 0.0;
    int order = 0; // K der Platte (Datei)
    GhostGlassData glass;
    std::vector<double> coefficients; // a_1 … a_K [mm]
    double sum_abs = 0.0;             // Σ|a_k| [mm]
    double max_abs = 0.0;             // max |a_k| [mm]
    double activity_threshold = 0.0;  // Schwelle für „aktiv“ [mm]
    bool active = false;              // max |a_k| > activity_threshold
    // Dominante Ordnungen: aktive Ordnungen mit |a_k| ≥ dominant_fraction · max |a|,
    // absteigend nach |a_k| (leer, wenn die Platte nicht aktiv ist).
    std::vector<int> dominant_orders;
    double power = 0.0;                // φ_G bei λ_ref [1/mm]
    double petzval = 0.0;              // φ_G/(n_m n_G) [1/mm]
    double marginal_height = 0.0;      // paraxialer Randstrahl (volle Pupille) an der Platte [mm]
    double chief_height_per_rad = 0.0; // paraxialer Hauptstrahl, Feldwinkel 1 rad [mm/rad]
    bool penalized = false;            // Terme in der Strafe
    double penalty = 0.0;              // Σ r² der Strafterme der Platte [Wellen²]
    double merit_without = 0.0;        // F mit a_k = 0 für diese Platte (NaN ohne Merit)
    double merit_gain = 0.0;           // merit_without − F (> 0: die Platte senkt F netto)
};

// Klassifikation einer Lage (alle Platten mit gleichem after_surface und distance).
struct GhostPositionReport {
    int after_surface = 0;
    double distance = 0.0;
    std::vector<int> plates;        // alle Platten an der Lage
    std::vector<int> active_plates; // davon aktiv
    // Farbanteil der gemeinsamen Korrektur: χ = ‖ΔT‖/‖T‖ mit T_k = Σ_g (n_g,ref − n_m) a_{g,k}
    // und ΔT_k = Σ_g Δn_g a_{g,k} über die aktiven Platten (Norm Σ_k x_k²/(2k + 1));
    // NaN bei einer Merit-Wellenlänge.
    double chromatic_ratio = 0.0;
    // Größtes |Δn_g|/(n_g,ref − n_m) der aktiven Gläser: χ einer einzelnen Platte.
    double single_glass_ratio = 0.0;
    double merit_gain = 0.0; // Σ merit_gain der aktiven Platten (NaN ohne Merit)
    double sum_abs = 0.0;    // Σ|a| der aktiven Platten
    GhostOperator recommendation = GhostOperator::None;
    std::string finding; // Befund (Tabelle in Abschnitt 5)
    std::string reason;  // Begründung der Klassifikation
};

struct GhostDiagnosis {
    double merit = 0.0;        // F am Punkt (NaN ohne Merit)
    double reference_nm = 0.0; // λ_ref
    int merit_wavelengths = 0; // Wellenlängen mit Gewicht > 0
    double short_nm = 0.0;     // kürzeste und längste davon
    double long_nm = 0.0;
    std::vector<GhostPlateReport> plates;
    std::vector<GhostPositionReport> positions;
    int recommended_position = -1; // Index in positions (−1: keine aktive Platte)
    GhostOperator recommendation = GhostOperator::None;
    std::string recommendation_text;
};

struct GhostDiagnosisOptions {
    // |a_k| darüber gilt als aktiv [mm]; ≤ 0: 10 ε der Strafe der Platte (ohne Strafe
    // 10 · kDefaultGhostSmoothing). Inaktive Koeffizienten liegen bei |a| ≲ ε (Abschnitt 3).
    double activity_threshold = 0.0;
    double dominant_fraction = 0.1; // dominante Ordnungen ab diesem Anteil von max |a|
    // Farbkorrektur (B), wenn χ > chromatic_factor · single_glass_ratio: die Muster der
    // Wellenlängen unterscheiden sich stärker, als es eine einzelne Platte aus einem der
    // aktiven Gläser könnte (nur über Gläser verschiedener Dispersion erreichbar).
    double chromatic_factor = 2.0;
    // Anomale Teildispersion (Fluorit-Platte aktiv): eine aktive Platte der Lage mit
    // ΔP_g,F ≥ anomalous_delta und einem Anteil Σ|a_g| / Σ|a| der Lage ≥ anomalous_share, bei
    // mindestens drei Merit-Wellenlängen (sekundäres Spektrum).
    double anomalous_delta = 0.02;
    double anomalous_share = 0.25;
    // Merit ohne Platte (ein zusätzlicher Trace pro Platte)
    bool merit_without = true;
};

// Merit F aller Varianten eines Batches (Strahl- und Zusatzresiduen, wie im LM).
using BatchMeritFn = std::function<std::vector<double>(const SystemBatchHost& batch)>;

// CPU-Merit wie CpuLmBackend::merit: merit_cpu (Quadratur `quadrature`) plus Σ e² der
// Zusatzresiduen `extra`, Varianten auf `threads` Host-Threads (≤ 0: Hardware).
BatchMeritFn make_cpu_merit(const ResidualSettings& quadrature, ExtraResiduals extra,
                            int threads = 0);

// Bericht für Variante v von `batch` (z. B. lm_point_batch am Ergebnis). `system` und
// `catalog` liefern Lage und Glas der Platten (die Datei, aus der der Batch gebaut ist),
// `penalty` die Strafterme, `merit` F am Punkt und bei a = 0 je Platte (Varianten
// 0 … G eines eigenen Batches; leer: ohne Merit-Werte). Wirft std::invalid_argument, wenn
// Datei und Batch nicht zusammenpassen.
GhostDiagnosis diagnose_ghost_plates(const SystemDescription& system, const GlassCatalog& catalog,
                                     const SystemBatchHost& batch, int v,
                                     const GhostPenalty& penalty, const BatchMeritFn& merit,
                                     const GhostDiagnosisOptions& options = {});

// Textbericht (Tabellen pro Platte und Lage, Empfehlung) bzw. JSON-Objekt (Text, Felder wie
// die Strukturen; nicht endliche Zahlen als null), docs/cli.md.
void write_ghost_diagnosis_text(const GhostDiagnosis& diagnosis, std::ostream& out);
std::string ghost_diagnosis_json(const GhostDiagnosis& diagnosis, int indent = 2);

} // namespace mko
