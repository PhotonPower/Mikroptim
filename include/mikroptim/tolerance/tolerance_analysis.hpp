#pragma once

// Monte-Carlo-Tolerierung (M8-T4, Host): Stichproben als Varianten eines nicht koaxialen Batches,
// Kompensator Arbeitsabstand (gebündelter LM, nur diese Variable), Bewertung pro Stichprobe aus
// der OPD der Pupillenquadratur (RMS-Wellenfront nach Abzug von Kolben und Kippung, Strehl als
// |⟨exp(2πi OPD/λ)⟩|² über die Quadratur) und Kennzahlen der Verteilungen. CPU und GPU über
// austauschbare Funktionen (Trace und Optimierer); GPU: mikroptim/cuda/tolerance.hpp.
// Verfahren: docs/optimierung.md, Abschnitt „Monte-Carlo-Tolerierung“; Strehl und RMS:
// docs/numerik.md, Abschnitt „Tolerierung: RMS und Strehl aus der Quadratur“.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/tolerances.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/tolerance/tolerance_model.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace mko {

// RMS und Strehl eines Tupels (Feld, Wellenlänge) aus OPD und Status der Quadratur.
struct TupleQuality {
    double rms = 0.0;    // RMS der OPD nach Abzug von Kolben und Kippung [mm]; NaN ohne Strahl
    double strehl = 0.0; // |Σ_ok W_k exp(2πi e_k/λ)|² / (Σ_alle W_k)², e_k = OPD nach Abzug
    double tilt_x = 0.0; // Kippung t_x, t_y der Anpassung OPD ≈ c₀ + t_x x + t_y y [mm]
    double tilt_y = 0.0;
    int failed = 0; // Strahlen mit Status ≠ Ok (hart: aus Kolben, Anpassung und Strehl
                    // ausgeschlossen, im Strehl undurchlässig)
};

// Bewertung eines Tupels: gewichtete kleinste Quadrate OPD_k ≈ c₀ + t_x x_k + t_y y_k über die
// gültigen Stützstellen (x = ρ sin φ, y = ρ cos φ der Quadratur `q`, Gewichte W_k;
// Normalgleichungen 3 × 3, Summen in aufsteigender Reihenfolge k, Cholesky; bei singulärer Matrix,
// z. B. auf der Achse oder mit weniger als drei Strahlen, nur der Kolben), Rest e_k, RMS = √(Σ W e²
// / Σ_ok W) und Strehl wie oben bei der Vakuumwellenlänge λ [mm] des Tupels. `opd` und `status`
// haben sample_count(q) Einträge (Ablage wie ResidualResult). Optional `rest` (leer oder
// sample_count(q) Einträge): der Rest e_k [mm] pro Stützstelle, 0 für versagte Strahlen und ohne
// gültigen Strahl (Empfindlichkeiten, M10-T3). Wirft std::invalid_argument bei falschen Längen
// oder λ ≤ 0.
TupleQuality tuple_quality(const PupilQuadrature<double>& q, std::span<const double> opd,
                           std::span<const std::uint8_t> status, double lambda_mm,
                           std::span<double> rest = {});

// Kennzahlen einer Stichprobe.
struct ToleranceSampleResult {
    std::vector<double> rms; // [f · L + l] RMS nach Abzug von Kolben und Kippung [Wellen bei λ_ref]
    std::vector<double> strehl;    // [f · L + l] Strehl bei der Wellenlänge des Tupels
    double rms_composite = 0.0;    // √(Σ ŵ_f ŵ_λ RMS²) [Wellen bei λ_ref], Gewichte normiert wie in
                                   // der Merit (docs/numerik.md, Abschnitt 7)
    double strehl_min = 0.0;       // kleinster Strehl der Tupel mit ŵ_f ŵ_λ > 0
    std::int64_t failed_rays = 0;  // hart versagte Strahlen der Bewertung
    double working_distance = 0.0; // Arbeitsabstand bei der Bewertung [mm]
    double compensation = 0.0;     // Änderung durch den Kompensator [mm] (0 ohne)
    LmStatus compensator_status = LmStatus::Running; // ohne Kompensator Running
    int compensator_iterations = 0;
};

// Kennzahlen einer Verteilung über die Stichproben.
struct DistributionStats {
    int count = 0;     // Stichproben mit endlichem Wert (NaN zählt in den Perzentilen als
                       // schlechtester Wert, nicht in Mittel, Std, Min, Max)
    double mean = 0.0; // Mittel (aufsteigend summiert)
    double std = 0.0;  // Standardabweichung mit n − 1 (zwei Durchläufe), 0 bei n = 1
    double min = 0.0;
    double max = 0.0;
    // Perzentile in der schlechten Richtung (Rang nach „nearest rank“, ohne Interpolation):
    // higher_is_worse (RMS): Wert, den p % der Stichproben nicht überschreiten, x_(⌈p n⌉) der
    // aufsteigend sortierten Werte; sonst (Strehl): Wert, den mindestens p % der Stichproben
    // erreichen, x_(n + 1 − ⌈p n⌉).
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

// Kennzahlen aus `values` (beliebige Reihenfolge, NaN erlaubt; leer: alles 0 bzw. NaN).
DistributionStats distribution_stats(std::span<const double> values, bool higher_is_worse);

// OPD und Status aller Varianten von `batch` in der Ablage von `layout` (hartes Versagen):
// `out` muss danach opd und status für alle Varianten enthalten (residuals darf leer sein).
using ToleranceTraceFn = std::function<void(const SystemBatchHost& batch,
                                            const ResidualLayout& layout, ResidualResult& out)>;
// CPU-Trace der Tolerierung: residuals_cpu pro Variante auf Host-Threads (zusammenhängende
// Blöcke; threads ≤ 0: nach der Arbeit, lm_host_threads), OPD und Status aller Varianten in der
// Ablage von `layout`, ohne Residuen. Bitgleich für jede Threadzahl.
void tolerance_trace_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                         ResidualResult& out, int threads = 0);

// Normierte Gewichte ŵ_f ŵ_λ der Tupel [f · L + l] wie in der Merit (Summen aufsteigend).
std::vector<double> tolerance_tuple_weights(const SystemBatchHost& batch);

// Optimierer des Kompensators (z. B. optimize_lm_cpu oder cuda::optimize_lm_gpu); `base` hat eine
// Variante pro Lauf.
using ToleranceOptimizerFn =
    std::function<LmResult(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const LmOptions& options)>;

// Standardgröße einer Tranche (Stichproben, die gemeinsam aufgebaut, kompensiert und getract
// werden). Nur Speicher und Laufzeit; die Ergebnisse sind für jede Größe bitgleich.
inline constexpr int kDefaultToleranceChunk = 4096;
inline constexpr double kDefaultStrehlThreshold = 0.8;

struct ToleranceSettings {
    int samples = 1000;     // Zahl der Stichproben (>= 1)
    std::uint64_t seed = 0; // Hauptstartwert; Stichprobe i nutzt variant_seed(seed, i)
    // Kompensator; leer: aus der Datei (tolerances.compensator, Standard Arbeitsabstand)
    std::optional<CompensatorType> compensator;
    // Pupillenquadratur der Bewertung und des Kompensators (rings, spokes); die Bewertung
    // rechnet immer mit hartem Versagen, der Kompensator mit failure/fail_weight von hier
    // (Standard weich wie in der Optimierung).
    ResidualSettings quadrature{};
    double strehl_threshold = kDefaultStrehlThreshold; // Ausbeute: Anteil mit strehl_min >= S
    int chunk = kDefaultToleranceChunk;                // Stichproben pro Tranche (>= 1)
    // Einstellungen des LM für den Kompensator (jacobian.quadrature wird aus `quadrature`
    // gesetzt, Zusatzresiduen gibt es nicht).
    LmOptions lm{};
    int threads = 0;        // Host-Threads des CPU-Traces (≤ 0: nach der Arbeit, lm_host_threads)
    ToleranceTraceFn trace; // leer: CPU (residuals_cpu pro Variante)
    ToleranceOptimizerFn optimizer; // leer: optimize_lm_cpu
    // Optional: Fortschritt nach jeder Tranche (fertige und alle Stichproben).
    std::function<void(int done, int total)> progress;
};

// Ergebnis der Tolerierung.
struct ToleranceResult {
    int samples = 0;
    std::uint64_t seed = 0;
    CompensatorType compensator = CompensatorType::None;
    int draws = 0;  // Zufallsgrößen pro Stichprobe
    int fields = 0; // F, L, Ablage der Tupel [f · L + l]
    int wavelengths = 0;
    std::vector<double> wavelength_nm;
    double reference_nm = 0.0;
    std::vector<double> tuple_weight; // ŵ_f ŵ_λ [f · L + l]
    double strehl_threshold = 0.0;
    int chunks = 0;                             // Tranchen
    ToleranceSampleResult nominal;              // ohne Störung, mit demselben Ablauf
    std::vector<ToleranceSampleResult> results; // pro Stichprobe i
    // Verteilungen
    DistributionStats rms_composite;
    DistributionStats strehl_min;
    DistributionStats compensation;
    std::vector<DistributionStats> rms;    // pro Tupel
    std::vector<DistributionStats> strehl; // pro Tupel
    double yield = 0.0;                    // Anteil der Stichproben mit strehl_min >= Schwelle
    int failed_samples = 0;                // Stichproben mit versagten Strahlen
};

// Tolerierung des Systems `system` mit Ausgangsbatch `nominal` (eine Variante, z. B.
// make_system_batch; koaxial oder nicht, wird mit make_non_coaxial in die volle Pupille gebracht):
// Toleranzsatz aus system.tolerances (resolve_tolerances; ohne Abschnitt: keine Störung). Ablauf
// pro Tranche [i₀, i₀ + T): Stichproben schreiben (ToleranceModel::write_samples), mit
// Kompensator den Arbeitsabstand aller Stichproben gemeinsam optimieren (Optimierer, Start beim
// Arbeitsabstand der Stichprobe, keine Schranken), dann OPD und Status über `trace`, Bewertung
// pro Tupel (tuple_quality) und Stichprobe. Die nominelle Bewertung läuft genauso mit ξ = 0. Die
// Ergebnisse jeder Stichprobe hängen nur von seed und ihrem Index ab (bitgleich für jede
// Tranchengröße und jede Aufteilung der Backends). Wirft FormatError (Toleranzsatz),
// std::invalid_argument (Einstellungen, Batch) und weiter, was Trace und Optimierer werfen.
ToleranceResult run_tolerance_analysis(const SystemDescription& system,
                                       const SystemBatchHost& nominal,
                                       const ToleranceSettings& settings);

// Kennzahlen der Verteilungen (rms_composite, strehl_min, compensation, rms, strehl, yield,
// failed_samples) aus result.results neu berechnen (von run_tolerance_analysis aufgerufen).
void summarize_tolerance_result(ToleranceResult& result);

} // namespace mko
