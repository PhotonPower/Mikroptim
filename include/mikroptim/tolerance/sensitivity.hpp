#pragma once

// Empfindlichkeiten (M10-T3, Host): Wirkung jeder einzelnen Toleranz eines Toleranzsatzes
// (Abschnitt `tolerances` der Systemdatei oder Mikroptim-Standardsatz) auf die Bildgüte, als
// direkte Rechnung bei ±δ und aus der Jacobi-Matrix der Wellenfrontreste nach den
// Toleranzparametern. Störungen über ToleranceModel::apply (M8-T4, ADR-0011), Bewertung wie
// in der Monte-Carlo-Tolerierung (tuple_quality: RMS nach Abzug von Kolben und Kippung).
// Verfahren und Formeln: docs/optimierung.md, Abschnitt „Empfindlichkeiten“; GPU-Trace über
// mikroptim/cuda/tolerance.hpp.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/tolerances.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"

#include <cstdint>
#include <vector>

namespace mko {

// Mikroptim-Standardsatz der Empfindlichkeiten (ohne Abschnitt `tolerances` oder auf Wunsch):
// Dezentrierung δ = 10 µm und Verkippung 1′ pro Element (docs/entwurfssteuerung.md, Abschnitt
// 3.4), Radius 0,1 % relativ, Dicke 20 µm, Brechzahl 5·10⁻⁴. Größenordnung üblicher
// Fertigungstoleranzen für Präzisionsoptik, ohne Bezug auf einen bestimmten Katalog.
struct SensitivityDefaults {
    double element_decenter_mm = 0.01;
    double element_tilt_arcmin = 1.0;
    ElementPivot element_pivot = ElementPivot::Center; // Drehpunkt der Elementverkippung
    double radius_relative = 1e-3;
    double thickness_mm = 0.02;
    double index = 5e-4;
};

// Standardsatz für `system` (Verteilung gleichförmig, ohne Kompensator; für die Empfindlichkeit
// zählen nur die Grenzen δ): element_decenter und element_tilt für alle Elemente (all_elements),
// radius für alle Flächen vom Typ sphere mit c ≠ 0, thickness für alle Flächen außer der letzten
// (Arbeitsabstand), index für alle Flächen mit Glas danach. Einträge mit Grenze 0 entfallen.
// Wirft std::invalid_argument bei negativen oder nicht endlichen Grenzen.
Tolerances default_sensitivity_tolerances(const SystemDescription& system,
                                          const SensitivityDefaults& defaults = {});

inline constexpr double kDefaultSensitivityStep = 1e-2;

struct SensitivitySettings {
    // Pupillenquadratur der Bewertung (rings, spokes); gerechnet wird immer mit hartem Versagen.
    ResidualSettings quadrature{};
    // Schrittweite h der zentralen Differenzen für die Jacobi-Matrix, relativ zur Grenze δ
    // (Störungen ±h·δ); 0 < h ≤ 1.
    double step = kDefaultSensitivityStep;
    int threads = 0;        // Host-Threads des CPU-Traces (≤ 0: nach der Arbeit)
    ToleranceTraceFn trace; // leer: CPU (tolerance_trace_cpu)
};

// Eine Zufallsgröße (Komponente) einer Toleranz: Q = Σ ŵ_f ŵ_λ (RMS_fλ/λ_ref)² [Wellen²] mit
// dem RMS nach Abzug von Kolben und Kippung (tuple_quality), also Q = RMS_gesamt² der
// Tolerierung. Q₀ nominal, Q± bei ξ = ±1 (Wert ±δ, alle übrigen 0).
struct SensitivityComponent {
    int tolerance = 0; // Index in SensitivityResult::tolerances
    int component = 0; // 0 oder 1 (Dezentrierung x, y; Verkippung α, β), sonst 0
    double q_plus = 0.0;
    double q_minus = 0.0;
    double linear = 0.0;    // a = (Q₊ − Q₋)/2
    double quadratic = 0.0; // b = (Q₊ + Q₋ − 2 Q₀)/2, Q(ξ) ≈ Q₀ + a ξ + b ξ²
    // Aus der Jacobi-Spalte j = ∂ẽ/∂ξ (zentrale Differenz mit ±h, ẽ = Wellenfrontreste, Σ ẽ² = Q):
    double jacobian_linear = 0.0;    // g = 2 ẽ₀ᵀ j
    double jacobian_quadratic = 0.0; // ‖j‖² (Gauß-Newton), Q(ξ) ≈ Q₀ + g ξ + ‖j‖² ξ²
    std::int64_t failed_rays = 0;    // hart versagte Strahlen in den vier gestörten Varianten
};

struct SensitivityResult {
    std::vector<ResolvedTolerance> tolerances;    // resolve_tolerances(system)
    std::vector<SensitivityComponent> components; // Reihenfolge der Ziehungen (ToleranceModel)
    double q0 = 0.0;                              // Q nominal [Wellen²]
    std::int64_t failed_rays = 0;                 // nominal
    int fields = 0;
    int wavelengths = 0;
    double reference_nm = 0.0;
    double step = 0.0;
    int variants = 0;  // getracte Varianten (1 + 4 · Zufallsgrößen)
    int residuals = 0; // Länge von ẽ (Strahlen aller Tupel einer Variante)
};

// Empfindlichkeiten aller Toleranzen von system.tolerances (ohne Abschnitt: leer) am Ausgangsbatch
// `nominal` (eine Variante, wird mit make_non_coaxial in die volle Pupille gebracht). Ein Batch
// mit 1 + 4 D Varianten (D Zufallsgrößen): nominal, dann pro Zufallsgröße ξ = +1, −1, +h, −h
// (ToleranceModel::apply ab der nominellen Variante), ein Trace (OPD, Status), Bewertung pro
// Tupel mit tuple_quality. Ohne Kompensator. Wellenfrontreste pro Strahl
// ẽ_k = √(ŵ_f ŵ_λ W_k / W_ok) e_k / λ_ref (versagte Strahlen 0), Jacobi-Spalte
// j = (ẽ(+h) − ẽ(−h)) / (2h). Deterministisch (bitgleich für jede Threadzahl). Wirft FormatError
// (Toleranzsatz), std::invalid_argument (Einstellungen, Batch) und weiter, was der Trace wirft.
SensitivityResult run_sensitivity_analysis(const SystemDescription& system,
                                           const SystemBatchHost& nominal,
                                           const SensitivitySettings& settings);

// Kennzahlen einer Toleranz über ihre Komponenten.
struct SensitivitySummary {
    int tolerance = 0;
    double worst_delta_q = 0.0; // max über Komponenten und Vorzeichen von Q± − Q₀ [Wellen²]
    double worst_rms = 0.0;     // √(Q₀ + worst_delta_q) [Wellen]
    // Mittel der quadratischen Terme über die Komponenten (x/y bzw. α/β): erwartete Zunahme von
    // Q bei einer Störung ±δ in zufälliger Richtung bzw. mit zufälligem Vorzeichen; direkt
    // (b) und aus der Jacobi-Spalte (Gauß-Newton ‖j‖², ohne die Krümmung der Reste).
    double mean_quadratic = 0.0;
    double mean_jacobian_quadratic = 0.0;
    std::int64_t failed_rays = 0;
};

// Eine Zusammenfassung pro Toleranz (Reihenfolge von result.tolerances).
std::vector<SensitivitySummary> summarize_sensitivity(const SensitivityResult& result);

} // namespace mko
