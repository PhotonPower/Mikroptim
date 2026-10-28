#pragma once

// Merit pro Variante eines SystemBatch: Ergebnis (MeritResult) und CPU-Referenz (merit_cpu)
// für den Kernel MERIT (M2-T3). Reines C++ ohne CUDA-Typen. Definitionen und
// Summationsreihenfolge: docs/architektur.md, Abschnitt "Kernel MERIT".

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"

#include <cstdint>
#include <vector>

namespace mko {

// Merit, versagte Strahlen und optional Max-Reduktionen für `variants` Varianten (CPU-Referenz
// und Download vom Gerät).
//
// - merit[v] = Σ r_k² über alle Strahlresiduen der Variante [Wellen² bei λ_ref, M4-T1]
//   (Ablage von ResidualLayout; hart versagte Strahlen haben r_k = 0 wie im Kernel RESIDUALS),
//   im weichen Modus (M4-T2) plus Σ p_k² der Strafresiduen, auf dem Gerät plus einem vom
//   Aufrufer gelieferten Anteil weiterer Residuen (mko::cuda::MeritRunOptions::extra_merit).
//   Im harten Modus hat eine Variante ohne gültigen Strahl die Merit 0 (plus Anteil); sie ist
//   nur an failed[v] erkennbar, das der Aufrufer deshalb prüfen muss.
// - failed[v] = Zahl der Strahlen mit Status ≠ Ok (weich und hart, inklusive Referenzsphäre).
// - hard_failed[v] = Zahl der hart versagten Strahlen (is_hard_failure); im harten Modus gleich
//   failed[v].
// - extrema: Max-Reduktionen pro Variante und Fläche (batch_extrema.hpp, nur Strahlen mit
//   Status Ok), leer, wenn nicht angefordert.
struct MeritResult {
    int variants = 0;
    std::int64_t rays_per_variant = 0;
    std::vector<double> merit;             // [V]
    std::vector<std::int64_t> failed;      // [V]
    std::vector<std::int64_t> hard_failed; // [V]
    SurfaceExtremaResult extrema;
};

// CPU-Referenz des Kernels MERIT für Variante v: Residuen wie residuals_cpu, Merit als
// Σ_(f, l) (Σ_k r_k² + Σ_k p_k²) (Strafresiduen p_k nur im weichen Modus), die inneren Summen
// in aufsteigender Reihenfolge k, außen in der Reihenfolge der Tupel (f-major, dann l) wie die
// zweite Stufe des Kernels. Mit `extrema` zusätzlich surface_extrema_cpu. Ergebnis mit
// variants = 1.
// Wirft wie residuals_cpu (std::out_of_range, std::invalid_argument).
MeritResult merit_cpu(const SystemBatchHost& batch, int v, const ResidualLayout& layout,
                      bool extrema = true, const ExtremaSmoothing& smoothing = {});

// Wie oben für alle Varianten (CPU-Pfad ohne GPU).
MeritResult merit_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                      bool extrema = true, const ExtremaSmoothing& smoothing = {});

} // namespace mko
