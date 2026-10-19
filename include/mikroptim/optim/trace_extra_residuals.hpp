#pragma once

// Trace-abhängige Zusatzresiduen (M4-T3): Residuen pro Variante, die aus den Max-Reduktionen
// derselben Trace-Auswertung entstehen (größte Strahlhöhe und kleinster cos I pro Fläche,
// SurfaceExtremaResult, M2-T3), z. B. Durchmesser und Einfallswinkel der mechanischen
// Nebenbedingungen. Im Residuenvektor einer Variante stehen sie hinter den übrigen
// Zusatzresiduen (ExtraResiduals): R = R_v + E + E_t. Schnittstelle, Pfade (CPU, GPU,
// FD-Jacobi, Merit) und Glattheit: docs/optimierung.md, Abschnitt 2.1.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/system_batch.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace mko {

// Sicht auf die Extrema einer Variante: 2 · S Werte in der Ablage von SurfaceExtremaResult
// ([s][0] größte Strahlhöhe [mm], [s][1] kleinster cos I) für die Tabellenflächen s.
struct VariantExtrema {
    int surfaces = 0;
    const double* values = nullptr;

    double max_height(int s) const { return values[2 * static_cast<std::size_t>(s)]; }
    double min_cos_incidence(int s) const { return values[2 * static_cast<std::size_t>(s) + 1]; }
};

// Extrema des Eintrags i (0 … extrema.variants − 1) von `extrema`. Wirft std::out_of_range.
VariantExtrema variant_extrema(const SurfaceExtremaResult& extrema, int i);

// Schreibt die Residuen der Variante v des Batches aus ihren Extrema nach out
// (out.size() == count). Muss threadsicher sein.
using TraceExtraResidualFn = std::function<void(
    const SystemBatchHost& batch, int v, const VariantExtrema& extrema, std::span<double> out)>;

struct TraceExtraResiduals {
    int count = 0;
    TraceExtraResidualFn evaluate;
};

// Prüft count ≥ 0 und eine Auswertefunktion bei count > 0; wirft std::invalid_argument mit
// `who` in der Meldung.
void check_trace_extra_residuals(const TraceExtraResiduals& extra, const char* who);

// Wertet die Varianten [first, first + count) des Batches aus (count < 0: bis zum Ende) auf
// `threads` Host-Threads (≤ 0: Hardware; Aufteilung wie evaluate_extra_residuals, bitgleich für
// jede Threadzahl). Die Extrema der Variante v stehen in `extrema` beim Eintrag
// v − extrema_first (z. B. Extrema einer Tranche der Engine ab ihrer ersten Variante). Ablage
// out[(v − first) · extra.count + i]. Wirft std::invalid_argument (Längen, Flächenzahl der
// Extrema), std::out_of_range (Bereich, fehlende Extrema einer Variante).
void evaluate_trace_extra_residuals(const TraceExtraResiduals& extra, const SystemBatchHost& batch,
                                    const SurfaceExtremaResult& extrema, int extrema_first,
                                    std::span<double> out, int first = 0, int count = -1,
                                    int threads = 0);

// Σ_i t_i² der trace-abhängigen Residuen der Variante v (aufsteigend in i; 0 bei count = 0).
double trace_extra_merit(const TraceExtraResiduals& extra, const SystemBatchHost& batch, int v,
                         const VariantExtrema& extrema);

// Σ_i t_i² der trace-abhängigen Residuen jeder Variante des Batches (aufsteigend in i; leer bei
// count = 0), mit den Extrema aller Varianten (extrema.variants == batch.variants()).
std::vector<double> trace_extra_merit_values(const TraceExtraResiduals& extra,
                                             const SystemBatchHost& batch,
                                             const SurfaceExtremaResult& extrema, int threads = 0);

// Fügt die Zeilen zweier Residuenblöcke pro Variante zusammen: out[v · (E + E_t) + i] ist
// a[v · E + i] für i < E und b[v · E_t + i − E] sonst (Ablage der Zusatzresiduen im
// Residuenvektor: erst ExtraResiduals, dann TraceExtraResiduals). Wirft std::invalid_argument
// bei unpassenden Längen.
std::vector<double> join_extra_rows(std::span<const double> a, int a_count,
                                    std::span<const double> b, int b_count, int variants);

} // namespace mko
