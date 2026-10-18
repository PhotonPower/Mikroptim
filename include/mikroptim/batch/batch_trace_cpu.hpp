#pragma once

// CPU-Referenztrace einer Variante eines SystemBatch (Referenz für die Kernel in M2).

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/pupil/quadrature.hpp"

namespace mko {

// Traced alle Felder × Wellenlängen × Stützstellen der Variante v mit batch_source_ray und
// trace_ray über surface_table(v); Ablage wie trace_system_cpu ((f · L + l) · N + k).
// Für eine Variante, die aus einem System gebaut wurde, ist das Ergebnis bitgleich zu
// trace_system_cpu für dieses System.
// Wirft std::out_of_range bei ungültigem v, std::invalid_argument bei der Punktquelle mit
// NA außerhalb von (0, index[v][0][l]).
CpuTraceResult trace_variant_cpu(const SystemBatchHost& batch, int v,
                                 const PupilQuadrature<double>& pupil);

} // namespace mko
