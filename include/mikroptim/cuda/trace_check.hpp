#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <vector>

namespace mko::cuda {

// Prüfkernel für den CPU/GPU-Abgleich von trace_ray (M1-T5, keine Kernel-Architektur):
// Die Flächentabelle `table` (Zeiger auf Host-Speicher) wird auf die GPU kopiert und dort von
// jedem Block in den Shared Memory geladen; jeder Thread traced einen Strahl aus `rays_in`
// (lokale Koordinaten von Fläche 0) mit trace_ray über eine SurfaceTableView auf den Shared
// Memory. Ergebnis pro Strahl: Endzustand, Status (TraceStatus als unsigned), versagende Fläche.
// Rückgabe false bei CUDA-Fehlern oder wenn die Tabelle nicht in 48 KB Shared Memory passt.
bool trace_rays_gpu(const SurfaceTableView<double>& table, int wavelength_index,
                    const std::vector<Ray<double>>& rays_in, std::vector<Ray<double>>& rays_out,
                    std::vector<unsigned>& status, std::vector<int>& failed_surface);

} // namespace mko::cuda
