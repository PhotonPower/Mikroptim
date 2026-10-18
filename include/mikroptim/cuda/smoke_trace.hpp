#pragma once

#include <vector>

namespace mko::cuda {

// Smoke-Test des Backends: achsparallele meridionale Strahlen der Höhen `heights`
// werden auf der GPU an einer Sphäre (Krümmung c, mu = n/n') gebrochen.
// Ergebnis: Achsschnittpunkt und Status (TraceStatus als unsigned) pro Strahl.
// Rückgabe false bei CUDA-Fehlern.
bool smoke_trace(const std::vector<double>& heights, double curvature, double mu,
                 std::vector<double>& axis_crossing, std::vector<unsigned>& status);

} // namespace mko::cuda
