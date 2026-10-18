#pragma once

// Zusatzresiduen pro Variante (Paraxial-Nebenbedingungen, später Strafterme), die neben den
// Strahlresiduen in den Residuenvektor eingehen (docs/optimierung.md, Abschnitt 2).

#include "mikroptim/batch/system_batch.hpp"

#include <functional>
#include <span>

namespace mko {

// Schreibt die Zusatzresiduen der Variante v nach out (out.size() == count).
using ExtraResidualFn =
    std::function<void(const SystemBatchHost& batch, int v, std::span<double> out)>;

struct ExtraResiduals {
    int count = 0;
    ExtraResidualFn evaluate;
};

} // namespace mko
