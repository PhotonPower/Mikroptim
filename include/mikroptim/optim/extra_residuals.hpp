#pragma once

// Zusatzresiduen pro Variante (Paraxial-Nebenbedingungen, Strafterme wie die L1-Strafe der
// Geisterkoeffizienten), die neben den Strahlresiduen in den Residuenvektor eingehen
// (docs/optimierung.md, Abschnitt 2).

#include "mikroptim/batch/system_batch.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mko {

// Schreibt die Zusatzresiduen der Variante v nach out (out.size() == count).
using ExtraResidualFn =
    std::function<void(const SystemBatchHost& batch, int v, std::span<double> out)>;

struct ExtraResiduals {
    int count = 0;
    ExtraResidualFn evaluate;
};

// Verkettet mehrere Zusatzresiduen (docs/optimierung.md, Abschnitt 2.1): count = Σ count_i, die
// Residuen von parts[i] stehen in der angegebenen Reihenfolge hintereinander. Teile mit
// count = 0 entfallen; bleibt genau einer übrig, wird er unverändert zurückgegeben, bleibt keiner,
// ist das Ergebnis leer (count = 0). Threadsicher, wenn alle Teile es sind. Wirft
// std::invalid_argument bei count < 0 oder count > 0 ohne Auswertefunktion.
inline ExtraResiduals concat_extra_residuals(std::vector<ExtraResiduals> parts) {
    std::vector<ExtraResiduals> used;
    int total = 0;
    for (ExtraResiduals& p : parts) {
        if (p.count < 0) {
            throw std::invalid_argument("Zusatzresiduen: negative Anzahl");
        }
        if (p.count == 0) {
            continue;
        }
        if (!p.evaluate) {
            throw std::invalid_argument("Zusatzresiduen: keine Auswertefunktion");
        }
        total += p.count;
        used.push_back(std::move(p));
    }
    if (used.empty()) {
        return {};
    }
    if (used.size() == 1) {
        return std::move(used.front());
    }
    ExtraResiduals out;
    out.count = total;
    out.evaluate = [used = std::move(used), total](const SystemBatchHost& batch, int v,
                                                   std::span<double> values) {
        if (values.size() != static_cast<std::size_t>(total)) {
            throw std::invalid_argument("Zusatzresiduen: Ausgabe hat falsche Länge");
        }
        std::size_t offset = 0;
        for (const ExtraResiduals& p : used) {
            const auto n = static_cast<std::size_t>(p.count);
            p.evaluate(batch, v, values.subspan(offset, n));
            offset += n;
        }
    };
    return out;
}

} // namespace mko
