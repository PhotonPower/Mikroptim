#pragma once

// Medium nach einer Fläche (docs/dateiformat.md, Feld `material` der Flächen).

#include <array>
#include <string>

namespace mko {

enum class MaterialKind { Air, Catalog, Constant, Buchdahl };

struct MaterialSpec {
    MaterialKind kind = MaterialKind::Air;
    std::string name;                        // Katalogname bei Catalog
    double n = 1.0;                          // bei Constant
    double n0 = 1.0;                         // bei Buchdahl
    std::array<double, 3> nu{0.0, 0.0, 0.0}; // bei Buchdahl: (nu1, nu2, nu3), fehlende = 0
};

} // namespace mko
