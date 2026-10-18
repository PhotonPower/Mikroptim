// Brechzahltabelle n[Fläche][Wellenlänge] (docs/architektur.md).

#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/glass/dispersion.hpp"

#include "format_util.hpp"

#include <cmath>
#include <string>

namespace mko {

double material_index(const MaterialSpec& medium, double lambda_nm, const GlassCatalog& catalog) {
    if (!(std::isfinite(lambda_nm) && lambda_nm > 0.0)) {
        throw GlassError("ungültige Wellenlänge " + detail::format_number(lambda_nm) + " nm");
    }
    switch (medium.kind) {
    case MaterialKind::Air:
        return 1.0;
    case MaterialKind::Constant:
        if (!(std::isfinite(medium.n) && medium.n > 0.0)) {
            throw GlassError("konstante Brechzahl " + detail::format_number(medium.n) +
                             " ist ungültig (endlich und > 0 erwartet)");
        }
        return medium.n;
    case MaterialKind::Catalog:
        return catalog.at(medium.name).index(lambda_nm);
    case MaterialKind::Buchdahl: {
        const bool finite = std::isfinite(medium.n0) && std::isfinite(medium.nu[0]) &&
                            std::isfinite(medium.nu[1]) && std::isfinite(medium.nu[2]);
        if (!finite) {
            throw GlassError("Buchdahl-Koeffizienten müssen endlich sein");
        }
        const double lambda_um = nm_to_um(lambda_nm);
        if (!(buchdahl_denominator(lambda_um) > 0.0)) {
            throw GlassError("Buchdahl-Koordinate ist bei " + detail::format_number(lambda_nm) +
                             " nm nicht definiert (erfordert lambda > 146,074 nm)");
        }
        return buchdahl_index(medium.n0, medium.nu[0], medium.nu[1], medium.nu[2], lambda_um);
    }
    }
    throw GlassError("unbekannte Materialart");
}

IndexTable make_index_table(std::span<const MaterialSpec> media,
                            std::span<const double> wavelengths_nm, const GlassCatalog& catalog) {
    IndexTable t;
    t.surfaces = static_cast<int>(media.size());
    t.wavelengths = static_cast<int>(wavelengths_nm.size());
    t.n.reserve(media.size() * wavelengths_nm.size());
    for (std::size_t s = 0; s < media.size(); ++s) {
        for (const double lambda_nm : wavelengths_nm) {
            try {
                t.n.push_back(material_index(media[s], lambda_nm, catalog));
            } catch (const GlassError& e) {
                throw GlassError("Medium nach Fläche " + std::to_string(s) + ", Wellenlänge " +
                                 detail::format_number(lambda_nm) + " nm: " + e.what());
            }
        }
    }
    return t;
}

} // namespace mko
