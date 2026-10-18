// Flächentabelle auf dem Host aus Systembeschreibung und Brechzahltabelle.

#include "mikroptim/cpu/host_surface_table.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko {

namespace {

SurfaceKind to_kind(SurfaceType type) {
    switch (type) {
    case SurfaceType::Stop:
        return SurfaceKind::Stop;
    case SurfaceType::Plane:
        return SurfaceKind::Plane;
    case SurfaceType::Sphere:
        return SurfaceKind::Sphere;
    }
    throw std::invalid_argument("Flächentabelle: unbekannter Flächentyp");
}

std::size_t to_size(int n) {
    return static_cast<std::size_t>(n);
}

} // namespace

IndexTable make_system_index_table(const SystemDescription& system, const GlassCatalog& catalog) {
    std::vector<MaterialSpec> media;
    media.reserve(system.surfaces.size());
    for (const Surface& s : system.surfaces) {
        media.push_back(s.material);
    }
    IndexTable t = make_index_table(media, system.spectrum.wavelengths_nm, catalog);
    t.ghost_plates = static_cast<int>(system.ghost_plates.size());
    t.ghost_n.reserve(system.ghost_plates.size() * system.spectrum.wavelengths_nm.size());
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        for (const double lambda_nm : system.spectrum.wavelengths_nm) {
            try {
                t.ghost_n.push_back(
                    material_index(system.ghost_plates[g].glass, lambda_nm, catalog));
            } catch (const GlassError& e) {
                throw GlassError("Glas der Geisterplatte " + std::to_string(g) + ", Wellenlänge " +
                                 std::to_string(lambda_nm) + " nm: " + e.what());
            }
        }
    }
    // Brechzahlen bei λ_ref (Paraxialrechnung): derselbe Ausdruck wie in den Spalten, daher
    // bitgleich zur Spalte, wenn λ_ref unter den Wellenlängen ist.
    const double ref = system.spectrum.reference_nm;
    t.reference_nm = ref;
    t.reference_n.reserve(system.surfaces.size());
    for (std::size_t s = 0; s < system.surfaces.size(); ++s) {
        try {
            t.reference_n.push_back(material_index(system.surfaces[s].material, ref, catalog));
        } catch (const GlassError& e) {
            throw GlassError("Medium nach Fläche " + std::to_string(s) + ", Referenzwellenlänge " +
                             std::to_string(ref) + " nm: " + e.what());
        }
    }
    t.reference_ghost_n.reserve(system.ghost_plates.size());
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        try {
            t.reference_ghost_n.push_back(
                material_index(system.ghost_plates[g].glass, ref, catalog));
        } catch (const GlassError& e) {
            throw GlassError("Glas der Geisterplatte " + std::to_string(g) +
                             ", Referenzwellenlänge " + std::to_string(ref) + " nm: " + e.what());
        }
    }
    return t;
}

HostSurfaceTable make_surface_table(const SystemDescription& system, const IndexTable& index) {
    const std::size_t count = system.surfaces.size();
    const std::size_t ghosts = system.ghost_plates.size();
    const std::size_t total = count + ghosts;
    if (count < 2 || total > static_cast<std::size_t>(kMaxSurfaces)) {
        throw std::invalid_argument("Flächentabelle: " + std::to_string(count) + " Flächen und " +
                                    std::to_string(ghosts) +
                                    " Geisterplatten, erlaubt sind 2 bis " +
                                    std::to_string(kMaxSurfaces) + " Tabellenflächen");
    }
    if (index.surfaces != static_cast<int>(count)) {
        throw std::invalid_argument("Flächentabelle: Brechzahltabelle hat " +
                                    std::to_string(index.surfaces) + " Flächen, das System " +
                                    std::to_string(count));
    }
    if (index.wavelengths < 1 || index.n.size() != count * to_size(index.wavelengths)) {
        throw std::invalid_argument("Flächentabelle: Brechzahltabelle ohne Wellenlängen oder "
                                    "mit falscher Größe");
    }
    const std::size_t L = to_size(index.wavelengths);
    if (index.ghost_plates != static_cast<int>(ghosts) || index.ghost_n.size() != ghosts * L) {
        throw std::invalid_argument("Flächentabelle: Brechzahltabelle hat Brechzahlen für " +
                                    std::to_string(index.ghost_plates) +
                                    " Geisterplatten, das System " + std::to_string(ghosts) +
                                    " (make_system_index_table verwenden)");
    }
    const SurfaceMap map = make_surface_map(system);

    HostSurfaceTable t;
    t.surfaces = static_cast<int>(total);
    t.wavelengths = index.wavelengths;
    t.type.reserve(total);
    t.curvature.reserve(total);
    t.thickness.reserve(total);
    t.semi_diameter.reserve(total);
    t.index.reserve(total * L);
    for (std::size_t j = 0; j < total; ++j) {
        const int file = map.file_of_table[j];
        const GhostPlate* plate =
            file >= 0 ? nullptr : &system.ghost_plates[to_size(map.ghost_of_table[j])];
        // Luftspalt bzw. Medium nach der Tabellenfläche: Dateifläche selbst oder die Fläche,
        // nach der die Geisterplatte liegt.
        const std::size_t gap = to_size(file >= 0 ? file : plate->after_surface);
        if (plate == nullptr) {
            const Surface& s = system.surfaces[gap];
            const SurfaceKind kind = to_kind(s.type);
            t.type.push_back(kind);
            t.curvature.push_back(kind == SurfaceKind::Sphere ? s.curvature : 0.0);
            t.semi_diameter.push_back(s.semi_diameter);
        } else {
            t.type.push_back(SurfaceKind::GhostPlate);
            t.curvature.push_back(0.0);
            t.semi_diameter.push_back(plate->norm_radius);
        }
        // Dicke: von der Lage dieser Fläche im Spalt (0 bzw. distance) bis zur nächsten
        // Geisterplatte desselben Spalts oder bis zum Ende des Spalts. Ohne Geisterplatten im
        // Spalt ist das exakt die Dicke aus der Datei (T − 0).
        const double start = plate == nullptr ? 0.0 : plate->distance;
        const bool next_is_ghost = j + 1 < total && map.ghost_of_table[j + 1] >= 0;
        const double end = next_is_ghost
                               ? system.ghost_plates[to_size(map.ghost_of_table[j + 1])].distance
                               : system.surfaces[gap].thickness;
        t.thickness.push_back(end - start);
        for (std::size_t l = 0; l < L; ++l) {
            t.index.push_back(index.n[gap * L + l]);
        }
    }

    t.ghost_plates = static_cast<int>(ghosts);
    for (const GhostPlate& gp : system.ghost_plates) {
        t.ghost_terms = std::max(t.ghost_terms, static_cast<int>(gp.coefficients.size()));
    }
    const std::size_t K = to_size(t.ghost_terms);
    t.ghost_coeff.assign(ghosts * K, 0.0);
    t.ghost_norm_radius.reserve(ghosts);
    for (std::size_t g = 0; g < ghosts; ++g) {
        const std::vector<double>& a = system.ghost_plates[g].coefficients;
        std::copy(a.begin(), a.end(), t.ghost_coeff.begin() + static_cast<std::ptrdiff_t>(g * K));
        t.ghost_norm_radius.push_back(system.ghost_plates[g].norm_radius);
    }
    t.ghost_index = index.ghost_n;
    return t;
}

} // namespace mko
