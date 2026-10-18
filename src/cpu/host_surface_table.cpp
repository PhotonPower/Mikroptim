// Flächentabelle auf dem Host aus Systembeschreibung und Brechzahltabelle.

#include "mikroptim/cpu/host_surface_table.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

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

} // namespace

HostSurfaceTable make_surface_table(const SystemDescription& system, const IndexTable& index) {
    const std::size_t count = system.surfaces.size();
    if (count < 2 || count > static_cast<std::size_t>(kMaxSurfaces)) {
        throw std::invalid_argument("Flächentabelle: " + std::to_string(count) +
                                    " Flächen, erlaubt sind 2 bis " + std::to_string(kMaxSurfaces));
    }
    if (index.surfaces != static_cast<int>(count)) {
        throw std::invalid_argument("Flächentabelle: Brechzahltabelle hat " +
                                    std::to_string(index.surfaces) + " Flächen, das System " +
                                    std::to_string(count));
    }
    if (index.wavelengths < 1 ||
        index.n.size() != count * static_cast<std::size_t>(index.wavelengths)) {
        throw std::invalid_argument("Flächentabelle: Brechzahltabelle ohne Wellenlängen oder "
                                    "mit falscher Größe");
    }

    HostSurfaceTable t;
    t.surfaces = static_cast<int>(count);
    t.wavelengths = index.wavelengths;
    t.type.reserve(count);
    t.curvature.reserve(count);
    t.thickness.reserve(count);
    t.semi_diameter.reserve(count);
    for (const Surface& s : system.surfaces) {
        const SurfaceKind kind = to_kind(s.type);
        t.type.push_back(kind);
        t.curvature.push_back(kind == SurfaceKind::Sphere ? s.curvature : 0.0);
        t.thickness.push_back(s.thickness);
        t.semi_diameter.push_back(s.semi_diameter);
    }
    t.index = index.n;
    return t;
}

} // namespace mko
