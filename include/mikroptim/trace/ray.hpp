#pragma once

#include "mikroptim/math/vec3.hpp"

namespace mko {

// Strahlzustand in lokalen Koordinaten der zuletzt getroffenen Fläche
// (Scheitel im Ursprung, z entlang der optischen Achse in Tracerichtung).
template <class R>
struct Ray {
    Vec3<R> p; // Position [mm]
    Vec3<R> d; // Richtung, normiert
    R opl;     // akkumulierte optische Weglänge [mm]
};

} // namespace mko
