#pragma once

// Verletzung der Sinusbedingung (OSC, Conrady) im Unendlichraum aus einem realen meridionalen
// Strahl im Endraum und den Paraxialdaten (docs/numerik.md, Abschnitt 14). Eine Formel für den
// Aberrationsbericht (M10-T1b) und das Residuum `sine_condition` (M11-T4).

#include "mikroptim/math/vec3.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#include <cmath>
#include <limits>

namespace mko {

// OSC eines realen Strahls, der parallel zur Achse in der Höhe H = ρ a einfällt. `p` und `d`
// sind Schnittpunkt und Richtung nach der letzten Fläche in deren lokalen Koordinaten, `pd` die
// Paraxialdaten derselben Flächen bei derselben Wellenlänge, n_last die Brechzahl des Endraums.
//
//   OSC = (f_real / f′) · (l′ − l′_P) / (L′ − l′_P) − 1,   f_real = −H / sin U′,
//
// mit der bildseitigen Brennweite f′ = n′_L / φ (= −a / u′_L des paraxialen Randstrahls), der
// paraxialen Schnittweite l′, der Lage l′_P der Austrittspupille (beide ab der letzten Fläche)
// und der realen Schnittweite L′ = z − y d_z / d_y; bildseitig telezentrisch entfällt der zweite
// Faktor. NaN, wenn der Strahl die Achse nicht schneidet (d_y = 0) oder das System afokal ist.
inline double conrady_osc(double height, const Vec3<double>& p, const Vec3<double>& d,
                          const ParaxialData& pd, double n_last) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (d.y == 0.0 || pd.afocal || !std::isfinite(pd.focal_length)) {
        return nan;
    }
    const double f_image = n_last * pd.focal_length;
    const double f_real = -height / d.y;
    double factor = 1.0;
    if (!pd.exit_pupil_at_infinity) {
        const double L = p.z - p.y * d.z / d.y;
        factor = (pd.back_focal_distance - pd.exit_pupil_position) / (L - pd.exit_pupil_position);
    }
    return f_real / f_image * factor - 1.0;
}

} // namespace mko
