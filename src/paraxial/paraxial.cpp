// Paraxiale Kenngrößen (docs/numerik.md, Abschnitt 9).

#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/paraxial/ynu_trace.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void check_finite(std::span<const double> values, const char* what) {
    for (std::size_t s = 0; s < values.size(); ++s) {
        if (!std::isfinite(values[s])) {
            throw std::invalid_argument(std::string("Paraxialtrace: ") + what + "[" +
                                        std::to_string(s) + "] ist nicht endlich");
        }
    }
}

struct SurfaceArrays {
    std::vector<double> c, t;
};

SurfaceArrays surface_arrays(const SystemDescription& system) {
    SurfaceArrays a;
    a.c.reserve(system.surfaces.size());
    a.t.reserve(system.surfaces.size());
    for (const Surface& s : system.surfaces) {
        a.c.push_back(s.type == SurfaceType::Sphere ? s.curvature : 0.0);
        a.t.push_back(s.thickness);
    }
    return a;
}

double stop_semi_diameter(const SystemDescription& system) {
    if (system.surfaces.empty()) {
        throw std::invalid_argument("Paraxialtrace: System ohne Flächen");
    }
    return system.surfaces.front().semi_diameter;
}

} // namespace

ParaxialData compute_paraxial(std::span<const double> curvatures,
                              std::span<const double> thicknesses, std::span<const double> indices,
                              double stop_semi_diameter) {
    if (curvatures.empty()) {
        throw std::invalid_argument("Paraxialtrace: keine Flächen");
    }
    if (thicknesses.size() != curvatures.size() || indices.size() != curvatures.size()) {
        throw std::invalid_argument(
            "Paraxialtrace: Krümmungen, Dicken und Brechzahlen müssen gleich lang sein");
    }
    check_finite(curvatures, "Krümmung");
    check_finite(thicknesses, "Dicke");
    check_finite(indices, "Brechzahl");
    for (std::size_t s = 0; s < indices.size(); ++s) {
        if (!(indices[s] > 0.0)) {
            throw std::invalid_argument("Paraxialtrace: Brechzahl[" + std::to_string(s) +
                                        "] muss > 0 sein");
        }
    }
    if (!(std::isfinite(stop_semi_diameter) && stop_semi_diameter > 0.0)) {
        throw std::invalid_argument("Paraxialtrace: Blendenradius muss endlich und > 0 sein");
    }

    const int count = static_cast<int>(curvatures.size());
    const double* c = curvatures.data();
    const double* t = thicknesses.data();
    const double* n = indices.data();
    const double n_image = n[count - 1];

    // Spalten der Systemmatrix: Randstrahl (y0 = 1, u0 = 0) und Hauptstrahl durch die
    // Blendenmitte (y0 = 0, u0 = 1); reduzierte Winkel n' u' im Bildraum.
    const ParaxialRay<double> marginal = paraxial_trace(c, t, n, count, ParaxialRay<double>{1, 0});
    const ParaxialRay<double> chief = paraxial_trace(c, t, n, count, ParaxialRay<double>{0, 1});
    const double A = marginal.y;
    const double C = n_image * marginal.u;
    const double B = chief.y;
    const double D = n_image * chief.u;

    ParaxialData p;
    p.abcd = {A, B, C, D};
    p.power = -C;
    p.afocal = (C == 0.0);
    if (p.afocal) {
        p.power = 0.0; // -0.0 vermeiden
        p.focal_length = kInf;
        p.back_focal_distance = kNaN;
        p.front_focal_distance = kNaN;
        p.front_principal_plane = kNaN;
        p.rear_principal_plane = kNaN;
        p.image_defocus = kNaN;
    } else {
        const double phi = p.power;
        p.focal_length = 1.0 / phi;
        // Schnittweite aus dem Randstrahl: Achsdurchgang z = -y_L / u'_L = n' A / phi.
        p.back_focal_distance = -marginal.y / marginal.u;
        // Strahl durch F (y0 = -z_F u0) verlässt das System parallel: C y0 + D u0 = 0.
        p.front_focal_distance = -D / phi;
        // H liegt um f hinter F, H' um n' f vor F' (Medium vor Fläche 0: Luft).
        p.front_principal_plane = (1.0 - D) / phi;
        p.rear_principal_plane = n_image * (A - 1.0) / phi;
        p.image_defocus = p.back_focal_distance - t[count - 1];
    }

    p.petzval_sum = petzval_sum(c, n, count);
    p.petzval_curvature = -n_image * p.petzval_sum;
    p.petzval_radius = p.petzval_curvature == 0.0 ? kInf : 1.0 / p.petzval_curvature;

    // Austrittspupille: Achsdurchgang des Hauptstrahls im Bildraum. Pupillenvergrößerung nach
    // Lagrange (n u)/(n' u') = 1/D; gleich der Randstrahlhöhe dort, weil det M = 1.
    p.exit_pupil_at_infinity = (D == 0.0);
    if (p.exit_pupil_at_infinity) {
        p.exit_pupil_position = kNaN;
        p.exit_pupil_magnification = kNaN;
        p.exit_pupil_semi_diameter = kInf;
    } else {
        p.exit_pupil_position = -chief.y / chief.u;
        p.exit_pupil_magnification = 1.0 / D;
        p.exit_pupil_semi_diameter = stop_semi_diameter * std::abs(p.exit_pupil_magnification);
    }
    return p;
}

ParaxialData compute_paraxial(const SystemDescription& system, const IndexTable& indices,
                              int wavelength_index) {
    const double stop = stop_semi_diameter(system);
    const std::size_t count = system.surfaces.size();
    if (indices.surfaces != static_cast<int>(count) || indices.wavelengths < 1 ||
        indices.n.size() != count * static_cast<std::size_t>(indices.wavelengths)) {
        throw std::invalid_argument(
            "Paraxialtrace: Brechzahltabelle passt nicht zur Flächenzahl des Systems");
    }
    if (wavelength_index < 0 || wavelength_index >= indices.wavelengths) {
        throw std::out_of_range("Paraxialtrace: Wellenlängenindex " +
                                std::to_string(wavelength_index) + " außerhalb von [0, " +
                                std::to_string(indices.wavelengths) + ")");
    }
    const SurfaceArrays a = surface_arrays(system);
    std::vector<double> n(count);
    for (std::size_t s = 0; s < count; ++s) {
        n[s] = indices.at(static_cast<int>(s), wavelength_index);
    }
    return compute_paraxial(a.c, a.t, n, stop);
}

ParaxialData compute_paraxial(const SystemDescription& system, const GlassCatalog& catalog,
                              double wavelength_nm) {
    const double stop = stop_semi_diameter(system);
    const SurfaceArrays a = surface_arrays(system);
    std::vector<double> n;
    n.reserve(system.surfaces.size());
    for (std::size_t s = 0; s < system.surfaces.size(); ++s) {
        try {
            n.push_back(material_index(system.surfaces[s].material, wavelength_nm, catalog));
        } catch (const GlassError& e) {
            throw GlassError("Medium nach Fläche " + std::to_string(s) + ": " + e.what());
        }
    }
    return compute_paraxial(a.c, a.t, n, stop);
}

ParaxialData compute_paraxial(const SystemDescription& system, const GlassCatalog& catalog) {
    return compute_paraxial(system, catalog, system.spectrum.reference_nm);
}

} // namespace mko
