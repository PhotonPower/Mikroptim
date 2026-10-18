// Paraxiale Kenngrößen (docs/numerik.md, Abschnitt 9).

#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/paraxial/ynu_trace.hpp"
#include "mikroptim/trace/ghost_plate.hpp"

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

// Flächendaten über die Tabellenflächen (Dateiflächen mit eingefügten Geisterplatten,
// make_surface_map): Krümmung, Dicke, Brechzahl nach der Fläche, Brechkraft und
// Petzvalbeitrag der Geisterplatten (0 bei Dateiflächen).
struct SurfaceArrays {
    std::vector<double> c, t, n, power, petzval;
};

// `file_index(i)` liefert die Brechzahl nach Dateifläche i, `ghost_index(g)` die des Glases
// der Geisterplatte g.
template <class FileIndex, class GhostIndex>
SurfaceArrays surface_arrays(const SystemDescription& system, FileIndex file_index,
                             GhostIndex ghost_index) {
    const SurfaceMap map = make_surface_map(system);
    const std::size_t total = static_cast<std::size_t>(map.table_surfaces());
    SurfaceArrays a;
    a.c.reserve(total);
    a.t.reserve(total);
    a.n.reserve(total);
    a.power.assign(total, 0.0);
    a.petzval.assign(total, 0.0);
    for (std::size_t j = 0; j < total; ++j) {
        const int file = map.file_of_table[j];
        const GhostPlate* plate =
            file >= 0 ? nullptr
                      : &system.ghost_plates[static_cast<std::size_t>(map.ghost_of_table[j])];
        const int gap = file >= 0 ? file : plate->after_surface;
        const Surface& s = system.surfaces[static_cast<std::size_t>(gap)];
        const double n_medium = file_index(gap);
        a.n.push_back(n_medium);
        // Dicke wie make_surface_table: von der Lage im Spalt bis zur nächsten Geisterplatte
        // desselben Spalts bzw. bis zum Ende des Spalts.
        const double start = plate == nullptr ? 0.0 : plate->distance;
        const bool next_is_ghost = j + 1 < total && map.ghost_of_table[j + 1] >= 0;
        const double end =
            next_is_ghost
                ? system.ghost_plates[static_cast<std::size_t>(map.ghost_of_table[j + 1])].distance
                : s.thickness;
        a.t.push_back(end - start);
        if (plate == nullptr) {
            a.c.push_back(s.type == SurfaceType::Sphere ? s.curvature : 0.0);
            continue;
        }
        a.c.push_back(0.0);
        const double n_glass = ghost_index(map.ghost_of_table[j]);
        const int K = static_cast<int>(plate->coefficients.size());
        a.power[j] =
            ghost_plate_power(plate->coefficients.data(), K, plate->norm_radius, n_glass, n_medium);
        a.petzval[j] = ghost_plate_petzval(plate->coefficients.data(), K, plate->norm_radius,
                                           n_glass, n_medium);
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
    return compute_paraxial(curvatures, thicknesses, indices, stop_semi_diameter, {}, {});
}

ParaxialData compute_paraxial(std::span<const double> curvatures,
                              std::span<const double> thicknesses, std::span<const double> indices,
                              double stop_semi_diameter, std::span<const double> power,
                              std::span<const double> petzval) {
    if (curvatures.empty()) {
        throw std::invalid_argument("Paraxialtrace: keine Flächen");
    }
    if (thicknesses.size() != curvatures.size() || indices.size() != curvatures.size()) {
        throw std::invalid_argument(
            "Paraxialtrace: Krümmungen, Dicken und Brechzahlen müssen gleich lang sein");
    }
    if ((!power.empty() && power.size() != curvatures.size()) ||
        (!petzval.empty() && petzval.size() != curvatures.size())) {
        throw std::invalid_argument("Paraxialtrace: Brechkräfte und Petzvalbeiträge der dünnen "
                                    "Elemente müssen so lang sein wie die Krümmungen");
    }
    check_finite(curvatures, "Krümmung");
    check_finite(thicknesses, "Dicke");
    check_finite(indices, "Brechzahl");
    check_finite(power, "Brechkraft");
    check_finite(petzval, "Petzvalbeitrag");
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
    const double* extra = power.empty() ? nullptr : power.data();
    ParaxialRay<double>* const no_record = nullptr;
    const ParaxialRay<double> marginal =
        paraxial_trace(c, t, n, count, ParaxialRay<double>{1, 0}, no_record, extra);
    const ParaxialRay<double> chief =
        paraxial_trace(c, t, n, count, ParaxialRay<double>{0, 1}, no_record, extra);
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

    p.petzval_sum = petzval_sum(c, n, count, petzval.empty() ? nullptr : petzval.data());
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
    const std::size_t ghosts = system.ghost_plates.size();
    const std::size_t L =
        indices.wavelengths < 1 ? 0 : static_cast<std::size_t>(indices.wavelengths);
    if (indices.surfaces != static_cast<int>(count) || L == 0 || indices.n.size() != count * L ||
        indices.ghost_plates != static_cast<int>(ghosts) || indices.ghost_n.size() != ghosts * L) {
        throw std::invalid_argument("Paraxialtrace: Brechzahltabelle passt nicht zur Flächen- "
                                    "oder Geisterplattenzahl des Systems");
    }
    if (wavelength_index < 0 || wavelength_index >= indices.wavelengths) {
        throw std::out_of_range("Paraxialtrace: Wellenlängenindex " +
                                std::to_string(wavelength_index) + " außerhalb von [0, " +
                                std::to_string(indices.wavelengths) + ")");
    }
    const SurfaceArrays a = surface_arrays(
        system, [&](int s) { return indices.at(s, wavelength_index); },
        [&](int g) { return indices.ghost_at(g, wavelength_index); });
    return compute_paraxial(a.c, a.t, a.n, stop, a.power, a.petzval);
}

ParaxialData compute_paraxial(const SystemDescription& system, const GlassCatalog& catalog,
                              double wavelength_nm) {
    const double stop = stop_semi_diameter(system);
    std::vector<double> n;
    n.reserve(system.surfaces.size());
    for (std::size_t s = 0; s < system.surfaces.size(); ++s) {
        try {
            n.push_back(material_index(system.surfaces[s].material, wavelength_nm, catalog));
        } catch (const GlassError& e) {
            throw GlassError("Medium nach Fläche " + std::to_string(s) + ": " + e.what());
        }
    }
    std::vector<double> n_ghost;
    n_ghost.reserve(system.ghost_plates.size());
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        try {
            n_ghost.push_back(material_index(system.ghost_plates[g].glass, wavelength_nm, catalog));
        } catch (const GlassError& e) {
            throw GlassError("Glas der Geisterplatte " + std::to_string(g) + ": " + e.what());
        }
    }
    const SurfaceArrays a = surface_arrays(
        system, [&](int s) { return n[static_cast<std::size_t>(s)]; },
        [&](int g) { return n_ghost[static_cast<std::size_t>(g)]; });
    return compute_paraxial(a.c, a.t, a.n, stop, a.power, a.petzval);
}

ParaxialData compute_paraxial(const SystemDescription& system, const GlassCatalog& catalog) {
    return compute_paraxial(system, catalog, system.spectrum.reference_nm);
}

} // namespace mko
