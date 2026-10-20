// Inhaltliche Validierung der Systembeschreibung (docs/dateiformat.md, „Einlesen und
// Validierung“). Die Pfade in den Meldungen entsprechen den JSON-Pfaden der Datei.

#include "mikroptim/model/system.hpp"

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

std::string num(double v) {
    std::ostringstream os;
    os.precision(15);
    os << v;
    return os.str();
}

std::string at(const std::string& path, std::size_t i) {
    return path + "[" + std::to_string(i) + "]";
}

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw FormatError(path, message);
}

void require_finite(const std::string& path, double v) {
    if (!std::isfinite(v)) {
        fail(path, "Zahl muss endlich sein");
    }
}

void require_positive(const std::string& path, double v) {
    require_finite(path, v);
    if (!(v > 0.0)) {
        fail(path, "muss größer als 0 sein, ist " + num(v));
    }
}

// Gewichte: gleiche Länge wie die Bezugsliste, jedes ≥ 0, Summe > 0.
void check_weights(const std::string& path, const std::vector<double>& weights,
                   const std::string& ref_path, std::size_t ref_size) {
    if (weights.size() != ref_size) {
        fail(path, "Länge " + std::to_string(weights.size()) + " passt nicht zu " + ref_path +
                       " (Länge " + std::to_string(ref_size) + ")");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        require_finite(at(path, i), weights[i]);
        if (weights[i] < 0.0) {
            fail(at(path, i), "Gewicht darf nicht negativ sein, ist " + num(weights[i]));
        }
        sum += weights[i];
    }
    if (!(sum > 0.0)) {
        fail(path, "Summe der Gewichte muss größer als 0 sein");
    }
}

void check_spectrum(const Spectrum& s) {
    const std::string wl = "spectrum.wavelengths_nm";
    if (s.wavelengths_nm.empty()) {
        fail(wl, "mindestens eine Wellenlänge nötig");
    }
    for (std::size_t i = 0; i < s.wavelengths_nm.size(); ++i) {
        require_positive(at(wl, i), s.wavelengths_nm[i]);
    }
    check_weights("spectrum.weights", s.weights, wl, s.wavelengths_nm.size());
    require_positive("spectrum.reference_nm", s.reference_nm);
}

void check_source(const Source& s) {
    if (s.type == SourceType::Collimated) {
        const std::string fa = "source.field_angles_deg";
        if (s.field_angles_deg.empty()) {
            fail(fa, "mindestens ein Feldwinkel nötig");
        }
        for (std::size_t i = 0; i < s.field_angles_deg.size(); ++i) {
            require_finite(at(fa, i), s.field_angles_deg[i]);
            if (!(std::abs(s.field_angles_deg[i]) < 90.0)) {
                fail(at(fa, i), "Feldwinkel muss betragsmäßig unter 90° liegen, ist " +
                                    num(s.field_angles_deg[i]));
            }
        }
        check_weights("source.field_weights", s.field_weights, fa, s.field_angles_deg.size());
        if (s.point != std::array<double, 3>{0.0, 0.0, 0.0}) {
            fail("source.point", "nur bei source.type \"point\" erlaubt");
        }
        if (s.numerical_aperture != 0.0) {
            fail("source.numerical_aperture", "nur bei source.type \"point\" erlaubt");
        }
    } else {
        for (std::size_t k = 0; k < 3; ++k) {
            require_finite(at("source.point", k), s.point[k]);
        }
        require_positive("source.numerical_aperture", s.numerical_aperture);
        if (!s.field_angles_deg.empty()) {
            fail("source.field_angles_deg", "nur bei source.type \"collimated\" erlaubt");
        }
        if (!s.field_weights.empty()) {
            fail("source.field_weights", "nur bei source.type \"collimated\" erlaubt");
        }
    }
}

void check_target(const Target& t) {
    if (t.focal_length) {
        require_finite("target.focal_length", *t.focal_length);
        if (*t.focal_length == 0.0) {
            fail("target.focal_length", "darf nicht 0 sein");
        }
    }
    if (t.parfocal_length) {
        require_positive("target.parfocal_length", *t.parfocal_length);
    }
    if (t.working_distance) {
        require_positive("target.working_distance", *t.working_distance);
    }
}

void check_material(const std::string& path, const MaterialSpec& m) {
    switch (m.kind) {
    case MaterialKind::Air:
        break;
    case MaterialKind::Catalog:
        if (m.name.empty()) {
            fail(path, "Glasname darf nicht leer sein");
        }
        if (m.name == "AIR") {
            fail(path, "\"AIR\" ist kein Katalogglas");
        }
        break;
    case MaterialKind::Constant:
        require_positive(path + ".n", m.n);
        break;
    case MaterialKind::Buchdahl:
        require_positive(path + ".n0", m.n0);
        for (std::size_t k = 0; k < m.nu.size(); ++k) {
            require_finite(at(path + ".nu", k), m.nu[k]);
        }
        break;
    }
}

void check_surfaces(const std::vector<Surface>& surfaces) {
    if (surfaces.size() < 2) {
        fail("surfaces", "mindestens zwei Flächen nötig (Blende und eine weitere), vorhanden: " +
                             std::to_string(surfaces.size()));
    }
    for (std::size_t i = 0; i < surfaces.size(); ++i) {
        const Surface& s = surfaces[i];
        const std::string p = at("surfaces", i);
        if (i == 0 && s.type != SurfaceType::Stop) {
            fail(p + ".type",
                 "die erste Fläche muss \"stop\" sein (Aperturblende, ADR-0006), ist \"" +
                     std::string(to_string(s.type)) + "\"");
        }
        if (i > 0 && s.type == SurfaceType::Stop) {
            fail(p + ".type", "\"stop\" ist nur als erste Fläche erlaubt");
        }
        require_finite(p + ".curvature", s.curvature);
        if (s.type != SurfaceType::Sphere && s.curvature != 0.0) {
            fail(p + ".curvature", "Krümmung ist nur bei \"sphere\" erlaubt");
        }
        require_finite(p + ".thickness", s.thickness);
        check_material(p + ".material", s.material);
        if (i == 0 && s.material.kind != MaterialKind::Air) {
            fail(p + ".material", "die Blende ist eine Ebene in Luft, Material muss \"AIR\" sein");
        }
        require_positive(p + ".semi_diameter", s.semi_diameter);
        // Dezentrierung und Verkippung (M8-T3, ADR-0011): endlich, |Winkel| < 90°, nicht an
        // der Blende (sie legt das Bezugssystem und die Pupille fest)
        for (std::size_t k = 0; k < 2; ++k) {
            require_finite(at(p + ".decenter", k), s.decenter[k]);
            require_finite(at(p + ".tilt_deg", k), s.tilt_deg[k]);
            if (!(std::abs(s.tilt_deg[k]) < 90.0)) {
                fail(at(p + ".tilt_deg", k), "Kippwinkel muss betragsmäßig kleiner als 90° sein");
            }
        }
        if (i == 0 && is_transformed(s)) {
            fail(p, "die Blende (Fläche 0) ist das Bezugssystem und darf nicht dezentriert oder "
                    "verkippt werden");
        }
    }
}

// Geisterplatten (docs/dateiformat.md, „Geisterplatten“): Luftspalt nach after_surface (nicht
// nach der letzten Fläche), 0 <= distance <= Dicke des Spalts, Glas (nicht Luft),
// norm_radius > 0, 1 <= K <= kMaxGhostTerms endliche Koeffizienten, sortiert nach
// (after_surface, distance).
void check_ghost_plates(const std::vector<GhostPlate>& plates,
                        const std::vector<Surface>& surfaces) {
    for (std::size_t g = 0; g < plates.size(); ++g) {
        const GhostPlate& gp = plates[g];
        const std::string p = at("ghost_plates", g);
        const int last = static_cast<int>(surfaces.size()) - 1;
        if (gp.after_surface < 0 || gp.after_surface > last) {
            fail(p + ".after_surface", "Fläche " + std::to_string(gp.after_surface) +
                                           " existiert nicht (erlaubt: 0 bis " +
                                           std::to_string(last - 1) + ")");
        }
        if (gp.after_surface == last) {
            fail(p + ".after_surface",
                 "nach der letzten Fläche liegt der Arbeitsabstand; Geisterplatten sind dort "
                 "nicht erlaubt (erlaubt: 0 bis " +
                     std::to_string(last - 1) + ")");
        }
        const Surface& gap = surfaces[static_cast<std::size_t>(gp.after_surface)];
        if (gap.material.kind != MaterialKind::Air) {
            fail(p + ".after_surface", "Geisterplatten liegen in einem Luftspalt; das Medium nach "
                                       "Fläche " +
                                           std::to_string(gp.after_surface) + " ist nicht \"AIR\"");
        }
        require_finite(p + ".distance", gp.distance);
        if (gp.distance < 0.0 || gp.distance > gap.thickness) {
            fail(p + ".distance", "muss zwischen 0 und der Dicke des Luftspalts (" +
                                      num(gap.thickness) + ") liegen, ist " + num(gp.distance));
        }
        check_material(p + ".glass", gp.glass);
        if (gp.glass.kind == MaterialKind::Air) {
            fail(p + ".glass", "Geisterplatte braucht ein Glas, \"AIR\" ist nicht erlaubt");
        }
        require_positive(p + ".norm_radius", gp.norm_radius);
        const std::string cp = p + ".coefficients";
        if (gp.coefficients.empty() ||
            gp.coefficients.size() > static_cast<std::size_t>(kMaxGhostTerms)) {
            fail(cp, "Ordnung (Zahl der Koeffizienten) muss zwischen 1 und " +
                         std::to_string(kMaxGhostTerms) + " liegen, ist " +
                         std::to_string(gp.coefficients.size()));
        }
        for (std::size_t k = 0; k < gp.coefficients.size(); ++k) {
            require_finite(at(cp, k), gp.coefficients[k]);
        }
        if (g > 0) {
            const GhostPlate& prev = plates[g - 1];
            if (gp.after_surface < prev.after_surface ||
                (gp.after_surface == prev.after_surface && gp.distance < prev.distance)) {
                fail(p, "Geisterplatten müssen nach after_surface und distance aufsteigend "
                        "sortiert sein");
            }
        }
    }
}

} // namespace

bool is_transformed(const Surface& surface) {
    return surface.decenter[0] != 0.0 || surface.decenter[1] != 0.0 || surface.tilt_deg[0] != 0.0 ||
           surface.tilt_deg[1] != 0.0;
}

bool is_coaxial(const SystemDescription& system) {
    for (const Surface& s : system.surfaces) {
        if (is_transformed(s)) {
            return false;
        }
    }
    return true;
}

std::string_view to_string(SurfaceType type) {
    switch (type) {
    case SurfaceType::Stop:
        return "stop";
    case SurfaceType::Plane:
        return "plane";
    case SurfaceType::Sphere:
        return "sphere";
    }
    return "?";
}

std::string_view to_string(SourceType type) {
    switch (type) {
    case SourceType::Collimated:
        return "collimated";
    case SourceType::Point:
        return "point";
    }
    return "?";
}

void validate_system(const SystemDescription& system) {
    if (system.format != kSystemFormat) {
        fail("format",
             "erwartet \"" + std::string(kSystemFormat) + "\", ist \"" + system.format + "\"");
    }
    if (system.name.empty()) {
        fail("name", "darf nicht leer sein");
    }
    if (system.units != "mm") {
        fail("units", "nur \"mm\" ist erlaubt, ist \"" + system.units + "\"");
    }
    for (std::size_t i = 0; i < system.glass_catalogs.size(); ++i) {
        if (system.glass_catalogs[i].empty()) {
            fail(at("glass_catalogs", i), "Pfad darf nicht leer sein");
        }
    }
    check_spectrum(system.spectrum);
    check_source(system.source);
    if (system.target) {
        check_target(*system.target);
    }
    check_surfaces(system.surfaces);
    check_ghost_plates(system.ghost_plates, system.surfaces);
    (void)resolve_variables(system); // wirft FormatError unter variables.items[i]…
    check_constraints(system);       // wirft FormatError unter constraints.items[i]…
}

void validate_materials(const SystemDescription& system,
                        const std::function<bool(std::string_view)>& glass_exists) {
    for (std::size_t i = 0; i < system.surfaces.size(); ++i) {
        const MaterialSpec& m = system.surfaces[i].material;
        if (m.kind == MaterialKind::Catalog && !glass_exists(m.name)) {
            fail(at("surfaces", i) + ".material",
                 "unbekanntes Glas \"" + m.name + "\" (in keinem der Glaskataloge enthalten)");
        }
    }
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        const MaterialSpec& m = system.ghost_plates[g].glass;
        if (m.kind == MaterialKind::Catalog && !glass_exists(m.name)) {
            fail(at("ghost_plates", g) + ".glass",
                 "unbekanntes Glas \"" + m.name + "\" (in keinem der Glaskataloge enthalten)");
        }
    }
}

int table_surface_count(const SystemDescription& system) {
    return static_cast<int>(system.surfaces.size() + system.ghost_plates.size());
}

SurfaceMap make_surface_map(const SystemDescription& system) {
    const int files = static_cast<int>(system.surfaces.size());
    SurfaceMap m;
    m.table_of_file.reserve(system.surfaces.size());
    m.table_of_ghost.reserve(system.ghost_plates.size());
    std::size_t g = 0;
    for (int i = 0; i < files; ++i) {
        m.table_of_file.push_back(static_cast<int>(m.file_of_table.size()));
        m.file_of_table.push_back(i);
        m.ghost_of_table.push_back(-1);
        for (; g < system.ghost_plates.size() && system.ghost_plates[g].after_surface == i; ++g) {
            const GhostPlate& gp = system.ghost_plates[g];
            if (i == files - 1 || (g > 0 && system.ghost_plates[g - 1].after_surface == i &&
                                   gp.distance < system.ghost_plates[g - 1].distance)) {
                throw std::invalid_argument("Flächenabbildung: Geisterplatte " + std::to_string(g) +
                                            " liegt nach der letzten Fläche oder ist nicht "
                                            "nach distance sortiert");
            }
            m.table_of_ghost.push_back(static_cast<int>(m.file_of_table.size()));
            m.file_of_table.push_back(-1);
            m.ghost_of_table.push_back(static_cast<int>(g));
        }
    }
    if (g != system.ghost_plates.size()) {
        throw std::invalid_argument("Flächenabbildung: Geisterplatte " + std::to_string(g) +
                                    " hat eine ungültige Fläche after_surface oder ist nicht "
                                    "nach after_surface sortiert");
    }
    return m;
}

int SurfaceMap::file_surface(int table_surface) const {
    const int n = table_surfaces();
    if (table_surface < 0 || table_surface > n) {
        throw std::out_of_range("Flächenabbildung: Tabellenfläche " +
                                std::to_string(table_surface) + " außerhalb von [0, " +
                                std::to_string(n) + "]");
    }
    return table_surface == n ? file_surfaces()
                              : file_of_table[static_cast<std::size_t>(table_surface)];
}

int SurfaceMap::table_surface(int file_surface) const {
    const int n = file_surfaces();
    if (file_surface < 0 || file_surface > n) {
        throw std::out_of_range("Flächenabbildung: Dateifläche " + std::to_string(file_surface) +
                                " außerhalb von [0, " + std::to_string(n) + "]");
    }
    return file_surface == n ? table_surfaces()
                             : table_of_file[static_cast<std::size_t>(file_surface)];
}

} // namespace mko
