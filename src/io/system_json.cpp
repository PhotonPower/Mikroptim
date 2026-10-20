// Einlesen und Schreiben von `mikroptim.system/1` (docs/dateiformat.md).
// nlohmann/json wird nur hier verwendet; das Datenmodell bleibt frei von JSON-Typen.

#include "mikroptim/io/system_json.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace mko {

namespace {

// ordered_json behält die Feldreihenfolge, auch in den unverändert durchgereichten Abschnitten.
using Json = nlohmann::ordered_json;

// Maximale Zeilenlänge, bis zu der Listen und Objekte beim Schreiben einzeilig bleiben.
constexpr std::size_t kLineWidth = 120;

std::string join(const std::string& path, std::string_view key) {
    return path.empty() ? std::string(key) : path + "." + std::string(key);
}

std::string at(const std::string& path, std::size_t i) {
    return path + "[" + std::to_string(i) + "]";
}

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw FormatError(path, message);
}

std::string kind_name(const Json& j) {
    switch (j.type()) {
    case Json::value_t::object:
        return "Objekt";
    case Json::value_t::array:
        return "Liste";
    case Json::value_t::string:
        return "Zeichenkette";
    case Json::value_t::boolean:
        return "Wahrheitswert";
    case Json::value_t::null:
        return "null";
    case Json::value_t::number_integer:
    case Json::value_t::number_unsigned:
    case Json::value_t::number_float:
        return "Zahl";
    default:
        return "unbekannter Typ";
    }
}

[[noreturn]] void fail_type(const std::string& path, const char* expected, const Json& found) {
    fail(path,
         std::string("falscher Typ: erwartet ") + expected + ", gefunden " + kind_name(found));
}

std::string display(const std::filesystem::path& p) {
    const std::u8string s = p.u8string();
    return std::string(s.begin(), s.end());
}

// --------------------------------------------------------------------------
// Einlesen
// --------------------------------------------------------------------------

class Reader {
public:
    explicit Reader(std::vector<std::string>* warnings) : warnings_(warnings) {}

    SystemDescription read(const Json& root) {
        expect_object(root, "");
        SystemDescription sys;

        // Format zuerst prüfen: Dateien anderer Hauptversion haben evtl. andere Felder.
        sys.format = read_format(required(root, "", "format"));
        sys.name = as_string(required(root, "", "name"), "name");
        if (const Json* d = member(root, "description")) {
            sys.description = as_string(*d, "description");
        }
        sys.units = as_string(required(root, "", "units"), "units");
        sys.glass_catalogs = read_glass_catalogs(required(root, "", "glass_catalogs"));
        sys.spectrum = read_spectrum(required(root, "", "spectrum"), "spectrum");
        sys.source = read_source(required(root, "", "source"), "source");
        if (const Json* t = member(root, "target")) {
            sys.target = read_target(*t, "target");
        }
        sys.surfaces = read_surfaces(required(root, "", "surfaces"), "surfaces");

        if (const Json* g = member(root, "ghost_plates")) {
            sys.ghost_plates = read_ghost_plates(*g, "ghost_plates");
        }
        if (const Json* v = member(root, "variables")) {
            sys.variables = read_variables(*v, "variables");
        }
        if (const Json* c = member(root, "constraints")) {
            sys.constraints = read_constraints(*c, "constraints");
        }
        warn_unknown(root, "",
                     {"format", "name", "description", "units", "glass_catalogs", "spectrum",
                      "source", "target", "surfaces", "ghost_plates", "variables", "constraints"});

        validate_system(sys);
        return sys;
    }

private:
    static const Json* member(const Json& obj, std::string_view key) {
        const auto it = obj.find(key);
        return it == obj.end() ? nullptr : &*it;
    }

    static const Json& required(const Json& obj, const std::string& path, std::string_view key) {
        const Json* m = member(obj, key);
        if (m == nullptr) {
            fail(join(path, key), "Pflichtfeld fehlt");
        }
        return *m;
    }

    static void expect_object(const Json& j, const std::string& path) {
        if (!j.is_object()) {
            if (path.empty()) {
                fail(path, "die Wurzel muss ein JSON-Objekt sein, gefunden " + kind_name(j));
            }
            fail_type(path, "Objekt", j);
        }
    }

    static double as_number(const Json& j, const std::string& path) {
        if (!j.is_number()) {
            fail_type(path, "Zahl", j);
        }
        const double v = j.get<double>();
        if (!std::isfinite(v)) {
            fail(path, "Zahl außerhalb des darstellbaren Bereichs");
        }
        return v;
    }

    static std::string as_string(const Json& j, const std::string& path) {
        if (!j.is_string()) {
            fail_type(path, "Zeichenkette", j);
        }
        return j.get<std::string>();
    }

    static std::vector<double> as_number_list(const Json& j, const std::string& path) {
        if (!j.is_array()) {
            fail_type(path, "Liste von Zahlen", j);
        }
        std::vector<double> v;
        v.reserve(j.size());
        for (std::size_t i = 0; i < j.size(); ++i) {
            v.push_back(as_number(j[i], at(path, i)));
        }
        return v;
    }

    void warn_unknown(const Json& obj, const std::string& path,
                      std::initializer_list<std::string_view> known) {
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            bool is_known = false;
            for (const std::string_view k : known) {
                is_known = is_known || it.key() == k;
            }
            if (!is_known) {
                warn(join(path, it.key()) + ": unbekanntes Feld wird ignoriert");
            }
        }
    }

    void warn(std::string message) {
        if (warnings_ != nullptr) {
            warnings_->push_back(std::move(message));
        }
    }

    static std::string read_format(const Json& j) {
        const std::string f = as_string(j, "format");
        const std::string prefix = "mikroptim.system/";
        if (f.rfind(prefix, 0) != 0) {
            fail("format",
                 "fremdes Format \"" + f + "\", erwartet \"" + std::string(kSystemFormat) + "\"");
        }
        const std::string version = f.substr(prefix.size());
        bool digits = !version.empty();
        for (const char ch : version) {
            digits = digits && ch >= '0' && ch <= '9';
        }
        if (!digits) {
            fail("format", "ungültige Versionsangabe in \"" + f + "\"");
        }
        if (version != "1") {
            fail("format", "unbekannte Hauptversion " + version + " in \"" + f +
                               "\" (unterstützt: " + std::string(kSystemFormat) + ")");
        }
        return f;
    }

    static std::vector<std::string> read_glass_catalogs(const Json& j) {
        const std::string path = "glass_catalogs";
        if (!j.is_array()) {
            fail_type(path, "Liste von Zeichenketten", j);
        }
        std::vector<std::string> v;
        for (std::size_t i = 0; i < j.size(); ++i) {
            v.push_back(as_string(j[i], at(path, i)));
        }
        return v;
    }

    Spectrum read_spectrum(const Json& j, const std::string& path) {
        expect_object(j, path);
        Spectrum s;
        s.wavelengths_nm =
            as_number_list(required(j, path, "wavelengths_nm"), join(path, "wavelengths_nm"));
        s.weights = as_number_list(required(j, path, "weights"), join(path, "weights"));
        s.reference_nm = as_number(required(j, path, "reference_nm"), join(path, "reference_nm"));
        warn_unknown(j, path, {"wavelengths_nm", "weights", "reference_nm"});
        return s;
    }

    Source read_source(const Json& j, const std::string& path) {
        expect_object(j, path);
        Source s;
        const std::string type_path = join(path, "type");
        const std::string type = as_string(required(j, path, "type"), type_path);
        if (type == "collimated") {
            s.type = SourceType::Collimated;
            s.field_angles_deg = as_number_list(required(j, path, "field_angles_deg"),
                                                join(path, "field_angles_deg"));
            s.field_weights =
                as_number_list(required(j, path, "field_weights"), join(path, "field_weights"));
            for (const char* key : {"point", "numerical_aperture"}) {
                if (member(j, key) != nullptr) {
                    warn(join(path, key) + ": wird bei source.type \"collimated\" ignoriert");
                }
            }
        } else if (type == "point") {
            s.type = SourceType::Point;
            const std::string point_path = join(path, "point");
            const std::vector<double> p = as_number_list(required(j, path, "point"), point_path);
            if (p.size() != 3) {
                fail(point_path,
                     "erwartet genau 3 Koordinaten, gefunden " + std::to_string(p.size()));
            }
            s.point = {p[0], p[1], p[2]};
            s.numerical_aperture = as_number(required(j, path, "numerical_aperture"),
                                             join(path, "numerical_aperture"));
            for (const char* key : {"field_angles_deg", "field_weights"}) {
                if (member(j, key) != nullptr) {
                    warn(join(path, key) + ": wird bei source.type \"point\" ignoriert");
                }
            }
        } else {
            fail(type_path, "unbekannter Quellentyp \"" + type + "\" (erlaubt: collimated, point)");
        }
        warn_unknown(j, path,
                     {"type", "field_angles_deg", "field_weights", "point", "numerical_aperture"});
        return s;
    }

    Target read_target(const Json& j, const std::string& path) {
        expect_object(j, path);
        Target t;
        if (const Json* v = member(j, "focal_length")) {
            t.focal_length = as_number(*v, join(path, "focal_length"));
        }
        if (const Json* v = member(j, "parfocal_length")) {
            t.parfocal_length = as_number(*v, join(path, "parfocal_length"));
        }
        if (const Json* v = member(j, "working_distance")) {
            t.working_distance = as_number(*v, join(path, "working_distance"));
        }
        warn_unknown(j, path, {"focal_length", "parfocal_length", "working_distance"});
        return t;
    }

    std::vector<Surface> read_surfaces(const Json& j, const std::string& path) {
        if (!j.is_array()) {
            fail_type(path, "Liste von Objekten", j);
        }
        std::vector<Surface> v;
        v.reserve(j.size());
        for (std::size_t i = 0; i < j.size(); ++i) {
            v.push_back(read_surface(j[i], at(path, i)));
        }
        return v;
    }

    Surface read_surface(const Json& j, const std::string& path) {
        expect_object(j, path);
        Surface s;
        const std::string type_path = join(path, "type");
        const std::string type = as_string(required(j, path, "type"), type_path);
        if (type == "stop") {
            s.type = SurfaceType::Stop;
        } else if (type == "plane") {
            s.type = SurfaceType::Plane;
        } else if (type == "sphere") {
            s.type = SurfaceType::Sphere;
        } else {
            fail(type_path,
                 "unbekannter Flächentyp \"" + type + "\" (erlaubt: stop, plane, sphere)");
        }
        const std::string curv_path = join(path, "curvature");
        if (s.type == SurfaceType::Sphere) {
            s.curvature = as_number(required(j, path, "curvature"), curv_path);
        } else if (member(j, "curvature") != nullptr) {
            fail(curv_path,
                 "Krümmung ist nur bei \"sphere\" erlaubt, Flächentyp ist \"" + type + "\"");
        }
        s.thickness = as_number(required(j, path, "thickness"), join(path, "thickness"));
        s.material = read_material(required(j, path, "material"), join(path, "material"));
        s.semi_diameter =
            as_number(required(j, path, "semi_diameter"), join(path, "semi_diameter"));
        // Dezentrierung und Verkippung (M8-T3, ADR-0011), optional
        if (const Json* d = member(j, "decenter")) {
            s.decenter = read_pair(*d, join(path, "decenter"));
        }
        if (const Json* t = member(j, "tilt_deg")) {
            s.tilt_deg = read_pair(*t, join(path, "tilt_deg"));
        }
        if (const Json* c = member(j, "comment")) {
            s.comment = as_string(*c, join(path, "comment"));
        }
        warn_unknown(j, path,
                     {"type", "curvature", "thickness", "material", "semi_diameter", "decenter",
                      "tilt_deg", "comment"});
        return s;
    }

    // Zahlenpaar [a, b].
    static std::array<double, 2> read_pair(const Json& j, const std::string& path) {
        const std::vector<double> v = as_number_list(j, path);
        if (v.size() != 2) {
            fail(path, "erwartet genau 2 Zahlen, gefunden " + std::to_string(v.size()));
        }
        return {v[0], v[1]};
    }

    // Nichtnegative ganze Zahl (als JSON-Zahl; 7 und 7.0 sind erlaubt).
    static int as_count(const Json& j, const std::string& path) {
        const double v = as_number(j, path);
        if (v < 0.0 || v != std::floor(v) || v > 1e9) {
            fail(path, "nichtnegative ganze Zahl erwartet, ist " + j.dump());
        }
        return static_cast<int>(v);
    }

    std::vector<GhostPlate> read_ghost_plates(const Json& j, const std::string& path) {
        if (!j.is_array()) {
            fail_type(path, "Liste", j);
        }
        std::vector<GhostPlate> v;
        v.reserve(j.size());
        for (std::size_t i = 0; i < j.size(); ++i) {
            v.push_back(read_ghost_plate(j[i], at(path, i)));
        }
        return v;
    }

    GhostPlate read_ghost_plate(const Json& j, const std::string& path) {
        expect_object(j, path);
        GhostPlate g;
        g.after_surface = as_count(required(j, path, "after_surface"), join(path, "after_surface"));
        g.distance = as_number(required(j, path, "distance"), join(path, "distance"));
        const std::string glass_path = join(path, "glass");
        g.glass = read_material(required(j, path, "glass"), glass_path);
        g.norm_radius = as_number(required(j, path, "norm_radius"), join(path, "norm_radius"));
        const std::string order_path = join(path, "order");
        const int order = as_count(required(j, path, "order"), order_path);
        g.coefficients =
            as_number_list(required(j, path, "coefficients"), join(path, "coefficients"));
        if (static_cast<std::size_t>(order) != g.coefficients.size()) {
            fail(order_path, "Ordnung " + std::to_string(order) +
                                 " passt nicht zur Länge von coefficients (" +
                                 std::to_string(g.coefficients.size()) + ")");
        }
        warn_unknown(
            j, path,
            {"after_surface", "distance", "glass", "norm_radius", "order", "coefficients"});
        return g;
    }

    MaterialSpec read_material(const Json& j, const std::string& path) {
        MaterialSpec m;
        if (j.is_string()) {
            const std::string name = j.get<std::string>();
            if (name == "AIR") {
                m.kind = MaterialKind::Air;
            } else {
                m.kind = MaterialKind::Catalog;
                m.name = name;
            }
            return m;
        }
        if (!j.is_object()) {
            fail_type(path, "Zeichenkette (Glasname oder \"AIR\") oder Objekt", j);
        }
        const std::string model_path = join(path, "model");
        const std::string model = as_string(required(j, path, "model"), model_path);
        if (model == "constant") {
            m.kind = MaterialKind::Constant;
            m.n = as_number(required(j, path, "n"), join(path, "n"));
            warn_unknown(j, path, {"model", "n"});
        } else if (model == "buchdahl") {
            m.kind = MaterialKind::Buchdahl;
            m.n0 = as_number(required(j, path, "n0"), join(path, "n0"));
            const std::string nu_path = join(path, "nu");
            const std::vector<double> nu = as_number_list(required(j, path, "nu"), nu_path);
            if (nu.size() < 2 || nu.size() > 3) {
                fail(nu_path, "erwartet 2 oder 3 Buchdahl-Koeffizienten, gefunden " +
                                  std::to_string(nu.size()));
            }
            for (std::size_t k = 0; k < nu.size(); ++k) {
                m.nu[k] = nu[k];
            }
            warn_unknown(j, path, {"model", "n0", "nu"});
        } else {
            fail(model_path,
                 "unbekanntes Materialmodell \"" + model + "\" (erlaubt: constant, buchdahl)");
        }
        return m;
    }

    // ---- Abschnitt `variables` (M3-T1) -------------------------------------------------

    Variables read_variables(const Json& j, const std::string& path) {
        expect_object(j, path);
        Variables vars;
        const std::string items_path = join(path, "items");
        const Json& items = required(j, path, "items");
        if (!items.is_array()) {
            fail_type(items_path, "Liste von Objekten", items);
        }
        vars.items.reserve(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) {
            vars.items.push_back(read_variable(items[i], at(items_path, i)));
        }
        warn_unknown(j, path, {"items"});
        return vars;
    }

    VariableSpec read_variable(const Json& j, const std::string& path) {
        expect_object(j, path);
        VariableSpec v;
        const std::string type_path = join(path, "type");
        const std::string type = as_string(required(j, path, "type"), type_path);
        const std::optional<VariableType> t = variable_type_from_string(type);
        if (!t) {
            fail(type_path, "unbekannter Variablentyp \"" + type +
                                "\" (erlaubt: curvature, thickness, working_distance, "
                                "ghost_coefficient, glass)");
        }
        v.type = *t;

        const Json* one = member(j, "surface");
        const Json* many = member(j, "surfaces");
        if (one != nullptr && many != nullptr) {
            fail(join(path, "surfaces"), "nur eines der Felder \"surface\" und \"surfaces\" "
                                         "angeben");
        }
        v.selection = read_selection(j, path, "surface", "surfaces", "Flächenindizes", v.surfaces);
        v.plate_selection =
            read_selection(j, path, "plate", "plates", "Geisterplatten-Indizes", v.plates);
        v.order_selection = read_selection(j, path, "order", "orders", "Ordnungen", v.orders);
        if (const Json* m = member(j, "min")) {
            v.min = as_number(*m, join(path, "min"));
        }
        if (const Json* m = member(j, "max")) {
            v.max = as_number(*m, join(path, "max"));
        }
        if (const Json* m = member(j, "typical")) {
            v.typical = as_number(*m, join(path, "typical"));
        }
        if (const Json* c = member(j, "comment")) {
            v.comment = as_string(*c, join(path, "comment"));
        }
        warn_unknown(j, path,
                     {"type", "surface", "surfaces", "plate", "plates", "order", "orders", "min",
                      "max", "typical", "comment"});
        return v;
    }

    // Auswahl über ein Einzelfeld (`one`, nichtnegative ganze Zahl) oder ein Listenfeld (`many`:
    // Liste nichtnegativer ganzer Zahlen oder "all"); beide zugleich sind ein Fehler.
    SurfaceSelection read_selection(const Json& j, const std::string& path, std::string_view one,
                                    std::string_view many, const char* what,
                                    std::vector<int>& out) {
        const Json* o = member(j, one);
        const Json* m = member(j, many);
        if (o != nullptr && m != nullptr) {
            fail(join(path, many), "nur eines der Felder \"" + std::string(one) + "\" und \"" +
                                       std::string(many) + "\" angeben");
        }
        if (o != nullptr) {
            out.push_back(as_count(*o, join(path, one)));
            return SurfaceSelection::One;
        }
        if (m != nullptr) {
            const std::string many_path = join(path, many);
            if (m->is_string() && m->get<std::string>() == "all") {
                return SurfaceSelection::All;
            }
            if (m->is_array()) {
                for (std::size_t k = 0; k < m->size(); ++k) {
                    out.push_back(as_count((*m)[k], at(many_path, k)));
                }
                return SurfaceSelection::List;
            }
            fail_type(many_path, (std::string("Liste von ") + what + " oder \"all\"").c_str(), *m);
        }
        return SurfaceSelection::None;
    }

    // ---- Abschnitt `constraints` (M3-T3) -----------------------------------------------

    Constraints read_constraints(const Json& j, const std::string& path) {
        expect_object(j, path);
        Constraints cons;
        const std::string items_path = join(path, "items");
        const Json& items = required(j, path, "items");
        if (!items.is_array()) {
            fail_type(items_path, "Liste von Objekten", items);
        }
        cons.items.reserve(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) {
            cons.items.push_back(read_constraint(items[i], at(items_path, i)));
        }
        warn_unknown(j, path, {"items"});
        return cons;
    }

    ConstraintSpec read_constraint(const Json& j, const std::string& path) {
        expect_object(j, path);
        ConstraintSpec c;
        const std::string type_path = join(path, "type");
        const std::string type = as_string(required(j, path, "type"), type_path);
        const std::optional<ConstraintType> t = constraint_type_from_string(type);
        if (!t) {
            fail(type_path, "unbekannter Typ einer Nebenbedingung \"" + type +
                                "\" (erlaubt: focal_length, parfocal_length, working_distance, "
                                "paraxial_focus, petzval, ghost_penalty, glass_hull, "
                                "max_semi_diameter, min_center_thickness, "
                                "max_center_thickness, min_edge_thickness, "
                                "max_incidence_angle)");
        }
        c.type = *t;
        if (const Json* m = member(j, "mode")) {
            const std::string mode_path = join(path, "mode");
            const std::string mode = as_string(*m, mode_path);
            const std::optional<ConstraintMode> cm = constraint_mode_from_string(mode);
            if (!cm) {
                fail(mode_path, "unbekannter Modus \"" + mode + "\" (erlaubt: equal, min, max)");
            }
            c.mode = *cm;
        }
        if (const Json* w = member(j, "weight")) {
            c.weight = as_number(*w, join(path, "weight"));
        }
        if (const Json* r = member(j, "reference_surface")) {
            c.reference_surface = as_count(*r, join(path, "reference_surface"));
        }
        if (const Json* o = member(j, "offset")) {
            c.offset = as_number(*o, join(path, "offset"));
        }
        if (const Json* k = member(j, "max_curvature")) {
            c.max_curvature = as_number(*k, join(path, "max_curvature"));
        }
        c.plate_selection =
            read_selection(j, path, "plate", "plates", "Geisterplatten-Indizes", c.plates);
        if (const Json* w = member(j, "order_weight")) {
            c.order_weight = as_number(*w, join(path, "order_weight"));
        }
        if (const Json* f = member(j, "order_factor")) {
            c.order_factor = as_number(*f, join(path, "order_factor"));
        }
        if (const Json* e = member(j, "smoothing")) {
            c.smoothing = as_number(*e, join(path, "smoothing"));
        }
        if (const Json* d = member(j, "margin")) {
            c.margin = as_number(*d, join(path, "margin"));
        }
        // Mechanische Bedingungen (M4-T3)
        c.surface_selection =
            read_selection(j, path, "surface", "surfaces", "Flächenindizes", c.surfaces);
        if (const Json* m = member(j, "medium")) {
            const std::string medium_path = join(path, "medium");
            const std::string medium = as_string(*m, medium_path);
            const std::optional<ConstraintMedium> cm = constraint_medium_from_string(medium);
            if (!cm) {
                fail(medium_path, "unbekanntes Medium \"" + medium + "\" (erlaubt: glass, air)");
            }
            c.medium = *cm;
        }
        if (const Json* d = member(j, "semi_diameter")) {
            c.semi_diameter = as_number(*d, join(path, "semi_diameter"));
        }
        if (const Json* th = member(j, "thickness")) {
            c.thickness = as_number(*th, join(path, "thickness"));
        }
        if (const Json* a = member(j, "angle")) {
            c.angle = as_number(*a, join(path, "angle"));
        }
        if (const Json* k = member(j, "clearance")) {
            c.clearance = as_number(*k, join(path, "clearance"));
        }
        if (const Json* h = member(j, "height")) {
            const std::string height_path = join(path, "height");
            const std::string height = as_string(*h, height_path);
            const std::optional<EdgeHeight> eh = edge_height_from_string(height);
            if (!eh) {
                fail(height_path,
                     "unbekannte Höhe \"" + height + "\" (erlaubt: semi_diameter, rays)");
            }
            c.height = *eh;
        }
        if (const Json* m = member(j, "comment")) {
            c.comment = as_string(*m, join(path, "comment"));
        }
        warn_unknown(j, path,
                     {"type",          "mode",      "weight",  "reference_surface", "offset",
                      "max_curvature", "plate",     "plates",  "order_weight",      "order_factor",
                      "smoothing",     "margin",    "surface", "surfaces",          "medium",
                      "semi_diameter", "thickness", "angle",   "clearance",         "height",
                      "comment"});
        return c;
    }

    std::vector<std::string>* warnings_;
};

// --------------------------------------------------------------------------
// Schreiben
// --------------------------------------------------------------------------

Json material_to_json(const MaterialSpec& m) {
    switch (m.kind) {
    case MaterialKind::Air:
        return "AIR";
    case MaterialKind::Catalog:
        return m.name;
    case MaterialKind::Constant: {
        Json j;
        j["model"] = "constant";
        j["n"] = m.n;
        return j;
    }
    case MaterialKind::Buchdahl: {
        Json j;
        j["model"] = "buchdahl";
        j["n0"] = m.n0;
        Json nu = Json::array({m.nu[0], m.nu[1]});
        if (m.nu[2] != 0.0) {
            nu.push_back(m.nu[2]);
        }
        j["nu"] = std::move(nu);
        return j;
    }
    }
    return "AIR";
}

// Schreibt eine Auswahl (Einzelfeld `one`, Liste oder "all" in `many`); None schreibt nichts.
void selection_to_json(Json& j, SurfaceSelection selection, const std::vector<int>& values,
                       const char* one, const char* many) {
    switch (selection) {
    case SurfaceSelection::None:
        break;
    case SurfaceSelection::One:
        j[one] = values.at(0);
        break;
    case SurfaceSelection::List:
        j[many] = values;
        break;
    case SurfaceSelection::All:
        j[many] = "all";
        break;
    }
}

Json variables_to_json(const Variables& vars) {
    Json items = Json::array();
    for (const VariableSpec& v : vars.items) {
        Json j;
        j["type"] = std::string(to_string(v.type));
        selection_to_json(j, v.selection, v.surfaces, "surface", "surfaces");
        selection_to_json(j, v.plate_selection, v.plates, "plate", "plates");
        selection_to_json(j, v.order_selection, v.orders, "order", "orders");
        if (v.min) {
            j["min"] = *v.min;
        }
        if (v.max) {
            j["max"] = *v.max;
        }
        if (v.typical) {
            j["typical"] = *v.typical;
        }
        if (v.comment) {
            j["comment"] = *v.comment;
        }
        items.push_back(std::move(j));
    }
    Json root;
    root["items"] = std::move(items);
    return root;
}

Json constraints_to_json(const Constraints& cons) {
    Json items = Json::array();
    for (const ConstraintSpec& c : cons.items) {
        Json j;
        j["type"] = std::string(to_string(c.type));
        if (c.mode) {
            j["mode"] = std::string(to_string(*c.mode));
        }
        if (c.weight) {
            j["weight"] = *c.weight;
        }
        if (c.reference_surface) {
            j["reference_surface"] = *c.reference_surface;
        }
        if (c.offset) {
            j["offset"] = *c.offset;
        }
        if (c.max_curvature) {
            j["max_curvature"] = *c.max_curvature;
        }
        selection_to_json(j, c.plate_selection, c.plates, "plate", "plates");
        if (c.order_weight) {
            j["order_weight"] = *c.order_weight;
        }
        if (c.order_factor) {
            j["order_factor"] = *c.order_factor;
        }
        if (c.smoothing) {
            j["smoothing"] = *c.smoothing;
        }
        if (c.margin) {
            j["margin"] = *c.margin;
        }
        selection_to_json(j, c.surface_selection, c.surfaces, "surface", "surfaces");
        if (c.medium) {
            j["medium"] = std::string(to_string(*c.medium));
        }
        if (c.semi_diameter) {
            j["semi_diameter"] = *c.semi_diameter;
        }
        if (c.thickness) {
            j["thickness"] = *c.thickness;
        }
        if (c.angle) {
            j["angle"] = *c.angle;
        }
        if (c.clearance) {
            j["clearance"] = *c.clearance;
        }
        if (c.height) {
            j["height"] = std::string(to_string(*c.height));
        }
        if (c.comment) {
            j["comment"] = *c.comment;
        }
        items.push_back(std::move(j));
    }
    Json root;
    root["items"] = std::move(items);
    return root;
}

Json to_json(const SystemDescription& sys) {
    Json root;
    root["format"] = sys.format;
    root["name"] = sys.name;
    if (sys.description) {
        root["description"] = *sys.description;
    }
    root["units"] = sys.units;
    root["glass_catalogs"] = sys.glass_catalogs;

    Json spectrum;
    spectrum["wavelengths_nm"] = sys.spectrum.wavelengths_nm;
    spectrum["weights"] = sys.spectrum.weights;
    spectrum["reference_nm"] = sys.spectrum.reference_nm;
    root["spectrum"] = std::move(spectrum);

    Json source;
    source["type"] = std::string(to_string(sys.source.type));
    if (sys.source.type == SourceType::Collimated) {
        source["field_angles_deg"] = sys.source.field_angles_deg;
        source["field_weights"] = sys.source.field_weights;
    } else {
        source["point"] = sys.source.point;
        source["numerical_aperture"] = sys.source.numerical_aperture;
    }
    root["source"] = std::move(source);

    if (sys.target) {
        Json target = Json::object();
        if (sys.target->focal_length) {
            target["focal_length"] = *sys.target->focal_length;
        }
        if (sys.target->parfocal_length) {
            target["parfocal_length"] = *sys.target->parfocal_length;
        }
        if (sys.target->working_distance) {
            target["working_distance"] = *sys.target->working_distance;
        }
        root["target"] = std::move(target);
    }

    Json surfaces = Json::array();
    for (const Surface& s : sys.surfaces) {
        Json js;
        js["type"] = std::string(to_string(s.type));
        if (s.type == SurfaceType::Sphere) {
            js["curvature"] = s.curvature;
        }
        js["thickness"] = s.thickness;
        js["material"] = material_to_json(s.material);
        js["semi_diameter"] = s.semi_diameter;
        if (s.decenter[0] != 0.0 || s.decenter[1] != 0.0) {
            js["decenter"] = Json::array({s.decenter[0], s.decenter[1]});
        }
        if (s.tilt_deg[0] != 0.0 || s.tilt_deg[1] != 0.0) {
            js["tilt_deg"] = Json::array({s.tilt_deg[0], s.tilt_deg[1]});
        }
        if (s.comment) {
            js["comment"] = *s.comment;
        }
        surfaces.push_back(std::move(js));
    }
    root["surfaces"] = std::move(surfaces);

    if (!sys.ghost_plates.empty()) {
        Json plates = Json::array();
        for (const GhostPlate& g : sys.ghost_plates) {
            Json jg;
            jg["after_surface"] = g.after_surface;
            jg["distance"] = g.distance;
            jg["glass"] = material_to_json(g.glass);
            jg["norm_radius"] = g.norm_radius;
            jg["order"] = g.coefficients.size();
            jg["coefficients"] = g.coefficients;
            plates.push_back(std::move(jg));
        }
        root["ghost_plates"] = std::move(plates);
    }
    if (sys.variables) {
        root["variables"] = variables_to_json(*sys.variables);
    }
    if (sys.constraints) {
        root["constraints"] = constraints_to_json(*sys.constraints);
    }
    return root;
}

// Einzeilige Darstellung im Stil der Beispieldateien: `{ "a": 1, "b": [1.0, 2.0] }`.
std::string inline_text(const Json& j) {
    if (j.is_object() && !j.empty()) {
        std::string s = "{ ";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) {
                s += ", ";
            }
            first = false;
            s += Json(it.key()).dump() + ": " + inline_text(it.value());
        }
        return s + " }";
    }
    if (j.is_array() && !j.empty()) {
        std::string s = "[";
        for (std::size_t i = 0; i < j.size(); ++i) {
            if (i > 0) {
                s += ", ";
            }
            s += inline_text(j[i]);
        }
        return s + "]";
    }
    return j.dump();
}

// Eingerückte Ausgabe: Listen und Objekte bleiben einzeilig, wenn sie in die Zeile passen.
// `column` ist die Spalte, an der der Wert beginnt.
void pretty(const Json& j, std::size_t indent, std::size_t column, bool expand, std::string& out) {
    if (!j.is_structured() || j.empty()) {
        out += inline_text(j);
        return;
    }
    if (!expand) {
        const std::string s = inline_text(j);
        if (column + s.size() + 1 <= kLineWidth) { // +1 für ein folgendes Komma
            out += s;
            return;
        }
    }
    const std::string inner(indent + 2, ' ');
    out += j.is_object() ? "{\n" : "[\n";
    std::size_t i = 0;
    for (auto it = j.begin(); it != j.end(); ++it, ++i) {
        out += inner;
        std::size_t col = inner.size();
        if (j.is_object()) {
            const std::string key = Json(it.key()).dump() + ": ";
            out += key;
            col += key.size();
        }
        pretty(it.value(), indent + 2, col, false, out);
        if (i + 1 < j.size()) {
            out += ',';
        }
        out += '\n';
    }
    out += std::string(indent, ' ');
    out += j.is_object() ? '}' : ']';
}

} // namespace

SystemDescription read_system_json(std::string_view text, std::vector<std::string>* warnings) {
    Json root;
    try {
        root = Json::parse(text.begin(), text.end());
    } catch (const Json::exception& e) {
        throw FormatError("", std::string("JSON-Syntaxfehler: ") + e.what());
    }
    try {
        return Reader(warnings).read(root);
    } catch (const Json::exception& e) {
        // Sollte nicht vorkommen, da Typen vorher geprüft werden.
        throw FormatError("", std::string("JSON-Fehler: ") + e.what());
    }
}

SystemDescription read_system_file(const std::filesystem::path& file,
                                   std::vector<std::string>* warnings) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Systemdatei kann nicht geöffnet werden: " + display(file));
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (in.bad()) {
        throw std::runtime_error("Systemdatei kann nicht gelesen werden: " + display(file));
    }
    try {
        return read_system_json(buffer.str(), warnings);
    } catch (const FormatError& e) {
        throw FormatError(e.json_path(), e.message(), display(file));
    }
}

std::string write_system_json(const SystemDescription& system) {
    validate_system(system);
    std::string out;
    try {
        pretty(to_json(system), 0, 0, true, out);
    } catch (const Json::exception& e) {
        // z. B. ungültiges UTF-8 in Zeichenketten des Modells
        throw FormatError("", std::string("JSON-Fehler beim Schreiben: ") + e.what());
    }
    out += '\n';
    return out;
}

void write_system_file(const SystemDescription& system, const std::filesystem::path& file) {
    const std::string text = write_system_json(system);
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Systemdatei kann nicht geschrieben werden: " + display(file));
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) {
        throw std::runtime_error("Fehler beim Schreiben der Systemdatei: " + display(file));
    }
}

std::vector<std::filesystem::path> resolve_glass_catalogs(const SystemDescription& system,
                                                          const std::filesystem::path& base_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path base = base_dir.empty() ? fs::current_path(ec) : fs::absolute(base_dir, ec);
    if (ec) {
        base = base_dir;
    }
    base = base.lexically_normal();

    std::vector<fs::path> result;
    result.reserve(system.glass_catalogs.size());
    for (std::size_t i = 0; i < system.glass_catalogs.size(); ++i) {
        const std::string& entry = system.glass_catalogs[i];
        const fs::path rel(std::u8string(entry.begin(), entry.end()));
        bool found = false;
        if (rel.is_absolute()) {
            if (fs::is_regular_file(rel, ec)) {
                result.push_back(rel.lexically_normal());
                found = true;
            }
        } else {
            for (fs::path dir = base;; dir = dir.parent_path()) {
                const fs::path candidate = dir / rel;
                if (fs::is_regular_file(candidate, ec)) {
                    result.push_back(candidate.lexically_normal());
                    found = true;
                    break;
                }
                if (dir == dir.parent_path()) {
                    break;
                }
            }
        }
        if (!found) {
            fail(at("glass_catalogs", i),
                 "Glaskatalog \"" + entry + "\" nicht gefunden (gesucht ab " + display(base) +
                     (rel.is_absolute() ? ")" : " und in den Elternverzeichnissen)"));
        }
    }
    return result;
}

} // namespace mko
