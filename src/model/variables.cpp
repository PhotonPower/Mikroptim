// Auflösung und Validierung des Abschnitts `variables` (docs/dateiformat.md, „Variablen“).
// Die Pfade in den Meldungen entsprechen den JSON-Pfaden der Datei.

#include "mikroptim/model/variables.hpp"

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
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

// JSON-Pfad des Werts einer Variable in der Systembeschreibung, z. B. `surfaces[2].thickness`
// oder `ghost_plates[1].coefficients[0]`.
std::string value_path(const ResolvedVariable& v) {
    if (v.type == VariableType::GhostCoefficient) {
        return at(at("ghost_plates", static_cast<std::size_t>(v.plate)) + ".coefficients",
                  static_cast<std::size_t>(v.order - 1));
    }
    const std::string surface = at("surfaces", static_cast<std::size_t>(v.surface));
    if (v.type == VariableType::Glass) {
        return surface + ".material";
    }
    return surface + (v.type == VariableType::Curvature ? ".curvature" : ".thickness");
}

// Geisterkoeffizienten eines Eintrags: (Platte, Ordnung) mit dem JSON-Pfad für Fehler.
struct GhostRef {
    int plate;
    int order;
    std::string path;
};

std::vector<GhostRef> ghost_targets_of(const VariableSpec& spec, const std::string& path,
                                       const std::vector<GhostPlate>& ghosts) {
    if (spec.selection != SurfaceSelection::None) {
        fail(path + (spec.selection == SurfaceSelection::One ? ".surface" : ".surfaces"),
             "ghost_coefficient bezieht sich auf Geisterplatten; \"plate\" bzw. \"plates\" "
             "statt Flächen angeben");
    }
    const int g_count = static_cast<int>(ghosts.size());
    struct PlateRef {
        int plate;
        std::string path;
    };
    std::vector<PlateRef> plates;
    switch (spec.plate_selection) {
    case SurfaceSelection::None:
        fail(path + ".plate", "Pflichtfeld fehlt (\"plate\" oder \"plates\" angeben)");
    case SurfaceSelection::One:
        if (spec.plates.size() != 1) {
            fail(path + ".plate", "genau eine Geisterplatte erwartet");
        }
        plates.push_back({spec.plates[0], path + ".plate"});
        break;
    case SurfaceSelection::List:
        if (spec.plates.empty()) {
            fail(path + ".plates", "mindestens eine Geisterplatte nötig");
        }
        for (std::size_t j = 0; j < spec.plates.size(); ++j) {
            plates.push_back({spec.plates[j], at(path + ".plates", j)});
        }
        break;
    case SurfaceSelection::All:
        if (!spec.plates.empty()) {
            fail(path + ".plates", "bei \"all\" keine einzelnen Geisterplatten");
        }
        for (int g = 0; g < g_count; ++g) {
            plates.push_back({g, path + ".plates"});
        }
        if (plates.empty()) {
            fail(path + ".plates", "\"all\" findet keine Geisterplatte (Abschnitt "
                                   "ghost_plates fehlt oder ist leer)");
        }
        break;
    }

    std::vector<GhostRef> out;
    for (const PlateRef& p : plates) {
        if (p.plate < 0 || p.plate >= g_count) {
            fail(p.path, "Geisterplatte " + std::to_string(p.plate) + " existiert nicht" +
                             (g_count == 0
                                  ? std::string(" (keine Geisterplatten)")
                                  : " (Geisterplatten 0 bis " + std::to_string(g_count - 1) + ")"));
        }
        const int k_count =
            static_cast<int>(ghosts[static_cast<std::size_t>(p.plate)].coefficients.size());
        switch (spec.order_selection) {
        case SurfaceSelection::None:
        case SurfaceSelection::All:
            if (spec.order_selection == SurfaceSelection::All && !spec.orders.empty()) {
                fail(path + ".orders", "bei \"all\" keine einzelnen Ordnungen");
            }
            for (int k = 1; k <= k_count; ++k) {
                out.push_back({p.plate, k, p.path});
            }
            break;
        case SurfaceSelection::One:
        case SurfaceSelection::List: {
            const bool one = spec.order_selection == SurfaceSelection::One;
            if (one && spec.orders.size() != 1) {
                fail(path + ".order", "genau eine Ordnung erwartet");
            }
            if (spec.orders.empty()) {
                fail(path + ".orders", "mindestens eine Ordnung nötig");
            }
            for (std::size_t j = 0; j < spec.orders.size(); ++j) {
                const int k = spec.orders[j];
                const std::string kpath = one ? path + ".order" : at(path + ".orders", j);
                if (k < 1 || k > k_count) {
                    fail(kpath, "Ordnung " + std::to_string(k) + " existiert bei Geisterplatte " +
                                    std::to_string(p.plate) + " nicht (Ordnungen 1 bis " +
                                    std::to_string(k_count) + ")");
                }
                out.push_back({p.plate, k, kpath});
            }
            break;
        }
        }
    }
    return out;
}

// Flächen eines Eintrags mit dem JSON-Pfad, unter dem Fehler zu einer Fläche gemeldet werden.
struct SurfaceRef {
    int surface;
    std::string path;
};

std::vector<SurfaceRef> targets_of(const VariableSpec& spec, const std::string& path,
                                   const std::vector<Surface>& surfaces) {
    const int n = static_cast<int>(surfaces.size());
    std::vector<SurfaceRef> out;
    if (spec.plate_selection != SurfaceSelection::None) {
        fail(path + (spec.plate_selection == SurfaceSelection::One ? ".plate" : ".plates"),
             "nur bei ghost_coefficient erlaubt");
    }
    if (spec.order_selection != SurfaceSelection::None) {
        fail(path + (spec.order_selection == SurfaceSelection::One ? ".order" : ".orders"),
             "nur bei ghost_coefficient erlaubt");
    }
    if (spec.type == VariableType::WorkingDistance) {
        if (spec.selection == SurfaceSelection::One) {
            fail(path + ".surface", "working_distance ist immer die Dicke der letzten Fläche; "
                                    "kein \"surface\" angeben");
        }
        if (spec.selection != SurfaceSelection::None) {
            fail(path + ".surfaces", "working_distance ist immer die Dicke der letzten Fläche; "
                                     "kein \"surfaces\" angeben");
        }
        out.push_back({n - 1, path + ".type"});
        return out;
    }
    switch (spec.selection) {
    case SurfaceSelection::None:
        fail(path + ".surface", "Pflichtfeld fehlt (\"surface\" oder \"surfaces\" angeben)");
    case SurfaceSelection::One:
        if (spec.surfaces.size() != 1) {
            fail(path + ".surface", "genau eine Fläche erwartet");
        }
        out.push_back({spec.surfaces[0], path + ".surface"});
        break;
    case SurfaceSelection::List:
        if (spec.surfaces.empty()) {
            fail(path + ".surfaces", "mindestens eine Fläche nötig");
        }
        for (std::size_t j = 0; j < spec.surfaces.size(); ++j) {
            out.push_back({spec.surfaces[j], at(path + ".surfaces", j)});
        }
        break;
    case SurfaceSelection::All:
        if (!spec.surfaces.empty()) {
            fail(path + ".surfaces", "bei \"all\" keine einzelnen Flächen");
        }
        // Krümmungen: alle Flächen vom Typ sphere; Dicken: alle außer der letzten
        // (deren Dicke ist der Arbeitsabstand); Glas: alle Flächen mit einem Glas danach.
        for (int s = 0; s < n; ++s) {
            const Surface& surf = surfaces[static_cast<std::size_t>(s)];
            bool take = s < n - 1;
            if (spec.type == VariableType::Curvature) {
                take = surf.type == SurfaceType::Sphere;
            } else if (spec.type == VariableType::Glass) {
                take = surf.material.kind != MaterialKind::Air;
            }
            if (take) {
                out.push_back({s, path + ".surfaces"});
            }
        }
        if (out.empty()) {
            const char* what = "\"all\" findet keine Fläche mit variabler Dicke";
            if (spec.type == VariableType::Curvature) {
                what = "\"all\" findet keine Fläche vom Typ \"sphere\"";
            } else if (spec.type == VariableType::Glass) {
                what = "\"all\" findet keine Fläche mit einem Glas (Material nicht \"AIR\")";
            }
            fail(path + ".surfaces", what);
        }
        break;
    }
    return out;
}

// Prüft, ob die Fläche als Ziel des Variablentyps taugt.
void check_target(VariableType type, const SurfaceRef& t, const std::vector<Surface>& surfaces) {
    const int n = static_cast<int>(surfaces.size());
    if (t.surface < 0 || t.surface >= n) {
        fail(t.path, "Fläche " + std::to_string(t.surface) + " existiert nicht (Flächen 0 bis " +
                         std::to_string(n - 1) + ")");
    }
    const Surface& s = surfaces[static_cast<std::size_t>(t.surface)];
    if (type == VariableType::Curvature && s.type != SurfaceType::Sphere) {
        if (s.type == SurfaceType::Stop) {
            fail(t.path, "die Blende (Fläche 0) ist eben und kann keine Krümmungsvariable sein");
        }
        fail(t.path, "Krümmungsvariable nur bei Flächen vom Typ \"sphere\"; Fläche " +
                         std::to_string(t.surface) + " ist \"" + std::string(to_string(s.type)) +
                         "\" (eine Planfläche, die gekrümmt werden darf, als \"sphere\" mit "
                         "\"curvature\": 0 angeben)");
    }
    if (type == VariableType::Glass && s.material.kind == MaterialKind::Air) {
        fail(t.path, "Glasvariable verlangt ein Glas nach Fläche " + std::to_string(t.surface) +
                         " (Material ist \"AIR\")");
    }
    if (type == VariableType::Thickness && t.surface == n - 1) {
        fail(t.path, "die Dicke der letzten Fläche ist der Arbeitsabstand; dafür die Variable "
                     "\"working_distance\" verwenden");
    }
}

} // namespace

double default_typical(VariableType type, int component) {
    switch (type) {
    case VariableType::Curvature:
        return kDefaultTypicalCurvature;
    case VariableType::Thickness:
        return kDefaultTypicalThickness;
    case VariableType::WorkingDistance:
        return kDefaultTypicalWorkingDistance;
    case VariableType::GhostCoefficient:
        return kDefaultTypicalGhostCoefficient;
    case VariableType::Glass:
        return component == 1   ? kDefaultTypicalGlassNu1
               : component == 2 ? kDefaultTypicalGlassNu2
                                : kDefaultTypicalGlassIndex;
    }
    return 1.0;
}

std::string_view to_string(VariableType type) {
    switch (type) {
    case VariableType::Curvature:
        return "curvature";
    case VariableType::Thickness:
        return "thickness";
    case VariableType::WorkingDistance:
        return "working_distance";
    case VariableType::GhostCoefficient:
        return "ghost_coefficient";
    case VariableType::Glass:
        return "glass";
    }
    return "?";
}

std::optional<VariableType> variable_type_from_string(std::string_view name) {
    for (const VariableType t :
         {VariableType::Curvature, VariableType::Thickness, VariableType::WorkingDistance,
          VariableType::GhostCoefficient, VariableType::Glass}) {
        if (name == to_string(t)) {
            return t;
        }
    }
    return std::nullopt;
}

double variable_value(const SystemDescription& system, const ResolvedVariable& variable) {
    if (variable.type == VariableType::GhostCoefficient) {
        return system.ghost_plates.at(static_cast<std::size_t>(variable.plate))
            .coefficients.at(static_cast<std::size_t>(variable.order - 1));
    }
    const Surface& s = system.surfaces.at(static_cast<std::size_t>(variable.surface));
    if (variable.type == VariableType::Glass) {
        const MaterialSpec& m = s.material;
        const int c = variable.component;
        if (c < 0 || c >= kGlassComponents) {
            throw std::invalid_argument("Glasvariable: Komponente " + std::to_string(c) +
                                        " außerhalb von [0, 3)");
        }
        switch (m.kind) {
        case MaterialKind::Buchdahl:
            return c == 0 ? m.n0 : m.nu[static_cast<std::size_t>(c - 1)];
        case MaterialKind::Constant:
            return c == 0 ? m.n : 0.0;
        case MaterialKind::Catalog:
        case MaterialKind::Air:
            break;
        }
        throw std::invalid_argument(
            "Glasvariable auf Fläche " + std::to_string(variable.surface) +
            ": Material ist kein Buchdahl-Glas (vorher mit convert_glass_variables umwandeln)");
    }
    return variable.type == VariableType::Curvature ? s.curvature : s.thickness;
}

std::vector<ResolvedVariable> resolve_variables(const SystemDescription& system) {
    std::vector<ResolvedVariable> out;
    if (!system.variables) {
        return out;
    }
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const std::vector<VariableSpec>& items = system.variables->items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const VariableSpec& spec = items[i];
        const std::string path = at("variables.items", i);

        // Schranken und typische Größe
        if (spec.min && !std::isfinite(*spec.min)) {
            fail(path + ".min", "Zahl muss endlich sein (offene Schranke: Feld weglassen)");
        }
        if (spec.max && !std::isfinite(*spec.max)) {
            fail(path + ".max", "Zahl muss endlich sein (offene Schranke: Feld weglassen)");
        }
        if (spec.min && spec.max && !(*spec.min < *spec.max)) {
            fail(path + ".max", "obere Schranke " + num(*spec.max) +
                                    " muss größer als die untere Schranke " + num(*spec.min) +
                                    " sein");
        }
        if (spec.typical && !(std::isfinite(*spec.typical) && *spec.typical > 0.0)) {
            fail(path + ".typical",
                 "typische Größe muss endlich und größer als 0 sein, ist " + num(*spec.typical));
        }

        // Startwert innerhalb der Schranken, sonst Fehler unter `min` bzw. `max`.
        const auto check_start = [&](const ResolvedVariable& v) {
            const double x = variable_value(system, v);
            if (x < v.lower) {
                fail(path + ".min", "Startwert " + num(x) + " (" + value_path(v) +
                                        ") liegt unter der unteren Schranke " + num(v.lower));
            }
            if (x > v.upper) {
                fail(path + ".max", "Startwert " + num(x) + " (" + value_path(v) +
                                        ") liegt über der oberen Schranke " + num(v.upper));
            }
        };
        if (spec.type == VariableType::Glass) {
            // Drei Komponenten mit verschiedenen Größenordnungen; die Hülle der Katalog-Gläser
            // wirkt als Strafterm (Nebenbedingung glass_hull), nicht als Schranke.
            const char* reason = "bei glass nicht erlaubt (die Glas-Hülle wirkt als Strafterm, "
                                 "Nebenbedingung \"glass_hull\"; typische Größen fest pro "
                                 "Komponente)";
            if (spec.min) {
                fail(path + ".min", reason);
            }
            if (spec.max) {
                fail(path + ".max", reason);
            }
            if (spec.typical) {
                fail(path + ".typical", reason);
            }
        }
        ResolvedVariable base;
        base.type = spec.type;
        base.lower = spec.min.value_or(-kInf);
        base.upper = spec.max.value_or(kInf);
        base.typical = spec.typical.value_or(default_typical(spec.type));
        base.item = static_cast<int>(i);

        if (spec.type == VariableType::GhostCoefficient) {
            for (const GhostRef& t : ghost_targets_of(spec, path, system.ghost_plates)) {
                for (const ResolvedVariable& prev : out) {
                    if (prev.type == spec.type && prev.plate == t.plate && prev.order == t.order) {
                        fail(t.path, "Variable doppelt: ghost_coefficient der Geisterplatte " +
                                         std::to_string(t.plate) + ", Ordnung " +
                                         std::to_string(t.order) + " steht schon in " +
                                         at("variables.items", prev.item));
                    }
                }
                ResolvedVariable v = base;
                v.plate = t.plate;
                v.order = t.order;
                v.surface = system.ghost_plates[static_cast<std::size_t>(t.plate)].after_surface;
                check_start(v);
                out.push_back(v);
            }
            continue;
        }

        for (const SurfaceRef& t : targets_of(spec, path, system.surfaces)) {
            check_target(spec.type, t, system.surfaces);
            for (const ResolvedVariable& prev : out) {
                if (prev.type == spec.type && prev.surface == t.surface) {
                    fail(t.path, "Variable doppelt: " + std::string(to_string(spec.type)) +
                                     (spec.type == VariableType::WorkingDistance
                                          ? std::string()
                                          : " von Fläche " + std::to_string(t.surface)) +
                                     " steht schon in " + at("variables.items", prev.item));
                }
            }
            ResolvedVariable v = base;
            v.surface = t.surface;
            if (spec.type == VariableType::Glass) {
                // n₀, ν₁, ν₂ hintereinander; keine Schranken, kein Startwert zu prüfen (das
                // Material darf noch ein Katalogglas sein)
                for (int c = 0; c < kGlassComponents; ++c) {
                    v.component = c;
                    v.typical = default_typical(VariableType::Glass, c);
                    out.push_back(v);
                }
                continue;
            }
            check_start(v);
            out.push_back(v);
        }
    }
    return out;
}

} // namespace mko
