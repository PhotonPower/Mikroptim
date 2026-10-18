// Auflösung und Validierung des Abschnitts `variables` (docs/dateiformat.md, „Variablen“).
// Die Pfade in den Meldungen entsprechen den JSON-Pfaden der Datei.

#include "mikroptim/model/variables.hpp"

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
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

// JSON-Pfad des Werts einer Variable in der Systembeschreibung, z. B. `surfaces[2].thickness`.
std::string value_path(VariableType type, int surface) {
    return at("surfaces", static_cast<std::size_t>(surface)) +
           (type == VariableType::Curvature ? ".curvature" : ".thickness");
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
        // (deren Dicke ist der Arbeitsabstand).
        for (int s = 0; s < n; ++s) {
            const bool take =
                spec.type == VariableType::Curvature
                    ? surfaces[static_cast<std::size_t>(s)].type == SurfaceType::Sphere
                    : s < n - 1;
            if (take) {
                out.push_back({s, path + ".surfaces"});
            }
        }
        if (out.empty()) {
            fail(path + ".surfaces", spec.type == VariableType::Curvature
                                         ? "\"all\" findet keine Fläche vom Typ \"sphere\""
                                         : "\"all\" findet keine Fläche mit variabler Dicke");
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
    if (type == VariableType::Thickness && t.surface == n - 1) {
        fail(t.path, "die Dicke der letzten Fläche ist der Arbeitsabstand; dafür die Variable "
                     "\"working_distance\" verwenden");
    }
}

} // namespace

double default_typical(VariableType type) {
    switch (type) {
    case VariableType::Curvature:
        return kDefaultTypicalCurvature;
    case VariableType::Thickness:
        return kDefaultTypicalThickness;
    case VariableType::WorkingDistance:
        return kDefaultTypicalWorkingDistance;
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
    }
    return "?";
}

std::optional<VariableType> variable_type_from_string(std::string_view name) {
    for (const VariableType t :
         {VariableType::Curvature, VariableType::Thickness, VariableType::WorkingDistance}) {
        if (name == to_string(t)) {
            return t;
        }
    }
    return std::nullopt;
}

double variable_value(const SystemDescription& system, const ResolvedVariable& variable) {
    const Surface& s = system.surfaces.at(static_cast<std::size_t>(variable.surface));
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
            ResolvedVariable v;
            v.type = spec.type;
            v.surface = t.surface;
            v.lower = spec.min.value_or(-kInf);
            v.upper = spec.max.value_or(kInf);
            v.typical = spec.typical.value_or(default_typical(spec.type));
            v.item = static_cast<int>(i);

            const double x = variable_value(system, v);
            if (x < v.lower) {
                fail(path + ".min", "Startwert " + num(x) + " (" + value_path(v.type, v.surface) +
                                        ") liegt unter der unteren Schranke " + num(v.lower));
            }
            if (x > v.upper) {
                fail(path + ".max", "Startwert " + num(x) + " (" + value_path(v.type, v.surface) +
                                        ") liegt über der oberen Schranke " + num(v.upper));
            }
            out.push_back(v);
        }
    }
    return out;
}

} // namespace mko
