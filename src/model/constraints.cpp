// Validierung des Abschnitts `constraints` (docs/dateiformat.md, „Nebenbedingungen“).
// Die Pfade in den Meldungen entsprechen den JSON-Pfaden der Datei.

#include "mikroptim/model/constraints.hpp"

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"

#include <cmath>
#include <cstddef>
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

// Feld von `target`, das den Sollwert des Typs hält; leer bei ParaxialFocus (Soll 0).
std::string_view target_field(ConstraintType type) {
    switch (type) {
    case ConstraintType::FocalLength:
        return "focal_length";
    case ConstraintType::ParfocalLength:
        return "parfocal_length";
    case ConstraintType::WorkingDistance:
        return "working_distance";
    case ConstraintType::ParaxialFocus:
    case ConstraintType::Petzval:
        return {};
    }
    return {};
}

// Typen, die nur als Gleichung definiert sind (einseitig ohne sinnvolle Bedeutung bzw. ohne
// stetige Formulierung, docs/optimierung.md, Abschnitt 2).
bool equality_only(ConstraintType type) {
    return type == ConstraintType::FocalLength || type == ConstraintType::ParaxialFocus;
}

} // namespace

std::string_view to_string(ConstraintType type) {
    switch (type) {
    case ConstraintType::FocalLength:
        return "focal_length";
    case ConstraintType::ParfocalLength:
        return "parfocal_length";
    case ConstraintType::WorkingDistance:
        return "working_distance";
    case ConstraintType::ParaxialFocus:
        return "paraxial_focus";
    case ConstraintType::Petzval:
        return "petzval";
    }
    return "?";
}

std::string_view to_string(ConstraintMode mode) {
    switch (mode) {
    case ConstraintMode::Equal:
        return "equal";
    case ConstraintMode::Min:
        return "min";
    case ConstraintMode::Max:
        return "max";
    }
    return "?";
}

std::optional<ConstraintType> constraint_type_from_string(std::string_view name) {
    for (const ConstraintType t : {ConstraintType::FocalLength, ConstraintType::ParfocalLength,
                                   ConstraintType::WorkingDistance, ConstraintType::ParaxialFocus,
                                   ConstraintType::Petzval}) {
        if (name == to_string(t)) {
            return t;
        }
    }
    return std::nullopt;
}

std::optional<ConstraintMode> constraint_mode_from_string(std::string_view name) {
    for (const ConstraintMode m :
         {ConstraintMode::Equal, ConstraintMode::Min, ConstraintMode::Max}) {
        if (name == to_string(m)) {
            return m;
        }
    }
    return std::nullopt;
}

bool is_paraxial(ConstraintType type) {
    switch (type) {
    case ConstraintType::FocalLength:
    case ConstraintType::ParfocalLength:
    case ConstraintType::WorkingDistance:
    case ConstraintType::ParaxialFocus:
    case ConstraintType::Petzval:
        return true;
    }
    return false;
}

std::optional<double> constraint_target(const SystemDescription& system, ConstraintType type) {
    if (type == ConstraintType::ParaxialFocus) {
        return 0.0;
    }
    if (type == ConstraintType::Petzval) {
        return std::nullopt; // Grenze im Eintrag (max_curvature)
    }
    if (!system.target) {
        return std::nullopt;
    }
    switch (type) {
    case ConstraintType::FocalLength:
        return system.target->focal_length;
    case ConstraintType::ParfocalLength:
        return system.target->parfocal_length;
    case ConstraintType::WorkingDistance:
        return system.target->working_distance;
    case ConstraintType::ParaxialFocus:
    case ConstraintType::Petzval:
        break;
    }
    return std::nullopt;
}

std::optional<double> constraint_bound(const SystemDescription& system,
                                       const ConstraintSpec& spec) {
    if (spec.type == ConstraintType::Petzval) {
        return spec.max_curvature;
    }
    return constraint_target(system, spec.type);
}

void check_constraints(const SystemDescription& system) {
    if (!system.constraints) {
        return;
    }
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    const int surfaces = static_cast<int>(system.surfaces.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        const ConstraintSpec& c = items[i];
        const std::string path = at("constraints.items", i);
        const std::string name(to_string(c.type));
        if (!constraint_type_from_string(name)) {
            fail(path + ".type", "unbekannter Typ einer Nebenbedingung");
        }

        if (c.mode) {
            if (!constraint_mode_from_string(to_string(*c.mode))) {
                fail(path + ".mode", "unbekannter Modus");
            }
            if (equality_only(c.type) && *c.mode != ConstraintMode::Equal) {
                fail(path + ".mode", name + " ist nur als Gleichung definiert (\"mode\": "
                                            "\"equal\" oder Feld weglassen)");
            }
            if (c.type == ConstraintType::Petzval && *c.mode != ConstraintMode::Max) {
                fail(path + ".mode", "petzval ist nur als Obergrenze |κ_P| ≤ max_curvature "
                                     "definiert (\"mode\": \"max\" oder Feld weglassen)");
            }
        }
        if (c.weight && !(std::isfinite(*c.weight) && *c.weight >= 0.0)) {
            fail(path + ".weight",
                 "Gewicht muss endlich und mindestens 0 sein, ist " + num(*c.weight));
        }
        if (c.type != ConstraintType::ParfocalLength) {
            if (c.reference_surface) {
                fail(path + ".reference_surface", "nur bei parfocal_length erlaubt");
            }
            if (c.offset) {
                fail(path + ".offset", "nur bei parfocal_length erlaubt");
            }
        }
        if (c.reference_surface && (*c.reference_surface < 0 || *c.reference_surface >= surfaces)) {
            fail(path + ".reference_surface", "Fläche " + std::to_string(*c.reference_surface) +
                                                  " existiert nicht (Flächen 0 bis " +
                                                  std::to_string(surfaces - 1) + ")");
        }
        if (c.offset && !std::isfinite(*c.offset)) {
            fail(path + ".offset", "Zahl muss endlich sein");
        }
        if (c.type == ConstraintType::Petzval) {
            // Grenze im Eintrag (kein Gegenstück in `target`)
            if (!c.max_curvature) {
                fail(path + ".max_curvature",
                     "Pflichtfeld bei petzval fehlt (größte zulässige |κ_P| in 1/mm)");
            }
            if (!(std::isfinite(*c.max_curvature) && *c.max_curvature >= 0.0)) {
                fail(path + ".max_curvature",
                     "muss endlich und mindestens 0 sein, ist " + num(*c.max_curvature));
            }
        } else if (c.max_curvature) {
            fail(path + ".max_curvature", "nur bei petzval erlaubt");
        }

        // Sollwert: eine Quelle der Wahrheit in `target` (Petzval: Grenze im Eintrag)
        if (!constraint_bound(system, c)) {
            fail(path + ".type", "Sollwert target." + std::string(target_field(c.type)) +
                                     " fehlt (Sollwerte stehen in \"target\")");
        }
        if (c.type == ConstraintType::ParaxialFocus &&
            system.source.type != SourceType::Collimated) {
            fail(path + ".type", "paraxial_focus (Bildebene im Brennpunkt F') verlangt eine "
                                 "kollimierte Quelle");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (items[j].type == c.type) {
                fail(path + ".type", "Nebenbedingung doppelt: " + name + " steht schon in " +
                                         at("constraints.items", j));
            }
        }
    }
}

} // namespace mko
