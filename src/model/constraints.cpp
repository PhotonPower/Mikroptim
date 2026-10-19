// Validierung des Abschnitts `constraints` (docs/dateiformat.md, „Nebenbedingungen“).
// Die Pfade in den Meldungen entsprechen den JSON-Pfaden der Datei.

#include "mikroptim/model/constraints.hpp"

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
    case ConstraintType::GhostPenalty:
    case ConstraintType::GlassHull:
    case ConstraintType::MaxSemiDiameter:
    case ConstraintType::MinCenterThickness:
    case ConstraintType::MaxCenterThickness:
    case ConstraintType::MinEdgeThickness:
    case ConstraintType::MaxIncidenceAngle:
        return {};
    }
    return {};
}

// Prüft eine Zahl > 0 und endlich (Felder der Geisterplatten-Strafe).
void check_positive(const std::optional<double>& v, const std::string& path) {
    if (v && !(std::isfinite(*v) && *v > 0.0)) {
        fail(path, "muss endlich und größer als 0 sein, ist " + num(*v));
    }
}

// Geisterplatten eines ghost_penalty-Eintrags mit dem JSON-Pfad für Fehler; fehlt die Auswahl,
// gilt der Eintrag für alle Platten.
struct PlateRef {
    int plate;
    std::string path;
};

std::vector<PlateRef> penalty_plates(const ConstraintSpec& c, const std::string& path,
                                     int ghost_count) {
    std::vector<PlateRef> out;
    switch (c.plate_selection) {
    case SurfaceSelection::None:
    case SurfaceSelection::All:
        if (!c.plates.empty()) {
            fail(path + ".plates", "bei \"all\" keine einzelnen Geisterplatten");
        }
        for (int g = 0; g < ghost_count; ++g) {
            out.push_back(
                {g, path + (c.plate_selection == SurfaceSelection::All ? ".plates" : ".type")});
        }
        break;
    case SurfaceSelection::One:
        if (c.plates.size() != 1) {
            fail(path + ".plate", "genau eine Geisterplatte erwartet");
        }
        out.push_back({c.plates[0], path + ".plate"});
        break;
    case SurfaceSelection::List:
        if (c.plates.empty()) {
            fail(path + ".plates", "mindestens eine Geisterplatte nötig");
        }
        for (std::size_t j = 0; j < c.plates.size(); ++j) {
            out.push_back({c.plates[j], at(path + ".plates", j)});
        }
        break;
    }
    for (const PlateRef& p : out) {
        if (p.plate < 0 || p.plate >= ghost_count) {
            fail(p.path, "Geisterplatte " + std::to_string(p.plate) + " existiert nicht" +
                             (ghost_count == 0 ? std::string(" (keine Geisterplatten)")
                                               : " (Geisterplatten 0 bis " +
                                                     std::to_string(ghost_count - 1) + ")"));
        }
    }
    return out;
}

// Dicken-Bedingungen (Mittendicke, Randdicke): gelten für Spalte i → i + 1, i ≤ S − 2.
bool is_gap_type(ConstraintType type) {
    return type == ConstraintType::MinCenterThickness ||
           type == ConstraintType::MaxCenterThickness || type == ConstraintType::MinEdgeThickness;
}

// Medium nach Dateifläche i: Luft ("AIR") oder Glas (jedes andere Material).
ConstraintMedium medium_after(const SystemDescription& system, int i) {
    return system.surfaces[static_cast<std::size_t>(i)].material.kind == MaterialKind::Air
               ? ConstraintMedium::Air
               : ConstraintMedium::Glass;
}

// Flächen eines mechanischen Eintrags (docs/dateiformat.md), Fehler mit dem JSON-Pfad unter
// `path` (Eintrag). Standardauswahl (Feld fehlt oder "all"): Flächen 1 … S − 1 bei
// max_semi_diameter und max_incidence_angle, bei den Dicken alle Spalte 0 … S − 2 mit passendem
// Medium. Einzelne Flächen müssen existieren, bei den Dicken im Bereich 0 … S − 2 liegen und
// zum Medium passen; keine Fläche doppelt, Ergebnis nicht leer.
std::vector<int> resolve_surfaces(const SystemDescription& system, const ConstraintSpec& c,
                                  const std::string& path) {
    const int S = static_cast<int>(system.surfaces.size());
    const bool gap = is_gap_type(c.type);
    const int last = gap ? S - 2 : S - 1; // größte erlaubte Fläche
    const auto medium_ok = [&](int i) { return !c.medium || medium_after(system, i) == *c.medium; };
    std::vector<int> out;
    switch (c.surface_selection) {
    case SurfaceSelection::None:
    case SurfaceSelection::All:
        if (!c.surfaces.empty()) {
            fail(path + ".surfaces", "bei \"all\" keine einzelnen Flächen");
        }
        for (int i = gap ? 0 : 1; i <= last; ++i) {
            if (medium_ok(i)) {
                out.push_back(i);
            }
        }
        if (out.empty()) {
            fail(path + ".type",
                 std::string(to_string(c.type)) + ": keine passende Fläche" +
                     (c.medium ? " mit Medium \"" + std::string(to_string(*c.medium)) + "\""
                               : std::string()));
        }
        return out;
    case SurfaceSelection::One:
    case SurfaceSelection::List: {
        const bool one = c.surface_selection == SurfaceSelection::One;
        if (one && c.surfaces.size() != 1) {
            fail(path + ".surface", "genau eine Fläche erwartet");
        }
        if (!one && c.surfaces.empty()) {
            fail(path + ".surfaces", "mindestens eine Fläche nötig");
        }
        for (std::size_t j = 0; j < c.surfaces.size(); ++j) {
            const int i = c.surfaces[j];
            const std::string p = one ? path + ".surface" : at(path + ".surfaces", j);
            if (i < 0 || i > last) {
                fail(p, "Fläche " + std::to_string(i) + " nicht erlaubt (Flächen 0 bis " +
                            std::to_string(last) +
                            (gap ? "; Dicken-Bedingungen gelten für den Spalt nach der Fläche, "
                                   "der Arbeitsabstand hat eigene Bedingungen)"
                                 : ")"));
            }
            if (!medium_ok(i)) {
                fail(p, "nach Fläche " + std::to_string(i) + " liegt " +
                            (medium_after(system, i) == ConstraintMedium::Air ? "Luft" : "Glas") +
                            ", der Eintrag verlangt \"medium\": \"" +
                            std::string(to_string(*c.medium)) + "\"");
            }
            for (std::size_t k = 0; k < j; ++k) {
                if (c.surfaces[k] == i) {
                    fail(p, "Fläche " + std::to_string(i) + " doppelt");
                }
            }
            out.push_back(i);
        }
        return out;
    }
    }
    return out;
}

// Prüft die Felder eines mechanischen Eintrags (ohne Modus, Grenzen im Eintrag).
void check_mechanical(const SystemDescription& system, const ConstraintSpec& c,
                      const std::string& path) {
    const std::string name(to_string(c.type));
    if (c.mode) {
        fail(path + ".mode", name + " hat keinen Modus (die Richtung steckt im Typ; Feld "
                                    "weglassen)");
    }
    const bool gap = is_gap_type(c.type);
    const auto forbid = [&](bool present, const char* field) {
        if (present) {
            fail(path + "." + field, std::string("bei ") + name + " nicht erlaubt");
        }
    };
    forbid(c.semi_diameter.has_value() && c.type != ConstraintType::MaxSemiDiameter,
           "semi_diameter");
    forbid(c.thickness.has_value() && !gap, "thickness");
    forbid(c.angle.has_value() && c.type != ConstraintType::MaxIncidenceAngle, "angle");
    forbid(c.medium.has_value() && !gap, "medium");
    forbid(c.clearance.has_value() && c.type != ConstraintType::MaxSemiDiameter &&
               c.type != ConstraintType::MinEdgeThickness,
           "clearance");
    forbid(c.height.has_value() && c.type != ConstraintType::MinEdgeThickness, "height");

    if (c.semi_diameter && !(std::isfinite(*c.semi_diameter) && *c.semi_diameter > 0.0)) {
        fail(path + ".semi_diameter",
             "muss endlich und größer als 0 sein, ist " + num(*c.semi_diameter));
    }
    if (c.clearance && !(std::isfinite(*c.clearance) && *c.clearance >= 0.0)) {
        fail(path + ".clearance", "muss endlich und mindestens 0 sein, ist " + num(*c.clearance));
    }
    if (gap) {
        if (!c.thickness) {
            fail(path + ".thickness", "Pflichtfeld bei " + name + " fehlt (Grenze in mm)");
        }
        const bool strict = c.type == ConstraintType::MaxCenterThickness;
        if (!(std::isfinite(*c.thickness) && (strict ? *c.thickness > 0.0 : *c.thickness >= 0.0))) {
            fail(path + ".thickness", std::string("muss endlich und ") +
                                          (strict ? "größer als 0" : "mindestens 0") +
                                          " sein, ist " + num(*c.thickness));
        }
    }
    if (c.type == ConstraintType::MaxIncidenceAngle) {
        if (!c.angle) {
            fail(path + ".angle", "Pflichtfeld bei max_incidence_angle fehlt (Grenze in Grad)");
        }
        if (!(std::isfinite(*c.angle) && *c.angle > 0.0 && *c.angle < 90.0)) {
            fail(path + ".angle",
                 "muss zwischen 0 und 90 Grad liegen (ausschließlich), ist " + num(*c.angle));
        }
    }
    resolve_surfaces(system, c, path);
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
    case ConstraintType::GhostPenalty:
        return "ghost_penalty";
    case ConstraintType::GlassHull:
        return "glass_hull";
    case ConstraintType::MaxSemiDiameter:
        return "max_semi_diameter";
    case ConstraintType::MinCenterThickness:
        return "min_center_thickness";
    case ConstraintType::MaxCenterThickness:
        return "max_center_thickness";
    case ConstraintType::MinEdgeThickness:
        return "min_edge_thickness";
    case ConstraintType::MaxIncidenceAngle:
        return "max_incidence_angle";
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

std::string_view to_string(ConstraintMedium medium) {
    switch (medium) {
    case ConstraintMedium::Glass:
        return "glass";
    case ConstraintMedium::Air:
        return "air";
    }
    return "?";
}

std::string_view to_string(EdgeHeight height) {
    switch (height) {
    case EdgeHeight::SemiDiameter:
        return "semi_diameter";
    case EdgeHeight::Rays:
        return "rays";
    }
    return "?";
}

std::optional<ConstraintType> constraint_type_from_string(std::string_view name) {
    for (const ConstraintType t :
         {ConstraintType::FocalLength, ConstraintType::ParfocalLength,
          ConstraintType::WorkingDistance, ConstraintType::ParaxialFocus, ConstraintType::Petzval,
          ConstraintType::GhostPenalty, ConstraintType::GlassHull, ConstraintType::MaxSemiDiameter,
          ConstraintType::MinCenterThickness, ConstraintType::MaxCenterThickness,
          ConstraintType::MinEdgeThickness, ConstraintType::MaxIncidenceAngle}) {
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

std::optional<ConstraintMedium> constraint_medium_from_string(std::string_view name) {
    for (const ConstraintMedium m : {ConstraintMedium::Glass, ConstraintMedium::Air}) {
        if (name == to_string(m)) {
            return m;
        }
    }
    return std::nullopt;
}

std::optional<EdgeHeight> edge_height_from_string(std::string_view name) {
    for (const EdgeHeight h : {EdgeHeight::SemiDiameter, EdgeHeight::Rays}) {
        if (name == to_string(h)) {
            return h;
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
    case ConstraintType::GhostPenalty:
    case ConstraintType::GlassHull:
    case ConstraintType::MaxSemiDiameter:
    case ConstraintType::MinCenterThickness:
    case ConstraintType::MaxCenterThickness:
    case ConstraintType::MinEdgeThickness:
    case ConstraintType::MaxIncidenceAngle:
        return false;
    }
    return false;
}

bool is_mechanical(ConstraintType type) {
    switch (type) {
    case ConstraintType::MaxSemiDiameter:
    case ConstraintType::MinCenterThickness:
    case ConstraintType::MaxCenterThickness:
    case ConstraintType::MinEdgeThickness:
    case ConstraintType::MaxIncidenceAngle:
        return true;
    case ConstraintType::FocalLength:
    case ConstraintType::ParfocalLength:
    case ConstraintType::WorkingDistance:
    case ConstraintType::ParaxialFocus:
    case ConstraintType::Petzval:
    case ConstraintType::GhostPenalty:
    case ConstraintType::GlassHull:
        return false;
    }
    return false;
}

std::optional<double> constraint_target(const SystemDescription& system, ConstraintType type) {
    if (type == ConstraintType::ParaxialFocus) {
        return 0.0;
    }
    if (type == ConstraintType::Petzval || type == ConstraintType::GhostPenalty ||
        type == ConstraintType::GlassHull || is_mechanical(type)) {
        return std::nullopt; // Grenze im Eintrag (max_curvature usw.) bzw. keine Sollgröße
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
    case ConstraintType::GhostPenalty:
    case ConstraintType::GlassHull:
    case ConstraintType::MaxSemiDiameter:
    case ConstraintType::MinCenterThickness:
    case ConstraintType::MaxCenterThickness:
    case ConstraintType::MinEdgeThickness:
    case ConstraintType::MaxIncidenceAngle:
        break;
    }
    return std::nullopt;
}

std::optional<double> constraint_bound(const SystemDescription& system,
                                       const ConstraintSpec& spec) {
    switch (spec.type) {
    case ConstraintType::Petzval:
        return spec.max_curvature;
    case ConstraintType::MaxSemiDiameter:
        return spec.semi_diameter;
    case ConstraintType::MinCenterThickness:
    case ConstraintType::MaxCenterThickness:
    case ConstraintType::MinEdgeThickness:
        return spec.thickness;
    case ConstraintType::MaxIncidenceAngle:
        return spec.angle;
    default:
        break;
    }
    return constraint_target(system, spec.type);
}

std::vector<int> constraint_surfaces(const SystemDescription& system, const ConstraintSpec& spec) {
    if (!is_mechanical(spec.type)) {
        return {};
    }
    return resolve_surfaces(system, spec, "constraints");
}

void check_constraints(const SystemDescription& system) {
    if (!system.constraints) {
        return;
    }
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    const int surfaces = static_cast<int>(system.surfaces.size());
    const int ghost_count = static_cast<int>(system.ghost_plates.size());
    std::vector<int> penalty_item(static_cast<std::size_t>(ghost_count), -1); // Eintrag je Platte
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
        if (!is_mechanical(c.type)) {
            // Felder der mechanischen Bedingungen (M4-T3) sind bei den übrigen Typen verboten
            if (c.surface_selection != SurfaceSelection::None) {
                fail(path +
                         (c.surface_selection == SurfaceSelection::One ? ".surface" : ".surfaces"),
                     "nur bei mechanischen Bedingungen erlaubt");
            }
            const std::pair<bool, const char*> fields[] = {
                {c.medium.has_value(), "medium"},
                {c.semi_diameter.has_value(), "semi_diameter"},
                {c.thickness.has_value(), "thickness"},
                {c.angle.has_value(), "angle"},
                {c.clearance.has_value(), "clearance"},
                {c.height.has_value(), "height"}};
            for (const auto& [present, field] : fields) {
                if (present) {
                    fail(path + "." + field, "nur bei mechanischen Bedingungen erlaubt");
                }
            }
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

        if (c.type == ConstraintType::GlassHull) {
            if (c.margin && !(std::isfinite(*c.margin) && *c.margin >= 0.0)) {
                fail(path + ".margin", "muss endlich und mindestens 0 sein, ist " + num(*c.margin));
            }
        } else if (c.margin) {
            fail(path + ".margin", "nur bei glass_hull erlaubt");
        }

        if (c.type == ConstraintType::GlassHull) {
            // Keine Sollgröße, kein Modus, höchstens ein Eintrag (gilt für alle Glasvariablen).
            if (c.mode) {
                fail(path + ".mode", "glass_hull hat keinen Modus (Feld weglassen)");
            }
            if (c.plate_selection != SurfaceSelection::None) {
                fail(path + (c.plate_selection == SurfaceSelection::One ? ".plate" : ".plates"),
                     "nur bei ghost_penalty erlaubt");
            }
            if (c.order_weight) {
                fail(path + ".order_weight", "nur bei ghost_penalty erlaubt");
            }
            if (c.order_factor) {
                fail(path + ".order_factor", "nur bei ghost_penalty erlaubt");
            }
            if (c.smoothing) {
                fail(path + ".smoothing", "nur bei ghost_penalty erlaubt");
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (items[j].type == c.type) {
                    fail(path + ".type", "Nebenbedingung doppelt: glass_hull steht schon in " +
                                             at("constraints.items", j));
                }
            }
            continue;
        }

        if (c.type == ConstraintType::GhostPenalty) {
            // Keine Sollgröße, kein Modus; jede Geisterplatte in höchstens einem Eintrag.
            if (c.mode) {
                fail(path + ".mode", "ghost_penalty hat keinen Modus (Feld weglassen)");
            }
            check_positive(c.order_weight, path + ".order_weight");
            check_positive(c.order_factor, path + ".order_factor");
            check_positive(c.smoothing, path + ".smoothing");
            for (const PlateRef& p : penalty_plates(c, path, ghost_count)) {
                int& owner = penalty_item[static_cast<std::size_t>(p.plate)];
                if (owner >= 0) {
                    fail(p.path, "Geisterplatte " + std::to_string(p.plate) + " steht schon in " +
                                     at("constraints.items", static_cast<std::size_t>(owner)));
                }
                owner = static_cast<int>(i);
            }
            continue;
        }
        if (c.plate_selection != SurfaceSelection::None) {
            fail(path + (c.plate_selection == SurfaceSelection::One ? ".plate" : ".plates"),
                 "nur bei ghost_penalty erlaubt");
        }
        if (c.order_weight) {
            fail(path + ".order_weight", "nur bei ghost_penalty erlaubt");
        }
        if (c.order_factor) {
            fail(path + ".order_factor", "nur bei ghost_penalty erlaubt");
        }
        if (c.smoothing) {
            fail(path + ".smoothing", "nur bei ghost_penalty erlaubt");
        }
        if (is_mechanical(c.type)) {
            // Grenzen im Eintrag, mehrere Einträge eines Typs erlaubt (M4-T3)
            check_mechanical(system, c, path);
            continue;
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
