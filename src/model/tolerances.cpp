// Auflösung und Validierung des Abschnitts `tolerances` (docs/dateiformat.md, „Toleranzen“).
// Die Pfade in den Meldungen entsprechen den JSON-Pfaden der Datei.

#include "mikroptim/model/tolerances.hpp"

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"

#include <cmath>
#include <cstddef>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>

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

bool is_surface_type(ToleranceType t) {
    return t != ToleranceType::ElementDecenter && t != ToleranceType::ElementTilt;
}

bool is_glass(const Surface& s) {
    return s.material.kind != MaterialKind::Air;
}

// Grenze ±δ ≥ 0 (endlich prüft der Leser; hier auch für Modelle aus dem Code).
void check_limit(const std::string& path, double v) {
    if (!std::isfinite(v) || v < 0.0) {
        fail(path, "muss endlich und >= 0 sein, ist " + num(v));
    }
}

// Ein Feld, das bei diesem Typ nicht erlaubt ist.
template <class T>
void forbid(const std::optional<T>& field, const std::string& path, std::string_view type) {
    if (field) {
        fail(path, "bei " + std::string(type) + " nicht erlaubt");
    }
}

// Flächen eines Eintrags mit Flächenauswahl (One, List, All) als Paare (Fläche, Pfad).
struct SurfaceRef {
    int surface;
    std::string path;
};

std::vector<SurfaceRef> surfaces_of(const ToleranceSpec& spec, const std::string& path,
                                    const SystemDescription& system) {
    const int count = static_cast<int>(system.surfaces.size());
    const std::string_view type = to_string(spec.type);
    std::vector<SurfaceRef> out;
    if (spec.selection == SurfaceSelection::None) {
        fail(path + ".surfaces",
             "\"surface\" oder \"surfaces\" fehlt (Pflicht bei " + std::string(type) + ")");
    }
    if (spec.selection == SurfaceSelection::All) {
        for (int s = 0; s < count; ++s) {
            const Surface& sf = system.surfaces[static_cast<std::size_t>(s)];
            bool take = false;
            switch (spec.type) {
            case ToleranceType::Radius:
                // relativ nur bei endlichem Radius (c ≠ 0)
                take = sf.type == SurfaceType::Sphere && (spec.fringes || sf.curvature != 0.0);
                break;
            case ToleranceType::Thickness:
                take = s < count - 1;
                break;
            case ToleranceType::Index:
            case ToleranceType::Abbe:
                take = is_glass(sf);
                break;
            default: // Decenter, Tilt
                take = s >= 1;
                break;
            }
            if (take) {
                out.push_back(SurfaceRef{s, path + ".surfaces"});
            }
        }
        if (out.empty()) {
            fail(path + ".surfaces",
                 "\"all\" findet keine passende Fläche für " + std::string(type));
        }
        return out;
    }
    if (spec.surfaces.empty()) {
        fail(path + ".surfaces", "Liste darf nicht leer sein");
    }
    for (std::size_t k = 0; k < spec.surfaces.size(); ++k) {
        const std::string p =
            spec.selection == SurfaceSelection::One ? path + ".surface" : at(path + ".surfaces", k);
        const int s = spec.surfaces[k];
        if (s < 0 || s >= count) {
            fail(p, "Fläche " + std::to_string(s) + " existiert nicht (0 … " +
                        std::to_string(count - 1) + ")");
        }
        const Surface& sf = system.surfaces[static_cast<std::size_t>(s)];
        switch (spec.type) {
        case ToleranceType::Radius:
            if (sf.type != SurfaceType::Sphere) {
                fail(p, "radius nur bei Flächen vom Typ sphere (Fläche " + std::to_string(s) +
                            " ist " + std::string(to_string(sf.type)) + ")");
            }
            if (spec.relative && sf.curvature == 0.0) {
                fail(p, "relative Radiustoleranz verlangt einen endlichen Radius (Fläche " +
                            std::to_string(s) + " hat die Krümmung 0; Passen verwenden)");
            }
            break;
        case ToleranceType::Thickness:
            if (s == count - 1) {
                fail(p, "thickness nicht für die letzte Fläche (der Arbeitsabstand ist der "
                        "Kompensator)");
            }
            break;
        case ToleranceType::Index:
        case ToleranceType::Abbe:
            if (!is_glass(sf)) {
                fail(p, std::string(type) + " verlangt Glas nach der Fläche (Fläche " +
                            std::to_string(s) + " hat \"AIR\")");
            }
            break;
        default: // Decenter, Tilt
            if (s == 0) {
                fail(p, "die Blende (Fläche 0) ist das Bezugssystem und wird nicht bewegt");
            }
            break;
        }
        out.push_back(SurfaceRef{s, p});
    }
    return out;
}

struct ElementRef {
    std::array<int, 2> range;
    std::string path;
};

std::vector<ElementRef> elements_of(const ToleranceSpec& spec, const std::string& path,
                                    const SystemDescription& system) {
    const int count = static_cast<int>(system.surfaces.size());
    std::vector<ElementRef> out;
    if (spec.element_selection == SurfaceSelection::None) {
        fail(path + ".elements", "\"element\" oder \"elements\" fehlt (Pflicht bei " +
                                     std::string(to_string(spec.type)) + ")");
    }
    if (spec.element_selection == SurfaceSelection::All) {
        for (const std::array<int, 2>& e : all_elements(system)) {
            out.push_back(ElementRef{e, path + ".elements"});
        }
        if (out.empty()) {
            fail(path + ".elements", "\"all\" findet kein Element (keine Fläche mit Glas)");
        }
        return out;
    }
    if (spec.elements.empty()) {
        fail(path + ".elements", "Liste darf nicht leer sein");
    }
    for (std::size_t k = 0; k < spec.elements.size(); ++k) {
        const std::string p = spec.element_selection == SurfaceSelection::One
                                  ? path + ".element"
                                  : at(path + ".elements", k);
        const std::array<int, 2>& e = spec.elements[k];
        if (e[0] < 1 || e[0] > e[1] || e[1] >= count) {
            fail(p, "Element [" + std::to_string(e[0]) + ", " + std::to_string(e[1]) +
                        "] ungültig: verlangt 1 <= erste <= letzte <= " +
                        std::to_string(count - 1) + " (die Blende bewegt sich nicht)");
        }
        out.push_back(ElementRef{e, p});
    }
    return out;
}

// Grenze des Eintrags in internen Einheiten (ohne die flächenabhängige Umrechnung der Passen).
double tilt_limit_rad(const ToleranceSpec& spec, const std::string& path) {
    const bool am = spec.arcmin.has_value();
    const bool dg = spec.deg.has_value();
    if (am == dg) {
        fail(path + (am ? ".deg" : ".arcmin"),
             "genau eines der Felder \"arcmin\" und \"deg\" angeben");
    }
    const double deg = am ? *spec.arcmin / 60.0 : *spec.deg;
    check_limit(path + (am ? ".arcmin" : ".deg"), am ? *spec.arcmin : *spec.deg);
    if (!(deg < 90.0)) {
        fail(path + (am ? ".arcmin" : ".deg"), "Kippwinkel muss kleiner als 90° sein");
    }
    return deg * (std::numbers::pi / 180.0);
}

// Prüft die Felder, die nur bei bestimmten Typen erlaubt sind, und liefert die Grenze (bei
// Radius in Passen die Zahl der Passen; die Umrechnung folgt pro Fläche).
double item_limit(const ToleranceSpec& spec, const std::string& path) {
    const std::string_view type = to_string(spec.type);
    const bool surface_type = is_surface_type(spec.type);
    if (!surface_type && spec.selection != SurfaceSelection::None) {
        fail(path + (spec.selection == SurfaceSelection::One ? ".surface" : ".surfaces"),
             "bei " + std::string(type) +
                 " nicht erlaubt (Elemente über \"element\" bzw. "
                 "\"elements\")");
    }
    if (surface_type && spec.element_selection != SurfaceSelection::None) {
        fail(path + (spec.element_selection == SurfaceSelection::One ? ".element" : ".elements"),
             "bei " + std::string(type) + " nicht erlaubt");
    }
    if (spec.type != ToleranceType::ElementTilt && spec.pivot) {
        fail(path + ".pivot", "nur bei element_tilt erlaubt");
    }
    if (spec.type != ToleranceType::Radius) {
        forbid(spec.fringes, path + ".fringes", type);
        forbid(spec.test_wavelength_nm, path + ".test_wavelength_nm", type);
    }
    if (spec.type != ToleranceType::Tilt && spec.type != ToleranceType::ElementTilt) {
        forbid(spec.arcmin, path + ".arcmin", type);
        forbid(spec.deg, path + ".deg", type);
    }
    if (spec.type != ToleranceType::Radius && spec.type != ToleranceType::Abbe) {
        forbid(spec.relative, path + ".relative", type);
    }
    if (spec.cutoff_sigma) {
        const double c = *spec.cutoff_sigma;
        if (!(c >= kMinToleranceCutoff && c <= kMaxToleranceCutoff)) {
            fail(path + ".cutoff_sigma", "muss in [" + num(kMinToleranceCutoff) + ", " +
                                             num(kMaxToleranceCutoff) + "] liegen, ist " + num(c));
        }
    }
    switch (spec.type) {
    case ToleranceType::Radius: {
        forbid(spec.delta, path + ".delta", type);
        if (spec.relative.has_value() == spec.fringes.has_value()) {
            fail(path + (spec.relative ? ".fringes" : ".relative"),
                 "genau eines der Felder \"relative\" und \"fringes\" angeben");
        }
        if (spec.test_wavelength_nm && !spec.fringes) {
            fail(path + ".test_wavelength_nm", "nur zusammen mit \"fringes\" erlaubt");
        }
        if (spec.test_wavelength_nm &&
            !(std::isfinite(*spec.test_wavelength_nm) && *spec.test_wavelength_nm > 0.0)) {
            fail(path + ".test_wavelength_nm", "muss > 0 sein");
        }
        if (spec.relative) {
            check_limit(path + ".relative", *spec.relative);
            if (!(*spec.relative < 1.0)) {
                fail(path + ".relative", "muss kleiner als 1 sein");
            }
            return *spec.relative;
        }
        check_limit(path + ".fringes", *spec.fringes);
        return *spec.fringes;
    }
    case ToleranceType::Abbe:
        forbid(spec.delta, path + ".delta", type);
        if (!spec.relative) {
            fail(path + ".relative", "Pflichtfeld fehlt (relative Abweichung Δν/ν)");
        }
        check_limit(path + ".relative", *spec.relative);
        if (!(*spec.relative < 1.0)) {
            fail(path + ".relative", "muss kleiner als 1 sein");
        }
        return *spec.relative;
    case ToleranceType::Thickness:
    case ToleranceType::Index:
    case ToleranceType::Decenter:
    case ToleranceType::ElementDecenter:
        if (!spec.delta) {
            fail(path + ".delta", "Pflichtfeld fehlt");
        }
        check_limit(path + ".delta", *spec.delta);
        return *spec.delta;
    case ToleranceType::Tilt:
    case ToleranceType::ElementTilt:
        forbid(spec.delta, path + ".delta", type);
        return tilt_limit_rad(spec, path);
    }
    return 0.0;
}

// Δc pro Passe: Eine Passe (Newtonring) ist eine Pfeilhöhendifferenz λ_t/2 über den
// Halbdurchmesser h; mit ds/dc = h²/(w (1 + w)), w = √(1 − c²h²) (exakte Ableitung der
// Pfeilhöhe am nominellen c) ist Δc = N · (λ_t/2) · w (1 + w) / h².
double fringe_curvature(double fringes, double lambda_nm, double c, double h,
                        const std::string& path) {
    const double ch = c * h;
    if (!(h > 0.0) || !(ch * ch < 1.0)) {
        fail(path, "Passen verlangen |c| · Halbdurchmesser < 1");
    }
    const double w = std::sqrt(1.0 - ch * ch);
    return fringes * (0.5 * lambda_nm * 1e-6) * w * (1.0 + w) / (h * h);
}

} // namespace

int ResolvedTolerance::draws() const {
    switch (type) {
    case ToleranceType::Decenter:
    case ToleranceType::Tilt:
    case ToleranceType::ElementDecenter:
    case ToleranceType::ElementTilt:
        return 2;
    default:
        return 1;
    }
}

ToleranceDistribution Tolerances::distribution_or_default() const {
    return distribution.value_or(kDefaultToleranceDistribution);
}

double Tolerances::cutoff_sigma_or_default() const {
    return cutoff_sigma.value_or(kDefaultToleranceCutoff);
}

CompensatorType Tolerances::compensator_or_default() const {
    return compensator.value_or(kDefaultCompensator);
}

double nominal_vertex_z(const SystemDescription& system, int surface) {
    double z = 0.0;
    for (int i = 0; i < surface && i < static_cast<int>(system.surfaces.size()); ++i) {
        z += system.surfaces[static_cast<std::size_t>(i)].thickness;
    }
    return z;
}

std::vector<std::array<int, 2>> all_elements(const SystemDescription& system) {
    std::vector<std::array<int, 2>> out;
    const int count = static_cast<int>(system.surfaces.size());
    int s = 1;
    while (s < count) {
        if (!is_glass(system.surfaces[static_cast<std::size_t>(s)])) {
            ++s;
            continue;
        }
        int last = s;
        while (last < count - 1 && is_glass(system.surfaces[static_cast<std::size_t>(last)])) {
            ++last;
        }
        // last: erste Fläche mit Luft danach (oder die letzte Fläche, wenn das System im Glas
        // endet; dann gehört sie zum Element)
        out.push_back({s, last});
        s = last + 1;
    }
    return out;
}

std::vector<ResolvedTolerance> resolve_tolerances(const SystemDescription& system) {
    std::vector<ResolvedTolerance> out;
    if (!system.tolerances) {
        return out;
    }
    const Tolerances& tol = *system.tolerances;
    if (tol.cutoff_sigma) {
        const double c = *tol.cutoff_sigma;
        if (!(c >= kMinToleranceCutoff && c <= kMaxToleranceCutoff)) {
            fail("tolerances.cutoff_sigma", "muss in [" + num(kMinToleranceCutoff) + ", " +
                                                num(kMaxToleranceCutoff) + "] liegen, ist " +
                                                num(c));
        }
    }
    const ToleranceDistribution section_distribution = tol.distribution_or_default();
    const double section_cutoff = tol.cutoff_sigma_or_default();
    // Erster Pfad jeder Fläche pro Typ (keine Fläche doppelt in einem Typ; bei Elementen jede
    // Fläche in höchstens einem Element pro Typ).
    std::map<std::pair<int, int>, std::string> seen;
    for (std::size_t i = 0; i < tol.items.size(); ++i) {
        const ToleranceSpec& spec = tol.items[i];
        const std::string path = at("tolerances.items", i);
        const double limit = item_limit(spec, path);
        ResolvedTolerance base;
        base.type = spec.type;
        base.item = static_cast<int>(i);
        base.limit = limit;
        base.distribution = spec.distribution.value_or(section_distribution);
        base.cutoff = spec.cutoff_sigma.value_or(section_cutoff);
        const int type_key = static_cast<int>(spec.type);
        const auto claim = [&](int s, const std::string& p) {
            const auto [it, inserted] = seen.emplace(std::make_pair(type_key, s), p);
            if (!inserted) {
                fail(p, "Fläche " + std::to_string(s) + " hat schon eine Toleranz vom Typ " +
                            std::string(to_string(spec.type)) + " (" + it->second + ")");
            }
        };
        if (is_surface_type(spec.type)) {
            for (const SurfaceRef& ref : surfaces_of(spec, path, system)) {
                claim(ref.surface, ref.path);
                ResolvedTolerance r = base;
                r.surface = ref.surface;
                if (spec.type == ToleranceType::Radius && spec.fringes) {
                    const Surface& sf = system.surfaces[static_cast<std::size_t>(ref.surface)];
                    r.radius_fringes = true;
                    r.limit = fringe_curvature(
                        *spec.fringes, spec.test_wavelength_nm.value_or(kDefaultTestWavelengthNm),
                        sf.curvature, sf.semi_diameter, ref.path);
                }
                out.push_back(r);
            }
        } else {
            for (const ElementRef& ref : elements_of(spec, path, system)) {
                for (int s = ref.range[0]; s <= ref.range[1]; ++s) {
                    claim(s, ref.path);
                }
                ResolvedTolerance r = base;
                r.first = ref.range[0];
                r.last = ref.range[1];
                if (spec.type == ToleranceType::ElementTilt) {
                    const double z_first = nominal_vertex_z(system, r.first);
                    const double z_last = nominal_vertex_z(system, r.last);
                    switch (spec.pivot.value_or(ElementPivot::Center)) {
                    case ElementPivot::First:
                        r.pivot_z = z_first;
                        break;
                    case ElementPivot::Last:
                        r.pivot_z = z_last;
                        break;
                    case ElementPivot::Center:
                        r.pivot_z = 0.5 * (z_first + z_last);
                        break;
                    case ElementPivot::Offset:
                        if (!std::isfinite(spec.pivot_offset)) {
                            fail(path + ".pivot", "Abstand muss endlich sein");
                        }
                        r.pivot_z = z_first + spec.pivot_offset;
                        break;
                    }
                }
                out.push_back(r);
            }
        }
    }
    return out;
}

std::string_view to_string(ToleranceType type) {
    switch (type) {
    case ToleranceType::Radius:
        return "radius";
    case ToleranceType::Thickness:
        return "thickness";
    case ToleranceType::Index:
        return "index";
    case ToleranceType::Abbe:
        return "abbe";
    case ToleranceType::Decenter:
        return "decenter";
    case ToleranceType::Tilt:
        return "tilt";
    case ToleranceType::ElementDecenter:
        return "element_decenter";
    case ToleranceType::ElementTilt:
        return "element_tilt";
    }
    return "?";
}

std::string_view to_string(ToleranceDistribution distribution) {
    return distribution == ToleranceDistribution::Uniform ? "uniform" : "normal";
}

std::string_view to_string(CompensatorType compensator) {
    return compensator == CompensatorType::None ? "none" : "working_distance";
}

std::string_view to_string(ElementPivot pivot) {
    switch (pivot) {
    case ElementPivot::First:
        return "first";
    case ElementPivot::Center:
        return "center";
    case ElementPivot::Last:
        return "last";
    case ElementPivot::Offset:
        return "offset";
    }
    return "?";
}

std::optional<ToleranceType> tolerance_type_from_string(std::string_view name) {
    for (const ToleranceType t :
         {ToleranceType::Radius, ToleranceType::Thickness, ToleranceType::Index,
          ToleranceType::Abbe, ToleranceType::Decenter, ToleranceType::Tilt,
          ToleranceType::ElementDecenter, ToleranceType::ElementTilt}) {
        if (to_string(t) == name) {
            return t;
        }
    }
    return std::nullopt;
}

std::optional<ToleranceDistribution> tolerance_distribution_from_string(std::string_view name) {
    if (name == "uniform") {
        return ToleranceDistribution::Uniform;
    }
    if (name == "normal") {
        return ToleranceDistribution::Normal;
    }
    return std::nullopt;
}

std::optional<CompensatorType> compensator_from_string(std::string_view name) {
    if (name == "none") {
        return CompensatorType::None;
    }
    if (name == "working_distance") {
        return CompensatorType::WorkingDistance;
    }
    return std::nullopt;
}

std::optional<ElementPivot> element_pivot_from_string(std::string_view name) {
    for (const ElementPivot p : {ElementPivot::First, ElementPivot::Center, ElementPivot::Last}) {
        if (to_string(p) == name) {
            return p;
        }
    }
    return std::nullopt;
}

} // namespace mko
