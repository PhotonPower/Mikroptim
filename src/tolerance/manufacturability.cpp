// Einfache Herstellbarkeitsprüfungen (M10-T3), docs/optimierung.md, Abschnitt
// „Herstellbarkeit“.

#include "mikroptim/tolerance/manufacturability.hpp"

#include "mikroptim/optim/mechanical_constraints.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

const Surface& at(const SystemDescription& system, int s) {
    return system.surfaces[static_cast<std::size_t>(s)];
}

// Krümmung der Fläche (Ebene und Blende: 0).
double curvature_of(const Surface& s) {
    return s.type == SurfaceType::Sphere ? s.curvature : 0.0;
}

bool over_hemisphere(const Surface& s, double h) {
    return std::abs(curvature_of(s)) * h >= 1.0;
}

// Grenzwert für Texte: höchstens drei signifikante Stellen, Dezimalkomma (0,1; 0,95; 1,5).
std::string format_limit(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3g", v);
    std::string s(buf);
    std::replace(s.begin(), s.end(), '.', ',');
    return s;
}

// Warnung, wenn value < limit (limit > 0).
CheckSeverity below(double value, double limit) {
    return limit > 0.0 && value < limit ? CheckSeverity::Warning : CheckSeverity::Ok;
}

// Warnung, wenn value > limit (limit > 0).
CheckSeverity above(double value, double limit) {
    return limit > 0.0 && value > limit ? CheckSeverity::Warning : CheckSeverity::Ok;
}

} // namespace

ManufacturabilityReport check_manufacturability(const SystemDescription& system,
                                                const ManufacturabilityLimits& limits) {
    for (const double v : {limits.max_hemisphere, limits.min_center_thickness,
                           limits.min_edge_thickness, limits.min_edge_thickness_relative,
                           limits.min_air_edge, limits.max_aspect_ratio, limits.min_centering}) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("Herstellbarkeit: Grenzwerte müssen endlich sein");
        }
    }
    ManufacturabilityReport r;
    r.limits = limits;
    const int count = static_cast<int>(system.surfaces.size());
    const auto add = [&r](ManufacturabilityCheckType type, int s, double value, double limit,
                          CheckSeverity severity) {
        r.checks.push_back(ManufacturabilityCheck{type, s, value, limit, severity});
        if (severity == CheckSeverity::Warning) {
            ++r.warnings;
        } else if (severity == CheckSeverity::Error) {
            ++r.errors;
        }
    };
    for (int s = 0; s < count; ++s) {
        const Surface& sf = at(system, s);
        if (sf.type == SurfaceType::Sphere) {
            const double v = std::abs(sf.curvature) * sf.semi_diameter;
            add(ManufacturabilityCheckType::Hemisphere, s, v, limits.max_hemisphere,
                v >= 1.0 ? CheckSeverity::Error : above(v, limits.max_hemisphere));
        }
        if (s >= count - 1) {
            continue;
        }
        const Surface& next = at(system, s + 1);
        const double h = std::max(sf.semi_diameter, next.semi_diameter);
        const double c1 = curvature_of(sf);
        const double c2 = curvature_of(next);
        const double t = sf.thickness;
        const bool impossible = over_hemisphere(sf, h) || over_hemisphere(next, h);
        const double e = edge_thickness(t, c1, c2, h);
        if (sf.material.kind != MaterialKind::Air) {
            add(ManufacturabilityCheckType::CenterThickness, s, t, limits.min_center_thickness,
                below(t, limits.min_center_thickness));
            const double edge_limit = edge_thickness_limit(limits, h);
            add(ManufacturabilityCheckType::EdgeThickness, s, e, edge_limit,
                impossible || e < 0.0 ? CheckSeverity::Error : below(e, edge_limit));
            const double ratio = t > 0.0 ? 2.0 * h / t : std::numeric_limits<double>::infinity();
            add(ManufacturabilityCheckType::AspectRatio, s, ratio, limits.max_aspect_ratio,
                above(ratio, limits.max_aspect_ratio));
            const double z = 0.5 * h * std::abs(c1 - c2);
            add(ManufacturabilityCheckType::Centering, s, z, limits.min_centering,
                below(z, limits.min_centering));
        } else {
            // Luftspalt zwischen zwei Flächen (der Arbeitsabstand nach der letzten zählt nicht)
            const CheckSeverity sev = impossible || e < 0.0 ? CheckSeverity::Error
                                      : limits.min_air_edge > 0.0 && e < limits.min_air_edge
                                          ? CheckSeverity::Warning
                                          : CheckSeverity::Ok;
            add(ManufacturabilityCheckType::AirEdge, s, e, limits.min_air_edge, sev);
        }
    }
    return r;
}

double edge_thickness_limit(const ManufacturabilityLimits& limits, double semi_diameter) {
    return std::max(limits.min_edge_thickness,
                    limits.min_edge_thickness_relative * 2.0 * semi_diameter);
}

std::string finding_message(const ManufacturabilityCheck& check) {
    if (check.severity == CheckSeverity::Ok) {
        return {};
    }
    const bool error = check.severity == CheckSeverity::Error;
    const std::string limit = format_limit(check.limit);
    switch (check.type) {
    case ManufacturabilityCheckType::Hemisphere:
        return error ? "Halbdurchmesser ≥ Radius (geometrisch unmöglich)"
                     : "Halbkugelgrad über " + limit + ", Linse schwer zu fassen";
    case ManufacturabilityCheckType::CenterThickness:
        return "Mittendicke unter " + limit + " mm";
    case ManufacturabilityCheckType::EdgeThickness:
        return error ? "Randdicke < 0 oder Fläche über der Halbkugel (geometrisch unmöglich)"
                     : "Randdicke unter " + limit + " mm";
    case ManufacturabilityCheckType::AirEdge:
        return error ? "Flächen berühren sich am Rand (geometrisch unmöglich)"
                     : "Luftrand unter " + limit + " mm";
    case ManufacturabilityCheckType::AspectRatio:
        return "Durchmesser/Mittendicke über " + limit;
    case ManufacturabilityCheckType::Centering:
        return "Glockenzentrierung ungeeignet (Z < " + limit + "), andere Zentrierverfahren nötig";
    }
    return {};
}

std::string_view to_string(ManufacturabilityCheckType type) {
    switch (type) {
    case ManufacturabilityCheckType::Hemisphere:
        return "hemisphere";
    case ManufacturabilityCheckType::CenterThickness:
        return "center_thickness";
    case ManufacturabilityCheckType::EdgeThickness:
        return "edge_thickness";
    case ManufacturabilityCheckType::AirEdge:
        return "air_edge";
    case ManufacturabilityCheckType::AspectRatio:
        return "aspect_ratio";
    case ManufacturabilityCheckType::Centering:
        return "centering";
    }
    return "?";
}

std::string_view to_string(CheckSeverity severity) {
    switch (severity) {
    case CheckSeverity::Ok:
        return "ok";
    case CheckSeverity::Warning:
        return "warning";
    case CheckSeverity::Error:
        return "error";
    }
    return "?";
}

} // namespace mko
