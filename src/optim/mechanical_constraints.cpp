// Mechanische Nebenbedingungen (docs/optimierung.md, Abschnitt 2.5).

#include "mikroptim/optim/mechanical_constraints.hpp"

#include "mikroptim/model/system.hpp"
#include "mikroptim/trace/surface_extrema.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::size_t to_size(int n) {
    return static_cast<std::size_t>(n);
}

[[noreturn]] void fail_constraint(int k, const MechanicalConstraint& c, const std::string& what) {
    throw std::invalid_argument("Mechanische Nebenbedingung " + std::to_string(k) + " (" +
                                std::string(to_string(c.type)) + ", Fläche " +
                                std::to_string(c.file_surface) + "): " + what);
}

bool is_gap(ConstraintType type) {
    return type == ConstraintType::MinCenterThickness ||
           type == ConstraintType::MaxCenterThickness || type == ConstraintType::MinEdgeThickness;
}

// Wirksame Krümmung einer Tabellenfläche der Variante v (Ebenen und Blende: 0).
double effective_curvature(const SystemBatchHost& batch, int v, int s) {
    return surface_curvature(batch.type(s), batch.curvature(v, s));
}

const VariantExtrema& require_extrema(const MechanicalConstraint& c, const VariantExtrema* e,
                                      int surfaces) {
    if (e == nullptr || e->values == nullptr) {
        throw std::invalid_argument("Mechanische Nebenbedingung " + std::string(to_string(c.type)) +
                                    ": Extrema fehlen");
    }
    if (e->surfaces != surfaces) {
        throw std::invalid_argument("Mechanische Nebenbedingung: Extrema mit " +
                                    std::to_string(e->surfaces) + " Flächen, Batch mit " +
                                    std::to_string(surfaces));
    }
    return *e;
}

} // namespace

double sag(double curvature, double height) {
    const double h2 = height * height;
    const double a = std::abs(curvature * height);
    if (!(a < 1.0)) {
        return curvature * h2; // Fortsetzung jenseits der Halbkugel (auch NaN bleibt NaN)
    }
    return curvature * h2 / (1.0 + std::sqrt((1.0 - a) * (1.0 + a)));
}

double edge_thickness(double thickness, double c_front, double c_back, double height) {
    return thickness - sag(c_front, height) + sag(c_back, height);
}

bool needs_extrema(const MechanicalConstraint& c) {
    switch (c.type) {
    case ConstraintType::MaxSemiDiameter:
    case ConstraintType::MaxIncidenceAngle:
        return true;
    case ConstraintType::MinEdgeThickness:
        return c.height == EdgeHeight::Rays;
    default:
        return false;
    }
}

bool MechanicalConstraints::needs_extrema() const {
    return std::any_of(items.begin(), items.end(),
                       [](const MechanicalConstraint& c) { return ::mko::needs_extrema(c); });
}

MechanicalConstraints make_mechanical_constraints(const SystemDescription& system) {
    check_constraints(system);
    MechanicalConstraints out;
    if (!system.constraints) {
        return out;
    }
    if (system.constraints->extrema_smoothing) {
        out.smoothing.height = system.constraints->extrema_smoothing->height_beta;
        out.smoothing.cos_incidence = system.constraints->extrema_smoothing->cos_beta;
    }
    const SurfaceMap map = make_surface_map(system);
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    for (std::size_t k = 0; k < items.size(); ++k) {
        const ConstraintSpec& spec = items[k];
        if (!is_mechanical(spec.type)) {
            continue;
        }
        for (const int i : constraint_surfaces(system, spec)) {
            MechanicalConstraint c;
            c.type = spec.type;
            c.file_surface = i;
            c.surface = map.table_surface(i);
            c.next_surface = is_gap(spec.type) ? map.table_surface(i + 1) : c.surface;
            c.weight = spec.weight_or_default();
            c.clearance = spec.clearance_or_default();
            c.height = spec.height_or_default();
            c.item = static_cast<int>(k);
            switch (spec.type) {
            case ConstraintType::MaxSemiDiameter:
                c.limit = spec.semi_diameter.value_or(
                    system.surfaces[static_cast<std::size_t>(i)].semi_diameter);
                break;
            case ConstraintType::MinCenterThickness:
            case ConstraintType::MaxCenterThickness:
                c.limit = *spec.thickness; // von check_constraints garantiert
                break;
            case ConstraintType::MinEdgeThickness:
                c.limit = *spec.thickness;
                c.semi_diameter = system.surfaces[static_cast<std::size_t>(i)].semi_diameter;
                c.next_semi_diameter =
                    system.surfaces[static_cast<std::size_t>(i) + 1].semi_diameter;
                break;
            case ConstraintType::MaxIncidenceAngle:
                c.angle_deg = *spec.angle;
                c.limit = std::cos(*spec.angle * kPi / 180.0);
                break;
            default:
                break;
            }
            out.items.push_back(c);
        }
    }
    return out;
}

void check_mechanical_constraints(const MechanicalConstraints& constraints,
                                  const BatchTopology& topology) {
    const int S = topology.surfaces;
    for (int k = 0; k < constraints.size(); ++k) {
        const MechanicalConstraint& c = constraints.items[to_size(k)];
        if (!is_mechanical(c.type)) {
            fail_constraint(k, c, "kein mechanischer Typ");
        }
        const auto check_surface = [&](int s, const char* what) {
            if (s < 0 || s >= S) {
                fail_constraint(k, c,
                                std::string(what) + " " + std::to_string(s) +
                                    " außerhalb von [0, " + std::to_string(S) + ")");
            }
            if (topology.type[to_size(s)] == SurfaceKind::GhostPlate) {
                fail_constraint(k, c, std::string(what) + " ist eine Geisterplatte");
            }
        };
        check_surface(c.surface, "Tabellenfläche");
        if (is_gap(c.type)) {
            check_surface(c.next_surface, "nächste Tabellenfläche");
            if (c.next_surface <= c.surface) {
                fail_constraint(k, c, "nächste Fläche liegt nicht hinter der Fläche");
            }
        }
        if (!(std::isfinite(c.weight) && c.weight >= 0.0)) {
            fail_constraint(k, c, "Gewicht muss endlich und mindestens 0 sein");
        }
        if (!std::isfinite(c.limit)) {
            fail_constraint(k, c, "Grenze muss endlich sein");
        }
        if (c.type == ConstraintType::MaxSemiDiameter && !(c.limit > 0.0)) {
            fail_constraint(k, c, "Halbdurchmesser muss größer als 0 sein");
        }
        if (c.type == ConstraintType::MaxIncidenceAngle && !(c.limit > 0.0 && c.limit < 1.0)) {
            fail_constraint(k, c, "cos θ_max muss in (0, 1) liegen");
        }
        if (!(std::isfinite(c.clearance) && c.clearance >= 0.0)) {
            fail_constraint(k, c, "clearance muss endlich und mindestens 0 sein");
        }
        if (c.type == ConstraintType::MinEdgeThickness) {
            if (c.height != EdgeHeight::SemiDiameter && c.height != EdgeHeight::Rays) {
                fail_constraint(k, c, "unbekannte Höhe");
            }
            if (c.height == EdgeHeight::SemiDiameter &&
                !(std::isfinite(c.semi_diameter) && c.semi_diameter > 0.0 &&
                  std::isfinite(c.next_semi_diameter) && c.next_semi_diameter > 0.0)) {
                fail_constraint(k, c, "Halbdurchmesser müssen endlich und größer als 0 sein");
            }
        }
    }
}

double mechanical_center_thickness(const MechanicalConstraint& c, const SystemBatchHost& batch,
                                   int v) {
    double t = 0.0;
    for (int s = c.surface; s < c.next_surface; ++s) {
        t += batch.thickness(v, s);
    }
    return t;
}

double mechanical_edge_height(const MechanicalConstraint& c, const VariantExtrema* extrema) {
    double h = 0.0;
    if (c.height == EdgeHeight::Rays) {
        if (extrema == nullptr || extrema->values == nullptr) {
            throw std::invalid_argument("Mechanische Nebenbedingung min_edge_thickness: Extrema "
                                        "fehlen");
        }
        h = std::max(extrema->max_height(c.surface), extrema->max_height(c.next_surface));
    } else {
        h = std::max(c.semi_diameter, c.next_semi_diameter);
    }
    return h + c.clearance;
}

double mechanical_value(const MechanicalConstraint& c, const SystemBatchHost& batch, int v,
                        const VariantExtrema* extrema) {
    if (v < 0 || v >= batch.variants()) {
        throw std::out_of_range("Mechanische Nebenbedingung: Variante " + std::to_string(v) +
                                " außerhalb von [0, " + std::to_string(batch.variants()) + ")");
    }
    switch (c.type) {
    case ConstraintType::MaxSemiDiameter: {
        const VariantExtrema& e = require_extrema(c, extrema, batch.surfaces());
        return e.max_height(c.surface) + c.clearance;
    }
    case ConstraintType::MinCenterThickness:
    case ConstraintType::MaxCenterThickness:
        return mechanical_center_thickness(c, batch, v);
    case ConstraintType::MinEdgeThickness: {
        if (c.height == EdgeHeight::Rays) {
            require_extrema(c, extrema, batch.surfaces());
        }
        const double h = mechanical_edge_height(c, extrema);
        return edge_thickness(mechanical_center_thickness(c, batch, v),
                              effective_curvature(batch, v, c.surface),
                              effective_curvature(batch, v, c.next_surface), h);
    }
    case ConstraintType::MaxIncidenceAngle: {
        const VariantExtrema& e = require_extrema(c, extrema, batch.surfaces());
        return e.min_cos_incidence(c.surface);
    }
    default:
        break;
    }
    throw std::invalid_argument("Mechanische Nebenbedingung: " + std::string(to_string(c.type)) +
                                " ist kein mechanischer Typ");
}

double mechanical_violation(const MechanicalConstraint& c, double value) {
    switch (c.type) {
    case ConstraintType::MaxSemiDiameter:
    case ConstraintType::MaxCenterThickness:
        return value - c.limit;
    default:
        return c.limit - value;
    }
}

double mechanical_residual(const MechanicalConstraint& c, const SystemBatchHost& batch, int v,
                           const VariantExtrema* extrema) {
    const double g = mechanical_violation(c, mechanical_value(c, batch, v, extrema));
    return std::sqrt(c.weight) * std::max(g, 0.0);
}

double incidence_angle_deg(double cos_incidence) {
    return std::acos(std::clamp(cos_incidence, -1.0, 1.0)) * 180.0 / kPi;
}

MechanicalResiduals make_mechanical_residuals(const MechanicalConstraints& constraints) {
    MechanicalResiduals out;
    std::vector<MechanicalConstraint> geometric;
    std::vector<MechanicalConstraint> traced;
    for (int k = 0; k < constraints.size(); ++k) {
        const MechanicalConstraint& c = constraints.items[to_size(k)];
        if (needs_extrema(c)) {
            traced.push_back(c);
            out.traced_items.push_back(k);
        } else {
            geometric.push_back(c);
            out.geometric_items.push_back(k);
        }
    }
    if (!geometric.empty()) {
        out.geometric.count = static_cast<int>(geometric.size());
        out.geometric.evaluate = [items = std::move(geometric)](const SystemBatchHost& batch, int v,
                                                                std::span<double> values) {
            if (values.size() != items.size()) {
                throw std::invalid_argument("Mechanische Nebenbedingungen: Ausgabe hat falsche "
                                            "Länge");
            }
            for (std::size_t k = 0; k < items.size(); ++k) {
                values[k] = mechanical_residual(items[k], batch, v, nullptr);
            }
        };
    }
    if (!traced.empty()) {
        out.traced.count = static_cast<int>(traced.size());
        out.traced.smoothing = constraints.smoothing;
        out.traced.evaluate = [items = std::move(traced)](const SystemBatchHost& batch, int v,
                                                          const VariantExtrema& extrema,
                                                          std::span<double> values) {
            if (values.size() != items.size()) {
                throw std::invalid_argument("Mechanische Nebenbedingungen: Ausgabe hat falsche "
                                            "Länge");
            }
            for (std::size_t k = 0; k < items.size(); ++k) {
                values[k] = mechanical_residual(items[k], batch, v, &extrema);
            }
        };
    }
    return out;
}

} // namespace mko
