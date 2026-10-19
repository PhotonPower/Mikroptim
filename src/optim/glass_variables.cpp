// Glasvariablen und Strafe außerhalb der Glas-Hülle (docs/optimierung.md, Abschnitte 1 und 2.4).

#include "mikroptim/optim/glass_variables.hpp"

#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Glas-Hülle: " + message);
}

} // namespace

std::vector<GlassConversion> convert_glass_variables(SystemDescription& system,
                                                     const GlassCatalog& catalog,
                                                     const BuchdahlFitOptions& fit,
                                                     int dispersion_terms) {
    std::vector<GlassConversion> out;
    for (const ResolvedVariable& r : resolve_variables(system)) {
        if (r.type != VariableType::Glass || r.component != 0) {
            continue;
        }
        MaterialSpec& m = system.surfaces[static_cast<std::size_t>(r.surface)].material;
        GlassConversion c;
        c.surface = r.surface;
        c.from = m.kind;
        switch (m.kind) {
        case MaterialKind::Catalog: {
            c.name = m.name;
            const BuchdahlFit f = fit_buchdahl(catalog.at(m.name), dispersion_terms, fit);
            m = to_buchdahl_material(f);
            c.max_fit_error = f.max_abs_error;
            break;
        }
        case MaterialKind::Constant: {
            const double n = m.n;
            m = MaterialSpec{};
            m.kind = MaterialKind::Buchdahl;
            m.n0 = n;
            break;
        }
        case MaterialKind::Buchdahl:
        case MaterialKind::Air: // von resolve_variables abgelehnt
            break;
        }
        c.n0 = m.n0;
        c.nu = m.nu;
        out.push_back(std::move(c));
    }
    return out;
}

GlassHullPenalty make_glass_hull_penalty(const SystemDescription& system, const VariableSet& set,
                                         GlassHull hull) {
    check_constraints(system);
    GlassHullPenalty p;
    p.hull = std::move(hull);
    if (!system.constraints) {
        return p;
    }
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (items[i].type == ConstraintType::GlassHull) {
            p.item = static_cast<int>(i);
            p.weight = items[i].weight_or_default();
            p.margin = items[i].margin_or_default();
            break;
        }
    }
    if (p.item < 0) {
        return p;
    }
    for (const Variable& var : set.variables()) {
        if (var.type == VariableType::Glass && var.glass_component == 0) {
            p.elements.push_back({var.file_surface, var.batch_surface});
        }
    }
    if (!p.elements.empty()) {
        if (p.hull.empty()) {
            invalid("keine Gläser für die Hülle");
        }
        if (p.margin > 0.0 && p.hull.dimension() < 3) {
            invalid("Sicherheitsabstand (margin) verlangt eine dreidimensionale Hülle, die "
                    "Gläser der Kataloge ergeben Dimension " +
                    std::to_string(p.hull.dimension()));
        }
    }
    return p;
}

void check_glass_hull_penalty(const GlassHullPenalty& penalty, const BatchTopology& topology) {
    if (!(std::isfinite(penalty.weight) && penalty.weight >= 0.0)) {
        invalid("Gewicht muss endlich und mindestens 0 sein");
    }
    if (!(std::isfinite(penalty.margin) && penalty.margin >= 0.0)) {
        invalid("Sicherheitsabstand muss endlich und mindestens 0 sein");
    }
    if (penalty.empty()) {
        return;
    }
    if (penalty.hull.empty()) {
        invalid("keine Gläser für die Hülle");
    }
    if (penalty.margin > 0.0 && penalty.hull.dimension() < 3) {
        invalid("Sicherheitsabstand verlangt eine dreidimensionale Hülle");
    }
    for (const GlassHullElement& e : penalty.elements) {
        if (e.batch_surface < 0 || e.batch_surface >= topology.surfaces) {
            invalid("Tabellenfläche " + std::to_string(e.batch_surface) + " außerhalb von [0, " +
                    std::to_string(topology.surfaces) + ")");
        }
    }
}

Point3 batch_glass_coordinates(const SystemBatchHost& batch, int v, int s) {
    return {batch.buchdahl_coeff(v, s, 0), batch.buchdahl_coeff(v, s, 1),
            batch.buchdahl_coeff(v, s, 2)};
}

namespace {

// φ(u) und die Projektion (normiert).
double excess_of(const GlassHullPenalty& penalty, const HullProjection& proj) {
    const double phi =
        penalty.hull.dimension() == 3 ? proj.signed_distance + penalty.margin : proj.distance;
    return std::max(0.0, phi);
}

} // namespace

double glass_hull_excess(const GlassHullPenalty& penalty, const Point3& x) {
    return excess_of(penalty, penalty.hull.project(x));
}

Point3 glass_hull_residual(const GlassHullPenalty& penalty, const Point3& x) {
    const HullProjection proj = penalty.hull.project(x);
    const double e = excess_of(penalty, proj);
    if (!(e > 0.0)) {
        return {0.0, 0.0, 0.0};
    }
    const double a = std::sqrt(penalty.weight) * e;
    return {a * proj.gradient[0], a * proj.gradient[1], a * proj.gradient[2]};
}

double glass_hull_penalty(const GlassHullPenalty& penalty, const Point3& x) {
    const double e = glass_hull_excess(penalty, x);
    return penalty.weight * e * e;
}

Point3 glass_hull_penalty_gradient(const GlassHullPenalty& penalty, const Point3& x) {
    const HullProjection proj = penalty.hull.project(x);
    const double e = excess_of(penalty, proj);
    if (!(e > 0.0)) {
        return {0.0, 0.0, 0.0};
    }
    const double a = 2.0 * penalty.weight * e;
    const Point3& s = penalty.hull.scale();
    return {a * proj.gradient[0] / s[0], a * proj.gradient[1] / s[1], a * proj.gradient[2] / s[2]};
}

void glass_hull_residuals(const GlassHullPenalty& penalty, const SystemBatchHost& batch, int v,
                          std::span<double> out) {
    if (out.size() != static_cast<std::size_t>(penalty.residual_count())) {
        invalid("Ausgabe hat " + std::to_string(out.size()) + " Einträge, erwartet " +
                std::to_string(penalty.residual_count()));
    }
    for (std::size_t i = 0; i < penalty.elements.size(); ++i) {
        const Point3 r = glass_hull_residual(
            penalty, batch_glass_coordinates(batch, v, penalty.elements[i].batch_surface));
        for (std::size_t k = 0; k < 3; ++k) {
            out[3 * i + k] = r[k];
        }
    }
}

ExtraResiduals make_glass_hull_residuals(GlassHullPenalty penalty) {
    ExtraResiduals out;
    out.count = penalty.residual_count();
    if (out.count == 0) {
        return out;
    }
    out.evaluate = [p = std::move(penalty)](const SystemBatchHost& batch, int v,
                                            std::span<double> values) {
        glass_hull_residuals(p, batch, v, values);
    };
    return out;
}

} // namespace mko
