// Glas-Hülle im normierten (n₀, ν₁, ν₂)-Raum (docs/optimierung.md, Abschnitt 1).

#include "mikroptim/glass/glass_hull.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mko {

GlassHull::GlassHull(std::vector<GlassHullPoint> points, const ConvexHullOptions& options,
                     std::vector<std::string> excluded)
    : points_(std::move(points)), excluded_(std::move(excluded)) {
    if (points_.empty()) {
        throw std::invalid_argument("Glas-Hülle: keine Gläser");
    }
    for (const GlassHullPoint& p : points_) {
        for (const double c : p.coordinates) {
            if (!std::isfinite(c)) {
                throw std::invalid_argument("Glas-Hülle: Koordinaten von " + p.name +
                                            " nicht endlich");
            }
        }
    }
    // Mittelwert (Summe in Lade-Reihenfolge) und Spannweite je Koordinate
    const double n = static_cast<double>(points_.size());
    for (std::size_t j = 0; j < 3; ++j) {
        double sum = 0.0;
        double lo = points_.front().coordinates[j];
        double hi = lo;
        for (const GlassHullPoint& p : points_) {
            sum += p.coordinates[j];
            lo = std::min(lo, p.coordinates[j]);
            hi = std::max(hi, p.coordinates[j]);
        }
        center_[j] = sum / n;
        scale_[j] = hi > lo ? hi - lo : 1.0;
    }
    std::vector<Point3> u;
    u.reserve(points_.size());
    for (const GlassHullPoint& p : points_) {
        u.push_back(normalize(p.coordinates));
    }
    hull_ = convex_hull(u, options);
}

std::vector<std::string> GlassHull::vertex_names() const {
    std::vector<std::string> out;
    for (const int i : hull_.vertex_source) {
        out.push_back(points_[static_cast<std::size_t>(i)].name);
    }
    return out;
}

Point3 GlassHull::normalize(const Point3& x) const {
    return {(x[0] - center_[0]) / scale_[0], (x[1] - center_[1]) / scale_[1],
            (x[2] - center_[2]) / scale_[2]};
}

Point3 GlassHull::denormalize(const Point3& u) const {
    return {center_[0] + scale_[0] * u[0], center_[1] + scale_[1] * u[1],
            center_[2] + scale_[2] * u[2]};
}

HullProjection GlassHull::project(const Point3& x) const {
    if (empty()) {
        throw std::invalid_argument("Glas-Hülle ist leer");
    }
    return project_onto_hull(hull_, normalize(x));
}

double GlassHull::distance(const Point3& x) const {
    return project(x).distance;
}

bool GlassHull::contains(const Point3& x, double tolerance) const {
    return project(x).distance <= tolerance;
}

Point3 glass_coordinates(const BuchdahlFit& fit) {
    return {fit.n0, fit.nu[0], fit.nu[1]};
}

GlassHull make_glass_hull(const GlassCatalog& catalog, const GlassHullOptions& options) {
    std::vector<GlassHullPoint> points;
    std::vector<std::string> excluded;
    for (const Glass& g : catalog.glasses()) {
        if (!(g.in_range(options.fit.lambda_min_nm) && g.in_range(options.fit.lambda_max_nm))) {
            excluded.push_back(g.name);
            continue;
        }
        const BuchdahlFit fit = fit_buchdahl(g, options.dispersion_terms, options.fit);
        GlassHullPoint p;
        p.name = g.name;
        p.coordinates = glass_coordinates(fit);
        p.nu3 = fit.nu[2];
        p.max_fit_error = fit.max_abs_error;
        points.push_back(std::move(p));
    }
    if (points.empty()) {
        throw GlassError("Glas-Hülle: kein Glas des Katalogs deckt den Fitbereich ab");
    }
    return GlassHull(std::move(points), options.hull, std::move(excluded));
}

} // namespace mko
