// Konvexe Hülle im ℝ³ mit Projektion (docs/optimierung.md, Abschnitt 1, „Glas-Hülle“).

#include "mikroptim/glass/convex_hull.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(int i) {
    return static_cast<std::size_t>(i);
}

Point3 sub(const Point3& a, const Point3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Point3 add(const Point3& a, const Point3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Point3 scale(const Point3& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}
double dot(const Point3& a, const Point3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point3 cross(const Point3& a, const Point3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double length(const Point3& a) {
    return std::sqrt(dot(a, a));
}

// Abstand von p zur Geraden durch a mit Einheitsrichtung u.
double line_distance(const Point3& p, const Point3& a, const Point3& u) {
    const Point3 d = sub(p, a);
    return length(sub(d, scale(u, dot(d, u))));
}

// Dreieck der Hülle während des Aufbaus (Indizes in die Eingabe).
struct Face {
    int a = 0;
    int b = 0;
    int c = 0;
    HullPlane plane;
};

HullPlane plane_of(const Point3& a, const Point3& b, const Point3& c) {
    Point3 n = cross(sub(b, a), sub(c, a));
    const double len = length(n);
    if (len > 0.0) {
        n = scale(n, 1.0 / len);
    }
    return HullPlane{n, dot(n, a)};
}

// Dimension 0: ein Punkt.
void finish_point(ConvexHull3& hull, std::span<const Point3> points, int i) {
    hull.dimension = 0;
    hull.vertices = {points[to_size(i)]};
    hull.vertex_source = {i};
}

// Dimension 1: Endpunkte der Projektion auf die Gerade durch p0 mit Richtung u.
void finish_segment(ConvexHull3& hull, std::span<const Point3> points, const Point3& p0,
                    const Point3& u) {
    int lo = 0;
    int hi = 0;
    double t_lo = std::numeric_limits<double>::infinity();
    double t_hi = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double t = dot(sub(points[i], p0), u);
        if (t < t_lo) {
            t_lo = t;
            lo = static_cast<int>(i);
        }
        if (t > t_hi) {
            t_hi = t;
            hi = static_cast<int>(i);
        }
    }
    hull.dimension = 1;
    hull.vertices = {points[to_size(lo)], points[to_size(hi)]};
    hull.vertex_source = {lo, hi};
}

// Dimension 2: Polygon in der Ebene durch p0 mit Einheitsnormale n (Andrew, monotone Kette, in
// ebenen Koordinaten zur Basis e1, e2 = n × e1). Liefert false, wenn weniger als drei Ecken
// bleiben (numerisch kollinear).
bool finish_polygon(ConvexHull3& hull, std::span<const Point3> points, const Point3& p0,
                    const Point3& e1, const Point3& n, double eps) {
    const Point3 e2 = cross(n, e1);
    struct P2 {
        double x;
        double y;
        int i;
    };
    std::vector<P2> q;
    q.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point3 d = sub(points[i], p0);
        q.push_back({dot(d, e1), dot(d, e2), static_cast<int>(i)});
    }
    std::sort(q.begin(), q.end(), [](const P2& a, const P2& b) {
        return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.i < b.i)));
    });
    // Linksdrehung o → a → b, wenn a mehr als ε links der Geraden o → b liegt:
    // cross = |b − o| · (Abstand von a zur Geraden, vorzeichenbehaftet).
    const auto keep = [eps](const P2& o, const P2& a, const P2& b) {
        const double cr = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        return -cr > eps * std::hypot(b.x - o.x, b.y - o.y);
    };
    std::vector<P2> chain;
    for (int pass = 0; pass < 2; ++pass) {
        const std::size_t start = chain.size();
        for (std::size_t k = 0; k < q.size(); ++k) {
            const P2& p = pass == 0 ? q[k] : q[q.size() - 1 - k];
            while (chain.size() >= start + 2 &&
                   !keep(chain[chain.size() - 2], chain[chain.size() - 1], p)) {
                chain.pop_back();
            }
            chain.push_back(p);
        }
        chain.pop_back(); // letzter Punkt ist der erste der anderen Kette
    }
    // Die Kette läuft im Uhrzeigersinn (keep verlangt Rechtsdrehung); umdrehen.
    std::reverse(chain.begin(), chain.end());
    if (chain.size() < 3) {
        return false;
    }
    hull.dimension = 2;
    hull.plane_normal = n;
    for (const P2& p : chain) {
        hull.vertices.push_back(points[to_size(p.i)]);
        hull.vertex_source.push_back(p.i);
    }
    for (int k = 1; k + 1 < static_cast<int>(chain.size()); ++k) {
        hull.triangles.push_back({0, k, k + 1});
    }
    return true;
}

} // namespace

Point3 closest_point_on_segment(const Point3& p, const Point3& a, const Point3& b) {
    const Point3 ab = sub(b, a);
    const double len2 = dot(ab, ab);
    if (!(len2 > 0.0)) {
        return a;
    }
    const double t = std::clamp(dot(sub(p, a), ab) / len2, 0.0, 1.0);
    return add(a, scale(ab, t));
}

Point3 closest_point_on_triangle(const Point3& p, const Point3& a, const Point3& b,
                                 const Point3& c) {
    const Point3 ab = sub(b, a);
    const Point3 ac = sub(c, a);
    const Point3 ap = sub(p, a);
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a; // Gebiet der Ecke a
    }
    const Point3 bp = sub(p, b);
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b; // Ecke b
    }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        return add(a, scale(ab, d1 / (d1 - d3))); // Kante ab
    }
    const Point3 cp = sub(p, c);
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c; // Ecke c
    }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        return add(a, scale(ac, d2 / (d2 - d6))); // Kante ac
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w)); // Kante bc
    }
    const double sum = va + vb + vc;
    if (!(sum > 0.0)) {
        // entartetes Dreieck (Fläche 0): nächster Punkt der drei Kanten
        Point3 best = closest_point_on_segment(p, a, b);
        double best_d = length(sub(p, best));
        for (const Point3& q :
             {closest_point_on_segment(p, b, c), closest_point_on_segment(p, a, c)}) {
            const double d = length(sub(p, q));
            if (d < best_d) {
                best_d = d;
                best = q;
            }
        }
        return best;
    }
    const double v = vb / sum;
    const double w = vc / sum;
    return add(a, add(scale(ab, v), scale(ac, w))); // Inneres der Fläche
}

ConvexHull3 convex_hull(std::span<const Point3> points, const ConvexHullOptions& options) {
    if (!(options.relative_tolerance > 0.0) || !std::isfinite(options.relative_tolerance)) {
        throw std::invalid_argument("Konvexe Hülle: relative Toleranz muss endlich und > 0 sein");
    }
    ConvexHull3 hull;
    if (points.empty()) {
        return hull;
    }
    Point3 lo = points[0];
    Point3 hi = points[0];
    double max_abs = 0.0;
    for (const Point3& p : points) {
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(p[to_size(k)])) {
                throw std::invalid_argument("Konvexe Hülle: Koordinate nicht endlich");
            }
            lo[to_size(k)] = std::min(lo[to_size(k)], p[to_size(k)]);
            hi[to_size(k)] = std::max(hi[to_size(k)], p[to_size(k)]);
            max_abs = std::max(max_abs, std::abs(p[to_size(k)]));
        }
    }
    const double eps = options.relative_tolerance * std::max(length(sub(hi, lo)), max_abs);
    hull.tolerance = eps;

    // Starttetraeder: i0 lexikographisch kleinster Punkt, i1 am weitesten von i0, i2 am
    // weitesten von der Geraden, i3 am weitesten von der Ebene.
    int i0 = 0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        if (points[i] < points[to_size(i0)]) {
            i0 = static_cast<int>(i);
        }
    }
    const Point3& p0 = points[to_size(i0)];
    int i1 = i0;
    double best = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double d = length(sub(points[i], p0));
        if (d > best) {
            best = d;
            i1 = static_cast<int>(i);
        }
    }
    if (!(best > eps)) {
        finish_point(hull, points, i0);
        return hull;
    }
    const Point3 u = scale(sub(points[to_size(i1)], p0), 1.0 / best);
    int i2 = i0;
    best = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double d = line_distance(points[i], p0, u);
        if (d > best) {
            best = d;
            i2 = static_cast<int>(i);
        }
    }
    if (!(best > eps)) {
        finish_segment(hull, points, p0, u);
        return hull;
    }
    const HullPlane base = plane_of(p0, points[to_size(i1)], points[to_size(i2)]);
    int i3 = i0;
    best = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double d = std::abs(dot(base.normal, points[i]) - base.offset);
        if (d > best) {
            best = d;
            i3 = static_cast<int>(i);
        }
    }
    if (!(best > eps)) {
        if (!finish_polygon(hull, points, p0, u, base.normal, eps)) {
            finish_segment(hull, points, p0, u);
        }
        return hull;
    }

    // Dimension 3: Tetraeder mit nach außen orientierten Dreiecken.
    const auto make_face = [&](int a, int b, int c) {
        Face f{a, b, c, plane_of(points[to_size(a)], points[to_size(b)], points[to_size(c)])};
        return f;
    };
    const int tet[4] = {i0, i1, i2, i3};
    std::vector<Face> faces;
    for (int k = 0; k < 4; ++k) {
        const int a = tet[(k + 1) % 4];
        int b = tet[(k + 2) % 4];
        int c = tet[(k + 3) % 4];
        const Point3& opposite = points[to_size(tet[k])];
        Face f = make_face(a, b, c);
        if (dot(f.plane.normal, opposite) - f.plane.offset > 0.0) {
            std::swap(b, c);
            f = make_face(a, b, c);
        }
        faces.push_back(f);
    }

    // Übrige Punkte nach fallendem Abstand vom Schwerpunkt des Tetraeders (Gleichstand: Index),
    // damit äußere Punkte zuerst kommen und innere früh als innen erkannt werden.
    Point3 centroid{};
    for (const int t : tet) {
        centroid = add(centroid, scale(points[to_size(t)], 0.25));
    }
    std::vector<int> order;
    std::vector<double> radius(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const int ii = static_cast<int>(i);
        radius[i] = length(sub(points[i], centroid));
        if (ii != i0 && ii != i1 && ii != i2 && ii != i3) {
            order.push_back(ii);
        }
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return radius[to_size(a)] > radius[to_size(b)]; });

    std::vector<char> visible;
    std::set<std::pair<int, int>> edges;
    for (const int idx : order) {
        const Point3& p = points[to_size(idx)];
        visible.assign(faces.size(), 0);
        bool any = false;
        for (std::size_t f = 0; f < faces.size(); ++f) {
            if (dot(faces[f].plane.normal, p) - faces[f].plane.offset > eps) {
                visible[f] = 1;
                any = true;
            }
        }
        if (!any) {
            continue; // innen oder auf dem Rand (innerhalb der Toleranz)
        }
        // Horizont: gerichtete Kanten sichtbarer Dreiecke, deren Gegenkante zu keinem
        // sichtbaren Dreieck gehört. Das neue Dreieck (a, b, p) übernimmt die Richtung a → b und
        // passt damit zur Gegenkante b → a des unsichtbaren Nachbarn (Orientierung nach außen).
        edges.clear();
        for (std::size_t f = 0; f < faces.size(); ++f) {
            if (visible[f] != 0) {
                edges.insert({faces[f].a, faces[f].b});
                edges.insert({faces[f].b, faces[f].c});
                edges.insert({faces[f].c, faces[f].a});
            }
        }
        std::vector<Face> next;
        next.reserve(faces.size() + 8);
        for (std::size_t f = 0; f < faces.size(); ++f) {
            if (visible[f] == 0) {
                next.push_back(faces[f]);
            }
        }
        for (const auto& [a, b] : edges) {
            if (edges.count({b, a}) == 0) {
                next.push_back(make_face(a, b, idx));
            }
        }
        faces = std::move(next);
    }

    // Ecken aufsteigend nach Eingabeindex, Dreiecke umnummerieren.
    std::vector<int> used;
    for (const Face& f : faces) {
        used.push_back(f.a);
        used.push_back(f.b);
        used.push_back(f.c);
    }
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());
    const auto local = [&used](int i) {
        return static_cast<int>(std::lower_bound(used.begin(), used.end(), i) - used.begin());
    };
    hull.dimension = 3;
    for (const int i : used) {
        hull.vertices.push_back(points[to_size(i)]);
        hull.vertex_source.push_back(i);
    }
    for (const Face& f : faces) {
        hull.triangles.push_back({local(f.a), local(f.b), local(f.c)});
        hull.planes.push_back(f.plane);
    }
    return hull;
}

HullProjection project_onto_hull(const ConvexHull3& hull, const Point3& p) {
    if (hull.dimension < 0) {
        throw std::invalid_argument("Projektion auf eine leere Hülle");
    }
    if (!(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))) {
        throw std::invalid_argument("Projektion auf die Hülle: Punkt nicht endlich");
    }
    HullProjection r;
    if (hull.dimension == 3) {
        // Innen, wenn p hinter allen Ebenen liegt; vorzeichenbehafteter Abstand über die nächste
        // Ebene (für ein konvexes Polyeder ist das der Abstand zum Rand).
        double worst = -std::numeric_limits<double>::infinity();
        std::size_t worst_plane = 0;
        for (std::size_t f = 0; f < hull.planes.size(); ++f) {
            const double s = dot(hull.planes[f].normal, p) - hull.planes[f].offset;
            if (s > worst) {
                worst = s;
                worst_plane = f;
            }
        }
        if (worst <= 0.0) {
            r.closest = p;
            r.distance = 0.0;
            r.signed_distance = worst;
            r.gradient = hull.planes[worst_plane].normal;
            r.inside = true;
            return r;
        }
    }
    const auto vertex = [&hull](int i) -> const Point3& { return hull.vertices[to_size(i)]; };
    double best = std::numeric_limits<double>::infinity();
    const auto consider = [&](const Point3& q) {
        const double d = length(sub(p, q));
        if (d < best) {
            best = d;
            r.closest = q;
        }
    };
    if (hull.dimension >= 2) {
        for (const std::array<int, 3>& t : hull.triangles) {
            consider(closest_point_on_triangle(p, vertex(t[0]), vertex(t[1]), vertex(t[2])));
        }
    } else if (hull.dimension == 1) {
        consider(closest_point_on_segment(p, vertex(0), vertex(1)));
    } else {
        consider(vertex(0));
    }
    r.distance = best;
    r.signed_distance = best;
    if (best > 0.0) {
        r.gradient = scale(sub(p, r.closest), 1.0 / best);
    } else {
        r.inside = true;
    }
    return r;
}

} // namespace mko
