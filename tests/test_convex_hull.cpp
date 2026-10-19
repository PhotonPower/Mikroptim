// Konvexe Hülle im ℝ³ (M6-T1, Abnahme „Hüllentests“): Würfel und Tetraeder gegen bekannte
// Facetten, zufällige Punktwolken gegen Brute-Force-Facetten, Abstand gegen den analytischen
// Würfelabstand und gegen eine unabhängige Projektion auf alle Facetten, Kanten und Ecken,
// entartete Fälle (Punkt, Strecke, Polygon) und Gradient gegen finite Differenzen.

#include "mikroptim/glass/convex_hull.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mko::test {
namespace {

using Tri = std::array<int, 3>;

Point3 sub(const Point3& a, const Point3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
double dot(const Point3& a, const Point3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point3 cross(const Point3& a, const Point3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Point3& a) {
    return std::sqrt(dot(a, a));
}

// Prüft die Struktur einer dreidimensionalen Hülle: jede Kante in genau zwei Dreiecken mit
// entgegengesetzter Richtung (geschlossene, orientierte Fläche), Euler V − E + F = 2, alle
// Eingabepunkte hinter allen Ebenen (bis auf die Toleranz), Normalen nach außen und zur Ebene
// des Dreiecks passend.
void expect_valid_polyhedron(const ConvexHull3& h, const std::vector<Point3>& points) {
    ASSERT_EQ(h.dimension, 3);
    ASSERT_EQ(h.planes.size(), h.triangles.size());
    std::map<std::pair<int, int>, int> directed;
    for (const Tri& t : h.triangles) {
        for (int e = 0; e < 3; ++e) {
            ++directed[{t[static_cast<std::size_t>(e)], t[static_cast<std::size_t>((e + 1) % 3)]}];
        }
    }
    for (const auto& [edge, count] : directed) {
        EXPECT_EQ(count, 1) << "Kante doppelt gerichtet";
        EXPECT_EQ(directed.count({edge.second, edge.first}), 1u) << "offene Kante";
    }
    const long v = static_cast<long>(h.vertices.size());
    const long e = static_cast<long>(directed.size() / 2);
    const long f = static_cast<long>(h.triangles.size());
    EXPECT_EQ(v - e + f, 2);
    Point3 centroid{};
    for (const Point3& p : h.vertices) {
        for (std::size_t k = 0; k < 3; ++k) {
            centroid[k] += p[k] / static_cast<double>(v);
        }
    }
    for (std::size_t i = 0; i < h.triangles.size(); ++i) {
        const Tri& t = h.triangles[i];
        const HullPlane& pl = h.planes[i];
        EXPECT_NEAR(norm(pl.normal), 1.0, kHullGeometry);
        for (const int k : t) {
            EXPECT_NEAR(dot(pl.normal, h.vertices[static_cast<std::size_t>(k)]), pl.offset,
                        kHullGeometry);
        }
        EXPECT_LT(dot(pl.normal, centroid), pl.offset); // Schwerpunkt innen
        for (const Point3& p : points) {
            EXPECT_LE(dot(pl.normal, p) - pl.offset, h.tolerance + kHullGeometry);
        }
    }
}

// Volumen einer geschlossenen, nach außen orientierten Dreiecksfläche (Divergenzsatz).
double polyhedron_volume(const ConvexHull3& h) {
    double vol = 0.0;
    for (const Tri& t : h.triangles) {
        vol += dot(h.vertices[static_cast<std::size_t>(t[0])],
                   cross(h.vertices[static_cast<std::size_t>(t[1])],
                         h.vertices[static_cast<std::size_t>(t[2])])) /
               6.0;
    }
    return vol;
}

// Unabhängige Projektion auf ein Dreieck: Fußpunkt in der Ebene, falls er im Dreieck liegt
// (baryzentrische Koordinaten über Teilflächen), sonst das Minimum über die drei Kanten und Ecken.
double brute_triangle_distance(const Point3& p, const Point3& a, const Point3& b, const Point3& c) {
    const Point3 n = cross(sub(b, a), sub(c, a));
    const double area2 = norm(n);
    const auto seg = [&](const Point3& s0, const Point3& s1) {
        const Point3 d = sub(s1, s0);
        double t = dot(sub(p, s0), d) / dot(d, d);
        t = std::min(1.0, std::max(0.0, t));
        return norm(sub(p, {s0[0] + t * d[0], s0[1] + t * d[1], s0[2] + t * d[2]}));
    };
    double best = std::min(
        {seg(a, b), seg(b, c), seg(c, a), norm(sub(p, a)), norm(sub(p, b)), norm(sub(p, c))});
    if (area2 > 0.0) {
        const Point3 un = {n[0] / area2, n[1] / area2, n[2] / area2};
        const double h = dot(sub(p, a), un);
        const Point3 q = {p[0] - h * un[0], p[1] - h * un[1], p[2] - h * un[2]};
        const double l0 = dot(cross(sub(b, q), sub(c, q)), un);
        const double l1 = dot(cross(sub(c, q), sub(a, q)), un);
        const double l2 = dot(cross(sub(a, q), sub(b, q)), un);
        if (l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0) {
            best = std::min(best, std::abs(h));
        }
    }
    return best;
}

// Abstand zu einem konvexen Polyeder aus seinen Randdreiecken: 0, wenn p hinter allen Ebenen der
// Brute-Force-Facetten liegt, sonst das Minimum über alle Dreiecke.
double brute_hull_distance(const ConvexHull3& h, const Point3& p) {
    bool inside = true;
    for (const HullPlane& pl : h.planes) {
        inside = inside && dot(pl.normal, p) <= pl.offset;
    }
    if (inside && h.dimension == 3) {
        return 0.0;
    }
    double best = 1e300;
    for (const Tri& t : h.triangles) {
        best = std::min(best, brute_triangle_distance(p, h.vertices[static_cast<std::size_t>(t[0])],
                                                      h.vertices[static_cast<std::size_t>(t[1])],
                                                      h.vertices[static_cast<std::size_t>(t[2])]));
    }
    return best;
}

// Analytischer Abstand zum Würfel [0, 1]³ (vorzeichenbehaftet: innen minus Abstand zum Rand).
double cube_signed_distance(const Point3& p) {
    double out2 = 0.0;
    double inner = 1e300;
    for (const double x : p) {
        const double o = std::max({0.0, -x, x - 1.0});
        out2 += o * o;
        inner = std::min({inner, x, 1.0 - x});
    }
    return out2 > 0.0 ? std::sqrt(out2) : -inner;
}

std::vector<Point3> cube_points(std::mt19937_64& rng) {
    std::vector<Point3> pts;
    std::uniform_real_distribution<double> u(0.05, 0.95);
    // innere Punkte, Flächenmitten und Kantenmitten zuerst, die Ecken zuletzt
    for (int i = 0; i < 40; ++i) {
        pts.push_back({u(rng), u(rng), u(rng)});
    }
    for (int k = 0; k < 3; ++k) {
        for (const double side : {0.0, 1.0}) {
            Point3 c{0.5, 0.5, 0.5};
            c[static_cast<std::size_t>(k)] = side;
            pts.push_back(c);
        }
    }
    pts.push_back({0.5, 0.0, 0.0});
    pts.push_back({1.0, 0.5, 1.0});
    for (int c = 0; c < 8; ++c) {
        pts.push_back({double(c & 1), double((c >> 1) & 1), double((c >> 2) & 1)});
    }
    return pts;
}

TEST(ConvexHull, CubeHasEightCornersAndSixPlanes) {
    std::mt19937_64 rng(1);
    const std::vector<Point3> pts = cube_points(rng);
    const ConvexHull3 h = convex_hull(pts);
    expect_valid_polyhedron(h, pts);
    // Ecken = die acht Würfelecken (die letzten acht Eingabepunkte)
    ASSERT_EQ(h.vertices.size(), 8u);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(h.vertex_source[i], static_cast<int>(pts.size() - 8 + i));
    }
    // 12 Dreiecke in genau 6 Ebenen ±e_k mit Abstand 0 bzw. 1
    EXPECT_EQ(h.triangles.size(), 12u);
    std::set<std::pair<int, int>> planes; // (Achse·2 + Seite, Anzahl)
    std::map<int, int> count;
    for (const HullPlane& pl : h.planes) {
        int axis = -1;
        for (int k = 0; k < 3; ++k) {
            if (std::abs(std::abs(pl.normal[static_cast<std::size_t>(k)]) - 1.0) < kHullGeometry) {
                axis = k;
            }
        }
        ASSERT_GE(axis, 0);
        const bool positive = pl.normal[static_cast<std::size_t>(axis)] > 0.0;
        EXPECT_NEAR(pl.offset, positive ? 1.0 : 0.0, kHullGeometry);
        ++count[axis * 2 + (positive ? 1 : 0)];
    }
    EXPECT_EQ(count.size(), 6u);
    for (const auto& [key, n] : count) {
        EXPECT_EQ(n, 2) << key;
    }
    EXPECT_NEAR(polyhedron_volume(h), 1.0, kHullGeometry);
}

TEST(ConvexHull, CubeDistanceMatchesAnalytic) {
    std::mt19937_64 rng(2);
    const std::vector<Point3> pts = cube_points(rng);
    const ConvexHull3 h = convex_hull(pts);
    std::uniform_real_distribution<double> u(-1.5, 2.5);
    double worst = 0.0;
    for (int i = 0; i < 2000; ++i) {
        const Point3 p{u(rng), u(rng), u(rng)};
        const HullProjection r = project_onto_hull(h, p);
        const double exact = cube_signed_distance(p);
        worst = std::max(worst, std::abs(r.signed_distance - exact));
        EXPECT_NEAR(r.signed_distance, exact, kHullDistance);
        EXPECT_NEAR(r.distance, std::max(0.0, exact), kHullDistance);
        EXPECT_EQ(r.inside, exact <= 0.0);
        EXPECT_NEAR(norm(sub(p, r.closest)), r.distance, kHullDistance);
        if (exact > 0.0) {
            // nächster Punkt = Klemmung auf den Würfel
            for (std::size_t k = 0; k < 3; ++k) {
                EXPECT_NEAR(r.closest[k], std::min(1.0, std::max(0.0, p[k])), kHullDistance);
            }
        }
    }
    std::printf("Würfel: max |d − d_analytisch| = %.3e\n", worst);
}

TEST(ConvexHull, TetrahedronKnownFacets) {
    const std::vector<Point3> pts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.1, 0.2, 0.3}};
    const ConvexHull3 h = convex_hull(pts);
    expect_valid_polyhedron(h, pts);
    ASSERT_EQ(h.vertices.size(), 4u);
    EXPECT_EQ(h.vertex_source, (std::vector<int>{0, 1, 2, 3}));
    ASSERT_EQ(h.triangles.size(), 4u);
    const double s = 1.0 / std::sqrt(3.0);
    int found = 0;
    for (const HullPlane& pl : h.planes) {
        const bool diag = std::abs(pl.normal[0] - s) < kHullGeometry &&
                          std::abs(pl.normal[1] - s) < kHullGeometry &&
                          std::abs(pl.normal[2] - s) < kHullGeometry;
        if (diag) {
            EXPECT_NEAR(pl.offset, s, kHullGeometry);
            ++found;
            continue;
        }
        // Koordinatenebenen x_k ≥ 0: Normale −e_k, Abstand 0
        EXPECT_NEAR(pl.offset, 0.0, kHullGeometry);
        EXPECT_NEAR(pl.normal[0] + pl.normal[1] + pl.normal[2], -1.0, kHullGeometry);
        ++found;
    }
    EXPECT_EQ(found, 4);
    EXPECT_NEAR(polyhedron_volume(h), 1.0 / 6.0, kHullGeometry);
    // (1, 1, 1): nächster Punkt (1/3, 1/3, 1/3) auf der schrägen Fläche, Abstand 2/√3
    const HullProjection r = project_onto_hull(h, {1, 1, 1});
    EXPECT_NEAR(r.distance, 2.0 * s, kHullDistance);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(r.closest[k], 1.0 / 3.0, kHullDistance);
        EXPECT_NEAR(r.gradient[k], s, kHullDistance);
    }
    // Ecke: (−1, −1, −1) → (0, 0, 0); Kante: (0.5, −1, −1) → (0.5, 0, 0)
    EXPECT_NEAR(project_onto_hull(h, {-1, -1, -1}).distance, std::sqrt(3.0), kHullDistance);
    const HullProjection edge = project_onto_hull(h, {0.5, -1, -1});
    EXPECT_NEAR(edge.distance, std::sqrt(2.0), kHullDistance);
    EXPECT_NEAR(edge.closest[0], 0.5, kHullDistance);
    // innen: (0.1, 0.2, 0.3) liegt 0,1 vor der Ebene x = 0
    const HullProjection in = project_onto_hull(h, {0.1, 0.2, 0.3});
    EXPECT_TRUE(in.inside);
    EXPECT_EQ(in.distance, 0.0);
    EXPECT_NEAR(in.signed_distance, -0.1, kHullDistance);
    EXPECT_NEAR(in.gradient[0], -1.0, kHullDistance);
}

// Brute-Force-Facetten einer Punktwolke in allgemeiner Lage: Tripel (i < j < k), deren Ebene
// alle übrigen Punkte auf einer Seite hat.
std::set<Tri> brute_facets(const std::vector<Point3>& pts) {
    std::set<Tri> out;
    const int n = static_cast<int>(pts.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                const auto& a = pts[static_cast<std::size_t>(i)];
                const Point3 nn = cross(sub(pts[static_cast<std::size_t>(j)], a),
                                        sub(pts[static_cast<std::size_t>(k)], a));
                int pos = 0;
                int neg = 0;
                for (int m = 0; m < n; ++m) {
                    if (m == i || m == j || m == k) {
                        continue;
                    }
                    const double s = dot(nn, sub(pts[static_cast<std::size_t>(m)], a));
                    (s > 0.0 ? pos : neg) += 1;
                }
                if (pos == 0 || neg == 0) {
                    out.insert({i, j, k});
                }
            }
        }
    }
    return out;
}

std::set<Tri> hull_facets(const ConvexHull3& h) {
    std::set<Tri> out;
    for (const Tri& t : h.triangles) {
        Tri s = {h.vertex_source[static_cast<std::size_t>(t[0])],
                 h.vertex_source[static_cast<std::size_t>(t[1])],
                 h.vertex_source[static_cast<std::size_t>(t[2])]};
        std::sort(s.begin(), s.end());
        out.insert(s);
    }
    return out;
}

TEST(ConvexHull, RandomCloudsMatchBruteForce) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int cloud = 0; cloud < 12; ++cloud) {
        const int n = 8 + 4 * cloud;
        std::vector<Point3> pts;
        for (int i = 0; i < n; ++i) {
            if (cloud % 3 == 2) {
                // auf einer Kugel: alle Punkte sind Ecken
                Point3 p{g(rng), g(rng), g(rng)};
                const double r = norm(p);
                pts.push_back({p[0] / r, p[1] / r, p[2] / r});
            } else {
                pts.push_back({u(rng), u(rng), 3.0 * u(rng)});
            }
        }
        const ConvexHull3 h = convex_hull(pts);
        expect_valid_polyhedron(h, pts);
        EXPECT_EQ(hull_facets(h), brute_facets(pts)) << "Wolke " << cloud;
        if (cloud % 3 == 2) {
            EXPECT_EQ(h.vertices.size(), pts.size());
        }
        // Abstand gegen die unabhängige Projektion auf alle Facetten, Kanten und Ecken
        for (int q = 0; q < 200; ++q) {
            const Point3 p{2.0 * u(rng), 2.0 * u(rng), 5.0 * u(rng)};
            EXPECT_NEAR(project_onto_hull(h, p).distance, brute_hull_distance(h, p), kHullDistance);
        }
    }
}

TEST(ConvexHull, GradientMatchesFiniteDifferences) {
    std::mt19937_64 rng(4);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<Point3> pts;
    for (int i = 0; i < 30; ++i) {
        pts.push_back({u(rng), 0.5 * u(rng), 2.0 * u(rng)});
    }
    const std::vector<ConvexHull3> hulls = {
        convex_hull(pts),
        convex_hull(std::vector<Point3>{{0, 0, 0}, {1, 0.2, 0}, {0.3, 1, 0.1}}), // Dreieck
        convex_hull(std::vector<Point3>{{0, 0, 0}, {1, 2, 3}}),                  // Strecke
        convex_hull(std::vector<Point3>{{0.2, 0.3, 0.4}}),                       // Punkt
    };
    const double h = 1e-6;
    for (const ConvexHull3& hull : hulls) {
        int checked = 0;
        for (int q = 0; q < 300; ++q) {
            const Point3 p{3.0 * u(rng), 3.0 * u(rng), 6.0 * u(rng)};
            const HullProjection r = project_onto_hull(hull, p);
            if (r.distance < 0.05) {
                continue; // nahe am Rand: Krümmung 1/d macht die zentrale Differenz ungenau
            }
            for (std::size_t k = 0; k < 3; ++k) {
                Point3 a = p;
                Point3 b = p;
                a[k] += h;
                b[k] -= h;
                const double fd =
                    (project_onto_hull(hull, a).distance - project_onto_hull(hull, b).distance) /
                    (2.0 * h);
                EXPECT_NEAR(r.gradient[k], fd, kHullGradientFd) << "Dimension " << hull.dimension;
            }
            EXPECT_NEAR(norm(r.gradient), 1.0, kHullGeometry);
            ++checked;
        }
        EXPECT_GT(checked, 100);
    }
}

TEST(ConvexHull, DegenerateSinglePoint) {
    const std::vector<Point3> pts = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3 + 1e-14}};
    const ConvexHull3 h = convex_hull(pts);
    EXPECT_EQ(h.dimension, 0);
    ASSERT_EQ(h.vertices.size(), 1u);
    const HullProjection r = project_onto_hull(h, {4, 6, 3});
    EXPECT_NEAR(r.distance, 5.0, kHullDistance);
    EXPECT_NEAR(r.signed_distance, 5.0, kHullDistance);
    EXPECT_NEAR(r.gradient[0], 0.6, kHullDistance);
    EXPECT_NEAR(r.gradient[1], 0.8, kHullDistance);
    EXPECT_TRUE(project_onto_hull(h, {1, 2, 3}).inside);
}

TEST(ConvexHull, DegenerateSegment) {
    // fünf kollineare Punkte: Strecke zwischen den äußeren
    std::vector<Point3> pts;
    for (const double t : {0.3, -1.0, 2.0, 0.0, 1.5}) {
        pts.push_back({1.0 + t, 2.0 * t, -t});
    }
    const ConvexHull3 h = convex_hull(pts);
    EXPECT_EQ(h.dimension, 1);
    ASSERT_EQ(h.vertices.size(), 2u);
    std::vector<int> ends = h.vertex_source;
    std::sort(ends.begin(), ends.end());
    EXPECT_EQ(ends, (std::vector<int>{1, 2}));
    // Punkt neben der Mitte (t = 0,5) senkrecht zur Richtung (1, 2, −1)/√6 im Abstand 3
    const Point3 dir{1.0, 2.0, -1.0};
    const Point3 perp{2.0 / std::sqrt(5.0), -1.0 / std::sqrt(5.0), 0.0};
    ASSERT_NEAR(dot(dir, perp), 0.0, 1e-15);
    const Point3 mid{1.5, 1.0, -0.5};
    const HullProjection r =
        project_onto_hull(h, {mid[0] + 3 * perp[0], mid[1] + 3 * perp[1], mid[2] + 3 * perp[2]});
    EXPECT_NEAR(r.distance, 3.0, kHullDistance);
    // jenseits des Endes t = 2: Abstand zum Endpunkt
    const HullProjection end = project_onto_hull(h, {1.0 + 3.0, 6.0, -3.0});
    EXPECT_NEAR(end.distance, std::sqrt(6.0), kHullDistance);
    EXPECT_TRUE(project_onto_hull(h, {1.5, 1.0, -0.5}).inside);
}

TEST(ConvexHull, DegenerateTriangleAndPolygon) {
    // drei Punkte: Dreieck in der Ebene z = 1
    const std::vector<Point3> tri = {{0, 0, 1}, {2, 0, 1}, {0, 2, 1}};
    const ConvexHull3 h3 = convex_hull(tri);
    EXPECT_EQ(h3.dimension, 2);
    EXPECT_EQ(h3.vertices.size(), 3u);
    EXPECT_EQ(h3.triangles.size(), 1u);
    EXPECT_NEAR(std::abs(h3.plane_normal[2]), 1.0, kHullGeometry);
    EXPECT_NEAR(project_onto_hull(h3, {0.5, 0.5, 4.0}).distance, 3.0, kHullDistance);  // über
    EXPECT_NEAR(project_onto_hull(h3, {-1.0, 0.5, 1.0}).distance, 1.0, kHullDistance); // Kante
    EXPECT_NEAR(project_onto_hull(h3, {2.0, 2.0, 1.0}).distance, std::sqrt(2.0),
                kHullDistance); // Hypotenuse
    EXPECT_TRUE(project_onto_hull(h3, {0.5, 0.5, 1.0}).inside);

    // Quadratgitter in einer schrägen Ebene: vier Ecken im Umlaufsinn um die Normale
    const Point3 e1{1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0};
    const Point3 e2{0.0, 0.0, 1.0};
    std::vector<Point3> grid;
    for (int i = 0; i <= 4; ++i) {
        for (int j = 0; j <= 4; ++j) {
            grid.push_back({0.25 * i * e1[0] + 0.25 * j * e2[0] + 1.0,
                            0.25 * i * e1[1] + 0.25 * j * e2[1] - 2.0,
                            0.25 * i * e1[2] + 0.25 * j * e2[2]});
        }
    }
    const ConvexHull3 hg = convex_hull(grid);
    EXPECT_EQ(hg.dimension, 2);
    ASSERT_EQ(hg.vertices.size(), 4u);
    std::vector<int> corners = hg.vertex_source;
    std::sort(corners.begin(), corners.end());
    EXPECT_EQ(corners, (std::vector<int>{0, 4, 20, 24}));
    // Umlaufsinn: (v1 − v0) × (v2 − v1) zeigt in Richtung der Normale
    for (std::size_t k = 0; k < 4; ++k) {
        const Point3 a = sub(hg.vertices[(k + 1) % 4], hg.vertices[k]);
        const Point3 b = sub(hg.vertices[(k + 2) % 4], hg.vertices[(k + 1) % 4]);
        EXPECT_GT(dot(cross(a, b), hg.plane_normal), 0.0);
    }
    EXPECT_EQ(hg.triangles.size(), 2u);
    // senkrecht über der Mitte im Abstand 0,7
    const Point3 n = cross(e1, e2);
    const Point3 c{1.0 + 0.5 * e1[0] + 0.7 * n[0], -2.0 + 0.5 * e1[1] + 0.7 * n[1],
                   0.5 + 0.7 * n[2]};
    EXPECT_NEAR(project_onto_hull(hg, c).distance, 0.7, kHullDistance);

    // fast koplanar (Abweichung 1e-13 bei Ausdehnung 1): Polygon
    std::vector<Point3> flat = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1e-13}};
    EXPECT_EQ(convex_hull(flat).dimension, 2);
    flat.back()[2] = 1e-6;
    EXPECT_EQ(convex_hull(flat).dimension, 3);
}

TEST(ConvexHull, EmptyAndInvalidInput) {
    const ConvexHull3 empty = convex_hull(std::vector<Point3>{});
    EXPECT_EQ(empty.dimension, -1);
    EXPECT_THROW(project_onto_hull(empty, {0, 0, 0}), std::invalid_argument);
    EXPECT_THROW(convex_hull(std::vector<Point3>{{0, 0, std::nan("")}}), std::invalid_argument);
    ConvexHullOptions bad;
    bad.relative_tolerance = 0.0;
    EXPECT_THROW(convex_hull(std::vector<Point3>{{0, 0, 0}}, bad), std::invalid_argument);
    const ConvexHull3 p = convex_hull(std::vector<Point3>{{0, 0, 0}});
    EXPECT_THROW(project_onto_hull(p, {0, std::nan(""), 0}), std::invalid_argument);
}

} // namespace
} // namespace mko::test
