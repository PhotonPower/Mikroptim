// Glas-Hülle (M6-T1, Abnahme „Hüllentests“): Normierung, Hülle der frei erfundenen Testgläser
// (data/glass/synthetisch.json) gegen eine unabhängige Referenz (numpy-Fit und
// scipy.spatial.ConvexHull, siehe Kommentar unten), Lage der Katalog-Gläser, Abstand außerhalb,
// entartete Kataloge und Fehler.

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

const std::string kGlassDir = MKO_SOURCE_DIR "/data/glass/";

Point3 sub(const Point3& a, const Point3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
double dot(const Point3& a, const Point3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point3 cross(const Point3& a, const Point3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Referenz für synthetisch.json, unabhängig berechnet mit Python: Buchdahl-Fit mit 3 Termen an
// denselben 301 Tschebyschow-Knoten über 400–700 nm per numpy.linalg.lstsq, Normierung mit
// Mittelwert und Spannweite, Hülle mit scipy.spatial.ConvexHull (Qhull).
constexpr std::array<double, 3> kRefCenter = {1.62724908, -0.09843656, 0.05023986};
constexpr std::array<double, 3> kRefScale = {0.38063772, 0.19489106, 0.21371309};
constexpr double kRefCenterDigits = 5e-9; // Referenz auf 8 Nachkommastellen angegeben
constexpr double kRefVolume = 0.02986044858479728;
constexpr double kRefArea = 1.1906401362816383;

TEST(GlassHull, SyntheticCatalogMatchesReference) {
    const GlassCatalog cat = GlassCatalog::from_file(kGlassDir + "synthetisch.json");
    const GlassHull h = make_glass_hull(cat);
    EXPECT_EQ(h.excluded(), (std::vector<std::string>{"SYN-UV", "SYN-IR"}));
    ASSERT_EQ(h.points().size(), 36u);
    ASSERT_EQ(h.dimension(), 3);
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(h.center()[j], kRefCenter[j], kRefCenterDigits);
        EXPECT_NEAR(h.scale()[j], kRefScale[j], kRefCenterDigits);
    }
    std::vector<std::string> names = h.vertex_names();
    std::sort(names.begin(), names.end());
    EXPECT_EQ(names, (std::vector<std::string>{"SYN-02", "SYN-03", "SYN-04", "SYN-05", "SYN-12",
                                               "SYN-19", "SYN-20", "SYN-26", "SYN-29", "SYN-30",
                                               "SYN-32", "SYN-33", "SYN-34", "SYN-35", "SYN-36"}));
    // Volumen (Divergenzsatz) und Oberfläche der Hülle in normierten Koordinaten
    const ConvexHull3& hull = h.hull();
    double volume = 0.0;
    double area = 0.0;
    for (const auto& t : hull.triangles) {
        const Point3& a = hull.vertices[static_cast<std::size_t>(t[0])];
        const Point3& b = hull.vertices[static_cast<std::size_t>(t[1])];
        const Point3& c = hull.vertices[static_cast<std::size_t>(t[2])];
        volume += dot(a, cross(b, c)) / 6.0;
        const Point3 n = cross(sub(b, a), sub(c, a));
        area += 0.5 * std::sqrt(dot(n, n));
    }
    std::printf("Hülle synthetisch.json: %zu Ecken, %zu Dreiecke, V = %.15g (Referenz %.15g), "
                "A = %.15g (Referenz %.15g)\n",
                hull.vertices.size(), hull.triangles.size(), volume, kRefVolume, area, kRefArea);
    EXPECT_NEAR(volume, kRefVolume, kGlassHullReference * kRefVolume);
    EXPECT_NEAR(area, kRefArea, kGlassHullReference * kRefArea);
    // Normierung: jede Koordinate hat Spannweite 1 und Mittel 0 über die Gläser
    for (std::size_t j = 0; j < 3; ++j) {
        double lo = 1e300;
        double hi = -1e300;
        double sum = 0.0;
        for (const GlassHullPoint& p : h.points()) {
            const double u = h.normalize(p.coordinates)[j];
            lo = std::min(lo, u);
            hi = std::max(hi, u);
            sum += u;
        }
        EXPECT_NEAR(hi - lo, 1.0, kHullGeometry);
        EXPECT_NEAR(sum / 36.0, 0.0, kHullGeometry);
    }
}

TEST(GlassHull, CatalogGlassesAreInside) {
    GlassCatalog cat = GlassCatalog::from_file(kGlassDir + "synthetisch.json");
    cat.merge(GlassCatalog::from_file(kGlassDir + "basis.json"));
    const GlassHull h = make_glass_hull(cat);
    EXPECT_EQ(h.points().size(), 39u);
    const std::vector<std::string> vertices = h.vertex_names();
    for (const GlassHullPoint& p : h.points()) {
        const HullProjection r = h.project(p.coordinates);
        const bool is_vertex =
            std::find(vertices.begin(), vertices.end(), p.name) != vertices.end();
        if (is_vertex) {
            EXPECT_LE(r.distance, kHullDistance) << p.name; // Ecke: auf dem Rand
        } else {
            EXPECT_TRUE(r.inside) << p.name;
            EXPECT_EQ(r.distance, 0.0) << p.name;
            EXPECT_LT(r.signed_distance, 0.0) << p.name;
        }
        EXPECT_TRUE(h.contains(p.coordinates, kHullDistance)) << p.name;
        EXPECT_GT(p.max_fit_error, 0.0) << p.name;
    }
    // CaF₂ hat die kleinste Dispersion |ν₁| und ist eine Ecke
    EXPECT_NE(std::find(vertices.begin(), vertices.end(), "CAF2"), vertices.end());
}

TEST(GlassHull, DistanceOutsideAlongFacetNormal) {
    const GlassHull h = make_glass_hull(GlassCatalog::from_file(kGlassDir + "synthetisch.json"));
    const ConvexHull3& hull = h.hull();
    // Schwerpunkt jeder Facette plus t·Normale (normiert): nächster Punkt ist der Schwerpunkt,
    // Abstand t; in Rohkoordinaten über denormalize.
    for (std::size_t f = 0; f < hull.triangles.size(); ++f) {
        Point3 c{};
        for (const int k : hull.triangles[f]) {
            for (std::size_t j = 0; j < 3; ++j) {
                c[j] += hull.vertices[static_cast<std::size_t>(k)][j] / 3.0;
            }
        }
        for (const double t : {1e-6, 0.01, 0.3}) {
            const Point3& n = hull.planes[f].normal;
            const Point3 u{c[0] + t * n[0], c[1] + t * n[1], c[2] + t * n[2]};
            const Point3 x = h.denormalize(u);
            EXPECT_NEAR(h.distance(x), t, kHullDistance) << "Facette " << f;
            EXPECT_FALSE(h.contains(x));
            const Point3 inner{c[0] - t * n[0], c[1] - t * n[1], c[2] - t * n[2]};
            if (t < 0.01) {
                EXPECT_TRUE(h.contains(h.denormalize(inner)));
            }
        }
    }
}

TEST(GlassHull, NormalizationAndManualPoints) {
    // ν₂ für alle gleich: Spannweite 0 → Skala 1; die Hülle ist dann eben (Dimension 2)
    std::vector<GlassHullPoint> pts = {
        {"A", {1.5, -0.05, 0.004}, 0.0, 0.0},
        {"B", {1.7, -0.10, 0.004}, 0.0, 0.0},
        {"C", {1.6, -0.15, 0.004}, 0.0, 0.0},
        {"D", {1.6, -0.10, 0.004}, 0.0, 0.0},
    };
    const GlassHull h(pts);
    EXPECT_NEAR(h.center()[0], 1.6, 1e-15);
    EXPECT_NEAR(h.center()[1], -0.1, 1e-15);
    EXPECT_EQ(h.scale()[2], 1.0);
    EXPECT_NEAR(h.scale()[0], 0.2, 1e-15);
    EXPECT_NEAR(h.scale()[1], 0.1, 1e-15);
    EXPECT_EQ(h.dimension(), 2);
    EXPECT_EQ(h.vertex_names().size(), 3u); // D liegt im Dreieck
    const Point3 x{1.55, -0.12, 0.0123};
    const Point3 back = h.denormalize(h.normalize(x));
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(back[j], x[j], 1e-15);
    }
    // senkrecht zur Ebene: Abstand = |Δν₂| (Skala 1)
    EXPECT_NEAR(h.distance({1.6, -0.1, 0.014}), 0.01, kHullDistance);
    EXPECT_THROW(GlassHull(std::vector<GlassHullPoint>{}), std::invalid_argument);
    EXPECT_THROW(GlassHull(std::vector<GlassHullPoint>{{"X", {1.5, std::nan(""), 0.0}, 0, 0}}),
                 std::invalid_argument);
    EXPECT_THROW(GlassHull().distance(x), std::invalid_argument);
}

TEST(GlassHull, BasisCatalogIsPlanarAndRangeIsChecked) {
    const GlassHull h = make_glass_hull(GlassCatalog::from_file(kGlassDir + "basis.json"));
    EXPECT_EQ(h.points().size(), 3u);
    EXPECT_EQ(h.dimension(), 2); // drei Gläser: Dreieck
    EXPECT_TRUE(h.excluded().empty());
    for (const GlassHullPoint& p : h.points()) {
        EXPECT_LE(h.distance(p.coordinates), kHullDistance);
    }
    // Kein Glas deckt den Fitbereich ab → GlassError
    const GlassCatalog only_uv = GlassCatalog::from_string(R"({
      "format": "mikroptim.glass/1",
      "glasses": [ { "name": "UV", "model": "sellmeier", "B": [1.1, 0.2, 1.0],
                     "C": [0.006, 0.03, 100.0], "range_nm": [250, 650] } ] })");
    EXPECT_THROW(make_glass_hull(only_uv), GlassError);
    // 2 Terme: ν₃ = 0, andere Koordinaten
    GlassHullOptions two;
    two.dispersion_terms = 2;
    const GlassHull h2 = make_glass_hull(GlassCatalog::from_file(kGlassDir + "basis.json"), two);
    EXPECT_EQ(h2.points()[0].nu3, 0.0);
    EXPECT_NE(h2.points()[0].coordinates[1], h.points()[0].coordinates[1]);
}

} // namespace
} // namespace mko::test
