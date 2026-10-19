#pragma once

// Konvexe Hülle einer Punktmenge im ℝ³ mit Abstand und Projektion (Host, M6-T1). Grundlage der
// Glas-Hülle im normierten (n₀, ν₁, ν₂)-Raum (mikroptim/glass/glass_hull.hpp,
// docs/optimierung.md, Abschnitt 1). Eigener Code ohne Abhängigkeiten: inkrementelle Hülle
// (Punkte nach fallendem Abstand vom Schwerpunkt des Starttetraeders eingefügt, Horizont über
// gerichtete Kanten). Entartete Punktmengen ergeben eine Hülle niedrigerer Dimension: ein Punkt
// (alle Punkte innerhalb der Toleranz gleich), eine Strecke (kollinear), ein konvexes Polygon
// (koplanar). Alle Rechnungen in FP64.

#include <array>
#include <span>
#include <vector>

namespace mko {

using Point3 = std::array<double, 3>;

// Ebene n·x = offset mit |n| = 1; das Innere der Hülle liegt bei n·x ≤ offset.
struct HullPlane {
    Point3 normal{};
    double offset = 0.0;
};

struct ConvexHullOptions {
    // Toleranz ε = relative_tolerance · Skala mit Skala = max(Diagonale der Hüllbox, größter
    // Koordinatenbetrag): Punkte näher als ε an einer Ebene, Geraden bzw. einem Punkt gelten als
    // darauf (Entscheidung über die Dimension und darüber, ob ein Punkt die Hülle erweitert).
    double relative_tolerance = 1e-10;
};

struct ConvexHull3 {
    int dimension = -1;     // −1: keine Punkte, 0: Punkt, 1: Strecke, 2: Polygon, 3: Polyeder
    double tolerance = 0.0; // verwendete absolute Toleranz ε
    // Ecken der Hülle (Kopien der Eingabepunkte) und ihr Index in der Eingabe. Dimension 3:
    // aufsteigend nach Eingabeindex; 2: Polygon im Umlaufsinn (gegen den Uhrzeigersinn um
    // plane_normal); 1: die beiden Endpunkte; 0: ein Punkt.
    std::vector<Point3> vertices;
    std::vector<int> vertex_source;
    // Dreiecke (Indizes in vertices). Dimension 3: Randdreiecke, von außen gesehen gegen den
    // Uhrzeigersinn; koplanare Nachbarn werden nicht zusammengefasst (ein Würfel hat 12
    // Dreiecke in 6 Ebenen). Dimension 2: Fächer (0, k, k + 1) des Polygons. Sonst leer.
    std::vector<std::array<int, 3>> triangles;
    // Nur Dimension 3: Ebene jedes Dreiecks (gleicher Index wie triangles), Normale nach außen.
    std::vector<HullPlane> planes;
    // Nur Dimension 2: Einheitsnormale der Ebene des Polygons.
    Point3 plane_normal{};
};

// Hülle der Punkte. Wirft std::invalid_argument bei nicht endlichen Koordinaten oder
// relative_tolerance ≤ 0. Keine Punkte: dimension = −1.
ConvexHull3 convex_hull(std::span<const Point3> points, const ConvexHullOptions& options = {});

// Nächster Punkt der Hülle zu p und Abstände.
struct HullProjection {
    Point3 closest{};      // nächster Punkt der Hülle (p selbst, wenn p in der Hülle liegt)
    double distance = 0.0; // euklidischer Abstand d = |p − closest| ≥ 0
    // Vorzeichenbehafteter Abstand: außen d; innen (nur Dimension 3) −min_i (offset_i − n_i·p),
    // also minus der Abstand zum Rand; bei niedrigerer Dimension gleich d.
    double signed_distance = 0.0;
    // Gradient des vorzeichenbehafteten Abstands: außen (p − closest)/d, innen (Dimension 3) die
    // Normale der nächsten Randebene; bei niedrigerer Dimension innen 0.
    Point3 gradient{};
    bool inside = false; // Dimension 3: n_i·p ≤ offset_i für alle Ebenen; sonst d = 0
};

// Projektion auf die Hülle: Dimension 3 innen ohne Rechnung, außen das Minimum über die
// Randdreiecke (der nächste Punkt eines konvexen Polyeders liegt auf seinem Rand); Dimension 2
// Minimum über die Fächerdreiecke, 1 Strecke, 0 Punkt. Wirft std::invalid_argument bei leerer
// Hülle oder nicht endlichem p.
HullProjection project_onto_hull(const ConvexHull3& hull, const Point3& p);

// Nächster Punkt eines Dreiecks bzw. einer Strecke zu p (Fallunterscheidung über die
// Voronoi-Gebiete von Ecken, Kanten und Fläche, Ericson: Real-Time Collision Detection, 5.1).
Point3 closest_point_on_triangle(const Point3& p, const Point3& a, const Point3& b,
                                 const Point3& c);
Point3 closest_point_on_segment(const Point3& p, const Point3& a, const Point3& b);

} // namespace mko
