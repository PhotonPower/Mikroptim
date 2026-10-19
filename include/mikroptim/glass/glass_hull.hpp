#pragma once

// Glas-Hülle (M6-T1, Host): konvexe Hülle der Katalog-Gläser im normierten Raum der
// Buchdahl-Koordinaten (n₀, ν₁, ν₂) (docs/optimierung.md, Abschnitt 1, „Glas-Hülle“;
// docs/numerik.md, Abschnitt 10). Die Strafe außerhalb der Hülle steht in
// mikroptim/optim/glass_variables.hpp.
//
// Koordinaten eines Glases: (n₀, ν₁, ν₂) des Buchdahl-Fits (Standard: 3 Terme über 400–700 nm,
// dieselben Einstellungen wie bei der Umwandlung des Startglases, convert_glass_variables).
// Normierung: u_j = (x_j − m_j)/s_j mit dem Mittelwert m_j und der Spannweite s_j = max − min
// der Koordinate j über die Gläser der Hülle (s_j = 1, wenn die Spannweite 0 ist). Eine Einheit
// in u ist also die Ausdehnung des Katalogs in dieser Koordinate; Abstände in u sind
// dimensionslos und gewichten n₀, ν₁ und ν₂ gleich.

#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/convex_hull.hpp"

#include <string>
#include <vector>

namespace mko {

// Ein Glas der Hülle.
struct GlassHullPoint {
    std::string name;
    Point3 coordinates{};       // (n₀, ν₁, ν₂)
    double nu3 = 0.0;           // ν₃ des Fits (0 bei 2 Termen), nur zur Information
    double max_fit_error = 0.0; // max |n_Buchdahl − n| des Fits
};

struct GlassHullOptions {
    BuchdahlFitOptions fit{}; // Fitbereich und Stützstellen (Standard 400–700 nm)
    int dispersion_terms = 3; // Termzahl des Fits (2 oder 3)
    ConvexHullOptions hull{}; // Toleranz der Hülle (in normierten Koordinaten)
};

class GlassHull {
public:
    GlassHull() = default;
    // Hülle der gegebenen Gläser; Normierung aus genau diesen Punkten. Wirft
    // std::invalid_argument ohne Punkte oder bei nicht endlichen Koordinaten.
    explicit GlassHull(std::vector<GlassHullPoint> points, const ConvexHullOptions& options = {},
                       std::vector<std::string> excluded = {});

    bool empty() const { return hull_.dimension < 0; }
    int dimension() const { return hull_.dimension; }
    const std::vector<GlassHullPoint>& points() const { return points_; }
    // Gläser des Katalogs, die nicht in die Hülle eingehen (Fitbereich nicht abgedeckt).
    const std::vector<std::string>& excluded() const { return excluded_; }
    const Point3& center() const { return center_; } // m_j
    const Point3& scale() const { return scale_; }   // s_j
    // Hülle in normierten Koordinaten.
    const ConvexHull3& hull() const { return hull_; }
    // Namen der Gläser auf den Ecken der Hülle (Reihenfolge von hull().vertices).
    std::vector<std::string> vertex_names() const;

    Point3 normalize(const Point3& x) const;   // u = (x − m)/s
    Point3 denormalize(const Point3& u) const; // x = m + s·u

    // Projektion von x (Rohkoordinaten) auf die Hülle; alle Größen des Ergebnisses
    // (closest, distance, signed_distance, gradient) in normierten Koordinaten.
    HullProjection project(const Point3& x) const;
    // Euklidischer Abstand d(u(x), Hülle) in normierten Koordinaten (0 innen).
    double distance(const Point3& x) const;
    // d ≤ tolerance (Standard 0: exakt innen oder auf dem Rand).
    bool contains(const Point3& x, double tolerance = 0.0) const;

private:
    std::vector<GlassHullPoint> points_;
    std::vector<std::string> excluded_;
    Point3 center_{};
    Point3 scale_{1.0, 1.0, 1.0};
    ConvexHull3 hull_;
};

// Hülle aller Gläser des Katalogs (Lade-Reihenfolge), deren Gültigkeitsbereich range_nm den
// Fitbereich abdeckt; die übrigen stehen in excluded(). Wirft GlassError, wenn kein Glas den
// Fitbereich abdeckt oder ein Fit scheitert, std::invalid_argument bei ungültigen Optionen.
GlassHull make_glass_hull(const GlassCatalog& catalog, const GlassHullOptions& options = {});

// Koordinaten (n₀, ν₁, ν₂) eines Fits.
Point3 glass_coordinates(const BuchdahlFit& fit);

} // namespace mko
