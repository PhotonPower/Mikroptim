// Mechanische Nebenbedingungen (M4-T3): Pfeilhöhe und Randdicke gegen geschlossene Formeln und
// Grenzfälle, Schema (Lesen, Schreiben, Roundtrip, Fehler mit JSON-Pfad), Auflösung der Flächen
// (auch mit Geisterplatten), Residuen aktiv/inaktiv, trace-abhängige Residuen gegen eine
// unabhängige Auswertung aller Strahlen, FD-Jacobi mit diesen Residuen und Merit = Σ r²
// (docs/optimierung.md, Abschnitte 2.1 und 2.5).

#include "fd_test_systems.hpp"
#include "mechanical_test_systems.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"
#include "mikroptim/trace/surface_extrema.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// ---- Pfeilhöhe und Randdicke ------------------------------------------------------------------

// Die Pfeilhöhe liegt auf der Sphäre: h² + (sag − R)² = R², also c (h² + sag²) − 2 sag = 0
// (unabhängig von der Form der Formel). Dazu Symmetrien und die Reihe für kleine c h.
TEST(MechanicalSag, OnSphereAndSymmetric) {
    for (const double c : {0.1, -0.1, 0.25, -0.031, 1e-6}) {
        for (const double h : {0.5, 2.0, 3.9, 0.01}) {
            if (std::abs(c * h) >= 1.0) {
                continue;
            }
            const double s = sag(c, h);
            const double defect = c * (h * h + s * s) - 2.0 * s;
            EXPECT_LE(std::abs(defect), 8.0 * kEps * std::abs(c) * h * h) << c << ' ' << h;
            EXPECT_EQ(sag(-c, h), -s);
            EXPECT_EQ(sag(c, -h), s);
            EXPECT_GT(s * c, 0.0); // c > 0: Pfeilhöhe in +z (Mittelpunkt auf der +z-Seite)
        }
    }
    // Reihe c h²/2 + c³h⁴/8 + c⁵h⁶/16 für kleine c h
    const double c = 1e-3;
    const double h = 2.0;
    const double series =
        c * h * h / 2 + std::pow(c, 3) * std::pow(h, 4) / 8 + std::pow(c, 5) * std::pow(h, 6) / 16;
    EXPECT_NEAR(sag(c, h), series, 1e-15);
    // Planfläche exakt 0
    EXPECT_EQ(sag(0.0, 3.0), 0.0);
    EXPECT_EQ(sag(0.0, 1e6), 0.0);
}

// Grenzfall c h → 1: Halbkugel mit Pfeilhöhe 1/c, stetige Fortsetzung c h² dahinter.
TEST(MechanicalSag, HemisphereLimitAndContinuation) {
    const double c = 0.25; // R = 4
    EXPECT_EQ(sag(c, 4.0), 4.0);
    const double below = sag(c, 4.0 * (1.0 - 1e-14));
    EXPECT_LT(below, 4.0);
    EXPECT_GT(below, 4.0 - 1e-6); // R − √(R² − h²) ≈ R − R √(2·10⁻¹⁴)
    EXPECT_DOUBLE_EQ(sag(c, 4.4), c * 4.4 * 4.4);
    EXPECT_GT(sag(c, 4.4), sag(c, 4.0));
    EXPECT_EQ(sag(-c, 4.4), -sag(c, 4.4));
    // Randdicke einer Halbkugellinse fällt weiter mit h (endliche Strafe statt NaN)
    const double e1 = edge_thickness(5.0, c, 0.0, 4.0);
    const double e2 = edge_thickness(5.0, c, 0.0, 4.4);
    EXPECT_EQ(e1, 1.0);
    EXPECT_LT(e2, e1);
    EXPECT_TRUE(std::isfinite(e2));
}

// Randdicke t − sag₁ + sag₂ mit Pfeilhöhen aus R − √(R² − h²) (bei diesen Werten ohne merkliche
// Auslöschung).
TEST(MechanicalSag, EdgeThicknessBiconvexMeniscusPlano) {
    const double h = 3.0;
    const double s10 = 10.0 - std::sqrt(100.0 - 9.0); // R = 10
    const double s20 = 20.0 - std::sqrt(400.0 - 9.0); // R = 20
    const double tol = 1e-14;
    // bikonvex: c₁ = 0,1, c₂ = −0,05: t − |sag₁| − |sag₂|
    EXPECT_NEAR(edge_thickness(4.0, 0.1, -0.05, h), 4.0 - s10 - s20, tol);
    // Meniskus: c₁ = 0,1, c₂ = 0,05: t − sag₁ + sag₂
    EXPECT_NEAR(edge_thickness(4.0, 0.1, 0.05, h), 4.0 - s10 + s20, tol);
    // bikonkav: Rand dicker als die Mitte
    EXPECT_NEAR(edge_thickness(1.0, -0.1, 0.05, h), 1.0 + s10 + s20, tol);
    // plankonvex
    EXPECT_NEAR(edge_thickness(4.0, 0.1, 0.0, h), 4.0 - s10, tol);
    EXPECT_EQ(edge_thickness(4.0, 0.0, 0.0, h), 4.0);
}

// ---- Schema -----------------------------------------------------------------------------------

const std::string kBase = R"({
  "format": "mikroptim.system/1",
  "name": "Mechanik",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
  "target": { "focal_length": 10.0 },
  "surfaces": [
    { "type": "stop", "thickness": 2.0, "material": "AIR", "semi_diameter": 3.0 },
    { "type": "sphere", "curvature": 0.1, "thickness": 4.0, "material": { "model": "constant", "n": 1.5 }, "semi_diameter": 4.0 },
    { "type": "sphere", "curvature": -0.05, "thickness": 1.0, "material": "AIR", "semi_diameter": 4.5 },
    { "type": "plane", "thickness": 20.0, "material": "AIR", "semi_diameter": 4.0 }
  ],
  "constraints": { "items": [ITEMS] }
}
)";

std::string with_items(const std::string& items) {
    std::string s = kBase;
    s.replace(s.find("ITEMS"), 5, items);
    return s;
}

void expect_error(const std::string& items, const std::string& path) {
    try {
        (void)read_system_json(with_items(items));
        ADD_FAILURE() << "keine Ausnahme, erwartet Fehler unter " << path;
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), path) << e.what();
    }
}

TEST(MechanicalSchema, RoundtripAllFields) {
    const std::string text = with_items(R"(
    { "type": "max_semi_diameter", "weight": 2.0, "surfaces": [1, 2], "semi_diameter": 3.5, "clearance": 0.2 },
    { "type": "max_semi_diameter" },
    { "type": "min_center_thickness", "medium": "glass", "thickness": 1.0 },
    { "type": "max_center_thickness", "surface": 2, "medium": "air", "thickness": 8.0 },
    { "type": "min_edge_thickness", "surfaces": "all", "thickness": 0.5, "height": "rays", "clearance": 0.3, "comment": "Rand" },
    { "type": "min_edge_thickness", "medium": "air", "thickness": 0.0, "height": "semi_diameter" },
    { "type": "max_incidence_angle", "angle": 55.0, "surface": 3 })");
    std::vector<std::string> warnings;
    const SystemDescription a = read_system_json(text, &warnings);
    EXPECT_TRUE(warnings.empty());
    const std::vector<ConstraintSpec>& c = a.constraints->items;
    ASSERT_EQ(c.size(), 7u);
    EXPECT_EQ(c[0].type, ConstraintType::MaxSemiDiameter);
    EXPECT_EQ(c[0].surface_selection, SurfaceSelection::List);
    EXPECT_EQ(c[0].surfaces, (std::vector<int>{1, 2}));
    EXPECT_EQ(c[0].semi_diameter, 3.5);
    EXPECT_EQ(c[0].clearance, 0.2);
    EXPECT_EQ(c[1].surface_selection, SurfaceSelection::None);
    EXPECT_FALSE(c[1].semi_diameter.has_value());
    EXPECT_EQ(c[2].medium, ConstraintMedium::Glass);
    EXPECT_EQ(c[2].thickness, 1.0);
    EXPECT_EQ(c[3].surface_selection, SurfaceSelection::One);
    EXPECT_EQ(c[3].medium, ConstraintMedium::Air);
    EXPECT_EQ(c[4].surface_selection, SurfaceSelection::All);
    EXPECT_EQ(c[4].height, EdgeHeight::Rays);
    EXPECT_EQ(c[4].height_or_default(), EdgeHeight::Rays);
    EXPECT_EQ(c[5].height, EdgeHeight::SemiDiameter);
    EXPECT_EQ(c[6].angle, 55.0);
    EXPECT_EQ(c[2].height_or_default(), EdgeHeight::SemiDiameter);
    EXPECT_EQ(c[2].clearance_or_default(), 0.0);
    // constraint_bound: Grenzen im Eintrag, kein Sollwert in target
    EXPECT_EQ(constraint_bound(a, c[0]), 3.5);
    EXPECT_FALSE(constraint_bound(a, c[1]).has_value());
    EXPECT_EQ(constraint_bound(a, c[3]), 8.0);
    EXPECT_EQ(constraint_bound(a, c[6]), 55.0);
    EXPECT_FALSE(constraint_target(a, ConstraintType::MinEdgeThickness).has_value());

    const std::string text1 = write_system_json(a);
    const SystemDescription b = read_system_json(text1);
    EXPECT_TRUE(a == b);
    EXPECT_EQ(text1, write_system_json(b));
    for (
        const char* expected :
        {R"({ "type": "max_semi_diameter", "weight": 2.0, "surfaces": [1, 2], "semi_diameter": 3.5, "clearance": 0.2 })",
         R"({ "type": "max_center_thickness", "surface": 2, "medium": "air", "thickness": 8.0 })",
         R"({ "type": "min_edge_thickness", "medium": "air", "thickness": 0.0, "height": "semi_diameter" })",
         R"({ "type": "max_incidence_angle", "surface": 3, "angle": 55.0 })"}) {
        EXPECT_NE(text1.find(expected), std::string::npos) << expected << '\n' << text1;
    }
}

TEST(MechanicalSchema, NamesAndKinds) {
    for (const ConstraintType t :
         {ConstraintType::MaxSemiDiameter, ConstraintType::MinCenterThickness,
          ConstraintType::MaxCenterThickness, ConstraintType::MinEdgeThickness,
          ConstraintType::MaxIncidenceAngle}) {
        EXPECT_EQ(constraint_type_from_string(to_string(t)), t);
        EXPECT_TRUE(is_mechanical(t));
        EXPECT_FALSE(is_paraxial(t));
    }
    EXPECT_FALSE(is_mechanical(ConstraintType::Petzval));
    EXPECT_FALSE(is_mechanical(ConstraintType::GhostPenalty));
    EXPECT_EQ(constraint_medium_from_string("glass"), ConstraintMedium::Glass);
    EXPECT_EQ(constraint_medium_from_string("air"), ConstraintMedium::Air);
    EXPECT_FALSE(constraint_medium_from_string("AIR").has_value());
    EXPECT_EQ(edge_height_from_string("rays"), EdgeHeight::Rays);
    EXPECT_EQ(edge_height_from_string("semi_diameter"), EdgeHeight::SemiDiameter);
    EXPECT_FALSE(edge_height_from_string("ray").has_value());
}

TEST(MechanicalSchema, RuleErrors) {
    const std::string p = "constraints.items[0]";
    // Felder: Typ und Wertebereich
    expect_error(R"({ "type": "max_semi_diameter", "semi_diameter": "3" })", p + ".semi_diameter");
    expect_error(R"({ "type": "min_center_thickness", "medium": "Glas", "thickness": 1 })",
                 p + ".medium");
    expect_error(R"({ "type": "min_edge_thickness", "height": "edge", "thickness": 1 })",
                 p + ".height");
    expect_error(R"({ "type": "max_semi_diameter", "surfaces": [1, -1] })", p + ".surfaces[1]");
    // kein Modus
    expect_error(R"({ "type": "max_semi_diameter", "mode": "max" })", p + ".mode");
    // Pflichtgrenzen
    expect_error(R"({ "type": "min_center_thickness" })", p + ".thickness");
    expect_error(R"({ "type": "min_edge_thickness", "medium": "glass" })", p + ".thickness");
    expect_error(R"({ "type": "max_incidence_angle" })", p + ".angle");
    // Wertebereiche
    expect_error(R"({ "type": "max_incidence_angle", "angle": 90 })", p + ".angle");
    expect_error(R"({ "type": "max_incidence_angle", "angle": 0 })", p + ".angle");
    expect_error(R"({ "type": "max_center_thickness", "thickness": 0 })", p + ".thickness");
    expect_error(R"({ "type": "min_center_thickness", "thickness": -0.1 })", p + ".thickness");
    expect_error(R"({ "type": "max_semi_diameter", "semi_diameter": 0 })", p + ".semi_diameter");
    expect_error(R"({ "type": "max_semi_diameter", "clearance": -1 })", p + ".clearance");
    expect_error(R"({ "type": "max_semi_diameter", "weight": -1 })", p + ".weight");
    // Felder beim falschen Typ
    expect_error(R"({ "type": "max_semi_diameter", "thickness": 1 })", p + ".thickness");
    expect_error(R"({ "type": "max_semi_diameter", "medium": "air" })", p + ".medium");
    expect_error(R"({ "type": "min_center_thickness", "thickness": 1, "clearance": 0.1 })",
                 p + ".clearance");
    expect_error(R"({ "type": "max_center_thickness", "thickness": 1, "height": "rays" })",
                 p + ".height");
    expect_error(R"({ "type": "max_incidence_angle", "angle": 50, "semi_diameter": 2 })",
                 p + ".semi_diameter");
    expect_error(R"({ "type": "min_edge_thickness", "thickness": 1, "angle": 50 })", p + ".angle");
    expect_error(R"({ "type": "max_semi_diameter", "max_curvature": 0.01 })", p + ".max_curvature");
    expect_error(R"({ "type": "max_semi_diameter", "plates": "all" })", p + ".plates");
    expect_error(R"({ "type": "focal_length", "surface": 1 })", p + ".surface");
    expect_error(R"({ "type": "focal_length", "thickness": 1 })", p + ".thickness");
    expect_error(R"({ "type": "petzval", "max_curvature": 0.01, "angle": 50 })", p + ".angle");
    // Flächen: Bereich, Medium, doppelt, leer
    expect_error(R"({ "type": "max_semi_diameter", "surface": 4 })", p + ".surface");
    expect_error(R"({ "type": "min_center_thickness", "surface": 3, "thickness": 1 })",
                 p + ".surface");
    expect_error(
        R"({ "type": "min_center_thickness", "surfaces": [0, 1], "medium": "air", "thickness": 1 })",
        p + ".surfaces[1]");
    expect_error(R"({ "type": "max_incidence_angle", "surfaces": [1, 1], "angle": 50 })",
                 p + ".surfaces[1]");
    expect_error(R"({ "type": "max_incidence_angle", "surfaces": [], "angle": 50 })",
                 p + ".surfaces");
    // Mehrere Einträge eines mechanischen Typs sind erlaubt
    EXPECT_NO_THROW((void)read_system_json(
        with_items(R"({ "type": "min_center_thickness", "medium": "glass", "thickness": 1 },
                      { "type": "min_center_thickness", "medium": "air", "thickness": 0.1 })")));
}

// ---- Auflösung und Residuen ohne Strahlbezug --------------------------------------------------

TEST(MechanicalConstraintsResolve, SurfacesTableSurfacesAndLimits) {
    const SystemDescription s = read_system_json(with_items(R"(
    { "type": "max_semi_diameter", "weight": 4.0 },
    { "type": "max_semi_diameter", "surface": 0, "semi_diameter": 2.5 },
    { "type": "min_center_thickness", "medium": "glass", "thickness": 1.0 },
    { "type": "min_center_thickness", "medium": "air", "thickness": 0.5 },
    { "type": "max_center_thickness", "thickness": 3.0 },
    { "type": "min_edge_thickness", "medium": "glass", "thickness": 1.0, "clearance": 0.25 },
    { "type": "max_incidence_angle", "angle": 60.0, "surfaces": [2, 1] })"));
    // Standardauswahl
    EXPECT_EQ(constraint_surfaces(s, s.constraints->items[0]), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(constraint_surfaces(s, s.constraints->items[2]), (std::vector<int>{1}));
    EXPECT_EQ(constraint_surfaces(s, s.constraints->items[3]), (std::vector<int>{0, 2}));
    EXPECT_EQ(constraint_surfaces(s, s.constraints->items[4]), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(constraint_surfaces(s, s.constraints->items[6]), (std::vector<int>{2, 1}));

    const MechanicalConstraints m = make_mechanical_constraints(s);
    ASSERT_EQ(m.size(), 3 + 1 + 1 + 2 + 3 + 1 + 2);
    // max_semi_diameter: Grenze aus semi_diameter der Flächen bzw. explizit
    EXPECT_EQ(m.items[0].limit, 4.0);
    EXPECT_EQ(m.items[1].limit, 4.5);
    EXPECT_EQ(m.items[2].limit, 4.0);
    EXPECT_EQ(m.items[0].weight, 4.0);
    EXPECT_EQ(m.items[3].file_surface, 0);
    EXPECT_EQ(m.items[3].limit, 2.5);
    EXPECT_EQ(m.items[3].next_surface, m.items[3].surface);
    // Randdicke: Halbdurchmesser der Flächen 1 und 2, clearance
    const MechanicalConstraint& edge = m.items[10];
    EXPECT_EQ(edge.type, ConstraintType::MinEdgeThickness);
    EXPECT_EQ(edge.semi_diameter, 4.0);
    EXPECT_EQ(edge.next_semi_diameter, 4.5);
    EXPECT_EQ(edge.clearance, 0.25);
    EXPECT_EQ(mechanical_edge_height(edge, nullptr), 4.75);
    EXPECT_EQ(edge.next_surface, 2);
    // Einfallswinkel als cos
    EXPECT_EQ(m.items[11].angle_deg, 60.0);
    EXPECT_NEAR(m.items[11].limit, 0.5, 1e-15);
    EXPECT_EQ(m.items[11].file_surface, 2);
    EXPECT_TRUE(m.needs_extrema());
    EXPECT_TRUE(needs_extrema(m.items[0]));
    EXPECT_FALSE(needs_extrema(m.items[4]));
    EXPECT_FALSE(needs_extrema(edge));

    const MechanicalResiduals r = make_mechanical_residuals(m);
    EXPECT_EQ(r.geometric.count, 1 + 2 + 3 + 1);
    EXPECT_EQ(r.traced.count, 3 + 1 + 2);
    EXPECT_EQ(r.geometric_items, (std::vector<int>{4, 5, 6, 7, 8, 9, 10}));
    EXPECT_EQ(r.traced_items, (std::vector<int>{0, 1, 2, 3, 11, 12}));
}

// Mittendicke eines geteilten Luftspalts (Geisterplatten) und Tabellenflächen.
TEST(MechanicalConstraintsResolve, GhostPlateGap) {
    LoadedSystem loaded = load_residual_system("einzellinse_geisterplatte.json");
    ConstraintSpec gap;
    gap.type = ConstraintType::MaxCenterThickness;
    gap.medium = ConstraintMedium::Air;
    gap.surface_selection = SurfaceSelection::One;
    gap.surfaces = {0};
    gap.thickness = 4.0;
    ConstraintSpec height;
    height.type = ConstraintType::MaxSemiDiameter;
    height.surface_selection = SurfaceSelection::List;
    height.surfaces = {1, 2};
    loaded.system.constraints = Constraints{{gap, height}};
    const MechanicalConstraints m = make_mechanical_constraints(loaded.system);
    ASSERT_EQ(m.size(), 3);
    EXPECT_EQ(m.items[0].surface, 0);
    EXPECT_EQ(m.items[0].next_surface, 3); // Blende, zwei Geisterplatten, dann Fläche 1
    EXPECT_EQ(m.items[1].surface, 3);
    EXPECT_EQ(m.items[2].surface, 4);
    const SystemBatchHost b = make_system_batch(loaded, 1);
    check_mechanical_constraints(m, b.topology());
    // 1,0 + 2,5 + 1,5 = 5,0 mm (Summe der Teilstücke), Grenze 4 → r = √1 · 1 mm
    EXPECT_EQ(mechanical_center_thickness(m.items[0], b, 0), 5.0);
    EXPECT_EQ(mechanical_residual(m.items[0], b, 0, nullptr), 1.0);
    // Prüfung gegen die Topologie: Geisterplatte, außerhalb, Gewicht
    MechanicalConstraints bad = m;
    bad.items[1].surface = 1; // Geisterplatte
    EXPECT_THROW(check_mechanical_constraints(bad, b.topology()), std::invalid_argument);
    bad = m;
    bad.items[2].surface = 5;
    EXPECT_THROW(check_mechanical_constraints(bad, b.topology()), std::invalid_argument);
    bad = m;
    bad.items[0].weight = -1.0;
    EXPECT_THROW(check_mechanical_constraints(bad, b.topology()), std::invalid_argument);
    bad = m;
    bad.items[0].next_surface = 0;
    EXPECT_THROW(check_mechanical_constraints(bad, b.topology()), std::invalid_argument);
    // Extrema gebraucht, aber nicht übergeben
    EXPECT_THROW((void)mechanical_value(m.items[1], b, 0, nullptr), std::invalid_argument);
}

// Residuen ohne Strahlbezug: aktiv √w·g, inaktiv exakt 0, an der Grenze 0.
TEST(MechanicalConstraintsResolve, GeometricResidualsActiveInactive) {
    const SystemDescription s = read_system_json(with_items(R"(
    { "type": "min_center_thickness", "surface": 1, "thickness": 4.5, "weight": 9.0 },
    { "type": "min_center_thickness", "surface": 1, "thickness": 4.0, "weight": 9.0 },
    { "type": "max_center_thickness", "surface": 1, "thickness": 3.0, "weight": 4.0 },
    { "type": "max_center_thickness", "surface": 1, "thickness": 5.0, "weight": 4.0 },
    { "type": "min_edge_thickness", "surface": 1, "thickness": 3.0, "weight": 16.0 },
    { "type": "min_edge_thickness", "surface": 1, "thickness": 2.0, "weight": 16.0, "clearance": 0.5 })"));
    const LoadedSystem loaded = prepare_system(s, std::filesystem::path(MKO_SOURCE_DIR));
    const SystemBatchHost b = make_system_batch(loaded, 1);
    const MechanicalConstraints m = make_mechanical_constraints(s);
    check_mechanical_constraints(m, b.topology());
    ASSERT_EQ(m.size(), 6);
    EXPECT_DOUBLE_EQ(mechanical_residual(m.items[0], b, 0, nullptr), 3.0 * 0.5);
    EXPECT_EQ(mechanical_residual(m.items[1], b, 0, nullptr), 0.0); // genau auf der Grenze
    EXPECT_DOUBLE_EQ(mechanical_residual(m.items[2], b, 0, nullptr), 2.0 * 1.0);
    EXPECT_EQ(mechanical_residual(m.items[3], b, 0, nullptr), 0.0);
    // Randdicke bei h = max(4, 4,5) = 4,5: 4 − sag(0,1; 4,5) + sag(−0,05; 4,5)
    const double e = 4.0 - sag(0.1, 4.5) + sag(-0.05, 4.5);
    EXPECT_EQ(mechanical_value(m.items[4], b, 0, nullptr), e);
    EXPECT_DOUBLE_EQ(mechanical_residual(m.items[4], b, 0, nullptr), 4.0 * (3.0 - e));
    const double e5 = 4.0 - sag(0.1, 5.0) + sag(-0.05, 5.0); // mit clearance 0,5
    EXPECT_EQ(mechanical_value(m.items[5], b, 0, nullptr), e5);
    EXPECT_EQ(mechanical_residual(m.items[5], b, 0, nullptr), 0.0);
    EXPECT_GT(e5, 2.0);
    // Zusatzresiduen in der Reihenfolge der Bedingungen
    const MechanicalResiduals r = make_mechanical_residuals(m);
    ASSERT_EQ(r.traced.count, 0);
    std::vector<double> out(6);
    r.geometric.evaluate(b, 0, out);
    for (int k = 0; k < 6; ++k) {
        EXPECT_EQ(out[static_cast<std::size_t>(k)],
                  mechanical_residual(m.items[static_cast<std::size_t>(k)], b, 0, nullptr));
    }
}

// ---- Trace-abhängige Residuen -----------------------------------------------------------------

// Unabhängige Auswertung: jeder Strahl einzeln mit eigenem Beobachter (Schnittpunkt, Richtung
// vor der Brechung), Höhe mit hypot und cos I mit explizit normierter Flächennormale; Maximum
// bzw. Minimum über die Strahlen mit Status Ok.
struct BruteObserver {
    double x[kMaxSurfaces];
    double y[kMaxSurfaces];
    double z[kMaxSurfaces];
    double c[kMaxSurfaces];
    Vec3<double> d[kMaxSurfaces];
    void surface(int s, double curv, const Vec3<double>& d_before, const Ray<double>& ray) {
        x[s] = ray.p.x;
        y[s] = ray.p.y;
        z[s] = ray.p.z;
        c[s] = curv;
        d[s] = d_before;
    }
};

struct BruteExtrema {
    std::vector<double> height; // [S]
    std::vector<double> cosine; // [S]
};

BruteExtrema brute_extrema(const SystemBatchHost& batch, int v, const ResidualLayout& layout) {
    const SystemBatchView<double> view = batch.view();
    const SurfaceTableView<double> table = view.surface_table(v);
    const int S = table.surfaces;
    BruteExtrema e{std::vector<double>(static_cast<std::size_t>(S), 0.0),
                   std::vector<double>(static_cast<std::size_t>(S), 1.0)};
    auto obs = std::make_unique<BruteObserver>();
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const Vec3<double> center = batch_image_point(view, table, v, f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            for (int k = 0; k < sample_count(q); ++k) {
                double opl = 0.0;
                if (batch_ray_opl(view, table, l, v, f, l, sample(q, k), center, &opl, obs.get()) !=
                    TraceStatus::Ok) {
                    continue;
                }
                for (int s = 0; s < S; ++s) {
                    const auto ss = static_cast<std::size_t>(s);
                    const double n0 = -obs->c[s] * obs->x[s];
                    const double n1 = -obs->c[s] * obs->y[s];
                    const double n2 = 1.0 - obs->c[s] * obs->z[s];
                    const double norm = std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);
                    const double cosine =
                        (obs->d[s].x * n0 + obs->d[s].y * n1 + obs->d[s].z * n2) / norm;
                    e.height[ss] = std::max(e.height[ss], std::hypot(obs->x[s], obs->y[s]));
                    e.cosine[ss] = std::min(e.cosine[ss], cosine);
                }
            }
        }
    }
    return e;
}

TEST(MechanicalTraceResiduals, MatchBruteForceOverAllRays) {
    const LoadedSystem loaded = load_residual_system("zweilinser_mechanik.json");
    const SystemBatchHost b = perturbed_batch(loaded, 12, 4711);
    const ResidualLayout layout(b.topology(), ResidualSettings{});
    const SurfaceExtremaResult ex = surface_extrema_cpu(b, layout);
    // Bedingungen aus der Datei, dazu Randdicke bei der freien Öffnung ohne clearance
    MechanicalConstraints m = make_mechanical_constraints(loaded.system);
    const MechanicalResiduals r = make_mechanical_residuals(m);
    ASSERT_EQ(r.traced.count, 1 + 2 + 2 + 4);
    int active = 0;
    for (int v = 0; v < b.variants(); ++v) {
        const BruteExtrema be = brute_extrema(b, v, layout);
        const VariantExtrema ve = variant_extrema(ex, v);
        for (int s = 0; s < b.surfaces(); ++s) {
            const auto ss = static_cast<std::size_t>(s);
            EXPECT_NEAR(ve.max_height(s), be.height[ss], 4 * kEps * be.height[ss]) << v << ' ' << s;
            EXPECT_NEAR(ve.min_cos_incidence(s), be.cosine[ss], 8 * kEps) << v << ' ' << s;
        }
        std::vector<double> out(static_cast<std::size_t>(r.traced.count));
        r.traced.evaluate(b, v, ve, out);
        for (std::size_t k = 0; k < out.size(); ++k) {
            const MechanicalConstraint& c = m.items[static_cast<std::size_t>(r.traced_items[k])];
            // Residuum aus den unabhängigen Extrema
            double value = 0.0;
            switch (c.type) {
            case ConstraintType::MaxSemiDiameter:
                value = be.height[static_cast<std::size_t>(c.surface)] + c.clearance;
                break;
            case ConstraintType::MaxIncidenceAngle:
                value = be.cosine[static_cast<std::size_t>(c.surface)];
                break;
            case ConstraintType::MinEdgeThickness: {
                const double h = std::max(be.height[static_cast<std::size_t>(c.surface)],
                                          be.height[static_cast<std::size_t>(c.next_surface)]) +
                                 c.clearance;
                value = b.thickness(v, c.surface) - sag(b.curvature(v, c.surface), h) +
                        sag(b.curvature(v, c.next_surface), h);
                break;
            }
            default:
                ADD_FAILURE() << "unerwarteter Typ";
            }
            const double g =
                c.type == ConstraintType::MaxSemiDiameter ? value - c.limit : c.limit - value;
            const double expected = std::sqrt(c.weight) * std::max(g, 0.0);
            EXPECT_NEAR(out[k], expected, 1e-12 * std::sqrt(c.weight) * (1.0 + std::abs(value)))
                << v << ' ' << to_string(c.type) << ' ' << c.file_surface;
            active += out[k] > 0.0 ? 1 : 0;
        }
    }
    EXPECT_GT(active, 10); // die Störungen machen viele Bedingungen aktiv
}

TEST(MechanicalTraceResiduals, EvaluateLayoutAndErrors) {
    const FdProblem p = mechanical_problem();
    SystemBatchHost b = p.base;
    b.resize(4);
    for (int v = 1; v < 4; ++v) {
        b.copy_variant_from(p.base, 0, v);
        b.thickness(v, 2) = 1.0 + v;
    }
    const ResidualLayout layout(b.topology(), ResidualSettings{});
    const SurfaceExtremaResult all = surface_extrema_cpu(b, layout);
    TraceExtraResiduals t;
    t.count = 2;
    t.evaluate = [](const SystemBatchHost& batch, int v, const VariantExtrema& e,
                    std::span<double> out) {
        out[0] = e.max_height(3);
        out[1] = batch.thickness(v, 2);
    };
    // Varianten 1 … 2 mit den Extrema aller Varianten (Eintrag v) bzw. nur ab Variante 1
    std::vector<double> out(4);
    evaluate_trace_extra_residuals(t, b, all, 0, out, 1, 2, 2);
    EXPECT_EQ(out[0], all.max_height(1, 3));
    EXPECT_EQ(out[1], 2.0);
    EXPECT_EQ(out[2], all.max_height(2, 3));
    EXPECT_EQ(out[3], 3.0);
    SurfaceExtremaResult part = surface_extrema_cpu(b, 1, layout);
    std::vector<double> one(2);
    evaluate_trace_extra_residuals(t, b, part, 1, one, 1, 1);
    EXPECT_EQ(one[0], all.max_height(1, 3));
    // Merit pro Variante
    const std::vector<double> merit = trace_extra_merit_values(t, b, all);
    ASSERT_EQ(merit.size(), 4u);
    for (int v = 0; v < 4; ++v) {
        const double h = all.max_height(v, 3);
        const double th = b.thickness(v, 2);
        EXPECT_EQ(merit[static_cast<std::size_t>(v)], h * h + th * th);
        EXPECT_EQ(trace_extra_merit(t, b, v, variant_extrema(all, v)), h * h + th * th);
    }
    // Fehler
    EXPECT_THROW(
        evaluate_trace_extra_residuals(t, b, all, 0, std::span<double>(out).first(3), 1, 2),
        std::invalid_argument);
    EXPECT_THROW(evaluate_trace_extra_residuals(t, b, part, 1, out, 1, 2), std::out_of_range);
    EXPECT_THROW(evaluate_trace_extra_residuals(t, b, all, 0, out, 3, 2), std::out_of_range);
    SurfaceExtremaResult wrong = all;
    wrong.surfaces = 3;
    EXPECT_THROW(evaluate_trace_extra_residuals(t, b, wrong, 0, out, 1, 2), std::invalid_argument);
    EXPECT_THROW((void)variant_extrema(all, 4), std::out_of_range);
    TraceExtraResiduals none;
    none.count = 1;
    EXPECT_THROW(check_trace_extra_residuals(none, "Test"), std::invalid_argument);
    // Zeilen zusammenfügen: erst a, dann b pro Variante
    const std::vector<double> a{1, 2, 3, 4};
    const std::vector<double> c{9, 8};
    EXPECT_EQ(join_extra_rows(a, 2, c, 1, 2), (std::vector<double>{1, 2, 9, 3, 4, 8}));
    EXPECT_THROW((void)join_extra_rows(a, 2, c, 2, 2), std::invalid_argument);
}

// ---- FD-Jacobi und Merit ----------------------------------------------------------------------

// Die Zeilen der trace-abhängigen Residuen in J: jede FD-Variante bekommt die Residuen aus
// ihren eigenen Extrema. Unabhängig nachgerechnet: Residuum an x⁺ und x⁻ jeder Spalte aus
// surface_extrema_cpu des gestörten Systems, Formel der Schablone. Dazu r₀, die Ablage
// (Strahl-, Zusatz-, Trace-Zeilen) und die analytische Ableitung einer Randdicke nach der Dicke.
TEST(MechanicalFdJacobian, TraceRowsMatchIndependentDifferences) {
    const FdProblem p = mechanical_problem();
    const SystemDescription d = with_mechanical_weight(p.loaded.system, 1.0);
    const LmOptions o = mechanical_options(d, 1);
    const MechanicalConstraints m = make_mechanical_constraints(d);
    const MechanicalResiduals mr = make_mechanical_residuals(m);
    const FdJacobian j = fd_jacobian_cpu(p.set, p.base, p.x0, o.jacobian);
    const ResidualLayout layout(p.base.topology(), o.jacobian.quadrature);
    const int P = p.set.size();
    const int E = o.jacobian.extra.count;
    const int Et = o.jacobian.trace_extra.count;
    ASSERT_EQ(Et, mr.traced.count);
    ASSERT_EQ(j.shape.extra_rows, E + Et);
    ASSERT_EQ(j.shape.ray_rows, layout.trace_rows());
    const std::int64_t rv = j.shape.ray_rows;

    // Residuen der trace-abhängigen Bedingungen an einem Punkt
    const auto traced_at = [&](std::span<const double> x) {
        SystemBatchHost b = p.base;
        p.set.apply(b, 0, x);
        const SurfaceExtremaResult e = surface_extrema_cpu(b, 0, layout);
        const VariantExtrema ve = variant_extrema(e, 0);
        std::vector<double> r;
        for (const int k : mr.traced_items) {
            r.push_back(mechanical_residual(m.items[static_cast<std::size_t>(k)], b, 0, &ve));
        }
        return r;
    };
    const std::vector<double> r0 = traced_at(p.x0);
    int nonzero = 0;
    for (int i = 0; i < Et; ++i) {
        EXPECT_EQ(j.residual(0)[static_cast<std::size_t>(rv + E + i)],
                  r0[static_cast<std::size_t>(i)]);
        nonzero += r0[static_cast<std::size_t>(i)] > 0.0 ? 1 : 0;
    }
    EXPECT_GE(nonzero, 3); // Strahlhöhe Fläche 1 und beide Glasränder am Start verletzt
    const FdPattern& pat = j.patterns[0];
    for (int k = 0; k < P; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        std::vector<double> xa = p.x0;
        std::vector<double> xb = p.x0;
        xa[kk] = pat.point_plus[kk];
        xb[kk] = pat.point_minus[kk];
        const std::vector<double> ra = traced_at(xa);
        const std::vector<double> rb = traced_at(xb);
        for (int i = 0; i < Et; ++i) {
            const auto ii = static_cast<std::size_t>(i);
            const double expected = fd_derivative(pat.stencil[kk], r0[ii], ra[ii], rb[ii],
                                                  pat.step_plus[kk], pat.step_minus[kk]);
            EXPECT_EQ(j.at(0, rv + E + i, k), expected) << "Zeile " << i << " Spalte " << k;
        }
    }
    // Mechanische Zusatzresiduen ohne Strahlbezug stehen am Ende der gewöhnlichen
    // Zusatzresiduen (hinter der Brennweite)
    for (std::size_t g = 0; g < mr.geometric_items.size(); ++g) {
        const MechanicalConstraint& c = m.items[static_cast<std::size_t>(mr.geometric_items[g])];
        const std::int64_t row =
            rv + (E - static_cast<int>(mr.geometric_items.size())) + static_cast<std::int64_t>(g);
        EXPECT_EQ(j.residual(0)[static_cast<std::size_t>(row)],
                  mechanical_residual(c, p.base, 0, nullptr));
    }
}

// Merit = Σ r² des Residuenvektors einschließlich der trace-abhängigen Residuen (CPU-Backend,
// make_cpu_merit, Summe über r₀ der Jacobi-Rechnung).
TEST(MechanicalLm, MeritIsSumOfSquares) {
    const FdProblem p = mechanical_problem();
    const LmOptions o = mechanical_options(with_mechanical_weight(p.loaded.system, 100.0), 1);
    const FdJacobian j = fd_jacobian_cpu(p.set, p.base, p.x0, o.jacobian);
    double sum = 0.0;
    for (const double r : j.residual(0)) {
        sum += r * r;
    }
    CpuLmBackend backend(p.set, p.base, o);
    std::vector<double> f(1);
    std::vector<std::int64_t> failed(1);
    std::vector<std::int64_t> hard(1);
    const int run = 0;
    backend.merit(std::span<const int>(&run, 1), p.x0, f, failed, hard);
    EXPECT_NEAR(f[0], sum, kMechanicalMeritRelative * sum);
    // Die trace-abhängigen Residuen tragen merklich bei (die Wellenfront des Starts dominiert)
    double traced = 0.0;
    const std::int64_t first = j.shape.ray_rows + o.jacobian.extra.count;
    for (std::int64_t i = first; i < j.shape.rows(); ++i) {
        traced +=
            j.residual(0)[static_cast<std::size_t>(i)] * j.residual(0)[static_cast<std::size_t>(i)];
    }
    EXPECT_GT(traced, 0.01 * sum);
    // make_cpu_merit (Diagnose) rechnet dasselbe in derselben Reihenfolge
    const BatchMeritFn merit =
        make_cpu_merit(o.jacobian.quadrature, o.jacobian.extra, 1, o.jacobian.trace_extra);
    SystemBatchHost b = p.base;
    p.set.apply(b, 0, p.x0);
    EXPECT_EQ(merit(b).at(0), f[0]);
}

} // namespace
} // namespace mko::test
