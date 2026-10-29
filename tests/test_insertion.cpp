// Einfüge-Operatoren A und B (M6-T2): zentrale Umnummerierung der Flächenbezüge, Aufbau der
// neuen Systeme, Merit-Invarianz (Abnahme |ΔF| ≤ 1e-12 relativ, CPU), Strafgewicht einer
// Geisterplatte, Hilfsfunktionen der Initialisierung (docs/optimierung.md, Abschnitt 6).

#include "insertion_test_systems.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mko::test {
namespace {

// Resolved-Variablen von `after` entsprechen denen von `before` mit über die Abbildung
// verschobenen Flächen (gleiche Reihenfolge, Schranken, typische Größen, Werte).
void expect_same_variables(const SystemDescription& before, const SystemDescription& after,
                           const std::vector<int>& new_of_old, std::size_t extra_after = 0) {
    const std::vector<ResolvedVariable> a = resolve_variables(before);
    const std::vector<ResolvedVariable> b = resolve_variables(after);
    ASSERT_GE(b.size(), a.size());
    std::size_t j = 0;
    for (const ResolvedVariable& va : a) {
        // eingefügte Teilstücke (gleicher Eintrag) überspringen
        while (j < b.size() &&
               !(b[j].type == va.type &&
                 (va.type == VariableType::WorkingDistance ||
                  va.type == VariableType::GhostCoefficient ||
                  b[j].surface == new_of_old[static_cast<std::size_t>(va.surface)]) &&
                 b[j].plate == va.plate && b[j].order == va.order &&
                 b[j].component == va.component)) {
            ++j;
        }
        ASSERT_LT(j, b.size()) << "Variable fehlt: " << to_string(va.type) << " " << va.surface;
        const ResolvedVariable& vb = b[j];
        EXPECT_EQ(vb.lower, va.lower);
        EXPECT_EQ(vb.upper, va.upper);
        EXPECT_EQ(vb.typical, va.typical);
        if (va.type != VariableType::Glass && va.type != VariableType::Thickness &&
            va.type != VariableType::GhostCoefficient) {
            EXPECT_EQ(variable_value(after, vb), variable_value(before, va));
        }
        ++j;
    }
    EXPECT_EQ(b.size(), a.size() + extra_after);
}

// ---------------------------------------------------------------------------------------------
// Umnummerierung

TEST(RenumberSurfaces, MapsAllReferences) {
    SystemDescription d = load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    // weitere Flächenbezüge: Dicken als Liste, Glas, Abgleichlänge mit Referenzfläche
    d.variables->items.push_back(
        variable_spec(VariableType::Thickness, SurfaceSelection::List, {0, 2}));
    d.variables->items.push_back(variable_spec(VariableType::Glass, SurfaceSelection::One, {1}));
    d.target->parfocal_length = 120.0;
    ConstraintSpec parfocal;
    parfocal.type = ConstraintType::ParfocalLength;
    parfocal.reference_surface = 1;
    parfocal.weight = 0.01;
    d.constraints->items.push_back(parfocal);
    validate_system(d);

    // Zwei Leerflächen hinter Fläche 1 (Abbildung 0→0, 1→1, 2→4, 3→5)
    std::vector<Surface> surfaces = d.surfaces;
    Surface dummy;
    dummy.type = SurfaceType::Plane;
    dummy.material = d.surfaces[1].material;
    dummy.semi_diameter = 12.5;
    surfaces.insert(surfaces.begin() + 2, 2, dummy);
    surfaces[1].thickness = 2.0;
    surfaces[2].thickness = 2.0;
    surfaces[3].thickness = 2.0;
    const std::vector<int> map{0, 1, 4, 5};
    const SystemDescription r = renumber_surfaces(d, map, surfaces);
    EXPECT_EQ(r.surfaces, surfaces);
    // "all" der Krümmungen → explizite Liste der alten Sphären (1, 2) → (1, 4)
    const VariableSpec& curv = r.variables->items[0];
    EXPECT_EQ(curv.selection, SurfaceSelection::List);
    EXPECT_EQ(curv.surfaces, (std::vector<int>{1, 4}));
    EXPECT_EQ(r.variables->items[1], d.variables->items[1]); // Arbeitsabstand
    EXPECT_EQ(r.variables->items[2], d.variables->items[2]); // Geisterkoeffizienten
    EXPECT_EQ(r.variables->items[3].surfaces, (std::vector<int>{0, 4}));
    EXPECT_EQ(r.variables->items[4].surfaces, (std::vector<int>{1}));
    EXPECT_EQ(r.variables->items[4].selection, SurfaceSelection::One);
    // Geisterplatten: Spalt 0 bleibt 0, Spalt 2 wird 4; Abstände bleiben
    ASSERT_EQ(r.ghost_plates.size(), d.ghost_plates.size());
    for (std::size_t g = 0; g < d.ghost_plates.size(); ++g) {
        EXPECT_EQ(r.ghost_plates[g].after_surface,
                  map[static_cast<std::size_t>(d.ghost_plates[g].after_surface)]);
        EXPECT_EQ(r.ghost_plates[g].distance, d.ghost_plates[g].distance);
        EXPECT_EQ(r.ghost_plates[g].coefficients, d.ghost_plates[g].coefficients);
    }
    EXPECT_EQ(r.constraints->items.back().reference_surface, 1);
    validate_system(r);
    // Variablenmenge: gleiche Variablen, nur Flächen verschoben; keine neuen Variablen, obwohl
    // die Leerflächen Glas haben und "all" sie sonst nicht beträfe
    expect_same_variables(d, r, map);
}

TEST(RenumberSurfaces, ReferenceSurfaceMoves) {
    SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    d.target->parfocal_length = 110.0;
    ConstraintSpec parfocal;
    parfocal.type = ConstraintType::ParfocalLength;
    parfocal.reference_surface = 2;
    d.constraints->items.push_back(parfocal);
    std::vector<Surface> s = d.surfaces;
    Surface plane;
    plane.semi_diameter = 1.0;
    s.insert(s.begin() + 1, plane);
    const std::vector<int> map{0, 2, 3};
    const SystemDescription r = renumber_surfaces(d, map, s);
    EXPECT_EQ(r.constraints->items.back().reference_surface, 3);
    EXPECT_EQ(r.variables->items[0].surfaces, (std::vector<int>{2, 3}));
}

// Mechanische Bedingungen (M4-T3) im Zweilinser: max_semi_diameter auf 1 und [3, 4],
// max_center_thickness und min_edge_thickness (Luft) mit Standardauswahl = Luftspalte 0, 2,
// min_edge_thickness (Glas) = Glasspalte 1, 3, max_incidence_angle = Flächen 1 … 4.
TEST(RenumberSurfaces, MechanicalConstraints) {
    const SystemDescription d = load_residual_system("zweilinser_mechanik.json").system;
    const auto surfaces_of = [](const SystemDescription& s) {
        std::vector<std::vector<int>> out;
        for (const ConstraintSpec& c : s.constraints->items) {
            out.push_back(constraint_surfaces(s, c));
        }
        return out;
    };
    const std::vector<std::vector<int>> before = surfaces_of(d);
    ASSERT_EQ(before[1], (std::vector<int>{1}));
    ASSERT_EQ(before[2], (std::vector<int>{3, 4}));
    ASSERT_EQ(before[3], (std::vector<int>{0, 2}));
    ASSERT_EQ(before[4], (std::vector<int>{1, 3}));
    ASSERT_EQ(before[6], (std::vector<int>{1, 2, 3, 4}));

    // Operator A zwischen den Linsen (Luftspalt 2): Abbildung 0, 1, 2, 5, 6; der geteilte
    // Luftspalt gilt für beide Teilstücke (2 und 4), die Meniskusflächen 3, 4 bekommen keine
    // Bedingungen (Standardauswahl → explizite Liste)
    {
        const InsertionResult r = insert_meniscus(d, meniscus(2, 0.5, -0.04, flint_glass()));
        ASSERT_EQ(r.new_of_old, (std::vector<int>{0, 1, 2, 5, 6}));
        const std::vector<std::vector<int>> after = surfaces_of(r.system);
        EXPECT_EQ(after[0], (std::vector<int>{}));        // focal_length
        EXPECT_EQ(after[1], (std::vector<int>{1}));       // vor der Einfügung
        EXPECT_EQ(after[2], (std::vector<int>{5, 6}));    // dahinter
        EXPECT_EQ(after[3], (std::vector<int>{0, 2, 4})); // Luft: geteilter Spalt 2 → 2 und 4
        EXPECT_EQ(after[4], (std::vector<int>{1, 5}));    // Glas
        EXPECT_EQ(after[5], (std::vector<int>{0, 2, 4}));
        EXPECT_EQ(after[6], (std::vector<int>{1, 2, 5, 6}));
        for (const ConstraintSpec& c : r.system.constraints->items) {
            if (is_mechanical(c.type)) {
                EXPECT_NE(c.surface_selection, SurfaceSelection::None);
                EXPECT_NE(c.surface_selection, SurfaceSelection::All);
            }
        }
        // Tabellenflächen der Bedingungen passen zum neuen System (Medium, Spalte)
        const MechanicalConstraints m = make_mechanical_constraints(r.system);
        EXPECT_EQ(m.size(), make_mechanical_constraints(d).size() + 2);
        const LoadedSystem l = prepare_test_system(r.system);
        EXPECT_NO_THROW(check_mechanical_constraints(m, make_system_batch(l, 1).topology()));
    }
    // Operator A vor allen referenzierten Flächen (Luftspalt 0): alles rückt um 2
    {
        const InsertionResult r = insert_meniscus(d, meniscus(0, 0.3, 0.02, constant_glass(1.6)));
        const std::vector<std::vector<int>> after = surfaces_of(r.system);
        EXPECT_EQ(after[1], (std::vector<int>{3}));
        EXPECT_EQ(after[2], (std::vector<int>{5, 6}));
        EXPECT_EQ(after[3], (std::vector<int>{0, 2, 4}));
        EXPECT_EQ(after[4], (std::vector<int>{3, 5}));
        EXPECT_EQ(after[6], (std::vector<int>{3, 4, 5, 6}));
    }
    // Operator B an der zweiten Linse (Glasspalt 3): Glasbedingung gilt für beide Teile
    {
        const InsertionResult r = split_lens(d, lens_split(3, 0.03, 0.5));
        ASSERT_EQ(r.new_of_old, (std::vector<int>{0, 1, 2, 3, 5}));
        const std::vector<std::vector<int>> after = surfaces_of(r.system);
        EXPECT_EQ(after[2], (std::vector<int>{3, 5}));
        EXPECT_EQ(after[3], (std::vector<int>{0, 2}));
        EXPECT_EQ(after[4], (std::vector<int>{1, 3, 4}));
        EXPECT_EQ(after[6], (std::vector<int>{1, 2, 3, 5}));
    }
}

TEST(RenumberSurfaces, RejectsInvalidMaps) {
    const SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    std::vector<Surface> s = d.surfaces;
    s.push_back(d.surfaces[2]);
    EXPECT_THROW(renumber_surfaces(d, std::vector<int>{0, 1}, s), std::invalid_argument);
    EXPECT_THROW(renumber_surfaces(d, std::vector<int>{1, 2, 3}, s), std::invalid_argument);
    EXPECT_THROW(renumber_surfaces(d, std::vector<int>{0, 1, 2}, s), std::invalid_argument);
    EXPECT_THROW(renumber_surfaces(d, std::vector<int>{0, 2, 2}, s), std::invalid_argument);
    EXPECT_NO_THROW(renumber_surfaces(d, std::vector<int>{0, 2, 3}, s));
}

// ---------------------------------------------------------------------------------------------
// Operator A

TEST(InsertMeniscus, BuildsSurfacesAndVariables) {
    SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    d.variables->items.push_back(
        variable_spec(VariableType::Thickness, SurfaceSelection::One, {0}));
    MeniscusInsertion m = meniscus(0, 1.5, 0.04, flint_glass());
    m.curvature_variables.min = -0.5;
    m.curvature_variables.max = 0.5;
    const InsertionResult r = insert_meniscus(d, m);
    EXPECT_EQ(r.first_new_surface, 1);
    EXPECT_EQ(r.new_surfaces, 2);
    EXPECT_EQ(r.new_of_old, (std::vector<int>{0, 3, 4}));
    const std::vector<Surface>& s = r.system.surfaces;
    ASSERT_EQ(s.size(), 5u);
    EXPECT_EQ(s[0].thickness, 1.5);
    EXPECT_EQ(s[1].type, SurfaceType::Sphere);
    EXPECT_EQ(s[1].curvature, 0.04);
    EXPECT_EQ(s[1].thickness, 0.0);
    EXPECT_EQ(s[1].material.kind, MaterialKind::Buchdahl);
    EXPECT_EQ(s[2].curvature, 0.04);
    EXPECT_EQ(s[2].material.kind, MaterialKind::Air);
    EXPECT_EQ(s[2].thickness, 5.0 - 1.5);
    EXPECT_EQ(s[1].semi_diameter, 12.5); // größerer Halbdurchmesser der Spaltflächen
    EXPECT_EQ(s[3], d.surfaces[1]);
    EXPECT_EQ(s[4], d.surfaces[2]);
    // Krümmungen (1, 2) → (3, 4); Dicke des Spalts gilt für beide Teilstücke; neuer Eintrag
    const std::vector<VariableSpec>& items = r.system.variables->items;
    ASSERT_EQ(items.size(), 4u);
    EXPECT_EQ(items[0].surfaces, (std::vector<int>{3, 4}));
    EXPECT_EQ(items[2].surfaces, (std::vector<int>{0, 2}));
    EXPECT_EQ(items[2].selection, SurfaceSelection::List);
    EXPECT_EQ(items[3].type, VariableType::Curvature);
    EXPECT_EQ(items[3].surfaces, (std::vector<int>{1, 2}));
    EXPECT_EQ(items[3].min, -0.5);
    EXPECT_EQ(items[3].max, 0.5);
    expect_same_variables(d, r.system, r.new_of_old, 3);

    // Ohne neue Krümmungsvariablen, mit Dicke
    m.curvature_variables.enabled = false;
    m.thickness = 0.5;
    const InsertionResult t = insert_meniscus(d, m);
    EXPECT_EQ(t.system.surfaces[1].thickness, 0.5);
    EXPECT_EQ(t.system.surfaces[2].thickness, 5.0 - 1.5 - 0.5);
    EXPECT_EQ(t.system.variables->items.size(), 3u);
}

TEST(InsertMeniscus, MovesGhostPlatesOfTheSplitGap) {
    const SystemDescription d =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    // Platten 0, 1 bei 1,0, Platte 2 bei 4,0 nach Fläche 0; Platte 3 bei 10 nach Fläche 2
    {
        const InsertionResult r = insert_meniscus(d, meniscus(0, 2.5, -0.1, flint_glass()));
        const std::vector<GhostPlate>& g = r.system.ghost_plates;
        EXPECT_EQ(g[0].after_surface, 0);
        EXPECT_EQ(g[0].distance, 1.0);
        EXPECT_EQ(g[1].after_surface, 0);
        EXPECT_EQ(g[2].after_surface, 2);
        EXPECT_EQ(g[2].distance, 4.0 - 2.5);
        EXPECT_EQ(g[3].after_surface, 4);
        EXPECT_EQ(g[3].distance, 10.0);
        EXPECT_EQ(g[2].coefficients, d.ghost_plates[2].coefficients);
    }
    {
        // An der Plattenlage: Platten mit distance = position bleiben vor dem Meniskus
        const InsertionResult r = insert_meniscus(d, meniscus(0, 1.0, 0.2, flint_glass()));
        EXPECT_EQ(r.system.ghost_plates[0].after_surface, 0);
        EXPECT_EQ(r.system.ghost_plates[0].distance, 1.0);
        EXPECT_EQ(r.system.ghost_plates[2].after_surface, 2);
        EXPECT_EQ(r.system.ghost_plates[2].distance, 3.0);
    }
    {
        // Mit Dicke: Platte im Glas ist ein Fehler, dahinter verschoben um position + Dicke
        MeniscusInsertion m = meniscus(0, 3.5, 0.0, flint_glass());
        m.thickness = 1.0;
        EXPECT_THROW(insert_meniscus(d, m), std::invalid_argument);
        m.position = 2.0;
        const InsertionResult r = insert_meniscus(d, m);
        EXPECT_EQ(r.system.ghost_plates[2].after_surface, 2);
        EXPECT_EQ(r.system.ghost_plates[2].distance, 1.0);
    }
    // Geisterkoeffizienten behalten ihre Indizes, Strafe gleich
    const InsertionResult r = insert_meniscus(d, meniscus(2, 15.0, -0.05, flint_glass()));
    EXPECT_EQ(make_ghost_penalty(r.system).terms, make_ghost_penalty(d).terms);
    expect_same_variables(d, r.system, r.new_of_old, 2);
}

TEST(InsertMeniscus, RejectsInvalidInsertions) {
    const SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    EXPECT_THROW(insert_meniscus(d, meniscus(1, 1.0, 0.0, flint_glass())),
                 std::invalid_argument); // Glas nach Fläche 1
    EXPECT_THROW(insert_meniscus(d, meniscus(-1, 1.0, 0.0, flint_glass())), std::invalid_argument);
    EXPECT_THROW(insert_meniscus(d, meniscus(3, 1.0, 0.0, flint_glass())), std::invalid_argument);
    EXPECT_THROW(insert_meniscus(d, meniscus(0, 5.5, 0.0, flint_glass())),
                 std::invalid_argument); // hinter dem Spalt
    EXPECT_THROW(insert_meniscus(d, meniscus(0, -0.1, 0.0, flint_glass())), std::invalid_argument);
    EXPECT_THROW(insert_meniscus(d, meniscus(0, 1.0, 0.0, MaterialSpec{})),
                 std::invalid_argument); // Luft
    EXPECT_THROW(insert_meniscus(
                     d, meniscus(0, 1.0, std::numeric_limits<double>::quiet_NaN(), flint_glass())),
                 std::invalid_argument);
    // Dickenvariable des Spalts mit Untergrenze 2: ein Teilstück wäre kürzer
    SystemDescription b = d;
    auto t = variable_spec(VariableType::Thickness, SurfaceSelection::One, {0});
    t.min = 2.0;
    b.variables->items.push_back(t);
    EXPECT_THROW(insert_meniscus(b, meniscus(0, 1.0, 0.0, flint_glass())), FormatError);
    EXPECT_NO_THROW(insert_meniscus(b, meniscus(0, 2.5, 0.0, flint_glass())));
}

TEST(InsertMeniscus, SaddleStarts) {
    const SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    const InsertionResult r = insert_meniscus(d, meniscus(0, 2.0, 0.05, flint_glass()));
    const auto starts = meniscus_saddle_starts(r.system, r.first_new_surface, 1e-3, -1e-3);
    EXPECT_EQ(starts[0].surfaces[1].curvature, 0.05 + 1e-3);
    EXPECT_EQ(starts[0].surfaces[2].curvature, 0.05 - 1e-3);
    EXPECT_EQ(starts[1].surfaces[1].curvature, 0.05 - 1e-3);
    EXPECT_EQ(starts[1].surfaces[2].curvature, 0.05 + 1e-3);
    EXPECT_EQ(starts[0].surfaces[3], r.system.surfaces[3]);
    EXPECT_THROW(meniscus_saddle_starts(r.system, 0, 1e-3, -1e-3), std::invalid_argument);
    // Beide Starts sind Abstiegsrichtungen weg vom Sattel nur mit Dicke > 0; mit Dicke 0 hat der
    // Meniskus die dünne Brechkraft ±2 (n − 1) δ, die Merit ändert sich in beiden Richtungen
    EXPECT_NE(system_merit(starts[0]), system_merit(r.system));
    EXPECT_NE(system_merit(starts[1]), system_merit(r.system));
}

// ---------------------------------------------------------------------------------------------
// Operator B

TEST(SplitLens, BuildsSurfacesAndVariables) {
    SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    d.variables->items.push_back(
        variable_spec(VariableType::Thickness, SurfaceSelection::List, {0, 1}));
    d.variables->items.push_back(variable_spec(VariableType::Glass, SurfaceSelection::All));
    const InsertionResult r = split_lens(d, lens_split(1, -0.03, 0.25));
    EXPECT_EQ(r.first_new_surface, 2);
    EXPECT_EQ(r.new_surfaces, 1);
    EXPECT_EQ(r.new_of_old, (std::vector<int>{0, 1, 3}));
    const std::vector<Surface>& s = r.system.surfaces;
    ASSERT_EQ(s.size(), 4u);
    EXPECT_EQ(s[1].thickness, 1.5);
    EXPECT_EQ(s[2].curvature, -0.03);
    EXPECT_EQ(s[2].thickness, 4.5);
    EXPECT_EQ(s[2].material.kind, MaterialKind::Catalog);
    EXPECT_EQ(s[2].material.name, "N-BK7");
    EXPECT_EQ(s[2].semi_diameter, 12.5);
    EXPECT_EQ(s[3], d.surfaces[2]);
    const std::vector<VariableSpec>& items = r.system.variables->items;
    ASSERT_EQ(items.size(), 5u);
    EXPECT_EQ(items[0].surfaces, (std::vector<int>{1, 3}));
    EXPECT_EQ(items[2].surfaces, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(items[3].surfaces, (std::vector<int>{1, 2})); // Glas für beide Teile
    EXPECT_EQ(items[4].type, VariableType::Curvature);
    EXPECT_EQ(items[4].surfaces, (std::vector<int>{2}));
    expect_same_variables(d, r.system, r.new_of_old, 1 + 3 + 1);
    EXPECT_THROW(split_lens(d, lens_split(0, 0.0, 0.5)), std::invalid_argument);
    EXPECT_THROW(split_lens(d, lens_split(2, 0.0, 0.5)), std::invalid_argument);
    EXPECT_THROW(split_lens(d, lens_split(1, 0.0, 1.5)), std::invalid_argument);
}

TEST(SplitLens, FreeGlass) {
    const SystemDescription d = load_residual_system("einzellinse_bk7_biegung.json").system;
    SystemDescription r = split_lens(d, lens_split(1, -0.03, 0.5)).system;
    free_glass(r, 2);
    ASSERT_EQ(r.variables->items.back().type, VariableType::Glass);
    EXPECT_EQ(r.variables->items.back().surfaces, (std::vector<int>{2}));
    const std::size_t n = r.variables->items.size();
    free_glass(r, 2); // schon frei
    EXPECT_EQ(r.variables->items.size(), n);
    free_glass(r, 1, catalog_glass("CAF2"));
    EXPECT_EQ(r.surfaces[1].material.name, "CAF2");
    EXPECT_THROW(free_glass(r, 3), std::invalid_argument); // Luft
    validate_system(r);
}

// ---------------------------------------------------------------------------------------------
// Merit-Invarianz (Abnahme)

TEST(InsertionMerit, UnchangedOnCpu) {
    double worst = 0.0;
    int within_plan = 0;
    const std::vector<InsertionCase> cases = insertion_cases();
    for (const InsertionCase& c : cases) {
        const double before = system_merit(c.before);
        const double after = system_merit(c.after);
        const double rel = relative_change(before, after);
        worst = std::max(worst, rel);
        within_plan += rel <= kInsertionMeritPlan ? 1 : 0;
        std::printf("[CPU] %-68s F = %.15e  |dF|/F = %.2e\n", c.name.c_str(), before, rel);
        EXPECT_LE(rel, kInsertionMeritRelative) << c.name;
        // Paraxial: Brennweite gleich bis auf Rundung
        const LoadedSystem lb = prepare_test_system(c.before);
        const LoadedSystem la = prepare_test_system(c.after);
        const double fb = compute_paraxial(lb.system, lb.catalog).focal_length;
        const double fa = compute_paraxial(la.system, la.catalog).focal_length;
        EXPECT_LE(std::abs(fa - fb), kParaxialRelative * std::abs(fb)) << c.name;
        // Variablen der neuen Datei bauen (Batch und Topologie passen)
        const SystemBatchHost batch = make_system_batch(la, 1);
        EXPECT_NO_THROW((void)VariableSet(la.system, batch.topology())) << c.name;
    }
    std::printf("[CPU] größte relative Änderung %.2e (Toleranz %.0e); %d von %zu Fällen <= %.0e\n",
                worst, kInsertionMeritRelative, within_plan, cases.size(), kInsertionMeritPlan);
}

TEST(InsertionMerit, ChangesWithThickness) {
    // Gegenprobe: derselbe Meniskus mit Dicke > 0 ändert die Merit deutlich
    const SystemDescription d = lens_with_aux_plane();
    MeniscusInsertion m = meniscus(2, 15.0, 0.3, flint_glass());
    m.thickness = 0.5;
    EXPECT_GT(relative_change(system_merit(d), system_merit(insert_meniscus(d, m).system)), 1e-3);
}

// ---------------------------------------------------------------------------------------------
// Strafgewicht und Initialisierung

TEST(GhostPenaltyScale, SplitsSharedEntry) {
    const SystemDescription d =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    const GhostPenalty before = make_ghost_penalty(d);
    SystemDescription s = d;
    scale_ghost_penalty(s, 2, 10.0);
    validate_system(s);
    EXPECT_EQ(ghost_penalty_weight(s, 2), 136.0);
    EXPECT_EQ(ghost_penalty_weight(s, 1), 13.6);
    const GhostPenalty after = make_ghost_penalty(s);
    ASSERT_EQ(after.terms.size(), before.terms.size());
    // Terme gleicher Platte und Ordnung vergleichen (die Reihenfolge der Terme folgt den
    // Einträgen)
    for (const GhostPenaltyTerm& b : before.terms) {
        bool found = false;
        for (const GhostPenaltyTerm& a : after.terms) {
            if (a.plate == b.plate && a.order == b.order) {
                found = true;
                EXPECT_DOUBLE_EQ(a.weight, b.plate == 2 ? 10.0 * b.weight : b.weight);
                EXPECT_EQ(a.smoothing, b.smoothing);
            }
        }
        EXPECT_TRUE(found);
    }
    // Zweites Erhöhen: eigener Eintrag wird nur skaliert
    const std::size_t entries = s.constraints->items.size();
    scale_ghost_penalty(s, 2, 2.0);
    EXPECT_EQ(s.constraints->items.size(), entries);
    EXPECT_EQ(ghost_penalty_weight(s, 2), 272.0);
    EXPECT_THROW(scale_ghost_penalty(s, 7, 2.0), std::invalid_argument);
    EXPECT_THROW(scale_ghost_penalty(s, 1, 0.0), std::invalid_argument);
    SystemDescription none = d;
    none.constraints->items.pop_back();
    EXPECT_THROW(scale_ghost_penalty(none, 0, 2.0), std::invalid_argument);
    EXPECT_FALSE(ghost_penalty_weight(none, 0).has_value());
}

TEST(InsertionInit, GhostProfileT4) {
    // Koeffizienten von u² in P_k(2u − 1): 0, 6, −30, 90, −210 (explizite Polynome)
    const double q[] = {0.0, 6.0, -30.0, 90.0, -210.0};
    for (int k = 1; k <= 5; ++k) {
        GhostPlate g;
        g.norm_radius = 2.0;
        g.coefficients.assign(static_cast<std::size_t>(k), 0.0);
        g.coefficients.back() = 1e-3;
        EXPECT_DOUBLE_EQ(ghost_profile_t4(g), 1e-3 * q[k - 1] / 16.0) << k;
    }
}

TEST(InsertionInit, AchromatCurvatureAndAbbe) {
    const double c1 = 0.0193;
    const double c2 = -0.0193;
    const double n = 1.5168;
    const double nu1 = 64.17;
    const double nu2 = 26.9;
    const double cs = achromat_split_curvature(c1, c2, nu1, nu2);
    const double phi1 = (n - 1.0) * (c1 - cs);
    const double phi2 = (n - 1.0) * (cs - c2);
    EXPECT_NEAR(phi1 + phi2, (n - 1.0) * (c1 - c2), 1e-15);
    EXPECT_NEAR(phi1 / nu1 + phi2 / nu2, 0.0, 1e-15);
    EXPECT_THROW(achromat_split_curvature(c1, c2, nu1, nu1), std::invalid_argument);
    const LoadedSystem l = load_residual_system("einzellinse_bk7_biegung.json");
    EXPECT_NEAR(abbe_number(catalog_glass("N-BK7"), l.catalog), 64.17, 0.01);
    EXPECT_TRUE(std::isinf(abbe_number(constant_glass(1.7), l.catalog)));
    EXPECT_NEAR(abbe_number(flint_glass(), l.catalog), 26.9, 0.1);
}

// Bericht von Hand: Platten an zwei Lagen, Lage 1 aktiv.
GhostDiagnosis hand_diagnosis(const SystemDescription& d, GhostOperator op) {
    GhostDiagnosis g;
    for (std::size_t p = 0; p < d.ghost_plates.size(); ++p) {
        GhostPlateReport r;
        r.plate = static_cast<int>(p);
        r.after_surface = d.ghost_plates[p].after_surface;
        r.distance = d.ghost_plates[p].distance;
        r.glass.n_ref = 1.5;
        g.plates.push_back(r);
    }
    GhostPositionReport pos;
    pos.after_surface = d.ghost_plates[6].after_surface;
    pos.distance = d.ghost_plates[6].distance;
    pos.plates = {6, 7, 8};
    pos.active_plates = {6, 8};
    pos.recommendation = op;
    g.positions.push_back(pos);
    g.recommended_position = 0;
    g.recommendation = op;
    return g;
}

TEST(InsertionInit, MeniscusFromDiagnosis) {
    SystemDescription d = load_residual_system("einzellinse_na05_geisterplatten.json").system;
    d.ghost_plates[6].coefficients = {0.0, 1e-4, 0.0, 0.0, 0.0, 0.0};
    d.ghost_plates[8].coefficients = {0.0, 7e-3, -1e-4, 0.0, 0.0, 0.0};
    GhostDiagnosis g = hand_diagnosis(d, GhostOperator::A);
    g.plates[6].sum_abs = 1e-4;
    g.plates[8].sum_abs = 7.1e-3;
    g.plates[8].glass.n_ref = 1.75;
    g.plates[8].glass.name = "Flint";
    const MeniscusPlan plan = meniscus_from_diagnosis(d, g);
    EXPECT_EQ(plan.plate, 8);
    EXPECT_EQ(plan.position, 0);
    EXPECT_EQ(plan.insertion.after_surface, 2);
    EXPECT_EQ(plan.insertion.position, 0.2);
    EXPECT_EQ(plan.insertion.glass.kind, MaterialKind::Buchdahl);
    EXPECT_EQ(plan.insertion.glass.n0, 1.76);
    EXPECT_DOUBLE_EQ(plan.t4, (6.0 * 7e-3 + 30.0 * 1e-4) / std::pow(0.85, 4));
    // Betrag: nähere Spaltfläche (0,2 ≤ 0,4/2 → Fläche 2, c = −0,145); Vorzeichen von t₄ > 0
    EXPECT_EQ(plan.insertion.curvature, 0.145);
    std::printf("%s\n", plan.reason.c_str());
    // Negatives t₄ → negatives c₀; Betrag vorgegeben
    d.ghost_plates[8].coefficients[1] = -7e-3;
    MeniscusInitOptions o;
    o.curvature_magnitude = 0.5;
    const MeniscusPlan neg = meniscus_from_diagnosis(d, g, o);
    EXPECT_EQ(neg.insertion.curvature, -0.5);
    // Der Plan lässt sich einfügen
    EXPECT_NO_THROW(insert_meniscus(d, neg.insertion));
    // Keine aktive Platte / Bericht passt nicht
    GhostDiagnosis inactive = g;
    inactive.positions[0].active_plates.clear();
    EXPECT_THROW(meniscus_from_diagnosis(d, inactive), std::invalid_argument);
    GhostDiagnosis wrong = g;
    wrong.positions[0].distance = 0.3;
    EXPECT_THROW(meniscus_from_diagnosis(d, wrong), std::invalid_argument);
}

TEST(InsertionInit, SplitFromDiagnosis) {
    const LoadedSystem l = load_residual_system("einzellinse_na05_geisterplatten.json");
    const SystemDescription& d = l.system;
    GhostDiagnosis g = hand_diagnosis(d, GhostOperator::B);
    // Lage hinter der Linse (nach Fläche 2): Linse 1/2 davor; aktiv CaF2 (6) und Flint (8)
    const SplitPlan plan = split_from_diagnosis(d, l.catalog, g);
    std::printf("%s\n", plan.reason.c_str());
    EXPECT_EQ(plan.split.lens_surface, 1);
    EXPECT_EQ(plan.free_surface, 2);
    EXPECT_EQ(plan.plate, 8); // Flint: größte Abweichung von 1/ν der Linse (N-BK7)
    EXPECT_NEAR(plan.abbe_lens, 64.17, 0.01);
    EXPECT_NEAR(plan.abbe_target, 26.9, 0.1);
    EXPECT_DOUBLE_EQ(plan.achromat_curvature,
                     achromat_split_curvature(0.82, -0.145, plan.abbe_lens, plan.abbe_target));
    // Die schnelle Linse (c₁ h ≈ 1) verträgt das dünne Achromat nicht: c_s wird zum Mittel hin
    // zurückgenommen, bis beide Teile Randdicke ≥ 0 haben
    const double mean_cs = 0.5 * (0.82 - 0.145);
    EXPECT_LE(plan.split.curvature, mean_cs);
    EXPECT_GT(plan.split.curvature, plan.achromat_curvature);
    EXPECT_NE(plan.reason.find("zurückgenommen"), std::string::npos);
    // anomal: größtes ΔP_g,F
    GhostDiagnosis an = hand_diagnosis(d, GhostOperator::BAnomalous);
    an.plates[6].glass.delta_partial = 0.055;
    an.plates[8].glass.delta_partial = 0.01;
    const SplitPlan anomalous = split_from_diagnosis(d, l.catalog, an);
    EXPECT_EQ(anomalous.plate, 6);
    EXPECT_EQ(anomalous.target_glass.name, "CAF2");
    // Mittel der Außenkrümmungen, vorderer Teil frei
    SplitInitOptions o;
    o.achromat_curvature = false;
    o.free_part = 0;
    const SplitPlan mean = split_from_diagnosis(d, l.catalog, g, o);
    EXPECT_DOUBLE_EQ(mean.split.curvature, 0.5 * (0.82 - 0.145));
    EXPECT_EQ(mean.free_surface, 1);
    // Ausführen: Merit unverändert, Glas freigeben
    InsertionResult r = split_lens(d, plan.split);
    free_glass(r.system, plan.free_surface);
    validate_system(r.system);
    EXPECT_LE(relative_change(system_merit(d), system_merit(split_lens(d, plan.split).system)),
              kInsertionMeritRelative);
}

// Glasvorauswahl in Operator B (M11-T5): Zielglas aus dem Katalog statt der Platte, c_s aus den
// Anteilen des dünnen Achromats in η; ohne Katalogglas der Linse wie bisher.
TEST(InsertionInit, SplitFromDiagnosisWithPreselection) {
    const LoadedSystem l = load_residual_system("einzellinse_na05_geisterplatten.json");
    const SystemDescription& d = l.system;
    const GhostDiagnosis g = hand_diagnosis(d, GhostOperator::B);
    SplitInitOptions o;
    o.preselect = true;
    o.preselect_max_power_sum = 50.0;
    const SplitPlan plan = split_from_diagnosis(d, l.catalog, g, o);
    ASSERT_TRUE(plan.preselected) << plan.reason;
    EXPECT_EQ(plan.target_glass.kind, MaterialKind::Catalog);
    EXPECT_NE(plan.target_glass.name, "N-BK7");
    EXPECT_NE(plan.reason.find("Glasvorauswahl"), std::string::npos) << plan.reason;
    // c_s des dünnen Achromats aus den Anteilen der Vorauswahl (Linse vorn, φ = 1)
    const std::vector<GlassDispersion> disp = catalog_dispersions(l.catalog);
    GlassPreselectionOptions po;
    po.max_power_sum = 50.0;
    for (std::size_t i = 0; i < disp.size(); ++i) {
        if (disp[i].name == "N-BK7") {
            po.fixed = {static_cast<int>(i)};
        }
    }
    po.max_results = 1;
    const std::vector<PreselectedGlasses> best = preselect_glasses(disp, po);
    ASSERT_EQ(best.size(), 1U);
    EXPECT_EQ(disp[static_cast<std::size_t>(best[0].glasses[1])].name, plan.target_glass.name);
    EXPECT_DOUBLE_EQ(plan.achromat_curvature, 0.82 - best[0].powers[0] * (0.82 + 0.145));
    EXPECT_DOUBLE_EQ(plan.secondary_spectrum, best[0].residual);
    // Linse ohne Katalogglas: Vorauswahl entfällt, Zielglas der Platte
    SystemDescription other = d;
    other.surfaces[1].material = MaterialSpec{MaterialKind::Constant, "", 1.6, 1.0, {}};
    const SplitPlan fallback = split_from_diagnosis(other, l.catalog, g, o);
    EXPECT_FALSE(fallback.preselected);
}

// Operator B ohne Geisterdiagnose (Farbregel, M11-T5c): Partner und Kittkrümmung aus der
// Glasvorauswahl, keine Platte; Fehler ohne Linse.
TEST(InsertionInit, SplitForColor) {
    const LoadedSystem l = load_residual_system("einzellinse_achromat_start.json");
    const SplitPlan plan = split_for_color(l.system, l.catalog, 1);
    EXPECT_TRUE(plan.preselected);
    EXPECT_EQ(plan.plate, -1);
    EXPECT_EQ(plan.split.lens_surface, 1);
    EXPECT_EQ(plan.free_surface, 2);
    EXPECT_EQ(plan.target_glass.kind, MaterialKind::Catalog);
    EXPECT_NE(plan.reason.find("Farbregel"), std::string::npos) << plan.reason;
    EXPECT_THROW((void)split_for_color(l.system, l.catalog, 0), std::invalid_argument);
    // Untere Schranke der Dickenvariablen: Anteil so begrenzt, dass beide Teile sie einhalten
    // (Linse 4 mm); zu dünne Linse wird nicht geteilt (M6-T6 p4_r3)
    SystemDescription thick = l.system;
    VariableSpec v;
    v.type = VariableType::Thickness;
    v.selection = SurfaceSelection::One;
    v.surfaces = {1};
    v.min = 1.8;
    if (!thick.variables) {
        thick.variables = Variables{};
    }
    thick.variables->items.push_back(v);
    SplitInitOptions o;
    o.fraction = 0.2;
    EXPECT_DOUBLE_EQ(split_for_color(thick, l.catalog, 1, o).split.fraction, 1.8 / 4.0);
    o.fraction = 0.5;
    EXPECT_DOUBLE_EQ(split_for_color(thick, l.catalog, 1, o).split.fraction, 0.5);
    thick.variables->items.back().min = 2.1;
    EXPECT_THROW((void)split_for_color(thick, l.catalog, 1, o), std::invalid_argument);
    EXPECT_THROW((void)split_for_color(l.system, l.catalog, 2), std::invalid_argument);
}

// Rundungsboden: Ein Luftspalt wird durch zwei Flächen mit n = 1 dazwischen geteilt (optisch
// nichts, nur Transfer in Teilstücken und Brechung mit μ = 1). Die Merit ändert sich in derselben
// Größenordnung wie beim Operator A – die Abweichungen sind Rundung des Traces, nicht des
// Operators.
TEST(InsertionMerit, RoundingFloorControl) {
    const SystemDescription aux = lens_with_aux_plane();
    const double f = system_merit(aux);
    double worst = 0.0;
    double worst_flint = 0.0;
    for (const double pos : {0.5, 1.0, 3.0, 7.0, 10.0, 15.0, 19.0}) {
        for (const double c : {0.0, -0.01, 0.05}) {
            const double rel = relative_change(
                f, system_merit(
                       insert_meniscus(aux, meniscus(2, pos, c, constant_glass(1.0))).system));
            const double flint = relative_change(
                f, system_merit(insert_meniscus(aux, meniscus(2, pos, c, flint_glass())).system));
            worst = std::max(worst, rel);
            worst_flint = std::max(worst_flint, flint);
            EXPECT_LE(rel, kInsertionMeritRelative) << pos << " " << c;
            EXPECT_LE(flint, kInsertionMeritRelative) << pos << " " << c;
        }
    }
    std::printf("[CPU] Rundungsboden: Flächen mit n = 1 im Spalt größte relative Änderung %.2e, "
                "Meniskus aus Flint an denselben Lagen %.2e\n",
                worst, worst_flint);
}

} // namespace
} // namespace mko::test
