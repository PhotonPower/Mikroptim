// Operatoren der Entwurfssteuerung (M10-T4, docs/optimierung.md, Abschnitt 13): Rückbau
// remove_element ∘ Operator A = Ausgangssystem, Rückbau-Homotopie bis zum Null-Element,
// Merit-Invarianz von place_ghost_dictionary und insert_null_doublet, aplanatische Schale ohne
// Beitrag zur sphärischen Aberration aller Ordnungen (Gruppensumme), Glastausch, Fehlerfälle.

#include "insertion_test_systems.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/design_operators.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

std::string sci(double v) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(2) << v;
    return os.str();
}

SystemDescription mehrlinser() {
    return load_residual_system("testsystem_mehrlinser.json").system;
}

// Mehrlinser mit Variablen (Krümmungen "all", Dicken der Luftspalte als Liste, Glas),
// mechanischen Bedingungen mit Standardauswahl und Brennweite.
SystemDescription mehrlinser_with_variables() {
    SystemDescription d = mehrlinser();
    Variables v;
    VariableSpec c;
    c.type = VariableType::Curvature;
    c.selection = SurfaceSelection::All;
    v.items.push_back(c);
    VariableSpec t;
    t.type = VariableType::Thickness;
    t.selection = SurfaceSelection::List;
    t.surfaces = {3, 5};
    t.min = 0.1;
    t.max = 20.0;
    v.items.push_back(t);
    VariableSpec w;
    w.type = VariableType::WorkingDistance;
    w.selection = SurfaceSelection::None;
    v.items.push_back(w);
    d.variables = v;
    Constraints cs;
    ConstraintSpec f;
    f.type = ConstraintType::FocalLength;
    f.weight = 1.0;
    cs.items.push_back(f);
    ConstraintSpec m;
    m.type = ConstraintType::MinCenterThickness;
    m.weight = 100.0;
    m.medium = ConstraintMedium::Air;
    m.thickness = 0.05;
    cs.items.push_back(m);
    d.constraints = cs;
    d.target = Target{};
    d.target->focal_length = 40.0;
    validate_system(d);
    return d;
}

// Vergleich „ergibt das Ausgangssystem“: alle Flächen gleich bis auf die Dicke des vereinigten
// Spalts (Rundung von p + fl(T − p)), Geisterplatten gleich bis auf distance, aufgelöste
// Variablen und Flächen der mechanischen Bedingungen gleich, Merit gleich.
void expect_same_system(const SystemDescription& expected, const SystemDescription& got, int gap) {
    ASSERT_EQ(got.surfaces.size(), expected.surfaces.size());
    for (std::size_t s = 0; s < expected.surfaces.size(); ++s) {
        Surface e = expected.surfaces[s];
        Surface g = got.surfaces[s];
        if (static_cast<int>(s) == gap) {
            EXPECT_NEAR(g.thickness, e.thickness, kRemovalRoundtripRelative * e.thickness);
            g.thickness = e.thickness;
        }
        EXPECT_EQ(g, e) << "Fläche " << s;
    }
    ASSERT_EQ(got.ghost_plates.size(), expected.ghost_plates.size());
    for (std::size_t p = 0; p < expected.ghost_plates.size(); ++p) {
        GhostPlate e = expected.ghost_plates[p];
        GhostPlate g = got.ghost_plates[p];
        EXPECT_NEAR(g.distance, e.distance, kRemovalRoundtripRelative * (1.0 + e.distance));
        g.distance = e.distance;
        EXPECT_EQ(g, e) << "Platte " << p;
    }
    const std::vector<ResolvedVariable> rv_e = resolve_variables(expected);
    const std::vector<ResolvedVariable> rv_g = resolve_variables(got);
    ASSERT_EQ(rv_g.size(), rv_e.size());
    for (std::size_t k = 0; k < rv_e.size(); ++k) {
        EXPECT_EQ(rv_g[k].type, rv_e[k].type) << k;
        EXPECT_EQ(rv_g[k].surface, rv_e[k].surface) << k;
        EXPECT_EQ(rv_g[k].plate, rv_e[k].plate) << k;
        EXPECT_EQ(rv_g[k].order, rv_e[k].order) << k;
        EXPECT_EQ(rv_g[k].lower, rv_e[k].lower) << k;
        EXPECT_EQ(rv_g[k].upper, rv_e[k].upper) << k;
    }
    if (expected.constraints) {
        ASSERT_TRUE(got.constraints);
        ASSERT_EQ(got.constraints->items.size(), expected.constraints->items.size());
        for (std::size_t c = 0; c < expected.constraints->items.size(); ++c) {
            const ConstraintSpec& e = expected.constraints->items[c];
            const ConstraintSpec& g = got.constraints->items[c];
            EXPECT_EQ(g.type, e.type);
            if (is_mechanical(e.type)) {
                EXPECT_EQ(constraint_surfaces(got, g), constraint_surfaces(expected, e)) << c;
            }
        }
    }
    EXPECT_LE(relative_change(system_merit(expected), system_merit(got)), kInsertionMeritRelative);
}

// --- remove_element ∘ insert_null_meniscus = Ausgangssystem (Abnahme) ---------------------------

TEST(DesignOperators, RemoveAfterNullMeniscusRestoresSystem) {
    struct Case {
        std::string name;
        SystemDescription d;
        int after;
        double position;
        double c0;
        MaterialSpec glass;
    };
    std::vector<Case> cases;
    cases.push_back({"Mehrlinser Blende", mehrlinser(), 0, 1.3, 0.05, catalog_glass("N-BK7")});
    cases.push_back({"Mehrlinser Spalt 3", mehrlinser(), 3, 0.37, -0.1, constant_glass(1.8)});
    cases.push_back({"Mehrlinser Spalt 5", mehrlinser(), 5, 1.9, 0.0, catalog_glass("CAF2")});
    cases.push_back({"Mehrlinser Variablen Spalt 3", mehrlinser_with_variables(), 3, 0.6, 0.02,
                     catalog_glass("N-BK7")});
    cases.push_back({"Mehrlinser Variablen Spalt 5", mehrlinser_with_variables(), 5, 0.3, -0.03,
                     flint_glass()});
    const SystemDescription geister =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    cases.push_back({"Geisterplatten vor/hinter", geister, 0, 2.5, -0.1, catalog_glass("N-BK7")});
    cases.push_back({"Geisterplatten an der Lage", geister, 0, 1.0, 0.07, catalog_glass("N-BK7")});
    cases.push_back({"Geisterplatten Spalt 2", geister, 2, 4.0, 0.01, flint_glass()});
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        const InsertionResult ins =
            insert_meniscus(c.d, meniscus(c.after, c.position, c.c0, c.glass));
        ASSERT_TRUE(is_null_element(ins.system, ins.first_new_surface));
        const RemovalResult rem = remove_element(ins.system, ins.first_new_surface);
        EXPECT_EQ(rem.first_removed, c.after + 1);
        EXPECT_EQ(rem.removed_surfaces, 2);
        EXPECT_EQ(rem.merged_gap, c.after);
        expect_same_system(c.d, rem.system, c.after);
    }
}

TEST(DesignOperators, RemoveAfterNullDoubletRestoresSystem) {
    const SystemDescription d = mehrlinser_with_variables();
    NullDoublet n;
    n.after_surface = 5;
    n.position = 0.8;
    n.curvature = 0.04;
    n.front_glass = catalog_glass("N-BK7");
    n.back_glass = flint_glass();
    const InsertionResult ins = insert_null_doublet(d, n);
    EXPECT_EQ(ins.new_surfaces, 3);
    EXPECT_EQ(ins.system.surfaces.size(), d.surfaces.size() + 3);
    EXPECT_LE(relative_change(system_merit(d), system_merit(ins.system)), kInsertionMeritRelative);
    EXPECT_EQ(glass_block_end(ins.system, 6), 8);
    const RemovalResult rem = remove_element(ins.system, 6);
    EXPECT_EQ(rem.removed_surfaces, 3);
    expect_same_system(d, rem.system, 5);
}

// --- Rückbau-Homotopie ---------------------------------------------------------------------------

TEST(DesignOperators, CollapseReachesNullElementAndRemoves) {
    const SystemDescription d = mehrlinser_with_variables();
    // Kittglied 1 … 3 (CaF₂ + Buchdahl) in vier Stufen
    SystemDescription s = d;
    const double vertex_before = d.surfaces[0].thickness + d.surfaces[1].thickness +
                                 d.surfaces[2].thickness + d.surfaces[3].thickness;
    const int K = 4;
    for (int k = 0; k < K; ++k) {
        s = collapse_element(s, 1, 1.0 / (K - k));
        EXPECT_EQ(is_null_element(s, 1), k == K - 1) << k;
    }
    EXPECT_EQ(s.surfaces[1].thickness, 0.0);
    EXPECT_EQ(s.surfaces[2].thickness, 0.0);
    EXPECT_EQ(s.surfaces[2].curvature, d.surfaces[1].curvature);
    EXPECT_EQ(s.surfaces[3].curvature, d.surfaces[1].curvature);
    // Lage der folgenden Scheitel bleibt
    const double vertex_after = s.surfaces[0].thickness + s.surfaces[1].thickness +
                                s.surfaces[2].thickness + s.surfaces[3].thickness;
    EXPECT_NEAR(vertex_after, vertex_before, 1e-14 * vertex_before);
    // Variablen des Blocks festgesetzt, die übrigen bleiben
    for (const ResolvedVariable& r : resolve_variables(s)) {
        if (r.type == VariableType::Curvature) {
            EXPECT_TRUE(r.surface < 1 || r.surface > 3) << r.surface;
        }
    }
    EXPECT_EQ(resolve_variables(s).size(), resolve_variables(d).size() - 3);
    // Null-Element entfernen: Merit gleich
    const RemovalResult rem = remove_element(s, 1);
    EXPECT_EQ(rem.system.surfaces.size(), d.surfaces.size() - 3);
    EXPECT_LE(relative_change(system_merit(s), system_merit(rem.system)), kInsertionMeritRelative);
    // vereinigter Spalt 0: Blende bis zur Einzellinse
    EXPECT_NEAR(rem.system.surfaces[0].thickness, vertex_before, 1e-14 * vertex_before);
    // Dickenvariable des Spalts nach dem Block (3) gilt jetzt für den vereinigten Spalt 0
    bool gap0 = false;
    for (const ResolvedVariable& r : resolve_variables(rem.system)) {
        gap0 = gap0 || (r.type == VariableType::Thickness && r.surface == 0);
    }
    EXPECT_TRUE(gap0);
}

TEST(DesignOperators, RemovalRaisesUpperBoundOfMergedGap) {
    SystemDescription d = mehrlinser_with_variables();
    d.variables->items[1].max = 3.0; // Spalte 3 (1 mm) und 5 (2 mm)
    const RemovalResult rem = remove_element(d, 4);
    EXPECT_EQ(rem.system.surfaces[3].thickness, 1.0 + 3.0 + 2.0);
    for (const ResolvedVariable& r : resolve_variables(rem.system)) {
        if (r.type == VariableType::Thickness && r.surface == 3) {
            EXPECT_EQ(r.upper, 6.0);
        }
    }
    const SystemDescription c = collapse_element(d, 4, 1.0);
    EXPECT_EQ(c.surfaces[5].thickness, 5.0);
    EXPECT_EQ(c.variables->items[1].max, 5.0);
}

TEST(DesignOperators, RemoveFrontElementMergesIntoWorkingDistance) {
    const SystemDescription d = mehrlinser_with_variables();
    const RemovalResult rem = remove_element(d, 6);
    ASSERT_EQ(rem.system.surfaces.size(), 6u);
    EXPECT_EQ(rem.system.surfaces[5].thickness, 2.0 + 4.0 + 14.45);
    // Dickenvariable des Spalts 5 entfällt (jetzt Arbeitsabstand, eigene Variable)
    for (const ResolvedVariable& r : resolve_variables(rem.system)) {
        EXPECT_FALSE(r.type == VariableType::Thickness && r.surface == 5);
    }
}

// --- Merit-Invarianz place_ghost_dictionary (Abnahme) ------------------------------------------

TEST(DesignOperators, GhostDictionaryLeavesMeritUnchanged) {
    struct Case {
        std::string name;
        SystemDescription d;
        GhostDictionaryPlacement p;
    };
    std::vector<Case> cases;
    // System mit Wörterbuch ("all"-Variablen und -Strafe), Koeffizienten ≠ 0, damit falsch
    // zugeordnete Plattenbezüge die Merit ändern würden
    SystemDescription geister =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    for (std::size_t g = 0; g < geister.ghost_plates.size(); ++g) {
        for (std::size_t k = 0; k < geister.ghost_plates[g].coefficients.size(); ++k) {
            geister.ghost_plates[g].coefficients[k] =
                1e-4 * static_cast<double>(g + 1) / static_cast<double>(k + 2);
        }
    }
    const auto place = [](int after, double dist, std::vector<MaterialSpec> glasses, int order,
                          std::optional<double> weight) {
        GhostDictionaryPlacement p;
        p.after_surface = after;
        p.distance = dist;
        p.glasses = std::move(glasses);
        p.order = order;
        p.penalty_weight = weight;
        return p;
    };
    cases.push_back({"neue Lage zwischen Platten", geister,
                     place(0, 2.5, {catalog_glass("N-BK7"), flint_glass()}, 6, std::nullopt)});
    cases.push_back(
        {"an vorhandener Lage", geister, place(0, 1.0, {catalog_glass("CAF2")}, 4, std::nullopt)});
    cases.push_back({"vor allen Platten, eigenes μ", geister,
                     place(0, 0.0, {flint_glass(), catalog_glass("N-BK7")}, 8, 50.0)});
    cases.push_back(
        {"hinter der Linse, eigenes μ", geister, place(2, 15.0, {catalog_glass("N-BK7")}, 6, 5.0)});
    // System ohne Platten und ohne Strafe: eigener Eintrag nötig
    cases.push_back({"Mehrlinser Spalt 3", mehrlinser_with_variables(),
                     place(3, 0.5, {catalog_glass("N-BK7"), flint_glass()}, 6, 10.0)});
    cases.push_back({"Mehrlinser Blende", mehrlinser_with_variables(),
                     place(0, 3.0, {constant_glass(1.6)}, 2, 1.0)});
    // Plattenbezüge als Listen (Platten 1 und 3), neue Platten davor
    SystemDescription lists = geister;
    lists.variables->items.back().plate_selection = SurfaceSelection::List;
    lists.variables->items.back().plates = {1, 3};
    lists.constraints->items.back().plate_selection = SurfaceSelection::List;
    lists.constraints->items.back().plates = {1, 3};
    cases.push_back({"Listen", lists, place(0, 0.5, {catalog_glass("N-BK7")}, 4, std::nullopt)});

    double worst = 0.0;
    int bitwise = 0;
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        const GhostDictionaryResult r = place_ghost_dictionary(c.d, c.p);
        const int m = static_cast<int>(c.p.glasses.size());
        ASSERT_EQ(r.system.ghost_plates.size(), c.d.ghost_plates.size() + c.p.glasses.size());
        // alte Platten unverändert an ihrer neuen Stelle, neue mit Koeffizienten 0
        for (std::size_t g = 0; g < c.d.ghost_plates.size(); ++g) {
            EXPECT_EQ(r.system.ghost_plates[static_cast<std::size_t>(r.plate_new_of_old[g])],
                      c.d.ghost_plates[g]);
        }
        for (int g : r.new_plates) {
            const GhostPlate& gp = r.system.ghost_plates[static_cast<std::size_t>(g)];
            EXPECT_EQ(gp.after_surface, c.p.after_surface);
            EXPECT_EQ(gp.distance, c.p.distance);
            EXPECT_EQ(gp.coefficients,
                      std::vector<double>(static_cast<std::size_t>(c.p.order), 0.0));
        }
        // neue Platten sind Variablen und haben eine Strafe
        int new_vars = 0;
        for (const ResolvedVariable& v : resolve_variables(r.system)) {
            for (int g : r.new_plates) {
                new_vars += v.type == VariableType::GhostCoefficient && v.plate == g;
            }
        }
        EXPECT_EQ(new_vars, m * c.p.order);
        for (int g : r.new_plates) {
            EXPECT_TRUE(ghost_penalty_weight(r.system, g).has_value()) << g;
            if (c.p.penalty_weight) {
                EXPECT_EQ(*ghost_penalty_weight(r.system, g), *c.p.penalty_weight);
            }
        }
        // alte Platten behalten ihr Gewicht und ihre Variablen
        for (std::size_t g = 0; g < c.d.ghost_plates.size(); ++g) {
            EXPECT_EQ(ghost_penalty_weight(r.system, r.plate_new_of_old[g]),
                      ghost_penalty_weight(c.d, static_cast<int>(g)));
        }
        const double rel = relative_change(system_merit(c.d), system_merit(r.system));
        worst = std::max(worst, rel);
        bitwise += rel == 0.0;
        EXPECT_LE(rel, kInsertionMeritRelative);
    }
    RecordProperty("worst_relative_merit_change", sci(worst));
    RecordProperty("bitwise_cases",
                   std::to_string(bitwise) + " von " + std::to_string(cases.size()));
}

TEST(DesignOperators, GhostDictionaryErrors) {
    const SystemDescription d = mehrlinser_with_variables();
    GhostDictionaryPlacement p;
    p.after_surface = 3;
    p.distance = 0.5;
    p.glasses = {catalog_glass("N-BK7")};
    EXPECT_THROW(place_ghost_dictionary(d, p), std::invalid_argument); // ohne Strafe
    p.penalty_weight = 1.0;
    p.after_surface = 1; // Glas
    EXPECT_THROW(place_ghost_dictionary(d, p), std::invalid_argument);
    p.after_surface = 7; // Arbeitsabstand
    EXPECT_THROW(place_ghost_dictionary(d, p), std::invalid_argument);
    p.after_surface = 3;
    p.distance = 1.5; // Spalt 1 mm
    EXPECT_THROW(place_ghost_dictionary(d, p), std::invalid_argument);
    p.distance = 0.5;
    p.glasses.clear();
    EXPECT_THROW(place_ghost_dictionary(d, p), std::invalid_argument);
}

// --- aplanatische Schale (Abnahme) ---------------------------------------------------------------

double group_sum(const AberrationReport& rep, int first, int last) {
    double s = 0.0;
    for (int i = first; i <= last; ++i) {
        s += rep.surfaces[static_cast<std::size_t>(i)].full_order_spherical;
    }
    return s;
}

TEST(DesignOperators, AplanaticShellKeepsFrontGroupSphericalSum) {
    // Schale vor dem Objekt (neue Frontlinse): Frontgruppe = Element 6–7 plus Schale. Die
    // Beiträge der Flächen davor bleiben bitgleich (gleiche Strahlen), die Schale trägt
    // mathematisch nichts zur sphärischen Aberration aller Ordnungen bei.
    const LoadedSystem l = prepare_test_system(mehrlinser());
    const AberrationReport before = compute_aberrations(l.system, l.catalog);
    ASSERT_TRUE(before.real_rays_valid);
    const double sum_before = group_sum(before, 6, 7);
    double worst = 0.0;
    double control = 1e300;
    for (const char* glass : {"N-BK7", "CAF2"}) {
        for (const double pos : {0.5, 2.0}) {
            SCOPED_TRACE(std::string(glass) + " bei " + std::to_string(pos));
            AplanaticShell sh;
            sh.after_surface = 7;
            sh.position = pos;
            sh.thickness = 1.5;
            sh.glass = catalog_glass(glass);
            const AplanaticShellResult r = insert_aplanatic_shell(l.system, l.catalog, sh);
            const SystemDescription& s = r.insertion.system;
            ASSERT_EQ(s.surfaces.size(), 10u);
            EXPECT_EQ(s.surfaces[7].thickness, pos);
            EXPECT_NEAR(r.image_intercept, r.intercept / r.n_glass, 1e-15 * r.intercept);
            const AberrationReport after = compute_aberrations(s, l.catalog);
            ASSERT_TRUE(after.real_rays_valid);
            for (int i = 0; i <= 7; ++i) {
                EXPECT_EQ(after.surfaces[static_cast<std::size_t>(i)].full_order_spherical,
                          before.surfaces[static_cast<std::size_t>(i)].full_order_spherical);
            }
            const double sum_after = group_sum(after, 6, 9);
            const double rel = std::abs(sum_after - sum_before) / std::abs(sum_before);
            worst = std::max(worst, rel);
            EXPECT_LT(rel, kAplanaticGroupSpherical);
            // Gegenprobe: eine nicht aplanatische Schale gleicher Lage (Krümmungen ×1,1) ändert
            // die Gruppensumme deutlich
            SystemDescription off = s;
            off.surfaces[8].curvature *= 1.1;
            off.surfaces[9].curvature *= 1.1;
            const AberrationReport bent = compute_aberrations(off, l.catalog);
            const double rel_bent =
                std::abs(group_sum(bent, 6, 9) - sum_before) / std::abs(sum_before);
            control = std::min(control, rel_bent);
            EXPECT_GT(rel_bent, 1e-2);
            // NA steigt um n_G (sin U′ des Randstrahls ∝ 1/f_real): paraxial kürzere Brennweite
            EXPECT_LT(compute_paraxial(s, l.catalog).focal_length,
                      compute_paraxial(l.system, l.catalog).focal_length);
        }
    }
    RecordProperty("worst_relative_group_change", sci(worst));
    RecordProperty("control_relative_group_change", sci(control));
}

TEST(DesignOperators, AplanaticShellInsideSystemAddsNoSphericalAberration) {
    // Schale im Spalt 3 (zwischen Kittglied und Einzellinse) und im Spalt 5: die beiden neuen
    // Flächen tragen (bis auf Rundung) nichts bei, die Flächen davor bleiben bitgleich.
    const LoadedSystem l = prepare_test_system(mehrlinser());
    const AberrationReport before = compute_aberrations(l.system, l.catalog);
    double scale = 0.0;
    for (const SurfaceAberration& s : before.surfaces) {
        scale = std::max(scale, std::abs(s.full_order_spherical));
    }
    double worst = 0.0;
    for (const int gap : {3, 5}) {
        SCOPED_TRACE(gap);
        AplanaticShell sh;
        sh.after_surface = gap;
        sh.position = 0.2;
        sh.thickness = 0.5;
        sh.glass = catalog_glass("N-BK7");
        sh.refocus = false;
        const AplanaticShellResult r = insert_aplanatic_shell(l.system, l.catalog, sh);
        const AberrationReport after = compute_aberrations(r.insertion.system, l.catalog);
        ASSERT_TRUE(after.real_rays_valid);
        for (int i = 0; i <= gap; ++i) {
            EXPECT_EQ(after.surfaces[static_cast<std::size_t>(i)].full_order_spherical,
                      before.surfaces[static_cast<std::size_t>(i)].full_order_spherical);
        }
        for (int i = gap + 1; i <= gap + 2; ++i) {
            const double c =
                std::abs(after.surfaces[static_cast<std::size_t>(i)].full_order_spherical);
            worst = std::max(worst, c / scale);
            EXPECT_LT(c, kAplanaticGroupSpherical * scale) << i;
        }
        // Spaltdicke aufgeteilt, Dickenbezüge und Lage der folgenden Scheitel bleiben
        EXPECT_NEAR(r.insertion.system.surfaces[static_cast<std::size_t>(gap)].thickness +
                        r.insertion.system.surfaces[static_cast<std::size_t>(gap + 1)].thickness +
                        r.insertion.system.surfaces[static_cast<std::size_t>(gap + 2)].thickness,
                    l.system.surfaces[static_cast<std::size_t>(gap)].thickness, 1e-14);
    }
    RecordProperty("worst_new_surface_contribution", sci(worst));
}

TEST(DesignOperators, AplanaticShellErrors) {
    const LoadedSystem l = prepare_test_system(mehrlinser());
    AplanaticShell sh;
    sh.after_surface = 3;
    sh.position = 0.5;
    sh.thickness = 0.6; // passt nicht in 1 mm
    sh.glass = catalog_glass("N-BK7");
    EXPECT_THROW(insert_aplanatic_shell(l.system, l.catalog, sh), std::invalid_argument);
    sh.thickness = 0.3;
    sh.after_surface = 0; // kollimiert: kein Bildpunkt
    EXPECT_THROW(insert_aplanatic_shell(l.system, l.catalog, sh), std::invalid_argument);
    sh.after_surface = 1; // Glas
    EXPECT_THROW(insert_aplanatic_shell(l.system, l.catalog, sh), std::invalid_argument);
}

// --- Glastausch und Fehlerfälle des Rückbaus
// ------------------------------------------------------

TEST(DesignOperators, ReplaceGlassFreezesGlassVariable) {
    SystemDescription d = mehrlinser_with_variables();
    free_glass(d, 4);
    ASSERT_EQ(d.variables->items.back().type, VariableType::Glass);
    replace_glass(d, 4, catalog_glass("CAF2"));
    EXPECT_EQ(d.surfaces[4].material.name, "CAF2");
    for (const VariableSpec& v : d.variables->items) {
        EXPECT_NE(v.type, VariableType::Glass);
    }
    EXPECT_THROW(replace_glass(d, 3, catalog_glass("CAF2")), std::invalid_argument); // Luft
    MaterialSpec air;
    EXPECT_THROW(replace_glass(d, 4, air), std::invalid_argument);
}

TEST(DesignOperators, RemovalErrors) {
    const SystemDescription d = mehrlinser();
    EXPECT_THROW(remove_element(d, 2), std::invalid_argument); // Kittfläche
    EXPECT_THROW(remove_element(d, 3), std::invalid_argument); // Luft danach
    EXPECT_THROW(remove_element(d, 0), std::invalid_argument); // Blende
    EXPECT_THROW(collapse_element(d, 4, 0.0), std::invalid_argument);
    EXPECT_THROW(collapse_element(d, 4, 1.5), std::invalid_argument);
    EXPECT_EQ(glass_block_end(d, 1), 3);
    EXPECT_FALSE(is_null_element(d, 4));
}

} // namespace
} // namespace mko::test
