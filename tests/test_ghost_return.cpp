// Rückkehr vom realen System zu Geisterplatten (M6-T10, docs/optimierung.md, Abschnitt 7,
// „Rückkehr zu Geisterplatten“): Merit unverändert, Lagen gegen den Aberrationsbericht als
// Orakel, Plattenspalte fest, Strafe wie im Startsystem der Zielspezifikation,
// Normierungsradien gegen die Geometrie im kollimierten Bündel, Fehlerfälle.

#include "ghost_return_test_systems.hpp"
#include "insertion_test_systems.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/synth/ghost_return.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

SystemDescription real_system() {
    return ghost_return_real_system();
}

GhostReturnOptions options(std::vector<GhostReturnLocation> locations) {
    GhostReturnOptions o;
    o.glasses = {catalog_glass("N-BK7"), catalog_glass("CAF2"), flint_glass()};
    o.locations = std::move(locations);
    return o;
}

// Dateiflächen mit Dickenvariable
std::vector<int> thickness_gaps(const SystemDescription& d) {
    std::vector<int> out;
    for (const ResolvedVariable& v : resolve_variables(d)) {
        if (v.type == VariableType::Thickness) {
            out.push_back(v.surface);
        }
    }
    return out;
}

// Orakel der Spaltwahl: innere Luftspalte (nach 1 … S − 2, Dicke ≥ min_gap) aus dem paraxialen
// Aberrationsbericht, Mittel der beiden Spaltflächen.
struct Expected {
    int aperture = -1;
    int field = -1;
    int front = -1;
};

Expected expected_gaps(const LoadedSystem& l, double min_gap) {
    AberrationOptions ao;
    ao.real_rays = false;
    ao.focal_shift = false;
    const AberrationReport rep = compute_aberrations(l.system, l.catalog, ao);
    Expected e;
    double best_y = -1.0;
    double best_r = -1.0;
    const int S = static_cast<int>(l.system.surfaces.size());
    for (int s = 1; s + 1 < S; ++s) {
        const Surface& g = l.system.surfaces[static_cast<std::size_t>(s)];
        if (g.material.kind != MaterialKind::Air || g.thickness < min_gap) {
            continue;
        }
        const SurfaceAberration& a = rep.surfaces[static_cast<std::size_t>(s)];
        const SurfaceAberration& b = rep.surfaces[static_cast<std::size_t>(s) + 1];
        const double y = 0.5 * (std::abs(a.y) + std::abs(b.y));
        const double r =
            0.5 * (std::abs(a.ybar) / std::abs(a.y) + std::abs(b.ybar) / std::abs(b.y));
        if (y > best_y) {
            best_y = y;
            e.aperture = s;
        }
        if (r > best_r) {
            best_r = r;
            e.field = s;
        }
        e.front = s;
    }
    return e;
}

TEST(GhostReturn, MeritUnchangedAndPlatesAtExpectedGaps) {
    const SystemDescription real = real_system();
    const LoadedSystem l = prepare_test_system(real);
    const Expected e = expected_gaps(l, GhostReturnOptions{}.min_gap);
    ASSERT_GE(e.aperture, 1);
    ASSERT_GE(e.field, 1);
    const double f_real = system_merit(real);

    struct Case {
        std::string name;
        std::vector<GhostReturnLocation> locations;
        std::vector<int> gaps; // erwartete Spalte in Reihenfolge der Lagen (ohne Doppelungen)
    };
    std::vector<Case> cases;
    cases.push_back({"Standard (aperture, field)",
                     {GhostReturnLocation::Aperture, GhostReturnLocation::Field},
                     e.aperture == e.field ? std::vector<int>{e.aperture}
                                           : std::vector<int>{e.aperture, e.field}});
    cases.push_back({"alle Lagen",
                     {GhostReturnLocation::Stop, GhostReturnLocation::Front,
                      GhostReturnLocation::Aperture, GhostReturnLocation::Field},
                     {}});
    {
        std::vector<int> g{0, e.front};
        for (const int x : {e.aperture, e.field}) {
            if (std::find(g.begin(), g.end(), x) == g.end()) {
                g.push_back(x);
            }
        }
        cases.back().gaps = g;
    }
    double worst = 0.0;
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        const GhostReturnOptions o = options(c.locations);
        const GhostReturnResult r = return_to_ghost_plates(l, o);
        ASSERT_EQ(r.positions.size(), c.gaps.size());
        // Doppelte Spalte werden vermerkt
        EXPECT_EQ(r.notes.size(), c.locations.size() - c.gaps.size());
        const SystemDescription& d = r.system;
        ASSERT_EQ(d.ghost_plates.size(), c.gaps.size() * o.glasses.size());
        for (std::size_t k = 0; k < c.gaps.size(); ++k) {
            const GhostReturnPosition& p = r.positions[k];
            EXPECT_EQ(p.after_surface, c.gaps[k]);
            EXPECT_EQ(p.distance,
                      0.5 * real.surfaces[static_cast<std::size_t>(p.after_surface)].thickness);
            ASSERT_EQ(p.plates.size(), o.glasses.size());
            for (std::size_t j = 0; j < p.plates.size(); ++j) {
                const GhostPlate& g = d.ghost_plates.at(static_cast<std::size_t>(p.plates[j]));
                EXPECT_EQ(g.after_surface, p.after_surface);
                EXPECT_EQ(g.distance, p.distance);
                EXPECT_EQ(g.glass.kind, o.glasses[j].kind);
                EXPECT_EQ(g.glass.name, o.glasses[j].name);
                EXPECT_EQ(g.glass.n0, o.glasses[j].n0);
                EXPECT_EQ(g.norm_radius, p.norm_radius);
                EXPECT_EQ(g.coefficients, std::vector<double>(10, 0.0));
            }
            EXPECT_GT(p.norm_radius, 0.0);
            // Spalt mit Dickenvariable wird fest, die übrigen Variablen bleiben
            const bool had = p.after_surface == 3 || p.after_surface == 5;
            EXPECT_EQ(p.gap_fixed, had);
        }
        std::vector<int> want = thickness_gaps(real);
        want.erase(std::remove_if(want.begin(), want.end(),
                                  [&](int s) {
                                      return std::find(c.gaps.begin(), c.gaps.end(), s) !=
                                             c.gaps.end();
                                  }),
                   want.end());
        EXPECT_EQ(thickness_gaps(d), want);
        // Geisterkoeffizienten: alle Ordnungen aller Platten sind Variablen
        int ghost_vars = 0;
        for (const ResolvedVariable& v : resolve_variables(d)) {
            ghost_vars += v.type == VariableType::GhostCoefficient ? 1 : 0;
        }
        EXPECT_EQ(ghost_vars, static_cast<int>(d.ghost_plates.size()) * 10);
        // Strafe wie im Startsystem der Zielspezifikation (ein Eintrag für alle Platten)
        int penalties = 0;
        for (const ConstraintSpec& s : d.constraints->items) {
            if (s.type == ConstraintType::GhostPenalty) {
                ++penalties;
                EXPECT_EQ(s.weight, 47.2);
                EXPECT_EQ(s.order_factor, 1.0);
                EXPECT_EQ(s.order_weight, 1.0);
                EXPECT_EQ(s.smoothing, 1e-6);
                EXPECT_NE(s.plate_selection, SurfaceSelection::List);
            }
        }
        EXPECT_EQ(penalties, 1);
        // Merit unverändert (unabhängig gerechnet und laut Bericht)
        const double f = system_merit(d);
        const double rel = relative_change(f_real, f);
        worst = std::max(worst, rel);
        EXPECT_LE(rel, kInsertionMeritRelative) << f_real << " " << f;
        EXPECT_LE(relative_change(r.merit_before, r.merit_after), kInsertionMeritRelative);
        const nlohmann::json j = nlohmann::json::parse(ghost_return_json_text(r));
        EXPECT_EQ(j["positions"].size(), c.gaps.size());
        EXPECT_LE(j["relative_change"].get<double>(), kInsertionMeritRelative);
    }
    std::cout << "[ Rückkehr ] größte relative Änderung der Merit " << worst << '\n';
}

TEST(GhostReturn, NormRadiusFromRealRaysInCollimatedBeam) {
    // Blendenspalt: kollimiertes Bündel vor der ersten Linse; Strahlen im Abstand d von der
    // Blende haben die Höhe |(x, y + d tan θ)| ≤ a + d tan θ (Pupillenradius a)
    const SystemDescription real = real_system();
    const LoadedSystem l = prepare_test_system(real);
    const double a = real.surfaces[0].semi_diameter;
    const double d = 0.5 * real.surfaces[0].thickness;
    const double tan_max = std::tan(real.source.field_angles_deg.back() * std::numbers::pi / 180.0);
    GhostReturnOptions o = options({GhostReturnLocation::Stop});
    o.norm_field_scale = 0.0; // nur die Achse
    const GhostReturnResult axis = return_to_ghost_plates(l, o);
    ASSERT_EQ(axis.positions.size(), 1u);
    const double h0 = axis.positions[0].norm_radius / o.norm_margin;
    EXPECT_LE(h0, a * (1.0 + 1e-12));
    EXPECT_GE(h0, 0.95 * a);
    o.norm_field_scale = 1.0;
    const GhostReturnResult full = return_to_ghost_plates(l, o);
    const double h1 = full.positions[0].norm_radius / o.norm_margin;
    EXPECT_LE(h1, a + d * tan_max + 1e-12);
    EXPECT_GE(h1, h0 + 0.9 * d * tan_max);
    // Feldwinkel der Datei bleiben unverändert
    EXPECT_EQ(full.system.source.field_angles_deg, real.source.field_angles_deg);
}

TEST(GhostReturn, ExistingPenaltyAndOrderOptions) {
    // Ein vorhandener Eintrag ghost_penalty ("all") deckt die neuen Platten ab
    SystemDescription real = real_system();
    ConstraintSpec p;
    p.type = ConstraintType::GhostPenalty;
    p.weight = 5.0;
    p.order_factor = 2.0;
    real.constraints->items.push_back(p);
    const LoadedSystem l = prepare_test_system(real);
    GhostReturnOptions o = options({GhostReturnLocation::Front});
    o.order = 4;
    const GhostReturnResult r = return_to_ghost_plates(l, o);
    int penalties = 0;
    for (const ConstraintSpec& s : r.system.constraints->items) {
        if (s.type == ConstraintType::GhostPenalty) {
            ++penalties;
            EXPECT_EQ(s.weight, 5.0);
            EXPECT_EQ(s.order_factor, 2.0);
        }
    }
    EXPECT_EQ(penalties, 1);
    for (const GhostPlate& g : r.system.ghost_plates) {
        EXPECT_EQ(g.coefficients.size(), 4u);
    }
    EXPECT_LE(relative_change(system_merit(real), system_merit(r.system)), kInsertionMeritRelative);
    // Ohne Spaltfixierung bleiben die Dickenvariablen
    o.fix_plate_gaps = false;
    const GhostReturnResult keep = return_to_ghost_plates(l, o);
    EXPECT_FALSE(keep.positions[0].gap_fixed);
    EXPECT_EQ(thickness_gaps(keep.system), thickness_gaps(real));
}

TEST(GhostReturn, LocationsParseAndErrors) {
    EXPECT_EQ(parse_ghost_return_locations("aperture,field"),
              (std::vector<GhostReturnLocation>{GhostReturnLocation::Aperture,
                                                GhostReturnLocation::Field}));
    EXPECT_EQ(
        parse_ghost_return_locations("stop,front"),
        (std::vector<GhostReturnLocation>{GhostReturnLocation::Stop, GhostReturnLocation::Front}));
    EXPECT_THROW(parse_ghost_return_locations("aperture,aperture"), std::invalid_argument);
    EXPECT_THROW(parse_ghost_return_locations("mitte"), std::invalid_argument);
    EXPECT_THROW(parse_ghost_return_locations(""), std::invalid_argument);
    EXPECT_EQ(to_string(GhostReturnLocation::Field), "field");

    const LoadedSystem l = prepare_test_system(real_system());
    GhostReturnOptions o = options({GhostReturnLocation::Aperture});
    // schon Geisterplatten
    const GhostReturnResult once = return_to_ghost_plates(l, o);
    EXPECT_THROW(return_to_ghost_plates(prepare_test_system(once.system), o),
                 std::invalid_argument);
    GhostReturnOptions bad = o;
    bad.glasses.clear();
    EXPECT_THROW(return_to_ghost_plates(l, bad), std::invalid_argument);
    bad = o;
    bad.glasses = {MaterialSpec{}}; // Luft
    EXPECT_THROW(return_to_ghost_plates(l, bad), std::invalid_argument);
    bad = o;
    bad.order = 0;
    EXPECT_THROW(return_to_ghost_plates(l, bad), std::invalid_argument);
    bad = o;
    bad.locations.clear();
    EXPECT_THROW(return_to_ghost_plates(l, bad), std::invalid_argument);
    bad = o;
    bad.penalty_weight = 0.0;
    EXPECT_THROW(return_to_ghost_plates(l, bad), std::invalid_argument);
    // Kein innerer Spalt dick genug und ohne Rückfall (M6-T14b): keine Lage passt
    bad = o;
    bad.min_gap = 100.0;
    bad.thin_gap_fallback = false;
    EXPECT_THROW(return_to_ghost_plates(l, bad), std::invalid_argument);
    // ohne Luftspalt im Inneren hilft auch der Rückfall nicht
    SystemDescription glued = real_system();
    for (const int s : {3, 5}) {
        glued.surfaces[static_cast<std::size_t>(s)].material = catalog_glass("N-BK7");
    }
    bad = o;
    bad.min_gap = 0.0;
    EXPECT_THROW(return_to_ghost_plates(prepare_test_system(glued), bad), std::invalid_argument);
}

// Befund F25 (M6-T14): Nach vielen Einfügungen liegen alle inneren Luftspalte an ihrer unteren
// Schranke (0,1 mm) unter min_gap = 0,2 mm; vorher meldete die Rückkehr „keine Lage passt“ und der
// Zyklus brach ab. Mit dem Rückfall gelten die Lagen unter allen inneren Luftspalten: Orakel ist
// die Spaltwahl des Aberrationsberichts ohne Mindestdicke, Merit unverändert, Notiz je Lage.
TEST(GhostReturn, ThinGapFallbackF25) {
    SystemDescription thin = real_system();
    thin.surfaces[3].thickness = 0.15; // innere Luftspalte (nach 3 und 5), beide < 0,2 mm
    thin.surfaces[5].thickness = 0.12;
    validate_system(thin);
    const LoadedSystem l = prepare_test_system(thin);
    const GhostReturnOptions o =
        options({GhostReturnLocation::Aperture, GhostReturnLocation::Field});
    ASSERT_GT(o.min_gap, 0.15);
    ASSERT_TRUE(o.thin_gap_fallback); // Standard

    // ohne Rückfall: Fehler wie vor M6-T14b, die Meldung nennt den dicksten Spalt
    GhostReturnOptions off = o;
    off.thin_gap_fallback = false;
    try {
        (void)return_to_ghost_plates(l, off);
        ADD_FAILURE() << "kein Fehler ohne Rückfall";
    } catch (const std::invalid_argument& e) {
        const std::string m = e.what();
        EXPECT_NE(m.find("keine Lage passt"), std::string::npos) << m;
        EXPECT_NE(m.find("0.15 mm nach Fläche 3"), std::string::npos) << m;
    }

    const Expected e = expected_gaps(l, 0.0);
    ASSERT_GE(e.aperture, 1);
    ASSERT_GE(e.field, 1);
    const GhostReturnResult r = return_to_ghost_plates(l, o);
    const std::vector<int> gaps =
        e.aperture == e.field ? std::vector<int>{e.aperture} : std::vector<int>{e.aperture, e.field};
    ASSERT_EQ(r.positions.size(), gaps.size());
    for (std::size_t k = 0; k < gaps.size(); ++k) {
        const GhostReturnPosition& p = r.positions[k];
        EXPECT_EQ(p.after_surface, gaps[k]);
        EXPECT_EQ(p.distance, 0.5 * thin.surfaces[static_cast<std::size_t>(gaps[k])].thickness);
        EXPECT_TRUE(p.gap_fixed); // Spalt hatte eine Dickenvariable
        EXPECT_GT(p.norm_radius, 0.0);
        for (const int g : p.plates) {
            EXPECT_EQ(r.system.ghost_plates.at(static_cast<std::size_t>(g)).after_surface,
                      gaps[k]);
        }
    }
    int fallback_notes = 0;
    for (const std::string& n : r.notes) {
        fallback_notes += n.find("F25") != std::string::npos ? 1 : 0;
    }
    EXPECT_EQ(fallback_notes, static_cast<int>(gaps.size()));
    EXPECT_LE(relative_change(system_merit(thin), system_merit(r.system)),
              kInsertionMeritRelative);
    EXPECT_LE(relative_change(r.merit_before, r.merit_after), kInsertionMeritRelative);

    // Passt eine Lage (hier der Blendenspalt), greift der Rückfall nicht: gleiches Ergebnis wie
    // ohne Rückfall
    GhostReturnOptions stop = options({GhostReturnLocation::Stop, GhostReturnLocation::Aperture});
    const GhostReturnResult with = return_to_ghost_plates(l, stop);
    stop.thin_gap_fallback = false;
    const GhostReturnResult without = return_to_ghost_plates(l, stop);
    ASSERT_EQ(with.positions.size(), 1u);
    EXPECT_EQ(with.positions[0].after_surface, 0);
    EXPECT_EQ(with.system.ghost_plates, without.system.ghost_plates);
    EXPECT_EQ(with.notes, without.notes);
    EXPECT_EQ(with.merit_after, without.merit_after);
}

} // namespace
} // namespace mko::test
