// Abschnitt `tolerances` der Systembeschreibung (M8-T4, docs/dateiformat.md, „Toleranzen“):
// Lesen und Schreiben (Rundreise), Auflösung in einzelne Toleranzen (Reihenfolge, "all",
// Elemente, Drehpunkt), Passen gegen die Pfeilhöhe der Kugel (unabhängige Rechnung) und die
// Fehlermeldungen mit JSON-Pfad.

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/tolerances.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

using mko::ResolvedTolerance;
using mko::ToleranceType;
using nlohmann::json;

namespace {

mko::SystemDescription read(const json& j) {
    return mko::read_system_json(j.dump());
}

// Pfeilhöhe einer Kugel mit Krümmung c bei Höhe h.
double sag(double c, double h) {
    return c * h * h / (1.0 + std::sqrt(1.0 - c * c * h * h));
}

} // namespace

TEST(ToleranceSchema, RoundTripOfTestSystem) {
    const mko::SystemDescription sys = read(mko::test::tolerance_system_json());
    ASSERT_TRUE(sys.tolerances.has_value());
    EXPECT_EQ(sys.tolerances->items.size(), 8u);
    EXPECT_EQ(sys.tolerances->compensator_or_default(), mko::CompensatorType::WorkingDistance);
    const mko::SystemDescription again = mko::read_system_json(mko::write_system_json(sys));
    EXPECT_EQ(again, sys);
    // Ohne Abschnitt bleibt er beim Schreiben weg
    json j = mko::test::tolerance_system_json();
    j.erase("tolerances");
    const mko::SystemDescription none = read(j);
    EXPECT_FALSE(none.tolerances.has_value());
    EXPECT_EQ(json::parse(mko::write_system_json(none)).count("tolerances"), 0u);
    EXPECT_TRUE(mko::resolve_tolerances(none).empty());
}

TEST(ToleranceSchema, DefaultsOfSection) {
    json j = mko::test::tolerance_system_json();
    j["tolerances"] = {
        {"items", json::array({{{"type", "index"}, {"surface", 1}, {"delta", 1e-3}}})}};
    const mko::SystemDescription sys = read(j);
    EXPECT_EQ(sys.tolerances->distribution_or_default(), mko::ToleranceDistribution::Normal);
    EXPECT_EQ(sys.tolerances->cutoff_sigma_or_default(), 2.0);
    EXPECT_EQ(sys.tolerances->compensator_or_default(), mko::CompensatorType::WorkingDistance);
    const std::vector<ResolvedTolerance> r = mko::resolve_tolerances(sys);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].distribution, mko::ToleranceDistribution::Normal);
    EXPECT_EQ(r[0].cutoff, 2.0);
    EXPECT_EQ(r[0].limit, 1e-3);
}

TEST(ToleranceSchema, ResolveOrderAndCounts) {
    const mko::SystemDescription sys = read(mko::test::tolerance_system_json());
    const std::vector<ResolvedTolerance> r = mko::resolve_tolerances(sys);
    ASSERT_EQ(r.size(), 18u);
    // Reihenfolge: Einträge in Dateireihenfolge, darin die Flächen
    const std::vector<std::pair<ToleranceType, int>> expect{{ToleranceType::Radius, 1},
                                                            {ToleranceType::Radius, 2},
                                                            {ToleranceType::Radius, 3},
                                                            {ToleranceType::Thickness, 0},
                                                            {ToleranceType::Thickness, 1},
                                                            {ToleranceType::Thickness, 2},
                                                            {ToleranceType::Index, 1},
                                                            {ToleranceType::Index, 2},
                                                            {ToleranceType::Abbe, 1},
                                                            {ToleranceType::Abbe, 2},
                                                            {ToleranceType::Decenter, 1},
                                                            {ToleranceType::Decenter, 2},
                                                            {ToleranceType::Decenter, 3},
                                                            {ToleranceType::Tilt, 1},
                                                            {ToleranceType::Tilt, 2},
                                                            {ToleranceType::Tilt, 3},
                                                            {ToleranceType::ElementDecenter, -1},
                                                            {ToleranceType::ElementTilt, -1}};
    int draws = 0;
    for (std::size_t k = 0; k < r.size(); ++k) {
        EXPECT_EQ(r[k].type, expect[k].first) << k;
        EXPECT_EQ(r[k].surface, expect[k].second) << k;
        draws += r[k].draws();
    }
    EXPECT_EQ(draws, 26);
    // Verteilung pro Eintrag (decenter gleichverteilt), sonst aus dem Abschnitt
    EXPECT_EQ(r[10].distribution, mko::ToleranceDistribution::Uniform);
    EXPECT_EQ(r[13].distribution, mko::ToleranceDistribution::Normal);
    // Kippwinkel: 1′ bzw. 2′ in rad
    EXPECT_NEAR(r[13].limit, std::numbers::pi / (180.0 * 60.0), mko::test::kUnitFp64);
    EXPECT_NEAR(r[17].limit, 2.0 * std::numbers::pi / (180.0 * 60.0), mko::test::kUnitFp64);
    // Elemente: "all" findet das Kittglied 1 … 3; Drehpunkt "center" zwischen den Scheiteln
    // z₁ = 2 und z₃ = 2 + 2,5 + 1,2 = 5,7
    EXPECT_EQ(r[16].first, 1);
    EXPECT_EQ(r[16].last, 3);
    EXPECT_EQ(r[17].first, 1);
    EXPECT_EQ(r[17].last, 3);
    EXPECT_NEAR(r[17].pivot_z, 0.5 * (2.0 + 5.7), mko::test::kLengthFp64);
    EXPECT_NEAR(mko::nominal_vertex_z(sys, 3), 5.7, mko::test::kLengthFp64);
}

TEST(ToleranceSchema, FringesMatchSagDifference) {
    // N Passen = Pfeilhöhenänderung N·λ_t/2 über den Halbdurchmesser. Zentrale Differenz der
    // Pfeilhöhe bei c ± Δc (Fehler O(Δc³)) als unabhängige Rechnung.
    json j = mko::test::with_tolerance_items(json::array(
        {{{"type", "radius"}, {"surfaces", json::array({1, 2, 3})}, {"fringes", 4.0}},
         {{"type", "radius"}, {"surface", 0}, {"fringes", 2.0}, {"test_wavelength_nm", 632.8}}}));
    // Fläche 0 ist die Blende (kein sphere): Fehler; deshalb zweiter Eintrag ohne Blende
    EXPECT_THROW(read(j), mko::FormatError);
    j["tolerances"]["items"][1] = {
        {"type", "radius"}, {"surfaces", json::array({})}, {"fringes", 2.0}};
    EXPECT_THROW(read(j), mko::FormatError);
    j["tolerances"]["items"].erase(1);
    j["tolerances"]["items"][0]["test_wavelength_nm"] = 632.8;
    const mko::SystemDescription sys = read(j);
    const std::vector<ResolvedTolerance> r = mko::resolve_tolerances(sys);
    ASSERT_EQ(r.size(), 3u);
    for (const ResolvedTolerance& t : r) {
        ASSERT_TRUE(t.radius_fringes);
        const mko::Surface& s = sys.surfaces[static_cast<std::size_t>(t.surface)];
        const double h = s.semi_diameter;
        const double ds = 0.5 * (sag(s.curvature + t.limit, h) - sag(s.curvature - t.limit, h));
        const double expect = 4.0 * 0.5 * 632.8e-6;
        EXPECT_NEAR(ds / expect, 1.0, mko::test::kFringeSag) << t.surface;
    }
    // Flächen vom Typ plane haben keine Radiustoleranz ("all" überspringt sie, die explizite
    // Angabe ist ein Fehler)
    json p = mko::test::tolerance_system_json();
    p["surfaces"][3]["type"] = "plane";
    p["surfaces"][3].erase("curvature");
    p["tolerances"]["items"] =
        json::array({{{"type", "radius"}, {"surfaces", "all"}, {"fringes", 1.0}}});
    const std::vector<ResolvedTolerance> rp = mko::resolve_tolerances(read(p));
    ASSERT_EQ(rp.size(), 2u); // Flächen 1 und 2; die Planfläche ist kein sphere
    p["tolerances"]["items"][0]["surfaces"] = json::array({3});
    EXPECT_THROW(read(p), mko::FormatError);
}

TEST(ToleranceSchema, RelativeRadiusNeedsFiniteRadius) {
    json j = mko::test::tolerance_system_json();
    j["surfaces"][3]["curvature"] = 0.0;
    j["tolerances"]["items"] =
        json::array({{{"type", "radius"}, {"surfaces", "all"}, {"relative", 0.001}}});
    // "all" überspringt c = 0 bei relativer Toleranz
    const std::vector<ResolvedTolerance> r = mko::resolve_tolerances(read(j));
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[1].surface, 2);
    EXPECT_FALSE(r[1].radius_fringes);
    j["tolerances"]["items"][0]["surfaces"] = json::array({3});
    try {
        (void)read(j);
        FAIL() << "keine Ausnahme";
    } catch (const mko::FormatError& e) {
        EXPECT_EQ(e.json_path(), "tolerances.items[0].surfaces[0]");
    }
}

TEST(ToleranceSchema, AllElementsAndPivots) {
    json j = mko::test::with_tolerance_items(json::array(
        {{{"type", "element_tilt"}, {"elements", "all"}, {"deg", 0.1}, {"pivot", "first"}},
         {{"type", "element_decenter"}, {"element", json::array({1, 2})}, {"delta", 0.01}}}));
    mko::SystemDescription sys = read(j);
    std::vector<ResolvedTolerance> r = mko::resolve_tolerances(sys);
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0].pivot_z, 2.0);
    EXPECT_NEAR(r[0].limit, 0.1 * std::numbers::pi / 180.0, mko::test::kUnitFp64);
    j["tolerances"]["items"][0]["pivot"] = "last";
    EXPECT_NEAR(mko::resolve_tolerances(read(j))[0].pivot_z, 5.7, mko::test::kLengthFp64);
    j["tolerances"]["items"][0]["pivot"] = 1.25;
    sys = read(j);
    EXPECT_NEAR(mko::resolve_tolerances(sys)[0].pivot_z, 3.25, mko::test::kLengthFp64);
    EXPECT_EQ(mko::read_system_json(mko::write_system_json(sys)), sys);
    EXPECT_EQ(mko::all_elements(sys), (std::vector<std::array<int, 2>>{{1, 3}}));
}

namespace {

struct BadItem {
    json item;
    std::string path;
};

} // namespace

TEST(ToleranceSchema, ErrorsCarryJsonPath) {
    const std::vector<BadItem> cases{
        {{{"type", "wobble"}, {"surface", 1}}, "tolerances.items[0].type"},
        {{{"type", "thickness"}, {"surface", 1}}, "tolerances.items[0].delta"},
        {{{"type", "thickness"}, {"surface", 3}, {"delta", 0.01}}, "tolerances.items[0].surface"},
        {{{"type", "thickness"}, {"delta", 0.01}}, "tolerances.items[0].surfaces"},
        {{{"type", "thickness"}, {"surface", 9}, {"delta", 0.01}}, "tolerances.items[0].surface"},
        {{{"type", "thickness"}, {"surface", 1}, {"delta", -0.01}}, "tolerances.items[0].delta"},
        {{{"type", "index"}, {"surface", 0}, {"delta", 1e-3}}, "tolerances.items[0].surface"},
        {{{"type", "index"}, {"surface", 1}, {"delta", 1e-3}, {"arcmin", 1.0}},
         "tolerances.items[0].arcmin"},
        {{{"type", "abbe"}, {"surface", 1}}, "tolerances.items[0].relative"},
        {{{"type", "abbe"}, {"surface", 1}, {"relative", 1.0}}, "tolerances.items[0].relative"},
        {{{"type", "radius"}, {"surface", 1}, {"relative", 0.01}, {"fringes", 1.0}},
         "tolerances.items[0].fringes"},
        {{{"type", "radius"}, {"surface", 1}}, "tolerances.items[0].relative"},
        {{{"type", "radius"}, {"surface", 1}, {"relative", 0.01}, {"test_wavelength_nm", 600.0}},
         "tolerances.items[0].test_wavelength_nm"},
        {{{"type", "decenter"}, {"surface", 0}, {"delta", 0.01}}, "tolerances.items[0].surface"},
        {{{"type", "tilt"}, {"surface", 1}}, "tolerances.items[0].arcmin"},
        {{{"type", "tilt"}, {"surface", 1}, {"arcmin", 1.0}, {"deg", 0.1}},
         "tolerances.items[0].deg"},
        {{{"type", "tilt"}, {"surface", 1}, {"deg", 90.0}}, "tolerances.items[0].deg"},
        {{{"type", "tilt"}, {"surface", 1}, {"arcmin", 1.0}, {"pivot", "first"}},
         "tolerances.items[0].pivot"},
        {{{"type", "element_tilt"}, {"surface", 1}, {"arcmin", 1.0}},
         "tolerances.items[0].surface"},
        {{{"type", "element_tilt"}, {"arcmin", 1.0}}, "tolerances.items[0].elements"},
        {{{"type", "element_decenter"}, {"element", json::array({0, 3})}, {"delta", 0.01}},
         "tolerances.items[0].element"},
        {{{"type", "element_decenter"},
          {"elements", json::array({json::array({1, 3}), json::array({2, 2})})},
          {"delta", 0.01}},
         "tolerances.items[0].elements[1]"},
        {{{"type", "element_tilt"},
          {"element", json::array({1, 3})},
          {"arcmin", 1.0},
          {"pivot", "middle"}},
         "tolerances.items[0].pivot"},
        {{{"type", "decenter"}, {"surface", 1}, {"delta", 0.01}, {"cutoff_sigma", 0.2}},
         "tolerances.items[0].cutoff_sigma"},
        {{{"type", "decenter"}, {"surface", 1}, {"delta", 0.01}, {"distribution", "cauchy"}},
         "tolerances.items[0].distribution"},
        {{{"type", "decenter"}, {"surfaces", json::array({1, 2, 1})}, {"delta", 0.01}},
         "tolerances.items[0].surfaces[2]"},
    };
    for (const BadItem& c : cases) {
        const json j = mko::test::with_tolerance_items(json::array({c.item}));
        try {
            (void)read(j);
            ADD_FAILURE() << "keine Ausnahme für " << c.item.dump();
        } catch (const mko::FormatError& e) {
            EXPECT_EQ(e.json_path(), c.path) << c.item.dump() << ": " << e.what();
        }
    }
    // Abschnitt: Abschneiden, Kompensator, fehlende items
    json j = mko::test::tolerance_system_json();
    j["tolerances"]["cutoff_sigma"] = 7.0;
    EXPECT_THROW(read(j), mko::FormatError);
    j = mko::test::tolerance_system_json();
    j["tolerances"]["compensator"] = "focus";
    EXPECT_THROW(read(j), mko::FormatError);
    j = mko::test::tolerance_system_json();
    j["tolerances"].erase("items");
    EXPECT_THROW(read(j), mko::FormatError);
    // Dieselbe Fläche in zwei Einträgen desselben Typs; verschiedene Typen sind erlaubt
    j = mko::test::with_tolerance_items(
        json::array({{{"type", "decenter"}, {"surface", 2}, {"delta", 0.01}},
                     {{"type", "tilt"}, {"surface", 2}, {"arcmin", 1.0}},
                     {{"type", "decenter"}, {"surfaces", json::array({3, 2})}, {"delta", 0.02}}}));
    try {
        (void)read(j);
        FAIL() << "keine Ausnahme";
    } catch (const mko::FormatError& e) {
        EXPECT_EQ(e.json_path(), "tolerances.items[2].surfaces[1]");
    }
}

TEST(ToleranceSchema, UnknownFieldsWarn) {
    json j = mko::test::tolerance_system_json();
    j["tolerances"]["items"][0]["colour"] = "blue";
    j["tolerances"]["seed"] = 3;
    std::vector<std::string> warnings;
    (void)mko::read_system_json(j.dump(), &warnings);
    ASSERT_EQ(warnings.size(), 2u);
}
