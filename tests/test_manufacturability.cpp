// Herstellbarkeitsprüfungen (M10-T3): frei erfundenes System mit bekannten Befunden (bikonvexe
// Linse mit dünnem Rand, flacher Meniskus, dünne Platte, Luftspalt mit Berührung,
// Hyperhemisphäre); Werte gegen die geschlossene Form der Pfeilhöhe R − √(R² − h²).

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/tolerance/manufacturability.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using mko::CheckSeverity;
using mko::ManufacturabilityCheck;
using mko::ManufacturabilityCheckType;
using mko::ManufacturabilityReport;
using nlohmann::json;

namespace {

json surface(const char* type, double c, double t, const char* material, double sd) {
    json s = {{"type", type}, {"thickness", t}, {"material", material}, {"semi_diameter", sd}};
    if (std::string(type) == "sphere") {
        s["curvature"] = c;
    }
    return s;
}

mko::SystemDescription test_system() {
    json j = {
        {"format", "mikroptim.system/1"},
        {"name", "Herstellbarkeit"},
        {"units", "mm"},
        {"glass_catalogs", {"data/glass/basis.json"}},
        {"spectrum", {{"wavelengths_nm", {546.07}}, {"weights", {1.0}}, {"reference_nm", 546.07}}},
        {"source", {{"type", "collimated"}, {"field_angles_deg", {0.0}}, {"field_weights", {1.0}}}},
        {"surfaces", json::array({
                         surface("stop", 0.0, 1.0, "AIR", 2.0),
                         surface("sphere", 0.1, 2.0, "CAF2", 4.0),  // 1: bikonvex, dünner Rand
                         surface("sphere", -0.1, 0.5, "AIR", 4.0),  // 2
                         surface("sphere", 0.1, 3.0, "CAF2", 3.0),  // 3: flacher Meniskus
                         surface("sphere", 0.09, 1.0, "AIR", 3.0),  // 4: Luftrand < 0
                         surface("plane", 0.0, 0.4, "CAF2", 5.0),   // 5: dünne Platte
                         surface("plane", 0.0, 1.0, "AIR", 5.0),    // 6
                         surface("sphere", 0.3, 4.0, "CAF2", 4.0),  // 7: h/R = 1,2
                         surface("sphere", -0.05, 5.0, "AIR", 4.0), // 8
                     })},
    };
    return mko::read_system_json(j.dump());
}

// Pfeilhöhe aus der geschlossenen Form (unabhängig von mko::sag).
double sag_closed(double c, double h) {
    if (c == 0.0) {
        return 0.0;
    }
    const double r = 1.0 / c;
    return r - (r > 0.0 ? 1.0 : -1.0) * std::sqrt(r * r - h * h);
}

const ManufacturabilityCheck& find(const ManufacturabilityReport& r, ManufacturabilityCheckType t,
                                   int s) {
    for (const ManufacturabilityCheck& c : r.checks) {
        if (c.type == t && c.surface == s) {
            return c;
        }
    }
    throw std::runtime_error("Prüfung fehlt");
}

} // namespace

TEST(Manufacturability, KnownFindingsOfInventedSystem) {
    const mko::SystemDescription sys = test_system();
    const ManufacturabilityReport r = mko::check_manufacturability(sys);
    const double tol = mko::test::kManufacturabilityGeometry;
    using T = ManufacturabilityCheckType;
    // Halbkugelgrad
    EXPECT_NEAR(find(r, T::Hemisphere, 1).value, 0.4, tol);
    EXPECT_EQ(find(r, T::Hemisphere, 1).severity, CheckSeverity::Ok);
    EXPECT_NEAR(find(r, T::Hemisphere, 7).value, 1.2, tol);
    EXPECT_EQ(find(r, T::Hemisphere, 7).severity, CheckSeverity::Error);
    // bikonvexe Linse: Rand 2 − 2 (R − √(R² − h²)) = 0,3303 mm < 0,5
    const ManufacturabilityCheck& e1 = find(r, T::EdgeThickness, 1);
    EXPECT_NEAR(e1.value, 2.0 - 2.0 * sag_closed(0.1, 4.0), tol);
    EXPECT_EQ(e1.severity, CheckSeverity::Warning);
    EXPECT_NEAR(find(r, T::AspectRatio, 1).value, 4.0, tol);
    EXPECT_NEAR(find(r, T::Centering, 1).value, 0.4, tol);
    EXPECT_EQ(find(r, T::Centering, 1).severity, CheckSeverity::Ok);
    // Luftrand nach der Blende und zwischen den Linsen
    EXPECT_NEAR(find(r, T::AirEdge, 0).value, 1.0 + sag_closed(0.1, 4.0), tol);
    EXPECT_NEAR(find(r, T::AirEdge, 2).value, 0.5 + 2.0 * sag_closed(0.1, 4.0), tol);
    // flacher Meniskus: Z = 1,5 · 0,01
    const ManufacturabilityCheck& z3 = find(r, T::Centering, 3);
    EXPECT_NEAR(z3.value, 0.015, tol);
    EXPECT_EQ(z3.severity, CheckSeverity::Warning);
    EXPECT_NEAR(find(r, T::EdgeThickness, 3).value,
                3.0 - sag_closed(0.1, 3.0) + sag_closed(0.09, 3.0), tol);
    // Luftspalt 4: Meniskusrückfläche bei h = 5 ragt in die Platte (Berührung)
    const ManufacturabilityCheck& a4 = find(r, T::AirEdge, 4);
    EXPECT_NEAR(a4.value, 1.0 - sag_closed(0.09, 5.0), tol);
    EXPECT_LT(a4.value, 0.0);
    EXPECT_EQ(a4.severity, CheckSeverity::Error);
    // dünne Platte: alle vier Warnungen
    EXPECT_EQ(find(r, T::CenterThickness, 5).severity, CheckSeverity::Warning);
    EXPECT_NEAR(find(r, T::EdgeThickness, 5).value, 0.4, tol);
    EXPECT_EQ(find(r, T::EdgeThickness, 5).severity, CheckSeverity::Warning);
    EXPECT_NEAR(find(r, T::AspectRatio, 5).value, 25.0, tol);
    EXPECT_EQ(find(r, T::AspectRatio, 5).severity, CheckSeverity::Warning);
    EXPECT_EQ(find(r, T::Centering, 5).value, 0.0);
    EXPECT_EQ(find(r, T::Centering, 5).severity, CheckSeverity::Warning);
    // Hyperhemisphäre: Luftspalt davor und Glas danach geometrisch unmöglich
    EXPECT_EQ(find(r, T::AirEdge, 6).severity, CheckSeverity::Error);
    EXPECT_EQ(find(r, T::EdgeThickness, 7).severity, CheckSeverity::Error);
    EXPECT_EQ(r.errors, 4);
    EXPECT_EQ(r.warnings, 6);
    // nach der letzten Fläche (Arbeitsabstand) nur noch ihr Halbkugelgrad, kein Luftrand
    for (const ManufacturabilityCheck& c : r.checks) {
        EXPECT_TRUE(c.surface < 8 || c.type == T::Hemisphere);
    }
    EXPECT_EQ(r.checks.size(), 6u + 4u * 4u + 4u); // 6 Kugeln, 4 Glasspalte, 4 Luftspalte
}

TEST(Manufacturability, LimitsAreConfigurable) {
    const mko::SystemDescription sys = test_system();
    mko::ManufacturabilityLimits off;
    off.max_hemisphere = 0.0;
    off.min_center_thickness = 0.0;
    off.min_edge_thickness = 0.0;
    off.min_edge_thickness_relative = 0.0;
    off.min_air_edge = 0.0;
    off.max_aspect_ratio = 0.0;
    off.min_centering = 0.0;
    const ManufacturabilityReport r = mko::check_manufacturability(sys, off);
    EXPECT_EQ(r.warnings, 0);
    EXPECT_EQ(r.errors, 4); // geometrisch unmögliche Fälle bleiben
    mko::ManufacturabilityLimits strict;
    strict.max_hemisphere = 0.35; // Flächen 1, 2 (0,4) zusätzlich
    strict.min_air_edge = 2.0;    // Luftrand 0 (1,83) zusätzlich, 2 (2,17) nicht
    const ManufacturabilityReport s = mko::check_manufacturability(sys, strict);
    EXPECT_EQ(s.warnings, 6 + 2 + 1);
    mko::ManufacturabilityLimits bad;
    bad.min_edge_thickness = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)mko::check_manufacturability(sys, bad), std::invalid_argument);
}

// Randdicke durchmesserabhängig (Entscheidung Maintainer 2026-11-19): Grenze
// max(0,5 mm; 0,05 · D) mit D = 2h. Eine große Bikonvexlinse (h = 15 mm, D = 30 mm) mit 1 mm Rand
// wird gewarnt (Grenze 1,5 mm), dieselbe Form auf h = 4 mm verkleinert (Grenze 0,5 mm) nicht.
TEST(Manufacturability, EdgeLimitScalesWithDiameter) {
    const auto lens = [](double h, double t, double c) {
        json j = {
            {"format", "mikroptim.system/1"},
            {"name", "Randdicke"},
            {"units", "mm"},
            {"glass_catalogs", {"data/glass/basis.json"}},
            {"spectrum",
             {{"wavelengths_nm", {546.07}}, {"weights", {1.0}}, {"reference_nm", 546.07}}},
            {"source",
             {{"type", "collimated"}, {"field_angles_deg", {0.0}}, {"field_weights", {1.0}}}},
            {"surfaces", json::array({
                             surface("stop", 0.0, 1.0, "AIR", h),
                             surface("sphere", c, t, "CAF2", h),
                             surface("sphere", -c, 10.0, "AIR", h),
                         })},
        };
        return mko::read_system_json(j.dump());
    };
    using T = ManufacturabilityCheckType;
    const double tol = mko::test::kManufacturabilityGeometry;
    // Mittendicke so, dass der Rand genau 1 mm ist
    const double c_big = 0.02;
    const mko::SystemDescription big = lens(15.0, 1.0 + 2.0 * sag_closed(c_big, 15.0), c_big);
    const ManufacturabilityReport rb = mko::check_manufacturability(big);
    const ManufacturabilityCheck& eb = find(rb, T::EdgeThickness, 1);
    EXPECT_NEAR(eb.value, 1.0, tol);
    EXPECT_DOUBLE_EQ(eb.limit, 1.5);
    EXPECT_EQ(eb.severity, CheckSeverity::Warning);
    EXPECT_EQ(mko::edge_thickness_limit(mko::ManufacturabilityLimits{}, 15.0), 1.5);
    EXPECT_EQ(mko::edge_thickness_limit(mko::ManufacturabilityLimits{}, 4.0), 0.5);
    // kleine Linse: Grenze 0,5 mm (0,05 · 8 mm = 0,4 mm liegt darunter)
    const double c_small = 0.075;
    const mko::SystemDescription small = lens(4.0, 1.0 + 2.0 * sag_closed(c_small, 4.0), c_small);
    const ManufacturabilityCheck es =
        find(mko::check_manufacturability(small), T::EdgeThickness, 1);
    EXPECT_NEAR(es.value, 1.0, tol);
    EXPECT_EQ(es.limit, 0.5);
    EXPECT_EQ(es.severity, CheckSeverity::Ok);
    // konfigurierbar: nur absolut (relativ 0) bzw. 10 % des Durchmessers
    mko::ManufacturabilityLimits absolute;
    absolute.min_edge_thickness_relative = 0.0;
    EXPECT_EQ(find(mko::check_manufacturability(big, absolute), T::EdgeThickness, 1).severity,
              CheckSeverity::Ok);
    mko::ManufacturabilityLimits ten;
    ten.min_edge_thickness_relative = 0.1;
    const ManufacturabilityCheck e10 =
        find(mko::check_manufacturability(small, ten), T::EdgeThickness, 1);
    EXPECT_DOUBLE_EQ(e10.limit, 0.8);
    EXPECT_EQ(e10.severity, CheckSeverity::Ok);
}

// Texte der Befunde (Entscheidung Maintainer 2026-11-19): ein flacher Meniskus ist nicht „nicht
// herstellbar“, nur die Glockenzentrierung ist ungeeignet.
TEST(Manufacturability, FindingMessages) {
    const ManufacturabilityReport r = mko::check_manufacturability(test_system());
    using T = ManufacturabilityCheckType;
    EXPECT_EQ(mko::finding_message(find(r, T::Centering, 3)),
              "Glockenzentrierung ungeeignet (Z < 0,1), andere Zentrierverfahren nötig");
    EXPECT_EQ(mko::finding_message(find(r, T::EdgeThickness, 1)), "Randdicke unter 0,5 mm");
    EXPECT_EQ(mko::finding_message(find(r, T::Centering, 1)), "");
    for (const ManufacturabilityCheck& c : r.checks) {
        EXPECT_EQ(mko::finding_message(c).find("nicht herstellbar"), std::string::npos);
        EXPECT_EQ(mko::finding_message(c).empty(), c.severity == CheckSeverity::Ok);
    }
}
