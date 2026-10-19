// Dateiformat der L1-Strafe der Geisterkoeffizienten (M4-T6): Variablentyp ghost_coefficient im
// Abschnitt `variables` und Nebenbedingung ghost_penalty im Abschnitt `constraints` – Lesen,
// Schreiben, Auflösen, Negativtests mit JSON-Pfad (docs/dateiformat.md).

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using mko::ConstraintType;
using mko::FormatError;
using mko::ResolvedVariable;
using mko::SurfaceSelection;
using mko::SystemDescription;
using mko::VariableType;

namespace {

const fs::path kSourceDir{MKO_SOURCE_DIR};

// Blende, Linse (Flächen 1, 2), drei Geisterplatten (K = 3, 2, 4) im Spalt nach der Blende;
// VARS und CONS werden ersetzt.
const std::string kBase = R"({
  "format": "mikroptim.system/1",
  "name": "Geistertest",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
  "surfaces": [
    { "type": "stop", "thickness": 4.0, "material": "AIR", "semi_diameter": 3.0 },
    { "type": "sphere", "curvature": 0.1, "thickness": 4.0, "material": { "model": "constant", "n": 1.5 }, "semi_diameter": 4.0 },
    { "type": "sphere", "curvature": -0.05, "thickness": 20.0, "material": "AIR", "semi_diameter": 4.0 }
  ],
  "ghost_plates": [
    { "after_surface": 0, "distance": 1.0, "glass": { "model": "constant", "n": 1.4 }, "norm_radius": 3.0, "order": 3, "coefficients": [0.001, -0.002, 0.0] },
    { "after_surface": 0, "distance": 1.0, "glass": { "model": "constant", "n": 1.7 }, "norm_radius": 3.0, "order": 2, "coefficients": [0.0, 0.0005] },
    { "after_surface": 0, "distance": 2.0, "glass": { "model": "constant", "n": 1.5 }, "norm_radius": 3.0, "order": 4, "coefficients": [0.0, 0.0, 0.0, 0.0] }
  ]VARS
}
)";

std::string make_text(const std::string& variables, const std::string& constraints = "") {
    std::string s = kBase;
    std::string tail;
    if (!variables.empty()) {
        tail += ",\n  \"variables\": { \"items\": [" + variables + "] }";
    }
    if (!constraints.empty()) {
        tail += ",\n  \"constraints\": { \"items\": [" + constraints + "] }";
    }
    s.replace(s.find("VARS"), 4, tail);
    return s;
}

struct Case {
    const char* name;
    std::string text;
    std::string path;
    std::string message;
};

void PrintTo(const Case& c, std::ostream* os) {
    *os << c.name;
}

} // namespace

// Auflösung: Platten in der angegebenen Reihenfolge, je Platte die Ordnungen wie angegeben
// (fehlend oder "all": 1 … K aufsteigend); Schranken und x_typ des Eintrags.
TEST(GhostSchema, ResolvesGhostCoefficients) {
    const SystemDescription s = mko::read_system_json(make_text(R"(
      { "type": "curvature", "surface": 1 },
      { "type": "ghost_coefficient", "plates": [2, 0], "orders": "all", "min": -0.1, "max": 0.1 },
      { "type": "ghost_coefficient", "plate": 1, "order": 2, "typical": 1e-5 })"));
    const std::vector<ResolvedVariable> r = mko::resolve_variables(s);
    ASSERT_EQ(r.size(), 1u + 4u + 3u + 1u);
    EXPECT_EQ(r[0].type, VariableType::Curvature);
    EXPECT_EQ(r[0].plate, -1);
    const int expected[][2] = {{2, 1}, {2, 2}, {2, 3}, {2, 4}, {0, 1}, {0, 2}, {0, 3}, {1, 2}};
    for (int i = 0; i < 8; ++i) {
        const ResolvedVariable& v = r[static_cast<std::size_t>(i + 1)];
        EXPECT_EQ(v.type, VariableType::GhostCoefficient) << i;
        EXPECT_EQ(v.plate, expected[i][0]) << i;
        EXPECT_EQ(v.order, expected[i][1]) << i;
        EXPECT_EQ(v.surface, 0) << i; // after_surface der Platte
    }
    EXPECT_EQ(r[1].lower, -0.1);
    EXPECT_EQ(r[1].upper, 0.1);
    EXPECT_EQ(r[1].typical, mko::kDefaultTypicalGhostCoefficient);
    EXPECT_EQ(r[8].typical, 1e-5);
    EXPECT_EQ(r[8].item, 2);
    EXPECT_EQ(mko::variable_value(s, r[6]), -0.002); // Platte 0, Ordnung 2
    EXPECT_EQ(mko::variable_value(s, r[8]), 0.0005);

    // Ohne Ordnungsfeld: alle Ordnungen; "plates": "all" alle Platten
    const SystemDescription all =
        mko::read_system_json(make_text(R"({ "type": "ghost_coefficient", "plates": "all" })"));
    EXPECT_EQ(mko::resolve_variables(all).size(), 3u + 2u + 4u);
    EXPECT_EQ(mko::to_string(VariableType::GhostCoefficient), "ghost_coefficient");
    EXPECT_EQ(mko::variable_type_from_string("ghost_coefficient"), VariableType::GhostCoefficient);
    EXPECT_EQ(mko::default_typical(VariableType::GhostCoefficient), 1e-4);
}

// Schreiben erhält die Form der Auswahl (Einzelfeld, Liste, "all", fehlend) und die
// Felder der Strafe; Lesen → Schreiben → Lesen ergibt dasselbe Modell.
TEST(GhostSchema, Roundtrip) {
    const std::string text = make_text(
        R"({ "type": "ghost_coefficient", "plates": "all" },
           { "type": "curvature", "surfaces": [1, 2] })",
        R"({ "type": "ghost_penalty", "plates": [0, 2], "weight": 13.6, "smoothing": 1e-7 },
           { "comment": "L1", "type": "ghost_penalty", "plate": 1, "order_weight": 0.5,
             "order_factor": 3.0 })");
    std::vector<std::string> warnings;
    const SystemDescription a = mko::read_system_json(text, &warnings);
    EXPECT_TRUE(warnings.empty());
    const auto& c = a.constraints->items;
    ASSERT_EQ(c.size(), 2u);
    EXPECT_EQ(c[0].type, ConstraintType::GhostPenalty);
    EXPECT_EQ(c[0].plate_selection, SurfaceSelection::List);
    EXPECT_EQ(c[0].plates, (std::vector<int>{0, 2}));
    EXPECT_EQ(c[0].weight_or_default(), 13.6);
    EXPECT_EQ(c[0].order_weight_or_default(), mko::kDefaultGhostOrderWeight);
    EXPECT_EQ(c[0].order_factor_or_default(), mko::kDefaultGhostOrderFactor);
    EXPECT_EQ(c[0].smoothing_or_default(), 1e-7);
    EXPECT_EQ(c[1].plate_selection, SurfaceSelection::One);
    EXPECT_EQ(c[1].weight_or_default(), mko::kDefaultConstraintWeight);
    EXPECT_EQ(c[1].order_weight_or_default(), 0.5);
    EXPECT_EQ(c[1].order_factor_or_default(), 3.0);
    EXPECT_EQ(c[1].smoothing_or_default(), mko::kDefaultGhostSmoothing);
    EXPECT_FALSE(mko::is_paraxial(ConstraintType::GhostPenalty));
    EXPECT_FALSE(mko::constraint_bound(a, c[0]).has_value());

    const std::string text1 = mko::write_system_json(a);
    const SystemDescription b = mko::read_system_json(text1);
    EXPECT_TRUE(a == b);
    EXPECT_EQ(text1, mko::write_system_json(b));
    for (
        const char* expected :
        {R"({ "type": "ghost_coefficient", "plates": "all" })",
         R"({ "type": "ghost_penalty", "weight": 13.6, "plates": [0, 2], "smoothing": 1e-07 })",
         R"({ "type": "ghost_penalty", "plate": 1, "order_weight": 0.5, "order_factor": 3.0, "comment": "L1" })"}) {
        EXPECT_NE(text1.find(expected), std::string::npos) << expected << '\n' << text1;
    }
    const SystemDescription one = mko::read_system_json(
        make_text(R"({ "type": "ghost_coefficient", "plate": 0, "orders": [3, 1] },
                     { "type": "ghost_coefficient", "plate": 1, "order": 2 })"));
    const std::string text2 = mko::write_system_json(one);
    EXPECT_NE(text2.find(R"({ "type": "ghost_coefficient", "plate": 0, "orders": [3, 1] })"),
              std::string::npos)
        << text2;
    EXPECT_NE(text2.find(R"({ "type": "ghost_coefficient", "plate": 1, "order": 2 })"),
              std::string::npos)
        << text2;
    EXPECT_TRUE(mko::read_system_json(text2) == one);
}

// Die Testsysteme der Abnahme sind gültig.
TEST(GhostSchema, ReadsTestSystems) {
    for (const char* file :
         {"einzellinse_geisterplatten_exakt.json", "einzellinse_na05_geisterplatten.json"}) {
        std::vector<std::string> warnings;
        const SystemDescription s =
            mko::read_system_file(kSourceDir / "data" / "systems" / file, &warnings);
        EXPECT_TRUE(warnings.empty()) << file;
        EXPECT_FALSE(s.ghost_plates.empty()) << file;
        EXPECT_GT(mko::resolve_variables(s).size(), s.ghost_plates.size()) << file;
    }
}

class GhostSchemaNegative : public ::testing::TestWithParam<Case> {};

TEST_P(GhostSchemaNegative, ThrowsWithJsonPath) {
    const Case& c = GetParam();
    try {
        (void)mko::read_system_json(c.text);
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), c.path) << e.what();
        EXPECT_NE(std::string(e.what()).find(c.message), std::string::npos) << e.what();
    }
}

namespace {

std::vector<Case> negative_cases() {
    const auto v = [](const std::string& items) { return make_text(items); };
    const auto c = [](const std::string& items) { return make_text("", items); };
    return {
        {"PlatteFehlt", v(R"({ "type": "ghost_coefficient" })"), "variables.items[0].plate",
         "Pflichtfeld fehlt"},
        {"FlaecheStattPlatte", v(R"({ "type": "ghost_coefficient", "surface": 1 })"),
         "variables.items[0].surface", "\"plate\" bzw. \"plates\""},
        {"PlatteExistiertNicht", v(R"({ "type": "ghost_coefficient", "plate": 3 })"),
         "variables.items[0].plate", "Geisterplatte 3 existiert nicht"},
        {"PlattenlisteLeer", v(R"({ "type": "ghost_coefficient", "plates": [] })"),
         "variables.items[0].plates", "mindestens eine Geisterplatte"},
        {"PlateUndPlates", v(R"({ "type": "ghost_coefficient", "plate": 0, "plates": [1] })"),
         "variables.items[0].plates", "nur eines der Felder"},
        {"OrdnungZuGross", v(R"({ "type": "ghost_coefficient", "plate": 1, "order": 3 })"),
         "variables.items[0].order", "Ordnung 3 existiert bei Geisterplatte 1 nicht"},
        {"OrdnungNull", v(R"({ "type": "ghost_coefficient", "plate": 0, "orders": [1, 0] })"),
         "variables.items[0].orders[1]", "Ordnung 0 existiert"},
        {"OrdnungenLeer", v(R"({ "type": "ghost_coefficient", "plate": 0, "orders": [] })"),
         "variables.items[0].orders", "mindestens eine Ordnung"},
        {"PlatteBeiKruemmung", v(R"({ "type": "curvature", "surface": 1, "plate": 0 })"),
         "variables.items[0].plate", "nur bei ghost_coefficient"},
        {"OrdnungBeiDicke", v(R"({ "type": "thickness", "surface": 1, "orders": "all" })"),
         "variables.items[0].orders", "nur bei ghost_coefficient"},
        {"KoeffizientDoppelt", v(R"({ "type": "ghost_coefficient", "plates": "all" },
              { "type": "ghost_coefficient", "plate": 2, "order": 4 })"),
         "variables.items[1].order", "doppelt: ghost_coefficient der Geisterplatte 2, Ordnung 4"},
        {"StartwertAusserhalb",
         v(R"({ "type": "ghost_coefficient", "plate": 0, "order": 2, "min": -0.001 })"),
         "variables.items[0].min", "(ghost_plates[0].coefficients[1]) liegt unter"},
        {"StrafePlatteExistiertNicht", c(R"({ "type": "ghost_penalty", "plates": [0, 5] })"),
         "constraints.items[0].plates[1]", "Geisterplatte 5 existiert nicht"},
        {"StrafePlatteDoppelt",
         c(R"({ "type": "ghost_penalty", "plate": 1 }, { "type": "ghost_penalty" })"),
         "constraints.items[1].type", "Geisterplatte 1 steht schon in constraints.items[0]"},
        {"StrafeMitModus", c(R"({ "type": "ghost_penalty", "mode": "min" })"),
         "constraints.items[0].mode", "keinen Modus"},
        {"GlaettungNull", c(R"({ "type": "ghost_penalty", "smoothing": 0 })"),
         "constraints.items[0].smoothing", "größer als 0"},
        {"OrdnungsfaktorNegativ", c(R"({ "type": "ghost_penalty", "order_factor": -2 })"),
         "constraints.items[0].order_factor", "größer als 0"},
        {"OrdnungsgewichtNull", c(R"({ "type": "ghost_penalty", "order_weight": 0 })"),
         "constraints.items[0].order_weight", "größer als 0"},
        {"StrafgewichtNegativ", c(R"({ "type": "ghost_penalty", "weight": -1 })"),
         "constraints.items[0].weight", "mindestens 0"},
        {"GlaettungBeiPetzval",
         c(R"({ "type": "petzval", "max_curvature": 0.01, "smoothing": 1e-6 })"),
         "constraints.items[0].smoothing", "nur bei ghost_penalty"},
        {"PlattenBeiPetzval", c(R"({ "type": "petzval", "max_curvature": 0.01, "plate": 0 })"),
         "constraints.items[0].plate", "nur bei ghost_penalty"},
        {"StrafeMitReferenzflaeche", c(R"({ "type": "ghost_penalty", "reference_surface": 0 })"),
         "constraints.items[0].reference_surface", "nur bei parfocal_length"},
    };
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Validierung, GhostSchemaNegative, ::testing::ValuesIn(negative_cases()),
                         [](const ::testing::TestParamInfo<Case>& case_info) {
                             return std::string(case_info.param.name);
                         });
