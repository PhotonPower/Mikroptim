// Abschnitt `variables` der Systemdatei (M3-T1): Lesen, Schreiben, Auflösen, Negativtests mit
// JSON-Pfad.

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using mko::FormatError;
using mko::ResolvedVariable;
using mko::SurfaceSelection;
using mko::SystemDescription;
using mko::VariableSpec;
using mko::VariableType;

namespace {

const fs::path kSourceDir{MKO_SOURCE_DIR};
constexpr double kInf = std::numeric_limits<double>::infinity();

// Basissystem mit Blende, zwei Sphären, Planfläche und Sphäre (Flächen 0 … 4); der Platzhalter
// VARS wird durch den Abschnitt `variables` ersetzt.
const std::string kBase = R"({
  "format": "mikroptim.system/1",
  "name": "Variablentest",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
  "surfaces": [
    { "type": "stop", "thickness": 2.0, "material": "AIR", "semi_diameter": 3.0 },
    { "type": "sphere", "curvature": 0.1, "thickness": 4.0, "material": { "model": "constant", "n": 1.5 }, "semi_diameter": 4.0 },
    { "type": "sphere", "curvature": -0.05, "thickness": 1.0, "material": "AIR", "semi_diameter": 4.0 },
    { "type": "plane", "thickness": 3.0, "material": { "model": "constant", "n": 1.6 }, "semi_diameter": 4.0 },
    { "type": "sphere", "curvature": -0.08, "thickness": 20.0, "material": "AIR", "semi_diameter": 4.0 }
  ]VARS
}
)";

std::string with_variables(const std::string& items) {
    std::string s = kBase;
    s.replace(s.find("VARS"), 4, ",\n  \"variables\": { \"items\": [" + items + "] }");
    return s;
}

std::string with_raw_variables(const std::string& raw) {
    std::string s = kBase;
    s.replace(s.find("VARS"), 4, ",\n  \"variables\": " + raw);
    return s;
}

} // namespace

// Beispieldatei aus data/systems: Kurzform "all", Schranken, typische Größe, Kommentar.
TEST(VariablesSchema, ReadsExampleFile) {
    std::vector<std::string> warnings;
    const SystemDescription s = mko::read_system_file(
        kSourceDir / "data" / "systems" / "einzellinse_bk7_variablen.json", &warnings);
    EXPECT_TRUE(warnings.empty());
    ASSERT_TRUE(s.variables.has_value());
    ASSERT_EQ(s.variables->items.size(), 3u);
    const VariableSpec& c = s.variables->items[0];
    EXPECT_EQ(c.type, VariableType::Curvature);
    EXPECT_EQ(c.selection, SurfaceSelection::All);
    EXPECT_TRUE(c.surfaces.empty());
    EXPECT_EQ(c.min, -0.05);
    EXPECT_EQ(c.max, 0.05);
    EXPECT_FALSE(c.typical.has_value());
    EXPECT_EQ(s.variables->items[1].comment, "Mittendicke");

    const std::vector<ResolvedVariable> r = mko::resolve_variables(s);
    ASSERT_EQ(r.size(), 4u);
    EXPECT_EQ(r[0], (ResolvedVariable{VariableType::Curvature, 1, -0.05, 0.05,
                                      mko::kDefaultTypicalCurvature, 0}));
    EXPECT_EQ(r[1], (ResolvedVariable{VariableType::Curvature, 2, -0.05, 0.05,
                                      mko::kDefaultTypicalCurvature, 0}));
    EXPECT_EQ(r[2], (ResolvedVariable{VariableType::Thickness, 1, 2.0, 12.0,
                                      mko::kDefaultTypicalThickness, 1}));
    EXPECT_EQ(r[3], (ResolvedVariable{VariableType::WorkingDistance, 2, 50.0, 150.0, 10.0, 2}));
    EXPECT_EQ(mko::variable_value(s, r[2]), 6.0);
    EXPECT_EQ(mko::variable_value(s, r[3]), 99.05);
}

// Alle Formen überstehen den Roundtrip; der Schreiber gibt jeden Eintrag einzeilig aus.
TEST(VariablesSchema, RoundtripAllForms) {
    const std::string text = with_variables(R"(
    { "type": "curvature", "surface": 1, "min": -0.5, "max": 0.5, "typical": 0.2, "comment": "vorn" },
    { "type": "curvature", "surfaces": [4, 2] },
    { "type": "thickness", "surfaces": "all", "min": 0.5 },
    { "type": "working_distance", "max": 25 })");
    std::vector<std::string> warnings;
    const SystemDescription a = mko::read_system_json(text, &warnings);
    EXPECT_TRUE(warnings.empty());
    ASSERT_TRUE(a.variables.has_value());
    const auto& items = a.variables->items;
    ASSERT_EQ(items.size(), 4u);
    EXPECT_EQ(items[0].selection, SurfaceSelection::One);
    EXPECT_EQ(items[0].surfaces, std::vector<int>{1});
    EXPECT_EQ(items[0].typical, 0.2);
    EXPECT_EQ(items[1].selection, SurfaceSelection::List);
    EXPECT_EQ(items[1].surfaces, (std::vector<int>{4, 2}));
    EXPECT_EQ(items[2].selection, SurfaceSelection::All);
    EXPECT_EQ(items[3].selection, SurfaceSelection::None);
    EXPECT_FALSE(items[3].min.has_value());
    EXPECT_EQ(items[3].max, 25.0);

    const std::string text1 = mko::write_system_json(a);
    const SystemDescription b = mko::read_system_json(text1);
    EXPECT_TRUE(a == b);
    EXPECT_EQ(text1, mko::write_system_json(b));
    EXPECT_NE(text1.find(R"(    "items": [
      { "type": "curvature", "surface": 1, "min": -0.5, "max": 0.5, "typical": 0.2, "comment": "vorn" },
      { "type": "curvature", "surfaces": [4, 2] },
      { "type": "thickness", "surfaces": "all", "min": 0.5 },
      { "type": "working_distance", "max": 25.0 }
    ])"),
              std::string::npos)
        << text1;

    // Auflösung: Reihenfolge der Einträge, Liste wie angegeben, "all" für Dicken ohne die
    // letzte Fläche, offene Schranken ±∞, Standardwerte für x_typ.
    const std::vector<ResolvedVariable> r = mko::resolve_variables(a);
    ASSERT_EQ(r.size(), 8u);
    const std::vector<std::pair<VariableType, int>> expected = {
        {VariableType::Curvature, 1}, {VariableType::Curvature, 4},
        {VariableType::Curvature, 2}, {VariableType::Thickness, 0},
        {VariableType::Thickness, 1}, {VariableType::Thickness, 2},
        {VariableType::Thickness, 3}, {VariableType::WorkingDistance, 4}};
    for (std::size_t k = 0; k < r.size(); ++k) {
        EXPECT_EQ(r[k].type, expected[k].first) << k;
        EXPECT_EQ(r[k].surface, expected[k].second) << k;
    }
    EXPECT_EQ(r[0].typical, 0.2);
    EXPECT_EQ(r[1].lower, -kInf);
    EXPECT_EQ(r[1].upper, kInf);
    EXPECT_EQ(r[1].typical, mko::kDefaultTypicalCurvature);
    EXPECT_EQ(r[3].lower, 0.5);
    EXPECT_EQ(r[3].upper, kInf);
    EXPECT_EQ(r[3].item, 2);
    EXPECT_EQ(r[7].lower, -kInf);
    EXPECT_EQ(r[7].upper, 25.0);
    EXPECT_EQ(r[7].typical, mko::kDefaultTypicalWorkingDistance);
}

// Eine Planfläche, die gekrümmt werden darf, steht als Sphäre mit Krümmung 0 in der Datei.
TEST(VariablesSchema, PlaneAsSphereWithZeroCurvature) {
    std::string text = with_variables(R"({ "type": "curvature", "surface": 3 })");
    text.replace(text.find(R"("type": "plane",)"), 16, R"("type": "sphere", "curvature": 0,)");
    const SystemDescription s = mko::read_system_json(text);
    const auto r = mko::resolve_variables(s);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(mko::variable_value(s, r[0]), 0.0);
    EXPECT_NE(mko::write_system_json(s).find(R"("curvature": 0.0,)"), std::string::npos);
}

TEST(VariablesSchema, EmptyItemsAndNoSection) {
    const SystemDescription a = mko::read_system_json(with_variables(""));
    ASSERT_TRUE(a.variables.has_value());
    EXPECT_TRUE(a.variables->items.empty());
    EXPECT_TRUE(mko::resolve_variables(a).empty());
    EXPECT_TRUE(a == mko::read_system_json(mko::write_system_json(a)));

    std::string plain = kBase;
    plain.replace(plain.find("VARS"), 4, "");
    const SystemDescription b = mko::read_system_json(plain);
    EXPECT_FALSE(b.variables.has_value());
    EXPECT_TRUE(mko::resolve_variables(b).empty());
    EXPECT_EQ(mko::write_system_json(b).find("variables"), std::string::npos);
}

TEST(VariablesSchema, UnknownFieldsProduceWarnings) {
    const std::string text = with_raw_variables(
        R"({ "items": [{ "type": "thickness", "surface": 1, "maximum": 3 }], "defaults": {} })");
    std::vector<std::string> warnings;
    const SystemDescription s = mko::read_system_json(text, &warnings);
    ASSERT_EQ(warnings.size(), 2u);
    const auto has = [&](const std::string& needle) {
        return std::any_of(warnings.begin(), warnings.end(), [&](const std::string& w) {
            return w.find(needle) != std::string::npos;
        });
    };
    EXPECT_TRUE(has("variables.items[0].maximum: unbekanntes Feld"));
    EXPECT_TRUE(has("variables.defaults: unbekanntes Feld"));
    EXPECT_FALSE(s.variables->items[0].max.has_value());
}

TEST(VariablesSchema, TypeNames) {
    for (const VariableType t :
         {VariableType::Curvature, VariableType::Thickness, VariableType::WorkingDistance}) {
        EXPECT_EQ(mko::variable_type_from_string(mko::to_string(t)), t);
    }
    EXPECT_FALSE(mko::variable_type_from_string("glass").has_value());
    EXPECT_EQ(mko::default_typical(VariableType::Thickness), mko::kDefaultTypicalThickness);
}

// Der Schreiber validiert auch den Abschnitt `variables` (keine unlesbaren Dateien).
TEST(VariablesSchema, WriterRejectsInvalidModel) {
    const SystemDescription s =
        mko::read_system_json(with_variables(R"({ "type": "curvature", "surface": 1 })"));
    const auto expect_path = [](const SystemDescription& t, const std::string& path) {
        try {
            (void)mko::write_system_json(t);
            FAIL() << "keine Ausnahme";
        } catch (const FormatError& e) {
            EXPECT_EQ(e.json_path(), path) << e.what();
        }
    };
    {
        SystemDescription t = s;
        t.variables->items[0].surfaces = {1, 2}; // One mit zwei Flächen
        expect_path(t, "variables.items[0].surface");
    }
    {
        SystemDescription t = s;
        t.variables->items[0].min = -kInf;
        expect_path(t, "variables.items[0].min");
    }
    {
        SystemDescription t = s;
        t.variables->items[0].selection = SurfaceSelection::All;
        expect_path(t, "variables.items[0].surfaces");
    }
    {
        SystemDescription t = s;
        t.surfaces[1].curvature = 0.7; // Startwert über der Schranke
        t.variables->items[0].max = 0.5;
        expect_path(t, "variables.items[0].max");
    }
}

// --------------------------------------------------------------------------
// Negativtests: Meldung enthält den JSON-Pfad
// --------------------------------------------------------------------------

namespace {

struct VariablesCase {
    const char* name;
    std::string text;
    std::string path;
    std::string message;
};

void PrintTo(const VariablesCase& c, std::ostream* os) {
    *os << c.name;
}

std::vector<VariablesCase> variables_cases() {
    const auto v = with_variables;
    return {
        {"AbschnittKeinObjekt", with_raw_variables("[]"), "variables", "falscher Typ"},
        {"ItemsFehlt", with_raw_variables("{}"), "variables.items", "Pflichtfeld fehlt"},
        {"ItemsKeineListe", with_raw_variables(R"({ "items": {} })"), "variables.items",
         "falscher Typ"},
        {"EintragKeinObjekt", v("1"), "variables.items[0]", "falscher Typ"},
        {"TypFehlt", v(R"({ "surface": 1 })"), "variables.items[0].type", "Pflichtfeld fehlt"},
        {"UnbekannterTyp", v(R"({ "type": "glass", "surface": 1 })"), "variables.items[0].type",
         "unbekannter Variablentyp \"glass\""},
        {"SurfaceUndSurfaces", v(R"({ "type": "curvature", "surface": 1, "surfaces": [2] })"),
         "variables.items[0].surfaces", "nur eines der Felder"},
        {"FlaecheFehlt", v(R"({ "type": "thickness", "min": 1 })"), "variables.items[0].surface",
         "Pflichtfeld fehlt"},
        {"FlaecheAlsKommazahl", v(R"({ "type": "curvature", "surface": 1.5 })"),
         "variables.items[0].surface", "nichtnegative ganze Zahl erwartet"},
        {"FlaecheAlsZeichenkette", v(R"({ "type": "curvature", "surface": "1" })"),
         "variables.items[0].surface", "falscher Typ: erwartet Zahl"},
        {"SurfacesFalscheZeichenkette", v(R"({ "type": "curvature", "surfaces": "alle" })"),
         "variables.items[0].surfaces", "falscher Typ"},
        {"SurfacesEintragKeineZahl", v(R"({ "type": "curvature", "surfaces": [1, null] })"),
         "variables.items[0].surfaces[1]", "falscher Typ"},
        {"SurfacesLeer", v(R"({ "type": "curvature", "surfaces": [] })"),
         "variables.items[0].surfaces", "mindestens eine Fläche"},
        {"FlaecheExistiertNicht", v(R"({ "type": "curvature", "surface": 5 })"),
         "variables.items[0].surface", "Fläche 5 existiert nicht"},
        {"NegativerFlaechenindex", v(R"({ "type": "thickness", "surface": -1 })"),
         "variables.items[0].surface", "nichtnegative ganze Zahl erwartet"},
        {"RiesigerFlaechenindex", v(R"({ "type": "thickness", "surface": 99999999999 })"),
         "variables.items[0].surface", "nichtnegative ganze Zahl erwartet"},
        {"ListenflaecheExistiertNicht", v(R"({ "type": "curvature", "surfaces": [1, 7] })"),
         "variables.items[0].surfaces[1]", "Fläche 7 existiert nicht"},
        {"KruemmungAnBlende", v(R"({ "type": "curvature", "surface": 0 })"),
         "variables.items[0].surface", "Blende"},
        {"KruemmungAnPlanflaeche", v(R"({ "type": "curvature", "surfaces": [1, 3] })"),
         "variables.items[0].surfaces[1]", "als \"sphere\" mit \"curvature\": 0"},
        {"DickeDerLetztenFlaeche", v(R"({ "type": "thickness", "surface": 4 })"),
         "variables.items[0].surface", "\"working_distance\" verwenden"},
        {"ArbeitsabstandMitFlaeche", v(R"({ "type": "working_distance", "surface": 4 })"),
         "variables.items[0].surface", "kein \"surface\""},
        {"ArbeitsabstandMitFlaechenliste",
         v(R"({ "type": "working_distance", "surfaces": "all" })"), "variables.items[0].surfaces",
         "kein \"surfaces\""},
        {"KruemmungDoppelt",
         v(R"({ "type": "curvature", "surface": 1 }, { "type": "curvature", "surfaces": [2, 1] })"),
         "variables.items[1].surfaces[1]",
         "doppelt: curvature von Fläche 1 steht schon in "
         "variables.items[0]"},
        {"DoppeltUeberAll",
         v(R"({ "type": "thickness", "surfaces": "all" }, { "type": "thickness", "surface": 2 })"),
         "variables.items[1].surface", "doppelt"},
        {"ArbeitsabstandDoppelt",
         v(R"({ "type": "working_distance" }, { "type": "working_distance", "min": 1 })"),
         "variables.items[1].type", "doppelt: working_distance steht schon"},
        {"MinGroesserMax", v(R"({ "type": "thickness", "surface": 1, "min": 5, "max": 3 })"),
         "variables.items[0].max", "größer als die untere Schranke"},
        {"MinGleichMax", v(R"({ "type": "thickness", "surface": 1, "min": 4, "max": 4 })"),
         "variables.items[0].max", "größer als die untere Schranke"},
        {"MinAlsZeichenkette", v(R"({ "type": "thickness", "surface": 1, "min": "1" })"),
         "variables.items[0].min", "falscher Typ"},
        {"TypischeGroesseNull", v(R"({ "type": "thickness", "surface": 1, "typical": 0 })"),
         "variables.items[0].typical", "größer als 0"},
        {"StartwertUnterMin", v(R"({ "type": "thickness", "surfaces": [0, 1], "min": 3 })"),
         "variables.items[0].min", "Startwert 2 (surfaces[0].thickness) liegt unter"},
        {"StartwertUeberMax", v(R"({ "type": "curvature", "surface": 1, "max": 0.05 })"),
         "variables.items[0].max", "Startwert 0.1 (surfaces[1].curvature) liegt über"},
        {"KommentarKeineZeichenkette", v(R"({ "type": "thickness", "surface": 1, "comment": 3 })"),
         "variables.items[0].comment", "falscher Typ"},
    };
}

} // namespace

class VariablesSchemaNegative : public ::testing::TestWithParam<VariablesCase> {};

TEST_P(VariablesSchemaNegative, ThrowsWithJsonPath) {
    const VariablesCase& c = GetParam();
    try {
        (void)mko::read_system_json(c.text);
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), c.path) << e.what();
        EXPECT_NE(std::string(e.what()).find(c.path + ": "), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find(c.message), std::string::npos) << e.what();
    }
}

INSTANTIATE_TEST_SUITE_P(Validierung, VariablesSchemaNegative,
                         ::testing::ValuesIn(variables_cases()),
                         [](const ::testing::TestParamInfo<VariablesCase>& case_info) {
                             return std::string(case_info.param.name);
                         });

// Ohne "sphere" findet "all" keine Krümmung.
TEST(VariablesSchema, AllWithoutSpheresIsRejected) {
    const SystemDescription s =
        mko::read_system_file(kSourceDir / "data" / "systems" / "planplatte_bk7.json");
    SystemDescription t = s;
    VariableSpec spec;
    spec.type = VariableType::Curvature;
    spec.selection = SurfaceSelection::All;
    t.variables = mko::Variables{{spec}};
    try {
        mko::validate_system(t);
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "variables.items[0].surfaces");
        EXPECT_NE(std::string(e.what()).find("keine Fläche vom Typ \"sphere\""), std::string::npos);
    }
}
