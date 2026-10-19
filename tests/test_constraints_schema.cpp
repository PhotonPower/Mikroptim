// Abschnitt `constraints` der Systemdatei (M3-T3, Petzval M4-T4): Lesen, Schreiben, Roundtrip,
// Negativtests mit JSON-Pfad.

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using mko::ConstraintMode;
using mko::ConstraintSpec;
using mko::ConstraintType;
using mko::FormatError;
using mko::SystemDescription;

namespace {

const fs::path kSourceDir{MKO_SOURCE_DIR};

// Basissystem (Flächen 0 … 3) mit allen Sollwerten; TARGET und CONS werden ersetzt.
const std::string kBase = R"({
  "format": "mikroptim.system/1",
  "name": "Nebenbedingungstest",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
  TARGET
  "surfaces": [
    { "type": "stop", "thickness": 2.0, "material": "AIR", "semi_diameter": 3.0 },
    { "type": "sphere", "curvature": 0.1, "thickness": 4.0, "material": { "model": "constant", "n": 1.5 }, "semi_diameter": 4.0 },
    { "type": "sphere", "curvature": -0.05, "thickness": 1.0, "material": "AIR", "semi_diameter": 4.0 },
    { "type": "plane", "thickness": 20.0, "material": "AIR", "semi_diameter": 4.0 }
  ]CONS
}
)";

const std::string kAllTargets =
    R"("target": { "focal_length": 10.0, "parfocal_length": 27.0, "working_distance": 19.5 },)";

std::string make_text(const std::string& constraints_raw, const std::string& target = kAllTargets) {
    std::string s = kBase;
    s.replace(s.find("TARGET"), 6, target);
    s.replace(s.find("CONS"), 4,
              constraints_raw.empty() ? "" : ",\n  \"constraints\": " + constraints_raw);
    return s;
}

std::string with_items(const std::string& items, const std::string& target = kAllTargets) {
    return make_text("{ \"items\": [" + items + "] }", target);
}

// Erwartet FormatError mit genau diesem JSON-Pfad.
void expect_error(const std::string& text, const std::string& path) {
    try {
        (void)mko::read_system_json(text);
        ADD_FAILURE() << "keine Ausnahme, erwartet Fehler unter " << path;
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), path) << e.what();
    }
}

} // namespace

// Beispieldatei aus data/systems: Sollwerte in target, drei Typen, Kommentar.
TEST(ConstraintsSchema, ReadsExampleFile) {
    std::vector<std::string> warnings;
    const SystemDescription s = mko::read_system_file(
        kSourceDir / "data" / "systems" / "einzellinse_bk7_nebenbedingungen.json", &warnings);
    EXPECT_TRUE(warnings.empty());
    ASSERT_TRUE(s.constraints.has_value());
    const auto& items = s.constraints->items;
    ASSERT_EQ(items.size(), 3u);
    EXPECT_EQ(items[0].type, ConstraintType::FocalLength);
    EXPECT_EQ(items[0].weight, 100.0);
    EXPECT_EQ(items[0].mode_or_default(), ConstraintMode::Equal);
    EXPECT_EQ(items[1].type, ConstraintType::ParfocalLength);
    EXPECT_EQ(items[1].reference_surface, 0);
    EXPECT_EQ(items[1].offset, 5.0);
    EXPECT_EQ(items[1].weight_or_default(), 1.0);
    EXPECT_TRUE(items[1].comment.has_value());
    EXPECT_EQ(items[2].type, ConstraintType::WorkingDistance);
    EXPECT_EQ(items[2].mode, ConstraintMode::Min);
    EXPECT_EQ(mko::constraint_target(s, ConstraintType::FocalLength), 100.0);
    EXPECT_EQ(mko::constraint_target(s, ConstraintType::ParfocalLength), 115.0);
    EXPECT_EQ(mko::constraint_target(s, ConstraintType::WorkingDistance), 95.0);
    EXPECT_EQ(mko::constraint_target(s, ConstraintType::ParaxialFocus), 0.0);
}

// Alle Typen und Felder überstehen den Roundtrip; fehlende Felder bleiben fehlend (Standardwerte
// werden nicht geschrieben), die Reihenfolge der Felder ist fest.
TEST(ConstraintsSchema, RoundtripAllFields) {
    const std::string text = with_items(R"(
    { "comment": "Brennweite", "weight": 2.5, "type": "focal_length" },
    { "type": "parfocal_length", "mode": "max", "reference_surface": 1.0, "offset": -1.5 },
    { "type": "working_distance", "mode": "min", "weight": 0 },
    { "type": "paraxial_focus", "mode": "equal" })");
    std::vector<std::string> warnings;
    const SystemDescription a = mko::read_system_json(text, &warnings);
    EXPECT_TRUE(warnings.empty());
    ASSERT_TRUE(a.constraints.has_value());
    const auto& items = a.constraints->items;
    ASSERT_EQ(items.size(), 4u);
    EXPECT_EQ(items[0], (ConstraintSpec{ConstraintType::FocalLength, std::nullopt, 2.5,
                                        std::nullopt, std::nullopt, "Brennweite", std::nullopt}));
    EXPECT_EQ(items[1], (ConstraintSpec{ConstraintType::ParfocalLength, ConstraintMode::Max,
                                        std::nullopt, 1, -1.5, std::nullopt, std::nullopt}));
    EXPECT_EQ(items[2].weight, 0.0);
    EXPECT_EQ(items[3].mode, ConstraintMode::Equal);
    EXPECT_FALSE(items[0].reference_surface.has_value());

    const std::string text1 = mko::write_system_json(a);
    const SystemDescription b = mko::read_system_json(text1);
    EXPECT_TRUE(a == b);
    EXPECT_EQ(text1, mko::write_system_json(b));
    for (const char* expected :
         {R"({ "type": "focal_length", "weight": 2.5, "comment": "Brennweite" })",
          R"({ "type": "parfocal_length", "mode": "max", "reference_surface": 1, "offset": -1.5 })",
          R"({ "type": "working_distance", "mode": "min", "weight": 0.0 })",
          R"({ "type": "paraxial_focus", "mode": "equal" })"}) {
        EXPECT_NE(text1.find(expected), std::string::npos) << expected << '\n' << text1;
    }

    // Leere Liste ist erlaubt und bleibt erhalten
    const SystemDescription empty = mko::read_system_json(make_text(R"({ "items": [] })"));
    ASSERT_TRUE(empty.constraints.has_value());
    EXPECT_TRUE(empty.constraints->items.empty());
    EXPECT_TRUE(mko::read_system_json(mko::write_system_json(empty)) == empty);
    // Ohne Abschnitt kein Abschnitt
    const SystemDescription none = mko::read_system_json(make_text(""));
    EXPECT_FALSE(none.constraints.has_value());
    EXPECT_EQ(mko::write_system_json(none).find("constraints"), std::string::npos);
}

TEST(ConstraintsSchema, UnknownFieldsWarn) {
    std::vector<std::string> warnings;
    (void)mko::read_system_json(
        make_text(R"({ "items": [{ "type": "focal_length", "tolerance": 1 }], "extra": 2 })"),
        &warnings);
    ASSERT_EQ(warnings.size(), 2u);
    EXPECT_NE(warnings[0].find("constraints.items[0].tolerance"), std::string::npos);
    EXPECT_NE(warnings[1].find("constraints.extra"), std::string::npos);
}

TEST(ConstraintsSchema, NamesRoundtrip) {
    for (const ConstraintType t : {ConstraintType::FocalLength, ConstraintType::ParfocalLength,
                                   ConstraintType::WorkingDistance, ConstraintType::ParaxialFocus,
                                   ConstraintType::Petzval}) {
        EXPECT_EQ(mko::constraint_type_from_string(mko::to_string(t)), t);
        EXPECT_TRUE(mko::is_paraxial(t));
    }
    for (const ConstraintMode m :
         {ConstraintMode::Equal, ConstraintMode::Min, ConstraintMode::Max}) {
        EXPECT_EQ(mko::constraint_mode_from_string(mko::to_string(m)), m);
    }
    EXPECT_EQ(mko::to_string(ConstraintType::Petzval), "petzval");
    EXPECT_FALSE(mko::constraint_type_from_string("petzval_radius").has_value());
    EXPECT_FALSE(mko::constraint_mode_from_string("MIN").has_value());
}

// Struktur: Abschnitt, Liste, Einträge, Typen der Felder.
TEST(ConstraintsSchema, StructureErrors) {
    expect_error(make_text("[]"), "constraints");
    expect_error(make_text("{}"), "constraints.items");
    expect_error(make_text(R"({ "items": {} })"), "constraints.items");
    expect_error(with_items("1"), "constraints.items[0]");
    expect_error(with_items("{}"), "constraints.items[0].type");
    expect_error(with_items(R"({ "type": 3 })"), "constraints.items[0].type");
    expect_error(with_items(R"({ "type": "Petzval" })"), "constraints.items[0].type");
    expect_error(with_items(R"({ "type": "petzval", "max_curvature": "0.01" })"),
                 "constraints.items[0].max_curvature");
    expect_error(with_items(R"({ "type": "Focal_Length" })"), "constraints.items[0].type");
    expect_error(with_items(R"({ "type": "working_distance", "mode": "above" })"),
                 "constraints.items[0].mode");
    expect_error(with_items(R"({ "type": "working_distance", "mode": 1 })"),
                 "constraints.items[0].mode");
    expect_error(with_items(R"({ "type": "focal_length", "weight": "1" })"),
                 "constraints.items[0].weight");
    expect_error(with_items(R"({ "type": "parfocal_length", "reference_surface": -1 })"),
                 "constraints.items[0].reference_surface");
    expect_error(with_items(R"({ "type": "parfocal_length", "reference_surface": 1.5 })"),
                 "constraints.items[0].reference_surface");
    expect_error(with_items(R"({ "type": "parfocal_length", "offset": "2" })"),
                 "constraints.items[0].offset");
    expect_error(with_items(R"({ "type": "focal_length", "comment": 7 })"),
                 "constraints.items[0].comment");
}

// Inhaltliche Regeln (check_constraints, auch beim Schreiben).
TEST(ConstraintsSchema, RuleErrors) {
    // Brennweite und paraxialer Fokus nur als Gleichung
    expect_error(with_items(R"({ "type": "focal_length", "mode": "min" })"),
                 "constraints.items[0].mode");
    expect_error(with_items(R"({ "type": "paraxial_focus", "mode": "max" })"),
                 "constraints.items[0].mode");
    // Gewicht endlich und >= 0
    expect_error(with_items(R"({ "type": "working_distance", "weight": -1 })"),
                 "constraints.items[0].weight");
    // Felder nur beim passenden Typ
    expect_error(with_items(R"({ "type": "focal_length", "reference_surface": 0 })"),
                 "constraints.items[0].reference_surface");
    expect_error(with_items(R"({ "type": "working_distance", "offset": 1 })"),
                 "constraints.items[0].offset");
    // Referenzfläche existiert
    expect_error(with_items(R"({ "type": "parfocal_length", "reference_surface": 4 })"),
                 "constraints.items[0].reference_surface");
    EXPECT_NO_THROW((void)mko::read_system_json(
        with_items(R"({ "type": "parfocal_length", "reference_surface": 3 })")));
    // Sollwert fehlt in target bzw. target fehlt ganz
    const std::string only_f = R"("target": { "focal_length": 10.0 },)";
    expect_error(with_items(R"({ "type": "focal_length" }, { "type": "parfocal_length" })", only_f),
                 "constraints.items[1].type");
    expect_error(with_items(R"({ "type": "working_distance" })", only_f),
                 "constraints.items[0].type");
    expect_error(with_items(R"({ "type": "focal_length" })", ""), "constraints.items[0].type");
    // paraxial_focus braucht keinen Sollwert in target
    EXPECT_NO_THROW((void)mko::read_system_json(with_items(R"({ "type": "paraxial_focus" })", "")));
    // jeder Typ höchstens einmal
    expect_error(with_items(R"({ "type": "working_distance" }, { "type": "focal_length" },
                               { "type": "working_distance", "mode": "max" })"),
                 "constraints.items[2].type");
    // paraxial_focus nur bei kollimierter Quelle
    std::string point = with_items(R"({ "type": "paraxial_focus" })");
    const std::string collimated =
        R"("source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] })";
    point.replace(
        point.find(collimated), collimated.size(),
        R"("source": { "type": "point", "point": [0, 0, -50], "numerical_aperture": 0.1 })");
    expect_error(point, "constraints.items[0].type");

    // Schreiben prüft dieselben Regeln am Modell
    SystemDescription s = mko::read_system_json(with_items(R"({ "type": "focal_length" })"));
    s.target->focal_length.reset();
    try {
        (void)mko::write_system_json(s);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "constraints.items[0].type");
    }
}

// ---- Petzval (M4-T4) -------------------------------------------------------------------------

// Grenze κ_max im Eintrag (max_curvature, einzige Quelle), Modus Max als Standard, kein Sollwert
// in target nötig; Roundtrip mit fester Feldreihenfolge.
TEST(ConstraintsSchema, PetzvalRoundtrip) {
    const std::string text = with_items(R"(
    { "comment": "R ≥ 130 mm", "max_curvature": 0.0076923, "type": "petzval", "weight": 1e3 })",
                                        "");
    std::vector<std::string> warnings;
    const SystemDescription a = mko::read_system_json(text, &warnings);
    EXPECT_TRUE(warnings.empty());
    ASSERT_TRUE(a.constraints.has_value());
    const ConstraintSpec& c = a.constraints->items.at(0);
    EXPECT_EQ(c.type, ConstraintType::Petzval);
    EXPECT_FALSE(c.mode.has_value());
    EXPECT_EQ(c.mode_or_default(), ConstraintMode::Max);
    EXPECT_EQ(c.max_curvature, 0.0076923);
    EXPECT_EQ(c.weight, 1000.0);
    EXPECT_FALSE(mko::constraint_target(a, ConstraintType::Petzval).has_value());
    EXPECT_EQ(mko::constraint_bound(a, c), 0.0076923);
    EXPECT_TRUE(mko::is_paraxial(ConstraintType::Petzval));

    const std::string text1 = mko::write_system_json(a);
    const SystemDescription b = mko::read_system_json(text1);
    EXPECT_TRUE(a == b);
    EXPECT_EQ(text1, mko::write_system_json(b));
    const char* expected =
        R"({ "type": "petzval", "weight": 1000.0, "max_curvature": 0.0076923, "comment": "R ≥ 130 mm" })";
    EXPECT_NE(text1.find(expected), std::string::npos) << text1;

    // Modus "max" ausdrücklich, κ_max = 0 (ebenes Feld) erlaubt
    const SystemDescription m = mko::read_system_json(
        with_items(R"({ "type": "petzval", "mode": "max", "max_curvature": 0 })"));
    EXPECT_EQ(m.constraints->items[0].mode, ConstraintMode::Max);
    EXPECT_EQ(m.constraints->items[0].max_curvature, 0.0);
    EXPECT_TRUE(mko::read_system_json(mko::write_system_json(m)) == m);

    // Beispieldatei der Abnahme
    const SystemDescription z =
        mko::read_system_file(kSourceDir / "data" / "systems" / "zweilinser_petzval.json");
    ASSERT_TRUE(z.constraints.has_value());
    ASSERT_EQ(z.constraints->items.size(), 2u);
    EXPECT_EQ(z.constraints->items[1].type, ConstraintType::Petzval);
    EXPECT_EQ(z.constraints->items[1].max_curvature, 0.01);
    EXPECT_EQ(z.constraints->items[1].weight, 1676769510.9367275);
}

TEST(ConstraintsSchema, PetzvalRuleErrors) {
    // Grenze ist Pflicht, endlich (JSON kennt kein inf) und >= 0
    expect_error(with_items(R"({ "type": "petzval" })"), "constraints.items[0].max_curvature");
    expect_error(with_items(R"({ "type": "petzval", "max_curvature": -0.01 })"),
                 "constraints.items[0].max_curvature");
    // nur Obergrenze
    expect_error(with_items(R"({ "type": "petzval", "mode": "equal", "max_curvature": 0.01 })"),
                 "constraints.items[0].mode");
    expect_error(with_items(R"({ "type": "petzval", "mode": "min", "max_curvature": 0.01 })"),
                 "constraints.items[0].mode");
    // max_curvature nur bei petzval, Felder der Abgleichlänge nicht bei petzval
    expect_error(with_items(R"({ "type": "focal_length", "max_curvature": 0.01 })"),
                 "constraints.items[0].max_curvature");
    expect_error(with_items(R"({ "type": "petzval", "max_curvature": 0.01, "offset": 1 })"),
                 "constraints.items[0].offset");
    // höchstens einmal
    expect_error(with_items(R"({ "type": "petzval", "max_curvature": 0.01 },
                               { "type": "petzval", "max_curvature": 0.02 })"),
                 "constraints.items[1].type");
    // Gewicht wie überall
    expect_error(with_items(R"({ "type": "petzval", "max_curvature": 0.01, "weight": -1 })"),
                 "constraints.items[0].weight");

    // Schreiben prüft dieselben Regeln am Modell (auch nicht endliche Werte)
    SystemDescription s =
        mko::read_system_json(with_items(R"({ "type": "petzval", "max_curvature": 0.01 })"));
    s.constraints->items[0].max_curvature = std::numeric_limits<double>::infinity();
    try {
        (void)mko::write_system_json(s);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "constraints.items[0].max_curvature");
    }
    s.constraints->items[0].max_curvature.reset();
    try {
        (void)mko::write_system_json(s);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "constraints.items[0].max_curvature");
    }
}
