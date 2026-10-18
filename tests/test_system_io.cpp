// Einlesen, Validieren und Schreiben von `mikroptim.system/1` (M1-T1).

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/system.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using mko::FormatError;
using mko::MaterialKind;
using mko::SourceType;
using mko::SurfaceType;
using mko::SystemDescription;

namespace {

const fs::path kSourceDir{MKO_SOURCE_DIR};

// Gültiges Basissystem für die Negativtests; die Bausteine dienen als Suchtexte.
const std::string kStop =
    R"({ "type": "stop", "semi_diameter": 10.0, "thickness": 5.0, "material": "AIR" })";
const std::string kFront =
    R"({ "type": "sphere", "curvature": 0.01, "thickness": 6.0, "material": "N-BK7", "semi_diameter": 12.5 })";
const std::string kBack =
    R"({ "type": "sphere", "curvature": -0.01, "thickness": 99.0, "material": "AIR", "semi_diameter": 12.5 })";
const std::string kSource =
    R"("source": { "type": "collimated", "field_angles_deg": [0.0, 1.0], "field_weights": [1.0, 1.0] })";
const std::string kBase = R"({
  "format": "mikroptim.system/1",
  "name": "Test",
  "units": "mm",
  "glass_catalogs": ["data/glass/basis.json"],
  "spectrum": { "wavelengths_nm": [486.13, 587.56, 656.27], "weights": [1.0, 1.0, 1.0], "reference_nm": 587.56 },
  )" + kSource + R"(,
  "surfaces": [
    )" + kStop + ",\n    " +
                          kFront + ",\n    " + kBack + R"(
  ]
}
)";

// Ersetzt das erste Vorkommen von `from` in `text` (Test schlägt fehl, wenn es fehlt).
std::string replaced(const std::string& text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    if (pos == std::string::npos) {
        ADD_FAILURE() << "Suchtext nicht gefunden: " << from;
        return text;
    }
    std::string s = text;
    s.replace(pos, from.size(), to);
    return s;
}

std::vector<fs::path> system_files() {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(kSourceDir / "data" / "systems")) {
        if (e.is_regular_file() && e.path().extension() == ".json") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool basis_glass(std::string_view name) {
    return name == "N-BK7" || name == "CAF2" || name == "F_SILICA";
}

// Eigenes temporäres Verzeichnis pro Test, wird im Destruktor gelöscht.
struct TempDir {
    fs::path path;
    TempDir() {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        path = fs::temp_directory_path() /
               (std::string("mikroptim_") + info->test_suite_name() + "_" + info->name());
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

void write_text(const fs::path& p, const std::string& text) {
    std::ofstream out(p, std::ios::binary);
    out << text;
}

} // namespace

// --------------------------------------------------------------------------
// Roundtrip und Positivtests
// --------------------------------------------------------------------------

TEST(SystemIo, BaseSystemIsValid) {
    std::vector<std::string> warnings;
    const SystemDescription s = mko::read_system_json(kBase, &warnings);
    EXPECT_TRUE(warnings.empty());
    EXPECT_EQ(s.surfaces.size(), 3u);
}

// Abnahme: lesen → schreiben → lesen verlustfrei, schreiben → lesen → schreiben textgleich.
TEST(SystemIo, RoundtripAllDataSystems) {
    const auto files = system_files();
    ASSERT_GE(files.size(), 2u);
    for (const fs::path& f : files) {
        SCOPED_TRACE(f.string());
        std::vector<std::string> warnings;
        const SystemDescription a = mko::read_system_file(f, &warnings);
        EXPECT_TRUE(warnings.empty());
        const std::string text1 = mko::write_system_json(a);
        const SystemDescription b = mko::read_system_json(text1);
        EXPECT_TRUE(a == b);
        const std::string text2 = mko::write_system_json(b);
        EXPECT_EQ(text1, text2);
        EXPECT_NO_THROW(mko::validate_materials(a, basis_glass));
    }
}

TEST(SystemIo, ReadsEinzellinseValues) {
    const SystemDescription s =
        mko::read_system_file(kSourceDir / "data" / "systems" / "einzellinse_bk7.json");
    EXPECT_EQ(s.format, "mikroptim.system/1");
    EXPECT_EQ(s.name, "Einzellinse N-BK7");
    ASSERT_TRUE(s.description.has_value());
    EXPECT_EQ(s.units, "mm");
    EXPECT_EQ(s.glass_catalogs, std::vector<std::string>{"data/glass/basis.json"});
    EXPECT_EQ(s.spectrum.wavelengths_nm, (std::vector<double>{486.13, 587.56, 656.27}));
    EXPECT_EQ(s.spectrum.weights, (std::vector<double>{1.0, 1.0, 1.0}));
    EXPECT_EQ(s.spectrum.reference_nm, 587.56);
    EXPECT_EQ(s.source.type, SourceType::Collimated);
    EXPECT_EQ(s.source.field_angles_deg, (std::vector<double>{0.0, 1.0, 2.0}));
    EXPECT_FALSE(s.target.has_value());
    ASSERT_EQ(s.surfaces.size(), 3u);
    EXPECT_EQ(s.surfaces[0].type, SurfaceType::Stop);
    EXPECT_EQ(s.surfaces[0].semi_diameter, 10.0);
    EXPECT_EQ(s.surfaces[0].material.kind, MaterialKind::Air);
    EXPECT_EQ(s.surfaces[1].type, SurfaceType::Sphere);
    EXPECT_EQ(s.surfaces[1].curvature, 0.00967);
    EXPECT_EQ(s.surfaces[1].material.kind, MaterialKind::Catalog);
    EXPECT_EQ(s.surfaces[1].material.name, "N-BK7");
    EXPECT_EQ(s.surfaces[2].curvature, -0.00967);
    EXPECT_EQ(s.surfaces[2].thickness, 99.05);
    EXPECT_FALSE(s.ghost_plates_json.has_value());
}

// Alle optionalen Felder und Materialvarianten überstehen den Roundtrip.
TEST(SystemIo, RoundtripAllFeatures) {
    const std::string text = R"({
  "format": "mikroptim.system/1",
  "name": "Alles",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [500], "weights": [2], "reference_nm": 500 },
  "source": { "type": "point", "point": [0.0, 0.1, -12.5], "numerical_aperture": 0.95 },
  "target": { "focal_length": 3.29, "working_distance": 0.35 },
  "surfaces": [
    { "type": "stop", "semi_diameter": 3.125, "thickness": 1.0, "material": "AIR", "comment": "Blende" },
    { "type": "sphere", "curvature": 0.1, "thickness": 2.0, "material": { "model": "constant", "n": 1.5 }, "semi_diameter": 4.0 },
    { "type": "plane", "thickness": 1.0, "material": { "model": "buchdahl", "n0": 1.6, "nu": [-0.1, 0.02] }, "semi_diameter": 4.0 },
    { "type": "sphere", "curvature": -0.05, "thickness": 0.5, "material": { "model": "buchdahl", "n0": 1.45, "nu": [-0.08, 0.01, -0.003] }, "semi_diameter": 4.0, "comment": "" }
  ],
  "ghost_plates": [ { "after_surface": 7, "distance": 0.5, "glass": "CAF2", "norm_radius": 4.0, "order": 8, "coefficients": [0, 0, 0, 0, 0, 0, 0, 0] } ],
  "variables": { "z": 1, "a": [1, 2] },
  "constraints": {}
}
)";
    std::vector<std::string> warnings;
    const SystemDescription a = mko::read_system_json(text, &warnings);
    EXPECT_TRUE(warnings.empty());
    EXPECT_FALSE(a.description.has_value());
    EXPECT_TRUE(a.glass_catalogs.empty());
    EXPECT_EQ(a.source.type, SourceType::Point);
    EXPECT_EQ(a.source.point[2], -12.5);
    EXPECT_EQ(a.source.numerical_aperture, 0.95);
    ASSERT_TRUE(a.target.has_value());
    EXPECT_EQ(a.target->focal_length, 3.29);
    EXPECT_FALSE(a.target->parfocal_length.has_value());
    EXPECT_EQ(a.target->working_distance, 0.35);
    EXPECT_EQ(a.surfaces[0].comment, "Blende");
    EXPECT_EQ(a.surfaces[1].material.kind, MaterialKind::Constant);
    EXPECT_EQ(a.surfaces[1].material.n, 1.5);
    EXPECT_EQ(a.surfaces[2].material.kind, MaterialKind::Buchdahl);
    EXPECT_EQ(a.surfaces[2].material.nu, (std::array<double, 3>{-0.1, 0.02, 0.0}));
    EXPECT_EQ(a.surfaces[3].material.nu, (std::array<double, 3>{-0.08, 0.01, -0.003}));
    EXPECT_EQ(a.surfaces[3].comment, "");
    ASSERT_TRUE(a.ghost_plates_json.has_value());
    ASSERT_TRUE(a.variables_json.has_value());
    EXPECT_EQ(*a.variables_json, R"({"z":1,"a":[1,2]})"); // Feldreihenfolge bleibt erhalten
    EXPECT_EQ(a.constraints_json, "{}");

    const std::string text1 = mko::write_system_json(a);
    const SystemDescription b = mko::read_system_json(text1);
    EXPECT_TRUE(a == b);
    EXPECT_EQ(text1, mko::write_system_json(b));
    EXPECT_NE(text1.find(R"("after_surface": 7)"), std::string::npos) << text1;
    EXPECT_NE(text1.find(R"("nu": [-0.1, 0.02])"), std::string::npos) << text1;
    EXPECT_NE(text1.find(R"("variables": { "z": 1, "a": [1, 2] })"), std::string::npos) << text1;
}

TEST(SystemIo, WriterFormatMatchesDocumentation) {
    const SystemDescription s =
        mko::read_system_file(kSourceDir / "data" / "systems" / "planplatte_bk7.json");
    const std::string text = mko::write_system_json(s);
    const std::string expected = R"({
  "format": "mikroptim.system/1",
  "name": "Planplatte N-BK7",
  "description": "Testsystem mit exakt bekannter optischer Weglänge.",
  "units": "mm",
  "glass_catalogs": ["data/glass/basis.json"],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0, 10.0], "field_weights": [1.0, 1.0] },
  "surfaces": [
    { "type": "stop", "thickness": 2.0, "material": "AIR", "semi_diameter": 5.0 },
    { "type": "plane", "thickness": 10.0, "material": "N-BK7", "semi_diameter": 10.0 },
    { "type": "plane", "thickness": 20.0, "material": "AIR", "semi_diameter": 10.0 }
  ]
}
)";
    EXPECT_EQ(text, expected);
}

TEST(SystemIo, IntegerNumbersAreAccepted) {
    const SystemDescription s =
        mko::read_system_json(replaced(kBase, R"("thickness": 6.0)", R"("thickness": 6)"));
    EXPECT_EQ(s.surfaces[1].thickness, 6.0);
}

TEST(SystemIo, UnknownFieldsProduceWarnings) {
    std::string text = replaced(kBase, R"("units": "mm",)", R"("units": "mm", "foo": 1,)");
    text = replaced(text, R"("curvature": 0.01,)", R"("curvature": 0.01, "bar": true,)");
    text = replaced(text, R"("type": "collimated",)",
                    R"("type": "collimated", "numerical_aperture": 0.5,)");
    std::vector<std::string> warnings;
    const SystemDescription s = mko::read_system_json(text, &warnings);
    EXPECT_TRUE(s == mko::read_system_json(kBase));
    ASSERT_EQ(warnings.size(), 3u);
    const auto has = [&](std::string_view needle) {
        return std::any_of(warnings.begin(), warnings.end(), [&](const std::string& w) {
            return w.find(needle) != std::string::npos;
        });
    };
    EXPECT_TRUE(has("foo: unbekanntes Feld"));
    EXPECT_TRUE(has("surfaces[1].bar: unbekanntes Feld"));
    EXPECT_TRUE(has("source.numerical_aperture: wird bei source.type \"collimated\" ignoriert"));
    // ohne Warnungsliste kein Fehler
    EXPECT_NO_THROW(mko::read_system_json(text));
}

TEST(SystemIo, FileWriteAndReadRoundtrip) {
    TempDir dir;
    const SystemDescription a = mko::read_system_json(kBase);
    const fs::path file = dir.path / "system.json";
    mko::write_system_file(a, file);
    EXPECT_TRUE(mko::read_system_file(file) == a);
}

// --------------------------------------------------------------------------
// Negativtests: Meldung enthält den JSON-Pfad
// --------------------------------------------------------------------------

struct NegativeCase {
    const char* name;
    std::string text;
    std::string path;    // erwarteter JSON-Pfad
    std::string message; // erwarteter Teil der Meldung
};

void PrintTo(const NegativeCase& c, std::ostream* os) {
    *os << c.name;
}

class SystemIoNegative : public ::testing::TestWithParam<NegativeCase> {};

TEST_P(SystemIoNegative, ThrowsWithJsonPath) {
    const NegativeCase& c = GetParam();
    try {
        (void)mko::read_system_json(c.text);
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), c.path) << e.what();
        EXPECT_NE(std::string(e.what()).find(c.path + ": "), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find(c.message), std::string::npos) << e.what();
    }
}

namespace {

std::vector<NegativeCase> negative_cases() {
    const std::string& b = kBase;
    return {
        {"FehlenderName", replaced(b, R"("name": "Test",)", ""), "name", "Pflichtfeld fehlt"},
        {"FehlendesSpektrum", replaced(b, R"(, "reference_nm": 587.56)", ""),
         "spectrum.reference_nm", "Pflichtfeld fehlt"},
        {"FehlendesMaterial", replaced(b, R"("material": "N-BK7", )", ""), "surfaces[1].material",
         "Pflichtfeld fehlt"},
        {"FehlendeKruemmungBeiSphere", replaced(b, R"("curvature": -0.01, )", ""),
         "surfaces[2].curvature", "Pflichtfeld fehlt"},
        {"DickeAlsZeichenkette", replaced(b, R"("thickness": 6.0)", R"("thickness": "6.0")"),
         "surfaces[1].thickness", "falscher Typ: erwartet Zahl, gefunden Zeichenkette"},
        {"WellenlaengenKeineListe",
         replaced(b, R"("wavelengths_nm": [486.13, 587.56, 656.27])",
                  R"("wavelengths_nm": 587.56)"),
         "spectrum.wavelengths_nm", "falscher Typ"},
        {"WellenlaengeAlsWahrheitswert", replaced(b, "[486.13, 587.56", "[486.13, true"),
         "spectrum.wavelengths_nm[1]", "gefunden Wahrheitswert"},
        {"GlaskatalogKeineZeichenkette", replaced(b, R"(["data/glass/basis.json"])", "[42]"),
         "glass_catalogs[0]", "falscher Typ"},
        {"FlaechenKeineListe",
         replaced(b, "\"surfaces\": [\n    " + kStop + ",",
                  "\"surfaces\": 1, \"x\": [\n    " + kStop + ","),
         "surfaces", "falscher Typ"},
        {"MaterialAlsZahl", replaced(b, R"("material": "N-BK7")", R"("material": 1.5)"),
         "surfaces[1].material", "falscher Typ"},
        {"UnbekannteHauptversion",
         replaced(b, R"("mikroptim.system/1")", R"("mikroptim.system/2")"), "format",
         "unbekannte Hauptversion 2"},
        {"FremdesFormat", replaced(b, R"("mikroptim.system/1")", R"("mikroptim.glass/1")"),
         "format", "fremdes Format"},
        {"UngueltigeVersion", replaced(b, R"("mikroptim.system/1")", R"("mikroptim.system/x")"),
         "format", "ungültige Versionsangabe"},
        {"FalscheEinheit", replaced(b, R"("units": "mm")", R"("units": "inch")"), "units",
         "nur \"mm\""},
        {"StopNichtAnErsterStelle",
         replaced(b, R"("type": "stop", "semi_diameter": 10.0)",
                  R"("type": "plane", "semi_diameter": 10.0)"),
         "surfaces[0].type", "erste Fläche muss \"stop\" sein"},
        {"ZweiterStop",
         replaced(b, R"("type": "sphere", "curvature": 0.01,)", R"("type": "stop",)"),
         "surfaces[1].type", "nur als erste Fläche"},
        {"UnbekannterFlaechentyp",
         replaced(b, R"("type": "sphere", "curvature": -0.01)",
                  R"("type": "asphere", "curvature": -0.01)"),
         "surfaces[2].type", "unbekannter Flächentyp"},
        {"KruemmungBeiPlanflaeche",
         replaced(b, R"({ "type": "stop",)", R"({ "type": "stop", "curvature": 0.1,)"),
         "surfaces[0].curvature", "nur bei \"sphere\""},
        {"BlendeNichtInLuft",
         replaced(b, R"("thickness": 5.0, "material": "AIR")",
                  R"("thickness": 5.0, "material": "N-BK7")"),
         "surfaces[0].material", "\"AIR\""},
        {"NurEineFlaeche", replaced(b, kStop + ",\n    " + kFront + ",\n    " + kBack, kStop),
         "surfaces", "mindestens zwei Flächen"},
        {"HalbdurchmesserNull",
         replaced(b, R"("material": "AIR", "semi_diameter": 12.5)",
                  R"("material": "AIR", "semi_diameter": 0.0)"),
         "surfaces[2].semi_diameter", "größer als 0"},
        {"UngleicheLaengeSpektrum",
         replaced(b, R"("weights": [1.0, 1.0, 1.0])", R"("weights": [1.0, 1.0])"),
         "spectrum.weights", "Länge 2 passt nicht zu spectrum.wavelengths_nm (Länge 3)"},
        {"UngleicheLaengeFelder",
         replaced(b, R"("field_weights": [1.0, 1.0])", R"("field_weights": [1.0])"),
         "source.field_weights", "Länge 1 passt nicht"},
        {"NegativeWellenlaenge", replaced(b, "[486.13, 587.56", "[-486.13, 587.56"),
         "spectrum.wavelengths_nm[0]", "größer als 0"},
        {"NegativesGewicht",
         replaced(b, R"("weights": [1.0, 1.0, 1.0])", R"("weights": [1.0, -1.0, 1.0])"),
         "spectrum.weights[1]", "nicht negativ"},
        {"GewichteAlleNull",
         replaced(b, R"("weights": [1.0, 1.0, 1.0])", R"("weights": [0.0, 0.0, 0.0])"),
         "spectrum.weights", "Summe"},
        {"ReferenzwellenlaengeNull",
         replaced(b, R"("reference_nm": 587.56)", R"("reference_nm": 0)"), "spectrum.reference_nm",
         "größer als 0"},
        {"FeldwinkelNeunzigGrad",
         replaced(b, R"("field_angles_deg": [0.0, 1.0])", R"("field_angles_deg": [0.0, 90.0])"),
         "source.field_angles_deg[1]", "unter 90°"},
        {"UnbekannterQuellentyp", replaced(b, R"("type": "collimated")", R"("type": "gaussian")"),
         "source.type", "unbekannter Quellentyp"},
        {"PunktquelleOhneNA",
         replaced(b, kSource, R"("source": { "type": "point", "point": [0.0, 0.0, -10.0] })"),
         "source.numerical_aperture", "Pflichtfeld fehlt"},
        {"PunktquelleFalscheLaenge",
         replaced(
             b, kSource,
             R"("source": { "type": "point", "point": [0.0, -10.0], "numerical_aperture": 0.5 })"),
         "source.point", "genau 3 Koordinaten"},
        {"UnbekanntesMaterialmodell",
         replaced(b, R"("material": "N-BK7")", R"("material": { "model": "sellmeier" })"),
         "surfaces[1].material.model", "unbekanntes Materialmodell"},
        {"KonstanteOhneN",
         replaced(b, R"("material": "N-BK7")", R"("material": { "model": "constant" })"),
         "surfaces[1].material.n", "Pflichtfeld fehlt"},
        {"KonstanteNegativ",
         replaced(b, R"("material": "N-BK7")", R"("material": { "model": "constant", "n": -1 })"),
         "surfaces[1].material.n", "größer als 0"},
        {"BuchdahlFalscheKoeffizientenzahl",
         replaced(b, R"("material": "N-BK7")",
                  R"("material": { "model": "buchdahl", "n0": 1.5, "nu": [0.1] })"),
         "surfaces[1].material.nu", "2 oder 3"},
        {"GeisterplattenKeineListe",
         replaced(b, R"("units": "mm",)", R"("units": "mm", "ghost_plates": {},)"), "ghost_plates",
         "falscher Typ"},
        {"ZielKeinObjekt", replaced(b, R"("units": "mm",)", R"("units": "mm", "target": 5,)"),
         "target", "falscher Typ"},
    };
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Validierung, SystemIoNegative, ::testing::ValuesIn(negative_cases()),
                         [](const ::testing::TestParamInfo<NegativeCase>& case_info) {
                             return std::string(case_info.param.name);
                         });

TEST(SystemIo, UnknownGlassReportsSurfacePath) {
    const SystemDescription s = mko::read_system_json(replaced(kBase, R"("N-BK7")", R"("N-BK8")"));
    try {
        mko::validate_materials(s, basis_glass);
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "surfaces[1].material");
        EXPECT_NE(std::string(e.what()).find("surfaces[1].material: unbekanntes Glas \"N-BK8\""),
                  std::string::npos)
            << e.what();
    }
}

TEST(SystemIo, SyntaxErrorIsReported) {
    try {
        (void)mko::read_system_json(replaced(kBase, R"("name": "Test",)", R"("name": "Test")"));
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_TRUE(e.json_path().empty());
        EXPECT_NE(std::string(e.what()).find("JSON-Syntaxfehler"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("line 4"), std::string::npos) << e.what();
    }
}

TEST(SystemIo, RootMustBeObject) {
    EXPECT_THROW((void)mko::read_system_json("[1, 2]"), FormatError);
}

TEST(SystemIo, FileErrorContainsFileNameAndPath) {
    TempDir dir;
    const fs::path file = dir.path / "kaputt.json";
    write_text(file, replaced(kBase, R"("thickness": 6.0)", R"("thickness": null)"));
    try {
        (void)mko::read_system_file(file);
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "surfaces[1].thickness");
        EXPECT_NE(e.file().find("kaputt.json"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("kaputt.json: surfaces[1].thickness: "),
                  std::string::npos)
            << e.what();
    }
}

TEST(SystemIo, MissingFileThrows) {
    EXPECT_THROW((void)mko::read_system_file(kSourceDir / "data" / "gibt_es_nicht.json"),
                 std::runtime_error);
}

// Der Schreiber validiert das Modell, damit keine unlesbaren Dateien entstehen.
TEST(SystemIo, WriterRejectsInvalidModel) {
    SystemDescription s = mko::read_system_json(kBase);
    {
        SystemDescription t = s;
        t.surfaces[1].type = SurfaceType::Plane; // Krümmung 0,01 ginge beim Schreiben verloren
        try {
            (void)mko::write_system_json(t);
            FAIL() << "keine Ausnahme";
        } catch (const FormatError& e) {
            EXPECT_EQ(e.json_path(), "surfaces[1].curvature");
        }
    }
    {
        SystemDescription t = s;
        t.spectrum.weights.push_back(1.0);
        EXPECT_THROW((void)mko::write_system_json(t), FormatError);
    }
    {
        SystemDescription t = s;
        t.ghost_plates_json = "[{oops";
        try {
            (void)mko::write_system_json(t);
            FAIL() << "keine Ausnahme";
        } catch (const FormatError& e) {
            EXPECT_EQ(e.json_path(), "ghost_plates");
        }
    }
}

// --------------------------------------------------------------------------
// Auflösung der Glaskatalog-Pfade
// --------------------------------------------------------------------------

TEST(SystemIo, ResolvesCatalogRelativeToRepositoryRoot) {
    const fs::path dir = kSourceDir / "data" / "systems";
    const SystemDescription s = mko::read_system_file(dir / "einzellinse_bk7.json");
    const auto catalogs = mko::resolve_glass_catalogs(s, dir);
    ASSERT_EQ(catalogs.size(), 1u);
    EXPECT_TRUE(fs::equivalent(catalogs[0], kSourceDir / "data" / "glass" / "basis.json"));
}

TEST(SystemIo, ResolvesCatalogRelativeToSystemFileFirst) {
    TempDir dir;
    fs::create_directories(dir.path / "sub");
    write_text(dir.path / "katalog.json", "{}");
    write_text(dir.path / "sub" / "katalog.json", "{}");
    SystemDescription s = mko::read_system_json(kBase);
    s.glass_catalogs = {"katalog.json"};
    const auto catalogs = mko::resolve_glass_catalogs(s, dir.path / "sub");
    ASSERT_EQ(catalogs.size(), 1u);
    EXPECT_TRUE(fs::equivalent(catalogs[0], dir.path / "sub" / "katalog.json"));
}

TEST(SystemIo, MissingCatalogReportsPath) {
    SystemDescription s = mko::read_system_json(kBase);
    s.glass_catalogs = {"data/glass/basis.json", "gibt/es/nicht_4711.json"};
    try {
        (void)mko::resolve_glass_catalogs(s, kSourceDir / "data" / "systems");
        FAIL() << "keine Ausnahme";
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), "glass_catalogs[1]");
        EXPECT_NE(std::string(e.what()).find("nicht gefunden"), std::string::npos) << e.what();
    }
}
