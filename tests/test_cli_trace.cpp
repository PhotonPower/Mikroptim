// Unterbefehl `mikroptim trace` (M1-T8): JSON-Ausgabe gegen die Bibliotheksfunktionen
// (compute_paraxial, evaluate_wavefront), afokales System, Punktquelle, Fehlerfälle und
// Textausgabe. Der Aufruf des fertigen Programms steht als ctest-Kommando in CMakeLists.txt.

#include "cli/trace_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using mko::test::kDocumentedValue;
using nlohmann::json;

namespace {

const std::string kEinzellinse = std::string(MKO_SOURCE_DIR) + "/data/systems/einzellinse_bk7.json";
const std::string kPlanplatte = std::string(MKO_SOURCE_DIR) + "/data/systems/planplatte_bk7.json";

struct RunResult {
    int code = -1;
    std::string out;
    std::string err;
};

RunResult run(std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> v(args);
    std::ostringstream out;
    std::ostringstream err;
    RunResult r;
    r.code = mko::cli::run_trace_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

// Temporäre Systemdatei, wird im Destruktor gelöscht.
class TempFile {
public:
    TempFile(const std::string& name, const std::string& content)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::ofstream f(path_, std::ios::binary);
        f << content;
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

// Einzellinse mit Ersetzungen für Quelle, Material der ersten Linsenfläche und einem
// zusätzlichen (unbekannten) Feld.
std::string lens_json(const std::string& source, const std::string& material,
                      const std::string& extra = {}) {
    return std::string(R"({
  "format": "mikroptim.system/1",
  "name": "CLI-Test",
  "units": "mm",
  "glass_catalogs": [")") +
           MKO_SOURCE_DIR + R"(/data/glass/basis.json"],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": )" +
           source + R"(,)" + extra + R"(
  "surfaces": [
    { "type": "stop", "semi_diameter": 10.0, "thickness": 5.0, "material": "AIR" },
    { "type": "sphere", "curvature": 0.00967, "thickness": 6.0, "material": ")" +
           material + R"(", "semi_diameter": 12.5 },
    { "type": "sphere", "curvature": -0.00967, "thickness": 99.05, "material": "AIR", "semi_diameter": 12.5 }
  ]
})";
}

const char* const kCollimated =
    R"({ "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] })";

} // namespace

// Abnahmekriterium: JSON-Ausgabe für einzellinse_bk7.json, Werte bitgleich zur Bibliothek.
TEST(CliTrace, EinzellinseJsonMatchesLibrary) {
    const RunResult r = run({kEinzellinse, "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_TRUE(r.err.empty()) << r.err;
    const json j = json::parse(r.out);

    for (const char* key :
         {"format", "mikroptim_version", "system", "file", "reference_nm", "wavelengths_nm",
          "source", "paraxial", "quadrature", "target_focal_length", "wavefront", "wavefront_error",
          "rays", "failed_rays", "warnings"}) {
        EXPECT_TRUE(j.contains(key)) << key;
    }
    EXPECT_EQ(j["format"], "mikroptim.trace/1");
    EXPECT_EQ(j["system"], "Einzellinse N-BK7");
    EXPECT_EQ(j["file"], kEinzellinse);
    EXPECT_EQ(j["reference_nm"].get<double>(), 587.56);
    EXPECT_EQ(j["wavelengths_nm"], json::parse("[486.13, 587.56, 656.27]"));
    EXPECT_EQ(j["source"]["type"], "collimated");
    EXPECT_EQ(j["source"]["field_angles_deg"], json::parse("[0.0, 1.0, 2.0]"));
    EXPECT_EQ(j["quadrature"]["rings"], 10);
    EXPECT_EQ(j["quadrature"]["spokes"], 8);
    EXPECT_TRUE(j["wavefront_error"].is_null());
    EXPECT_TRUE(j["warnings"].empty());

    const mko::LoadedSystem loaded = mko::load_system(kEinzellinse);
    const mko::ParaxialData p = mko::compute_paraxial(loaded.system, loaded.catalog);
    const json& jp = j["paraxial"];
    EXPECT_EQ(jp["wavelength_nm"].get<double>(), 587.56);
    EXPECT_FALSE(jp["afocal"].get<bool>());
    EXPECT_NEAR(jp["focal_length"].get<double>(), 101.0497, kDocumentedValue);
    EXPECT_EQ(jp["focal_length"].get<double>(), p.focal_length);
    EXPECT_EQ(jp["power"].get<double>(), p.power);
    EXPECT_EQ(jp["back_focal_distance"].get<double>(), p.back_focal_distance);
    EXPECT_EQ(jp["front_focal_distance"].get<double>(), p.front_focal_distance);
    EXPECT_EQ(jp["front_principal_plane"].get<double>(), p.front_principal_plane);
    EXPECT_EQ(jp["rear_principal_plane"].get<double>(), p.rear_principal_plane);
    EXPECT_EQ(jp["image_defocus"].get<double>(), p.image_defocus);
    EXPECT_EQ(jp["petzval_sum"].get<double>(), p.petzval_sum);
    EXPECT_EQ(jp["petzval_curvature"].get<double>(), p.petzval_curvature);
    EXPECT_EQ(jp["petzval_radius"].get<double>(), p.petzval_radius);
    EXPECT_FALSE(jp["exit_pupil"]["at_infinity"].get<bool>());
    EXPECT_EQ(jp["exit_pupil"]["position"].get<double>(), p.exit_pupil_position);
    EXPECT_EQ(jp["exit_pupil"]["magnification"].get<double>(), p.exit_pupil_magnification);
    EXPECT_EQ(jp["exit_pupil"]["semi_diameter"].get<double>(), p.exit_pupil_semi_diameter);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(jp["abcd"][i].get<double>(), p.abcd[i]);
    }

    const mko::WavefrontResult w = mko::evaluate_wavefront(loaded);
    EXPECT_EQ(j["target_focal_length"].get<double>(), w.focal_length);
    const json& entries = j["wavefront"];
    ASSERT_TRUE(entries.is_array());
    ASSERT_EQ(entries.size(), 9U); // 3 Felder x 3 Wellenlängen
    int rays = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const json& e = entries[i];
        const mko::WavefrontMap& m = w.maps[i];
        EXPECT_EQ(e["field"], m.field);
        EXPECT_EQ(e["wavelength"], m.wavelength);
        EXPECT_EQ(e["field_angle_deg"].get<double>(),
                  loaded.system.source.field_angles_deg[static_cast<std::size_t>(m.field)]);
        EXPECT_EQ(e["wavelength_nm"].get<double>(), m.wavelength_nm);
        EXPECT_EQ(e["image_point"][0].get<double>(), m.image_point.x);
        EXPECT_EQ(e["image_point"][1].get<double>(), m.image_point.y);
        EXPECT_EQ(e["image_point"][2].get<double>(), m.image_point.z);
        EXPECT_EQ(e["pupil_mode"], m.field == 0 ? "axis" : "half");
        EXPECT_EQ(e["rays"], m.samples());
        EXPECT_EQ(e["failed_rays"], 0);
        EXPECT_EQ(e["rms_waves"].get<double>(), m.rms_waves); // bitgleich
        EXPECT_EQ(e["rms_mm"].get<double>(), m.rms);
        EXPECT_EQ(e["piston_mm"].get<double>(), m.piston);
        EXPECT_GT(m.rms_waves, 0.0);
        rays += m.samples();
    }
    // Achse: 10 x 1, außer der Achse: 10 x 8 Stützstellen, je 3 Wellenlängen
    EXPECT_EQ(rays, 3 * (10 + 2 * 80));
    EXPECT_EQ(j["rays"], rays);
    EXPECT_EQ(j["failed_rays"], 0);
}

TEST(CliTrace, QuadratureOptions) {
    const RunResult r = run({"--rings", "4", kEinzellinse, "--spokes", "6", "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["quadrature"]["rings"], 4);
    EXPECT_EQ(j["quadrature"]["spokes"], 6);

    mko::WavefrontOptions opt;
    opt.rings = 4;
    opt.spokes = 6;
    const mko::WavefrontResult w = mko::evaluate_wavefront(mko::load_system(kEinzellinse), opt);
    ASSERT_EQ(j["wavefront"].size(), w.maps.size());
    for (std::size_t i = 0; i < w.maps.size(); ++i) {
        EXPECT_EQ(j["wavefront"][i]["rays"], w.maps[i].samples());
        EXPECT_EQ(j["wavefront"][i]["rms_waves"].get<double>(), w.maps[i].rms_waves);
    }
}

// Afokales System: nicht endliche Paraxialwerte werden null, die Wellenfront ist ohne
// target.focal_length nicht auswertbar; die Paraxialdaten werden trotzdem ausgegeben.
TEST(CliTrace, PlanplatteAfocalGivesNull) {
    const RunResult r = run({kPlanplatte, "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    const json& jp = j["paraxial"];
    EXPECT_TRUE(jp["afocal"].get<bool>());
    EXPECT_EQ(jp["power"].get<double>(), 0.0);
    for (const char* key :
         {"focal_length", "back_focal_distance", "front_focal_distance", "front_principal_plane",
          "rear_principal_plane", "image_defocus", "petzval_radius"}) {
        EXPECT_TRUE(jp[key].is_null()) << key;
    }
    EXPECT_EQ(jp["petzval_sum"].get<double>(), 0.0);
    EXPECT_TRUE(j["wavefront"].is_null());
    EXPECT_TRUE(j["target_focal_length"].is_null());
    EXPECT_TRUE(j["rays"].is_null());
    EXPECT_TRUE(j["failed_rays"].is_null());
    ASSERT_TRUE(j["wavefront_error"].is_string());
    EXPECT_NE(j["wavefront_error"].get<std::string>().find("target.focal_length"),
              std::string::npos);

    const RunResult t = run({kPlanplatte});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    EXPECT_NE(t.out.find("unendlich (afokal)"), std::string::npos) << t.out;
    EXPECT_NE(t.out.find("Wellenfront nicht auswertbar"), std::string::npos) << t.out;
    // Petzvalkrümmung ist hier -0: kein Vorzeichen vor einer Null
    EXPECT_EQ(t.out.find("-0.000000"), std::string::npos) << t.out;
}

// Punktquelle: ein Feld ohne Feldwinkel, keine Sollbrennweite; unbekannte Felder erscheinen
// als Warnung in "warnings" und auf stderr.
TEST(CliTrace, PointSourceAndWarnings) {
    const TempFile file("mikroptim_cli_trace_point.json",
                        lens_json(R"({ "type": "point", "point": [0.0, 0.0, -50.0],
                                       "numerical_aperture": 0.05 })",
                                  "N-BK7", R"( "unbekannt": 1,)"));
    const RunResult r = run({file.path(), "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.err.find("unbekannt"), std::string::npos) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["source"]["type"], "point");
    EXPECT_EQ(j["source"]["point"], json::parse("[0.0, 0.0, -50.0]"));
    EXPECT_EQ(j["source"]["numerical_aperture"].get<double>(), 0.05);
    EXPECT_TRUE(j["target_focal_length"].is_null());
    ASSERT_EQ(j["wavefront"].size(), 1U);
    EXPECT_TRUE(j["wavefront"][0]["field_angle_deg"].is_null());
    EXPECT_EQ(j["wavefront"][0]["pupil_mode"], "axis");
    ASSERT_EQ(j["warnings"].size(), 1U);
    EXPECT_NE(j["warnings"][0].get<std::string>().find("unbekannt"), std::string::npos);

    const mko::WavefrontResult w = mko::evaluate_wavefront(mko::load_system(file.path()));
    EXPECT_EQ(j["wavefront"][0]["rms_waves"].get<double>(), w.maps[0].rms_waves);
}

TEST(CliTrace, MissingFileFails) {
    const std::string missing = std::string(MKO_SOURCE_DIR) + "/data/systems/gibt_es_nicht.json";
    const RunResult r = run({missing, "--json"});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_TRUE(r.out.empty());
    EXPECT_NE(r.err.find("gibt_es_nicht.json"), std::string::npos) << r.err;
}

// Unbekanntes Glas: Meldung mit JSON-Pfad, Exit-Code 1.
TEST(CliTrace, UnknownGlassReportsJsonPath) {
    const TempFile file("mikroptim_cli_trace_glass.json", lens_json(kCollimated, "N-GIBTSNICHT"));
    const RunResult r = run({file.path()});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_TRUE(r.out.empty());
    EXPECT_NE(r.err.find("surfaces[1].material"), std::string::npos) << r.err;
}

TEST(CliTrace, SyntaxErrorFails) {
    const TempFile file("mikroptim_cli_trace_syntax.json", "{ \"format\": ");
    const RunResult r = run({file.path(), "--json"});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_TRUE(r.out.empty());
    EXPECT_FALSE(r.err.empty());
}

TEST(CliTrace, UsageErrors) {
    for (const std::vector<std::string_view>& args : std::vector<std::vector<std::string_view>>{
             {},
             {"--json"},
             {kEinzellinse, "--gpu"},
             {kEinzellinse, kPlanplatte},
             {kEinzellinse, "--rings"},
             {kEinzellinse, "--rings", "0"},
             {kEinzellinse, "--rings", "17"},
             {kEinzellinse, "--rings", "4x"},
             {kEinzellinse, "--spokes", "0"},
         }) {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(mko::cli::run_trace_command(args, out, err), mko::cli::kExitUsage)
            << (args.empty() ? std::string("<leer>") : std::string(args.back()));
        EXPECT_TRUE(out.str().empty());
        EXPECT_NE(err.str().find("Aufruf"), std::string::npos);
    }
    const RunResult help = run({"--help"});
    EXPECT_EQ(help.code, mko::cli::kExitOk);
    EXPECT_NE(help.out.find("--rings"), std::string::npos);
}

// Textausgabe: Kopf, Paraxialdaten und eine Tabellenzeile pro Feld und Wellenlänge.
TEST(CliTrace, EinzellinseText) {
    const RunResult r = run({kEinzellinse});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const std::string& s = r.out;
    EXPECT_EQ(s.rfind("System: Einzellinse N-BK7\n", 0), 0U) << s;
    EXPECT_NE(s.find("Brennweite f'"), std::string::npos);
    EXPECT_NE(s.find("101.049"), std::string::npos);
    EXPECT_NE(s.find("Versagte Strahlen: 0 von 510"), std::string::npos) << s;
    // Keine plattformabhängigen Darstellungen nicht endlicher Zahlen
    EXPECT_EQ(s.find("nan"), std::string::npos);
    EXPECT_EQ(s.find("inf"), std::string::npos);
    // Feld, Winkel, Wellenlänge, Strahlen, versagt, RMS [Wellen], RMS [mm]
    const std::regex row(
        R"(^ +\d+ +\d+\.\d{3} +\d+\.\d{3} +\d+ +0 +\d+\.\d{6} +\d\.\d{6}e[-+]\d+$)");
    std::size_t rows = 0;
    std::istringstream lines(s);
    for (std::string line; std::getline(lines, line);) {
        if (std::regex_match(line, row)) {
            ++rows;
        }
    }
    EXPECT_EQ(rows, 9U) << s;
}
