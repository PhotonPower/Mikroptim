// Unterbefehl `mikroptim trace` (M1-T8): JSON-Ausgabe gegen die Bibliotheksfunktionen
// (compute_paraxial, evaluate_wavefront), afokales System, Punktquelle, Fehlerfälle und
// Textausgabe. Option --gpu (M2-T5): im CUDA-Build GPU- gegen CPU-Ausgabe (übersprungen ohne
// Gerät), sonst die Meldung "GPU-Backend nicht verfügbar". Der Aufruf des fertigen Programms
// steht als ctest-Kommando in tests/CMakeLists.txt.

#include "cli/trace_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "temp_paths.hpp"
#include "tolerances.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
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
        : path_(mko::test::temp_dir() / name) {
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

    for (const char* key : {"format", "mikroptim_version", "system", "file", "reference_nm",
                            "wavelengths_nm", "source", "paraxial", "quadrature", "backend",
                            "device", "precision", "target_focal_length", "wavefront",
                            "wavefront_error", "rays", "failed_rays", "warnings"}) {
        EXPECT_TRUE(j.contains(key)) << key;
    }
    EXPECT_EQ(j["format"], "mikroptim.trace/1");
    EXPECT_EQ(j["backend"], "cpu");
    EXPECT_TRUE(j["device"].is_null());
    EXPECT_EQ(j["precision"], "fp64");
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
             {kEinzellinse, "--unbekannt"},
             {kEinzellinse, "--device", "0"}, // --device ohne --gpu
             {kEinzellinse, "--gpu", "--device"},
             {kEinzellinse, "--gpu", "--device", "-1"},
             {kEinzellinse, "--gpu", "--device", "x"},
             {kEinzellinse, kPlanplatte},
             {kEinzellinse, "--rings"},
             {kEinzellinse, "--rings", "0"},
             {kEinzellinse, "--rings", "17"},
             {kEinzellinse, "--rings", "4x"},
             {kEinzellinse, "--spokes", "0"},
             {kEinzellinse, "--precision", "df64"}, // df64 nur mit --gpu
             {kEinzellinse, "--gpu", "--precision"},
             {kEinzellinse, "--gpu", "--precision", "fp32"},
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
    // Schlüssel auf 44 Zeichen aufgefüllt
    EXPECT_NE(s.find("\n  Backend" + std::string(44 - 7, ' ') + "CPU, FP64\n"), std::string::npos)
        << s;
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

// ------------------------------------------------------------------------------------------
// --gpu (M2-T5)
// ------------------------------------------------------------------------------------------

#ifndef MIKROPTIM_HAS_CUDA

// Ohne CUDA gebaut: klare Meldung, Exit-Code kExitNoGpu, keine Ausgabe (läuft in der CI).
TEST(CliTrace, GpuUnavailableWithoutCuda) {
    for (const std::vector<std::string_view>& args : std::vector<std::vector<std::string_view>>{
             {kEinzellinse, "--gpu"},
             {kEinzellinse, "--gpu", "--json"},
             {kEinzellinse, "--gpu", "--device", "1"},
         }) {
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(mko::cli::run_trace_command(args, out, err), mko::cli::kExitNoGpu);
        EXPECT_TRUE(out.str().empty()) << out.str();
        EXPECT_NE(err.str().find("ohne CUDA gebaut"), std::string::npos) << err.str();
    }
}

#else

namespace {

using mko::test::kCpuGpuFp64;
using mko::test::kCpuGpuFp64Relative;
using mko::test::kDf64VsFp64;
using mko::test::kDf64VsFp64Relative;

const std::string kMehrlinser =
    std::string(MKO_SOURCE_DIR) + "/data/systems/testsystem_mehrlinser.json";

bool have_gpu() {
    return mko::cuda::device_count() > 0;
}

// Schranke CPU gegen GPU für einen Wert in mm (tests/tolerances.hpp).
double cpu_gpu_bound_mm(double cpu_mm) {
    return std::max(kCpuGpuFp64, kCpuGpuFp64Relative * std::abs(cpu_mm));
}

// Größte beobachtete Abweichung der RMS-Werte (für das Testprotokoll).
struct Deviation {
    double abs_mm = 0.0;
    double rel = 0.0;
    int values = 0;
};

// RMS in Einheiten von `unit_mm` (1 für mm, λ_ref in mm für Wellen).
void expect_rms_near(const json& gpu, const json& cpu, double unit_mm, const std::string& what,
                     Deviation& dev) {
    if (cpu.is_null()) {
        EXPECT_TRUE(gpu.is_null()) << what;
        return;
    }
    ASSERT_TRUE(gpu.is_number()) << what;
    const double c = cpu.get<double>() * unit_mm;
    const double g = gpu.get<double>() * unit_mm;
    const double d = std::abs(g - c);
    EXPECT_LE(d, cpu_gpu_bound_mm(c)) << what << ": GPU " << g << " mm, CPU " << c << " mm";
    dev.abs_mm = std::max(dev.abs_mm, d);
    if (c != 0.0) {
        dev.rel = std::max(dev.rel, d / std::abs(c));
    }
    ++dev.values;
}

std::vector<std::string> keys_of(const json& j) {
    std::vector<std::string> keys;
    for (auto it = j.begin(); it != j.end(); ++it) {
        keys.push_back(it.key());
    }
    return keys;
}

// GPU-JSON gegen CPU-JSON: gleiche Felder in gleicher Reihenfolge, alles bitgleich außer
// backend/device, rms_waves/rms_mm (innerhalb der CPU/GPU-Toleranz) und piston_mm (die Engine
// liefert keinen Kolben, GPU null). Versagte Strahlen und Strahlzahlen sind damit identisch.
Deviation expect_gpu_json_matches_cpu(const json& gpu, const json& cpu) {
    Deviation dev;
    EXPECT_EQ(keys_of(gpu), keys_of(cpu));
    EXPECT_EQ(cpu["backend"], "cpu");
    EXPECT_TRUE(cpu["device"].is_null());
    EXPECT_EQ(gpu["backend"], "gpu");
    EXPECT_EQ(gpu["device"]["index"], 0);
    EXPECT_FALSE(gpu["device"]["name"].get<std::string>().empty());
    const double lambda_ref_mm = cpu["reference_nm"].get<double>() * 1e-6;
    for (auto it = cpu.begin(); it != cpu.end(); ++it) {
        const std::string& key = it.key();
        if (key == "backend" || key == "device") {
            continue;
        }
        if (key != "wavefront" || it.value().is_null()) {
            EXPECT_EQ(gpu[key], it.value()) << key;
            continue;
        }
        const json& ce = it.value();
        const json& ge = gpu[key];
        if (!ge.is_array() || ge.size() != ce.size()) {
            ADD_FAILURE() << "wavefront: GPU " << ge.dump() << ", CPU " << ce.dump();
            continue;
        }
        for (std::size_t i = 0; i < ce.size(); ++i) {
            EXPECT_EQ(keys_of(ge[i]), keys_of(ce[i]));
            for (auto e = ce[i].begin(); e != ce[i].end(); ++e) {
                const std::string what = "wavefront[" + std::to_string(i) + "]." + e.key();
                if (e.key() == "rms_waves") {
                    expect_rms_near(ge[i][e.key()], e.value(), lambda_ref_mm, what, dev);
                } else if (e.key() == "rms_mm") {
                    expect_rms_near(ge[i][e.key()], e.value(), 1.0, what, dev);
                } else if (e.key() == "piston_mm") {
                    EXPECT_TRUE(ge[i][e.key()].is_null()) << what;
                } else {
                    EXPECT_EQ(ge[i][e.key()], e.value()) << what;
                }
            }
        }
    }
    return dev;
}

void print_deviation(const std::string& name, const Deviation& d) {
    std::cout << "[          ] " << name << ": " << d.values
              << " RMS-Werte, max |GPU - CPU| = " << d.abs_mm << " mm, relativ " << d.rel << '\n';
}

json run_json(std::initializer_list<std::string_view> args) {
    const RunResult r = run(args);
    EXPECT_EQ(r.code, mko::cli::kExitOk) << r.err;
    return json::parse(r.out);
}

std::vector<std::string> lines_of(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream in(s);
    for (std::string line; std::getline(in, line);) {
        lines.push_back(line);
    }
    return lines;
}

// Zeile der Wellenfronttabelle (Feld, Winkel, Wellenlänge, Strahlen, versagt, RMS [Wellen]
// %.6f, RMS [mm] %.6e), bei der die Rundung der letzten Stelle zwischen GPU und CPU kippen
// darf: Die ersten fünf Spalten sind gleich, die RMS-Werte unterscheiden sich um höchstens
// eine Einheit der letzten gedruckten Stelle. Abweichungen von ~1e-12 relativ ändern die
// gedruckte Zahl nur, wenn der Wert fast genau auf einer Rundungsgrenze liegt.
bool same_row_within_rounding(const std::string& gpu, const std::string& cpu) {
    std::istringstream gs(gpu);
    std::istringstream cs(cpu);
    std::vector<std::string> g;
    std::vector<std::string> c;
    for (std::string t; gs >> t;) {
        g.push_back(t);
    }
    for (std::string t; cs >> t;) {
        c.push_back(t);
    }
    if (g.size() != 7 || c.size() != 7) {
        return false;
    }
    for (std::size_t i = 0; i < 5; ++i) {
        if (g[i] != c[i]) {
            return false;
        }
    }
    const double gw = std::stod(g[5]);
    const double cw = std::stod(c[5]);
    const double gm = std::stod(g[6]);
    const double cm = std::stod(c[6]);
    return std::abs(gw - cw) <= 1.5e-6 && std::abs(gm - cm) <= 1.5e-6 * std::abs(cm);
}

// Textausgabe GPU gegen CPU: identisch bis auf die Zeile "Backend" (und höchstens gekippte
// Rundung in einer Tabellenzeile, siehe same_row_within_rounding).
void expect_gpu_text_matches_cpu(const std::string& gpu, const std::string& cpu) {
    const std::vector<std::string> g = lines_of(gpu);
    const std::vector<std::string> c = lines_of(cpu);
    ASSERT_EQ(g.size(), c.size()) << gpu << "\n---\n" << cpu;
    const std::string backend = "  Backend" + std::string(44 - 7, ' ');
    int backend_lines = 0;
    int rounding_flips = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (c[i].rfind("  Backend ", 0) == 0) {
            ++backend_lines;
            EXPECT_EQ(c[i], backend + "CPU, FP64");
            EXPECT_EQ(g[i].rfind(backend + "GPU 0 (", 0), 0U) << g[i];
            EXPECT_TRUE(g[i].ends_with("), FP64")) << g[i];
            continue;
        }
        if (g[i] != c[i]) {
            ++rounding_flips;
            EXPECT_TRUE(same_row_within_rounding(g[i], c[i]))
                << "Zeile " << i << "\nGPU: " << g[i] << "\nCPU: " << c[i];
        }
    }
    EXPECT_EQ(backend_lines, 1);
    std::cout << "[          ] Text: " << c.size() << " Zeilen, gekippte Rundung in "
              << rounding_flips << " Zeilen\n";
}

// Plankonvexlinse mit R = 9,09 mm hinter einer Blende mit Radius 10 mm: Die äußeren Strahlen
// treffen die erste Fläche nicht (Missed), unter Feldwinkel auch weitere. Frei erfundenes
// Testsystem für versagte Strahlen.
std::string vignetting_json() {
    return std::string(R"({
  "format": "mikroptim.system/1",
  "name": "CLI-Test Vignettierung",
  "units": "mm",
  "glass_catalogs": [")") +
           MKO_SOURCE_DIR + R"(/data/glass/basis.json"],
  "spectrum": { "wavelengths_nm": [486.13, 587.56], "weights": [1.0, 1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0, 5.0, 10.0], "field_weights": [1.0, 1.0, 1.0] },
  "surfaces": [
    { "type": "stop", "semi_diameter": 10.0, "thickness": 2.0, "material": "AIR" },
    { "type": "sphere", "curvature": 0.11, "thickness": 6.0, "material": "N-BK7", "semi_diameter": 9.0 },
    { "type": "plane", "thickness": 13.6, "material": "AIR", "semi_diameter": 9.0 }
  ]
})";
}

} // namespace

// Abnahmekriterium M2-T5: `--gpu --json` gleich der CPU-Ausgabe innerhalb der Toleranz.
TEST(CliTraceGpu, JsonMatchesCpu) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    for (const std::string& file : {kEinzellinse, kMehrlinser}) {
        SCOPED_TRACE(file);
        const json cpu = run_json({file, "--json"});
        const json gpu = run_json({file, "--json", "--gpu"});
        ASSERT_TRUE(gpu["wavefront"].is_array());
        print_deviation(file, expect_gpu_json_matches_cpu(gpu, cpu));
    }
    // Andere Quadratur und ausdrücklich gewähltes Gerät 0
    const json cpu = run_json({kMehrlinser, "--json", "--rings", "16", "--spokes", "12"});
    const json gpu = run_json(
        {kMehrlinser, "--json", "--rings", "16", "--spokes", "12", "--gpu", "--device", "0"});
    print_deviation("Mehrlinser 16 x 12", expect_gpu_json_matches_cpu(gpu, cpu));
}

// Abnahmekriterium M2-T5: Textausgabe identisch bis auf die Backend-Zeile.
TEST(CliTraceGpu, TextMatchesCpu) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    for (const std::string& file : {kEinzellinse, kMehrlinser}) {
        SCOPED_TRACE(file);
        const RunResult cpu = run({file});
        const RunResult gpu = run({file, "--gpu"});
        ASSERT_EQ(cpu.code, mko::cli::kExitOk) << cpu.err;
        ASSERT_EQ(gpu.code, mko::cli::kExitOk) << gpu.err;
        EXPECT_TRUE(gpu.err.empty()) << gpu.err;
        expect_gpu_text_matches_cpu(gpu.out, cpu.out);
    }
}

// Versagte Strahlen pro Feld und Wellenlänge identisch.
TEST(CliTraceGpu, FailedRaysMatchCpu) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const TempFile file("mikroptim_cli_trace_vignetting.json", vignetting_json());
    const json cpu = run_json({file.path(), "--json"});
    const json gpu = run_json({file.path(), "--json", "--gpu"});
    ASSERT_TRUE(cpu["failed_rays"].is_number());
    EXPECT_GT(cpu["failed_rays"].get<int>(), 0);
    EXPECT_LT(cpu["failed_rays"].get<int>(), cpu["rays"].get<int>());
    print_deviation("Vignettierung", expect_gpu_json_matches_cpu(gpu, cpu));
    std::cout << "[          ] Vignettierung: " << cpu["failed_rays"] << " von " << cpu["rays"]
              << " Strahlen versagt\n";
}

// Punktquelle auf der Achse (axis), in der y-z-Ebene (half) und außerhalb (full).
TEST(CliTraceGpu, PointSourceMatchesCpu) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const std::vector<std::vector<std::string>> cases = {{"[0.0, 0.0, -50.0]", "axis"},
                                                         {"[0.0, 2.0, -50.0]", "half"},
                                                         {"[1.0, 2.0, -50.0]", "full"}};
    for (const std::vector<std::string>& c : cases) {
        SCOPED_TRACE(c[0]);
        const TempFile file("mikroptim_cli_trace_gpu_point.json",
                            lens_json(R"({ "type": "point", "point": )" + c[0] +
                                          R"(, "numerical_aperture": 0.05 })",
                                      "N-BK7"));
        const json cpu = run_json({file.path(), "--json"});
        const json gpu = run_json({file.path(), "--json", "--gpu"});
        ASSERT_EQ(cpu["wavefront"].size(), 1U);
        EXPECT_EQ(cpu["wavefront"][0]["pupil_mode"], c[1]);
        print_deviation("Punktquelle " + c[0], expect_gpu_json_matches_cpu(gpu, cpu));
    }
}

// Afokal ohne target.focal_length: wie im CPU-Pfad keine Wellenfront, gleiche Meldung.
TEST(CliTraceGpu, AfocalSameAsCpu) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const json cpu = run_json({kPlanplatte, "--json"});
    const json gpu = run_json({kPlanplatte, "--json", "--gpu"});
    EXPECT_TRUE(gpu["wavefront"].is_null());
    EXPECT_EQ(gpu["wavefront_error"], cpu["wavefront_error"]);
    expect_gpu_json_matches_cpu(gpu, cpu);

    const RunResult t = run({kPlanplatte, "--gpu"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    EXPECT_NE(t.out.find("Wellenfront nicht auswertbar"), std::string::npos) << t.out;
}

// `--gpu --precision df64`: Wellenfront in df64, gegen die CPU (FP64) innerhalb der
// df64-Toleranz (tests/tolerances.hpp); Strahlzahlen und versagte Strahlen identisch.
TEST(CliTraceGpu, Df64MatchesCpuWithinDf64Tolerance) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    for (const std::string& file : {kEinzellinse, kMehrlinser}) {
        SCOPED_TRACE(file);
        const json cpu = run_json({file, "--json"});
        const json gpu = run_json({file, "--json", "--gpu", "--precision", "df64"});
        EXPECT_EQ(cpu["precision"], "fp64");
        EXPECT_EQ(gpu["precision"], "df64");
        EXPECT_EQ(gpu["backend"], "gpu");
        EXPECT_EQ(gpu["rays"], cpu["rays"]);
        EXPECT_EQ(gpu["failed_rays"], cpu["failed_rays"]);
        ASSERT_TRUE(gpu["wavefront"].is_array());
        ASSERT_EQ(gpu["wavefront"].size(), cpu["wavefront"].size());
        double max_dev = 0.0;
        for (std::size_t i = 0; i < cpu["wavefront"].size(); ++i) {
            const json& c = cpu["wavefront"][i];
            const json& g = gpu["wavefront"][i];
            EXPECT_EQ(g["failed_rays"], c["failed_rays"]) << i;
            const double cr = c["rms_mm"].get<double>();
            const double gr = g["rms_mm"].get<double>();
            const double d = std::abs(gr - cr);
            EXPECT_LE(d, std::max(kDf64VsFp64, kDf64VsFp64Relative * std::abs(cr))) << i;
            max_dev = std::max(max_dev, d);
        }
        std::cout << "[          ] " << file << ": df64, max |GPU - CPU| = " << max_dev << " mm\n";
    }
    const RunResult t = run({kEinzellinse, "--gpu", "--precision", "df64"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    EXPECT_NE(t.out.find("), df64"), std::string::npos) << t.out;
}

TEST(CliTraceGpu, InvalidDeviceFails) {
    if (!have_gpu()) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const std::string device = std::to_string(mko::cuda::device_count());
    const RunResult r = run({kEinzellinse, "--gpu", "--device", device});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_TRUE(r.out.empty());
    EXPECT_NE(r.err.find("--device " + device), std::string::npos) << r.err;
}

#endif
