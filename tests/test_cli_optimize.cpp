// Unterbefehl `mikroptim optimize` (M3-T5): Merit sinkt, Ergebnis gleich der Bibliothek
// (optimize_lm_cpu), Ergebnisdatei lädt und reproduziert die Endmerit bitgleich (auch mit
// Geisterplatten), JSON-Protokoll gegen den Verlauf, Nebenbedingungen, Multistart
// deterministisch, Text, Fehlerfälle. Im CUDA-Build zusätzlich --gpu (übersprungen ohne Gerät).
// Der Aufruf des fertigen Programms steht als ctest-Kommando in tests/CMakeLists.txt.

#include "cli/optimize_command.hpp"
#include "cli/trace_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const std::string kSystems = std::string(MKO_SOURCE_DIR) + "/data/systems/";
const std::string kBiegung = kSystems + "einzellinse_bk7_biegung.json";
const std::string kNebenbedingungen = kSystems + "einzellinse_bk7_nebenbedingungen.json";
const std::string kVariablen = kSystems + "einzellinse_bk7_variablen.json";
const std::string kOhneVariablen = kSystems + "einzellinse_bk7.json";
const std::string kGeister = kSystems + "einzellinse_geisterplatte.json";
const std::string kBasisGlas = std::string(MKO_SOURCE_DIR) + "/data/glass/basis.json";

struct RunResult {
    int code = -1;
    std::string out;
    std::string err;
};

RunResult optimize(const std::vector<std::string>& args) {
    const std::vector<std::string_view> v(args.begin(), args.end());
    std::ostringstream out;
    std::ostringstream err;
    RunResult r;
    r.code = mko::cli::run_optimize_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

// Eigenes temporäres Verzeichnis pro Test (ctest startet Tests parallel), im Destruktor gelöscht.
class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_(fs::temp_directory_path() / ("mikroptim_cli_optimize_" + name)) {
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    std::string file(const std::string& name) const { return (path_ / name).string(); }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    f << content;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

std::vector<double> doubles(const json& a) {
    std::vector<double> v;
    for (const json& x : a) {
        v.push_back(x.get<double>());
    }
    return v;
}

// Merit wie im LM (CpuLmBackend: Strahlresiduen plus paraxiale Zusatzresiduen) für das System
// in `file` an seinem eigenen Variablenvektor, und dieser Vektor.
struct FileMerit {
    double merit = 0.0;
    std::int64_t failed = 0;
    std::vector<double> x;
};

FileMerit file_merit(const std::string& file, int rings = 10, int spokes = 8) {
    const mko::LoadedSystem loaded = mko::load_system(fs::path(file));
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::LmOptions options;
    options.jacobian.quadrature.rings = rings;
    options.jacobian.quadrature.spokes = spokes;
    // wie mikroptim optimize: paraxiale Nebenbedingungen, dahinter die L1-Strafe (M4-T6)
    options.jacobian.extra = mko::concat_extra_residuals(
        {mko::make_paraxial_residuals(mko::make_paraxial_constraints(loaded.system)),
         mko::make_ghost_penalty_residuals(mko::make_ghost_penalty(loaded.system))});
    FileMerit r;
    r.x = set.extract(base, 0);
    std::vector<double> f(1);
    std::vector<std::int64_t> failed(1);
    std::vector<std::int64_t> hard_failed(1);
    const int run = 0;
    mko::CpuLmBackend backend(set, base, options);
    backend.merit(std::span<const int>(&run, 1), r.x, f, failed, hard_failed);
    r.merit = f[0];
    r.failed = failed[0];
    return r;
}

// Einzellinse mit Geisterplatten (einzellinse_geisterplatte.json) mit absolutem Glaskatalog,
// Variablen c (alle), Dicke des geteilten Spalts hinter der Blende in [4; 8] und w;
// `with_target`: target.focal_length = 100 (feste Feldziele beim erneuten Laden).
std::string ghost_system_json(bool with_target) {
    json j = json::parse(read_file(kGeister));
    j["glass_catalogs"] = json::array({fs::path(kBasisGlas).generic_string()});
    if (with_target) {
        j["target"] = {{"focal_length", 100.0}};
    }
    j["variables"] = {
        {"items", json::array({{{"type", "curvature"}, {"surfaces", "all"}},
                               {{"type", "thickness"}, {"surface", 0}, {"min", 4.0}, {"max", 8.0}},
                               {{"type", "working_distance"}, {"min", 50.0}, {"max", 150.0}}})}};
    return j.dump(2);
}

// Formfaktor q = (c₁ + c₂)/(c₁ − c₂).
double shape_factor(const std::vector<double>& x) {
    return (x[0] + x[1]) / (x[0] - x[1]);
}

} // namespace

// Abnahme: Merit sinkt; CLI-Ergebnis bitgleich zur Bibliothek; Ergebnisdatei lädt (auch mit
// `mikroptim trace`) und liefert beim Neuladen dieselbe Merit bitgleich: Die Datei enthält x
// verlustfrei (kürzeste Darstellung), der Lader baut daraus denselben Batch wie
// lm_point_batch, und die Merit ist eine deterministische Funktion des Batches.
TEST(CliOptimize, MeritDecreasesAndResultReproducesMerit) {
    const TempDir dir("reload");
    const std::string out_file = dir.file("ergebnis.json");
    const RunResult r = optimize({kBiegung, "-o", out_file, "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_EQ(r.err, "");
    const json j = json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.optimize/1");
    EXPECT_EQ(j["backend"], "cpu");
    EXPECT_EQ(j["output"], out_file);
    const json& res = j["result"];
    EXPECT_TRUE(res["converged"].get<bool>()) << res["status"];
    const double f0 = res["start_merit"].get<double>();
    const double f1 = res["merit"].get<double>();
    EXPECT_LT(f1, f0);
    EXPECT_LT(f1, 1e-4 * f0); // dokumentiert: 3,13e-2 → 1,56e-6 Wellen²
    EXPECT_EQ(res["failed_rays"], 0);

    // Gleich der Bibliothek (optimize_lm_cpu mit denselben Einstellungen)
    const mko::LoadedSystem loaded = mko::load_system(fs::path(kBiegung));
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::LmOptions options;
    options.jacobian.extra =
        mko::make_paraxial_residuals(mko::make_paraxial_constraints(loaded.system));
    const std::vector<double> x0 = set.extract(base, 0);
    const mko::LmResult lib = mko::optimize_lm_cpu(set, base, x0, options);
    const std::vector<double> x = doubles(j["runs"][0]["x"]);
    ASSERT_EQ(x.size(), lib.runs[0].x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
        EXPECT_TRUE(same_bits(x[k], lib.runs[0].x[k])) << k;
        EXPECT_TRUE(same_bits(j["variables"][k]["result"].get<double>(), x[k])) << k;
        EXPECT_TRUE(same_bits(j["variables"][k]["start"].get<double>(), x0[k])) << k;
    }
    EXPECT_TRUE(same_bits(f1, lib.runs[0].merit));
    EXPECT_TRUE(same_bits(f0, lib.runs[0].history[0]));
    EXPECT_EQ(res["iterations"], lib.runs[0].iterations);
    // Biegung minimaler sphärischer Aberration (docs/optimierung.md: q = 0,7208)
    EXPECT_NEAR(shape_factor(x), 0.7208, mko::test::kDocumentedValue);

    // Ergebnisdatei: gleicher Variablenvektor und bitgleiche Merit beim Neuladen
    const FileMerit reloaded = file_merit(out_file);
    ASSERT_EQ(reloaded.x.size(), x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
        EXPECT_TRUE(same_bits(reloaded.x[k], x[k])) << k;
    }
    EXPECT_TRUE(same_bits(reloaded.merit, f1)) << reloaded.merit << " " << f1;
    EXPECT_EQ(reloaded.failed, 0);

    // Dasselbe über die CLI: optimize mit 0 Zyklen wertet nur aus
    const RunResult again = optimize({out_file, "--max-iterations", "0", "--json"});
    ASSERT_EQ(again.code, mko::cli::kExitOk) << again.err;
    const json ja = json::parse(again.out);
    EXPECT_TRUE(same_bits(ja["result"]["start_merit"].get<double>(), f1));
    EXPECT_TRUE(same_bits(ja["result"]["merit"].get<double>(), f1));
    EXPECT_EQ(ja["result"]["iterations"], 0);
    EXPECT_EQ(ja["result"]["status"], "max_iterations");
    EXPECT_EQ(ja["iterations"].size(), 0U);

    // mikroptim trace lädt die Ergebnisdatei (Glaskatalog relativ zum neuen Verzeichnis)
    std::ostringstream tout;
    std::ostringstream terr;
    const std::vector<std::string_view> targs{out_file, "--json"};
    ASSERT_EQ(mko::cli::run_trace_command(targs, tout, terr), mko::cli::kExitOk) << terr.str();
    const json jt = json::parse(tout.str());
    EXPECT_NEAR(jt["paraxial"]["focal_length"].get<double>(), 100.0, mko::test::kDocumentedValue);
    // Wellenfront „nachher“ der Optimierung = Wellenfront von trace auf der Ergebnisdatei
    EXPECT_TRUE(same_bits(jt["wavefront"][0]["rms_waves"].get<double>(),
                          j["wavefront"]["result"][0]["rms_waves"].get<double>()));
    // Eingabedatei unverändert, nur die Variablen der Ergebnisdatei unterscheiden sich
    const mko::SystemDescription written = mko::read_system_file(fs::path(out_file));
    EXPECT_EQ(written.surfaces[1].curvature, x[0]);
    EXPECT_EQ(written.surfaces[2].curvature, x[1]);
    EXPECT_EQ(written.surfaces[2].thickness, x[2]);
    EXPECT_EQ(written.variables, loaded.system.variables);
    EXPECT_EQ(written.constraints, loaded.system.constraints);
    EXPECT_EQ(written.surfaces[1].thickness, loaded.system.surfaces[1].thickness);
}

// Das JSON-Protokoll enthält jeden Zyklus; die Merit nach Zyklus k ist history[k].
TEST(CliOptimize, JsonIterationsMatchLibraryHistory) {
    const RunResult r = optimize({kBiegung, "--json", "--max-iterations", "20"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    const json& its = j["iterations"];
    ASSERT_EQ(its.size(), 20U);
    EXPECT_EQ(j["result"]["status"], "max_iterations");

    const mko::LoadedSystem loaded = mko::load_system(fs::path(kBiegung));
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::LmOptions options;
    options.max_iterations = 20;
    options.jacobian.extra =
        mko::make_paraxial_residuals(mko::make_paraxial_constraints(loaded.system));
    const mko::LmResult lib = mko::optimize_lm_cpu(set, base, set.extract(base, 0), options);
    ASSERT_EQ(lib.runs[0].history.size(), 21U);
    int accepted = 0;
    for (std::size_t k = 0; k < its.size(); ++k) {
        EXPECT_EQ(its[k]["run"], 0);
        EXPECT_EQ(its[k]["iteration"], static_cast<int>(k) + 1);
        EXPECT_TRUE(same_bits(its[k]["merit"].get<double>(), lib.runs[0].history[k + 1])) << k;
        EXPECT_TRUE(its[k]["evaluated"].get<bool>());
        EXPECT_GT(its[k]["damping"].get<double>(), 0.0);
        EXPECT_GE(its[k]["step_norm"].get<double>(), 0.0);
        accepted += its[k]["accepted"].get<bool>() ? 1 : 0;
    }
    EXPECT_EQ(accepted, lib.runs[0].accepted);

    // --quiet: kein Protokoll
    const RunResult q = optimize({kBiegung, "--json", "--quiet", "--max-iterations", "3"});
    ASSERT_EQ(q.code, mko::cli::kExitOk) << q.err;
    EXPECT_TRUE(json::parse(q.out)["iterations"].is_null());
}

// Nebenbedingungen: Istwerte und Residuen am Start gegen Paraxialdaten und Dicken.
TEST(CliOptimize, ConstraintsReportedBeforeAndAfter) {
    const RunResult r = optimize({kNebenbedingungen, "--json", "--max-iterations", "10"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    const json& c = j["constraints"];
    ASSERT_EQ(c.size(), 3U);
    const mko::LoadedSystem loaded = mko::load_system(fs::path(kNebenbedingungen));
    const mko::ParaxialData p = mko::compute_paraxial(loaded.system, loaded.catalog);

    EXPECT_EQ(c[0]["type"], "focal_length");
    EXPECT_EQ(c[0]["mode"], "equal");
    EXPECT_EQ(c[0]["target"].get<double>(), 100.0);
    EXPECT_TRUE(same_bits(c[0]["value_start"].get<double>(), p.focal_length));
    const double q_focal = 100.0 * (1.0 - 100.0 * p.power);
    EXPECT_NEAR(c[0]["residual_start"].get<double>(), 10.0 * q_focal,
                mko::test::kParaxialRelative * 100.0 * std::abs(q_focal));

    EXPECT_EQ(c[1]["type"], "parfocal_length");
    EXPECT_EQ(c[1]["value_start"].get<double>(), ((5.0 + 5.0) + 6.0) + 99.05);
    EXPECT_EQ(c[1]["residual_start"].get<double>(), ((5.0 + 5.0) + 6.0) + 99.05 - 115.0);

    EXPECT_EQ(c[2]["type"], "working_distance");
    EXPECT_EQ(c[2]["mode"], "min");
    EXPECT_EQ(c[2]["value_start"].get<double>(), 99.05);
    EXPECT_EQ(c[2]["residual_start"].get<double>(), 0.0); // Untergrenze 95 inaktiv

    // Nachher: Werte passen zur Ergebnisvariante (w = Arbeitsabstand)
    const std::vector<double> x = doubles(j["runs"][0]["x"]);
    EXPECT_EQ(c[2]["value_result"].get<double>(), x[3]);
    EXPECT_LT(std::abs(c[0]["residual_result"].get<double>()),
              std::abs(c[0]["residual_start"].get<double>()));
    EXPECT_LT(j["result"]["merit"].get<double>(), j["result"]["start_merit"].get<double>());
}

// Mit Geisterplatten: Die Dicke des geteilten Spalts wird als Dateidicke geschrieben; das
// erneute Laden ergibt dasselbe letzte Teilstück fl(x − d) und damit bitgleich dieselbe Merit.
TEST(CliOptimize, GhostPlateResultReproducesMerit) {
    const TempDir dir("ghost");
    const std::string in_file = dir.file("geister.json");
    write_file(in_file, ghost_system_json(true));
    const std::string out_file = dir.file("geister_opt.json");
    const RunResult r = optimize({in_file, "-o", out_file, "--json", "--max-iterations", "5",
                                  "--rings", "4", "--spokes", "4"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    const double f1 = j["result"]["merit"].get<double>();
    EXPECT_LT(f1, j["result"]["start_merit"].get<double>());
    EXPECT_TRUE(j["warnings"].empty());
    const FileMerit reloaded = file_merit(out_file, 4, 4);
    EXPECT_TRUE(same_bits(reloaded.merit, f1)) << reloaded.merit << " " << f1;
    const std::vector<double> x = doubles(j["runs"][0]["x"]);
    EXPECT_EQ(reloaded.x[2], x[2]); // Dicke des Spalts mit Geisterplatten
    // Absoluter Glaskatalog bleibt unverändert
    const mko::SystemDescription written = mko::read_system_file(fs::path(out_file));
    ASSERT_EQ(written.glass_catalogs.size(), 1U);
    EXPECT_EQ(written.glass_catalogs[0], fs::path(kBasisGlas).generic_string());
}

// L1-Strafe und Diagnosebericht (M4-T6): Geisterkoeffizienten als Variablen, Strafe als
// Zusatzresiduen (Ergebnisdatei reproduziert die Merit einschließlich Strafe bitgleich),
// Abschnitt ghost_diagnosis im JSON; ohne Geisterplatten null.
TEST(CliOptimize, GhostPenaltyAndDiagnosis) {
    const TempDir dir("ghost_penalty");
    const std::string in_file = dir.file("exakt.json");
    json in = json::parse(read_file(kSystems + "einzellinse_geisterplatten_exakt.json"));
    in["glass_catalogs"] = json::array({fs::path(kBasisGlas).generic_string()});
    write_file(in_file, in.dump(2));
    const std::string out_file = dir.file("exakt_opt.json");
    const RunResult r =
        optimize({in_file, "-o", out_file, "--json", "--quiet", "--max-iterations", "1000"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_TRUE(j["result"]["converged"].get<bool>());
    // Variablen: Geisterkoeffizienten mit Platte und Ordnung, alle |a| < 1e-8 (Abnahme
    // Exaktheit auch über die Kommandozeile)
    int ghosts = 0;
    for (const json& v : j["variables"]) {
        if (v["type"] == "ghost_coefficient") {
            ++ghosts;
            EXPECT_TRUE(v["surface"].is_null());
            EXPECT_TRUE(v["plate"].is_number_integer());
            EXPECT_GE(v["order"].get<int>(), 1);
            EXPECT_LT(std::abs(v["result"].get<double>()), mko::test::kGhostExactCoefficient)
                << v["name"];
        } else {
            EXPECT_TRUE(v["plate"].is_null());
        }
    }
    EXPECT_EQ(ghosts, 16);
    // Diagnose: vier Platten, keine aktiv, keine Empfehlung
    const json& d = j["ghost_diagnosis"];
    ASSERT_TRUE(d.is_object());
    EXPECT_EQ(d["plates"].size(), 4U);
    EXPECT_EQ(d["positions"].size(), 3U);
    EXPECT_EQ(d["recommendation"], "none");
    EXPECT_TRUE(d["recommended_position"].is_null());
    EXPECT_TRUE(same_bits(d["merit"].get<double>(), j["result"]["merit"].get<double>()));
    for (const json& p : d["plates"]) {
        EXPECT_FALSE(p["active"].get<bool>());
    }
    // Ergebnisdatei: Koeffizienten verlustfrei, Merit mit Strafe bitgleich
    const FileMerit reloaded = file_merit(out_file);
    EXPECT_TRUE(same_bits(reloaded.merit, j["result"]["merit"].get<double>()))
        << reloaded.merit << " " << j["result"]["merit"];
    EXPECT_EQ(reloaded.x, doubles(j["runs"][0]["x"]));

    // Text: Abschnitt der Diagnose mit Empfehlung (Bedarf bei NA 0,5)
    const RunResult t = optimize({kSystems + "einzellinse_na05_geisterplatten.json", "--quiet"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    for (const char* s : {"Geisterplatten                              9 (L1-Strafe: 54",
                          "\nGeisterplatten (Diagnose", "  a[8,2]  mm", "Lagen:",
                          "Empfehlung: Operator A (Meniskus ohne Brechkraft) nach Fläche 2"}) {
        EXPECT_NE(t.out.find(s), std::string::npos) << s << "\n" << t.out;
    }
    // Ohne Geisterplatten kein Abschnitt
    const RunResult n = optimize({kBiegung, "--json", "--quiet", "--max-iterations", "2"});
    ASSERT_EQ(n.code, mko::cli::kExitOk) << n.err;
    EXPECT_TRUE(json::parse(n.out)["ghost_diagnosis"].is_null());
    const RunResult nt = optimize({kBiegung, "--quiet", "--max-iterations", "2"});
    EXPECT_EQ(nt.out.find("Geisterplatten"), std::string::npos);
}

// Glasvariable und Glas-Hülle (M6-T1): Das Startglas wird ein Buchdahl-Material, n0, nu1, nu2
// sind Variablen, die Strafe glass_hull geht in die Merit ein; die Ergebnisdatei enthält das
// Glas als {"model": "buchdahl"} mit festem ν₃ und reproduziert die Merit bitgleich.
TEST(CliOptimize, GlassVariableAndHull) {
    const TempDir dir("glass");
    const std::string out_file = dir.file("glas_opt.json");
    const std::string in_file = kSystems + "einzellinse_glasvariable.json";
    const RunResult r = optimize({in_file, "-o", out_file, "--json", "--quiet", "--max-iterations",
                                  "200", "--rings", "6", "--spokes", "4"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_LT(j["result"]["merit"].get<double>(), j["result"]["start_merit"].get<double>());
    std::vector<std::string> names;
    for (const json& v : j["variables"]) {
        names.push_back(v["name"].get<std::string>());
    }
    EXPECT_EQ(names, (std::vector<std::string>{"c[1]", "c[2]", "n0[1]", "nu1[1]", "nu2[1]", "w"}));
    EXPECT_EQ(j["variables"][2]["type"], "glass");
    EXPECT_EQ(j["variables"][2]["surface"], 1);
    const json& g = j["glass_hull"];
    ASSERT_TRUE(g.is_object());
    EXPECT_EQ(g["glasses"], 39);
    EXPECT_EQ(g["dimension"], 3);
    EXPECT_EQ(g["weight"].get<double>(), 1e8);
    ASSERT_EQ(g["elements"].size(), 1U);
    EXPECT_EQ(g["elements"][0]["surface"], 1);
    EXPECT_LE(g["elements"][0]["excess_start"].get<double>(), mko::test::kHullDistance);
    std::printf("Glas-Hülle über die Kommandozeile: Überschreitung am Ende %.3e\n",
                g["elements"][0]["excess_result"].get<double>());

    // Ergebnisdatei: Buchdahl-Material mit den Ergebniskoordinaten, ν₃ des N-BK7-Fits bleibt
    const std::vector<double> x = doubles(j["runs"][0]["x"]);
    const mko::LoadedSystem loaded = mko::load_system(fs::path(out_file));
    const mko::MaterialSpec& m = loaded.system.surfaces[1].material;
    ASSERT_EQ(m.kind, mko::MaterialKind::Buchdahl);
    EXPECT_EQ(m.n0, x[2]);
    EXPECT_EQ(m.nu[0], x[3]);
    EXPECT_EQ(m.nu[1], x[4]);
    const mko::BuchdahlFit fit = mko::fit_buchdahl(loaded.catalog.at("N-BK7"), 3);
    EXPECT_EQ(m.nu[2], fit.nu[2]);
    // Merit der Ergebnisdatei (Strahlen, Brennweite, Glas-Hülle) bitgleich
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::LmOptions options;
    options.jacobian.quadrature.rings = 6;
    options.jacobian.quadrature.spokes = 4;
    options.jacobian.extra = mko::concat_extra_residuals(
        {mko::make_paraxial_residuals(mko::make_paraxial_constraints(loaded.system)),
         mko::make_glass_hull_residuals(mko::make_glass_hull_penalty(
             loaded.system, set, mko::make_glass_hull(loaded.catalog)))});
    const std::vector<double> x_file = set.extract(base, 0);
    EXPECT_EQ(x_file, x);
    std::vector<double> f(1);
    std::vector<std::int64_t> failed(1);
    std::vector<std::int64_t> hard_failed(1);
    const int run = 0;
    mko::CpuLmBackend backend(set, base, options);
    backend.merit(std::span<const int>(&run, 1), x_file, f, failed, hard_failed);
    EXPECT_TRUE(same_bits(f[0], j["result"]["merit"].get<double>()))
        << f[0] << " " << j["result"]["merit"];

    // Text: Abschnitte Glasvariablen und Glas-Hülle; ohne Glasvariablen kein Abschnitt
    const RunResult t = optimize({in_file, "--quiet", "--max-iterations", "2"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    for (const char* s :
         {"Glasvariablen (Startglas als Buchdahl-Material", "  Fläche 1: N-BK7 (Fit, max. Fehler",
          "Glas-Hülle (39 Gläser, Dimension 3", "  n0[1]   1   "}) {
        EXPECT_NE(t.out.find(s), std::string::npos) << s << "\n" << t.out;
    }
    const RunResult n = optimize({kBiegung, "--json", "--quiet", "--max-iterations", "2"});
    ASSERT_EQ(n.code, mko::cli::kExitOk) << n.err;
    EXPECT_TRUE(json::parse(n.out)["glass_hull"].is_null());
}

// Ohne target.focal_length mit Feldwinkeln ≠ 0 hängen die Feldziele von der Brennweite des
// Startsystems ab: Warnung auf stderr und im JSON.
TEST(CliOptimize, WarnsAboutFieldTargetsFromStartFocalLength) {
    const TempDir dir("warn");
    const std::string in_file = dir.file("geister.json");
    write_file(in_file, ghost_system_json(false));
    const RunResult r =
        optimize({in_file, "--json", "--max-iterations", "1", "--rings", "2", "--spokes", "2"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.err.find("mikroptim optimize: Warnung: Feldziele"), std::string::npos) << r.err;
    const json j = json::parse(r.out);
    ASSERT_EQ(j["warnings"].size(), 1U);
}

// Multistart: deterministisch (zwei Aufrufe gleiche Ausgabe), Lauf 0 = x₀, Läufe j ≥ 1 aus
// perturbed_vector, bester Lauf nach der dokumentierten Regel.
TEST(CliOptimize, MultistartIsDeterministic) {
    const std::vector<std::string> args{kBiegung,         "--json", "--quiet", "--runs", "4",
                                        "--perturbation", "0.1",    "--seed",  "7"};
    const RunResult a = optimize(args);
    const RunResult b = optimize(args);
    ASSERT_EQ(a.code, mko::cli::kExitOk) << a.err;
    EXPECT_EQ(a.out, b.out);
    const json j = json::parse(a.out);
    const json& runs = j["runs"];
    ASSERT_EQ(runs.size(), 4U);
    EXPECT_EQ(j["settings"]["runs"], 4);
    EXPECT_EQ(j["settings"]["seed"], 7);

    const mko::LoadedSystem loaded = mko::load_system(fs::path(kBiegung));
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    const std::vector<double> x0 = set.extract(base, 0);
    mko::PerturbationOptions p;
    p.curvature = {0.1, 0.0};
    p.thickness = {0.1, 0.0};
    p.working_distance = {0.1, 0.0};
    p.seed = 7;
    p.first_is_base = true;
    int best = 0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const std::vector<double> start = doubles(runs[i]["start"]);
        const std::vector<double> expected = mko::perturbed_vector(set, x0, p, i);
        ASSERT_EQ(start.size(), expected.size());
        for (std::size_t k = 0; k < start.size(); ++k) {
            EXPECT_TRUE(same_bits(start[k], expected[k])) << i << " " << k;
        }
        const json& rb = runs[static_cast<std::size_t>(best)];
        if (runs[i]["failed_rays"] < rb["failed_rays"] ||
            (runs[i]["failed_rays"] == rb["failed_rays"] &&
             runs[i]["merit"].get<double>() < rb["merit"].get<double>())) {
            best = static_cast<int>(i);
        }
    }
    EXPECT_EQ(j["best_run"], best);
    EXPECT_TRUE(same_bits(j["result"]["merit"].get<double>(),
                          runs[static_cast<std::size_t>(best)]["merit"].get<double>()));
    EXPECT_TRUE(
        same_bits(j["result"]["start_merit"].get<double>(), runs[0]["start_merit"].get<double>()));
    // Lauf 0 = Einzellauf
    const RunResult single = optimize({kBiegung, "--json", "--quiet"});
    ASSERT_EQ(single.code, mko::cli::kExitOk);
    EXPECT_TRUE(same_bits(json::parse(single.out)["result"]["merit"].get<double>(),
                          runs[0]["merit"].get<double>()));

    // Anderer Startwert, andere Startpunkte
    const RunResult other = optimize(
        {kBiegung, "--json", "--quiet", "--runs", "2", "--perturbation", "0.1", "--seed", "8"});
    ASSERT_EQ(other.code, mko::cli::kExitOk);
    EXPECT_NE(json::parse(other.out)["runs"][1]["start"], runs[1]["start"]);
}

// Textausgabe: Kopf, Protokoll, Zusammenfassung; --quiet ohne Protokoll.
TEST(CliOptimize, TextOutput) {
    const TempDir dir("text");
    const std::string out_file = dir.file("ergebnis.json");
    const RunResult r = optimize({kBiegung, "-o", out_file});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    for (const char* s :
         {"Optimierung: Einzellinse N-BK7, Biegung\n", "Iterationsprotokoll", "\nErgebnis\n",
          "\n  Status    ", "\n  Zyklen    ", "\nVariablen\n", "  c[1]    1/mm",
          "\nNebenbedingungen", "focal_length", "\nWellenfront", "Ergebnis geschrieben: "}) {
        EXPECT_NE(r.out.find(s), std::string::npos) << s << "\n" << r.out;
    }
    // Eine Zeile pro Zyklus
    std::size_t lines = 0;
    for (std::size_t pos = r.out.find("angenommen"); pos != std::string::npos;
         pos = r.out.find("angenommen", pos + 1)) {
        ++lines;
    }
    EXPECT_GE(lines, 40U);

    const RunResult q = optimize({kBiegung, "--quiet", "--max-iterations", "2"});
    ASSERT_EQ(q.code, mko::cli::kExitOk);
    EXPECT_EQ(q.out.find("Iterationsprotokoll"), std::string::npos);
    EXPECT_NE(q.out.find("Merit nachher"), std::string::npos);
    EXPECT_NE(q.out.find("- (ohne -o)"), std::string::npos);
}

// Weiches Strahlversagen (M4-T2): Das Startsystem mit 30 % versagenden Strahlen wird ohne
// Eingriff versagensfrei; Ergebnis bitgleich zu optimize_lm_cpu (äußeres Strafverfahren mit
// w_fail ab 1e6, x10 pro Runde); Protokoll mit fortlaufenden Zyklen und hart versagten
// Strahlen; --hard-failure behält die versagten Strahlen (eine Runde, w_fail null in JSON);
// --fail-weight setzt das Startgewicht.
TEST(CliOptimize, SoftFailureStartBecomesFailureFree) {
    const std::string file = kSystems + "weiches_versagen_start.json";
    const RunResult r = optimize({file, "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["settings"]["failure"], "soft");
    EXPECT_EQ(j["settings"]["fail_weight"].get<double>(), mko::kDefaultFailWeight);
    const json& res = j["result"];
    EXPECT_EQ(res["start_failed_rays"], 27);
    EXPECT_EQ(res["start_hard_failed_rays"], 0);
    EXPECT_EQ(res["failed_rays"], 0);
    EXPECT_EQ(res["hard_failed_rays"], 0);
    EXPECT_TRUE(res["converged"].get<bool>()) << res["status"];
    const int rounds = res["penalty_rounds"].get<int>();
    EXPECT_GE(rounds, 2);
    EXPECT_EQ(res["fail_weight"].get<double>(),
              mko::kDefaultFailWeight * std::pow(10.0, rounds - 1));
    EXPECT_EQ(j["runs"][0]["penalty_rounds"], rounds);
    EXPECT_EQ(j["runs"][0]["hard_failed_rays"], 0);

    const mko::LoadedSystem loaded = mko::load_system(fs::path(file));
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    const mko::LmResult lib = mko::optimize_lm_cpu(set, base, set.extract(base, 0), {});
    const mko::LmRunResult& run = lib.runs[0];
    EXPECT_EQ(run.penalty_rounds, rounds);
    EXPECT_EQ(run.iterations, res["iterations"].get<int>());
    EXPECT_TRUE(same_bits(run.merit, res["merit"].get<double>()));
    const std::vector<double> x = doubles(j["runs"][0]["x"]);
    ASSERT_EQ(x.size(), run.x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
        EXPECT_TRUE(same_bits(x[k], run.x[k])) << k;
    }
    // Protokoll: Zyklen über die Runden fortlaufend, am Ende ohne versagte Strahlen
    const json& its = j["iterations"];
    ASSERT_EQ(its.size(), static_cast<std::size_t>(run.iterations));
    for (std::size_t k = 0; k < its.size(); ++k) {
        EXPECT_EQ(its[k]["iteration"], static_cast<int>(k) + 1);
        EXPECT_LE(its[k]["hard_failed_rays"].get<std::int64_t>(),
                  its[k]["failed_rays"].get<std::int64_t>());
    }
    EXPECT_EQ(its.back()["failed_rays"], 0);

    const RunResult h = optimize({file, "--json", "--quiet", "--hard-failure"});
    ASSERT_EQ(h.code, mko::cli::kExitOk) << h.err;
    const json jh = json::parse(h.out);
    EXPECT_EQ(jh["settings"]["failure"], "hard");
    EXPECT_TRUE(jh["settings"]["fail_weight"].is_null());
    EXPECT_EQ(jh["result"]["start_hard_failed_rays"], 27);
    EXPECT_GT(jh["result"]["hard_failed_rays"].get<std::int64_t>(), 0);
    EXPECT_EQ(jh["result"]["hard_failed_rays"], jh["result"]["failed_rays"]);
    EXPECT_EQ(jh["result"]["penalty_rounds"], 1);
    EXPECT_TRUE(jh["result"]["fail_weight"].is_null());

    const RunResult w = optimize({file, "--json", "--quiet", "--fail-weight", "1e9"});
    ASSERT_EQ(w.code, mko::cli::kExitOk) << w.err;
    const json jw = json::parse(w.out);
    EXPECT_EQ(jw["settings"]["fail_weight"].get<double>(), 1e9);
    EXPECT_GE(jw["result"]["fail_weight"].get<double>(), 1e9);
    EXPECT_EQ(jw["result"]["failed_rays"], 0);

    // Text: Kopf und Zusammenfassung nennen Versagensart, Runden und w_fail
    const RunResult t = optimize({file, "--quiet"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    for (const char* s :
         {"Strahlversagen", "weich, w_fail ab 1.000e+06", "davon hart vorher/nachher",
          "Runden des Strafverfahrens", "w_fail am Ende"}) {
        EXPECT_NE(t.out.find(s), std::string::npos) << s << "\n" << t.out;
    }
}

TEST(CliOptimize, Errors) {
    const TempDir dir("errors");
    // keine Variablen
    RunResult r = optimize({kOhneVariablen});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("keine Variablen"), std::string::npos) << r.err;
    EXPECT_EQ(r.out, "");
    // Datei fehlt
    r = optimize({kSystems + "gibt_es_nicht.json"});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_EQ(r.out, "");
    // Ergebnis = Eingabe: Kommandozeilenfehler, Eingabe unverändert
    const std::string in_file = dir.file("eingabe.json");
    write_file(in_file, read_file(kVariablen));
    const std::string before = read_file(in_file);
    r = optimize({in_file, "-o", in_file});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    EXPECT_NE(r.err.find("Eingabedatei"), std::string::npos) << r.err;
    r = optimize({in_file, "-o", (dir.path() / "." / "eingabe.json").string()});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    EXPECT_EQ(read_file(in_file), before);
    // Verzeichnis der Ergebnisdatei existiert nicht
    r = optimize(
        {kBiegung, "--quiet", "--max-iterations", "1", "-o", dir.file("fehlt/ergebnis.json")});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("mikroptim optimize: Fehler: "), std::string::npos);
    // Kommandozeile
    for (const std::vector<std::string>& args :
         std::vector<std::vector<std::string>>{{},
                                               {kBiegung, "--unbekannt"},
                                               {kBiegung, kBiegung},
                                               {kBiegung, "-o"},
                                               {kBiegung, "--runs", "0"},
                                               {kBiegung, "--max-iterations", "-1"},
                                               {kBiegung, "--perturbation", "0.1"},
                                               {kBiegung, "--seed", "3"},
                                               {kBiegung, "--runs", "2", "--perturbation", "-1"},
                                               {kBiegung, "--runs", "2", "--perturbation", "x"},
                                               {kBiegung, "--runs", "2", "--seed", "-1"},
                                               {kBiegung, "--rings", "17"},
                                               {kBiegung, "--spokes", "0"},
                                               {kBiegung, "--fail-weight", "0"},
                                               {kBiegung, "--fail-weight", "-1"},
                                               {kBiegung, "--fail-weight", "x"},
                                               {kBiegung, "--fail-weight", "1e8", "--hard-failure"},
                                               {kBiegung, "--device", "0"}}) {
        r = optimize(args);
        EXPECT_EQ(r.code, mko::cli::kExitUsage) << (args.empty() ? "" : args.back());
        EXPECT_NE(r.err.find("Aufruf: mikroptim optimize"), std::string::npos);
    }
    r = optimize({"--help"});
    EXPECT_EQ(r.code, mko::cli::kExitOk);
    EXPECT_NE(r.out.find("Aufruf: mikroptim optimize"), std::string::npos);
}

#ifndef MIKROPTIM_HAS_CUDA
TEST(CliOptimize, GpuUnavailableWithoutCuda) {
    const RunResult r = optimize({kBiegung, "--gpu"});
    EXPECT_EQ(r.code, mko::cli::kExitNoGpu);
    EXPECT_NE(r.err.find("--gpu: ohne CUDA gebaut"), std::string::npos) << r.err;
    EXPECT_EQ(r.out, "");
}
#else

// Schranke für |F_GPU - F_CPU| derselben Variante aus der Toleranz der OPD: Weicht die OPD jedes
// Strahls um höchstens d = kCpuGpuFp64 ab, gilt für die Strahlresiduen (Wellen, M4-T1)
// ||dr||_2 <= N d mit N = residual_scale_norm, also |F' - F| <= 2 sqrt(F) N d + (N d)^2
// (Cauchy-Schwarz). Die paraxialen Zusatzresiduen rechnet in beiden Fällen der Host (bitgleich).
// Eine relative Schranke passt nicht: Am Optimum ist die OPD ~1e-6 mm, ihre absolute Abweichung
// aber ~1e-14 mm (Kolbenabzug von OPL ~100 mm).
namespace {
double cpu_gpu_merit_bound(double merit, const std::string& file) {
    const mko::LoadedSystem loaded = mko::load_system(fs::path(file));
    const mko::SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    const mko::ResidualLayout layout(batch.topology());
    const double nd = mko::test::residual_scale_norm(batch, layout) * mko::test::kCpuGpuFp64;
    return 2.0 * std::sqrt(merit) * nd + nd * nd;
}
} // namespace

// GPU gegen CPU: gleiches Optimum (Formfaktor), Ergebnisdatei reproduziert die GPU-Endmerit
// bitgleich auf der GPU (deterministische Kernel) und innerhalb cpu_gpu_merit_bound auf der CPU.
TEST(CliOptimizeGpu, MatchesCpuAndReproducesMerit) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const TempDir dir("gpu");
    const std::string out_file = dir.file("ergebnis_gpu.json");
    const RunResult r = optimize({kBiegung, "--gpu", "-o", out_file, "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["backend"], "gpu");
    EXPECT_EQ(j["device"]["index"], 0);
    const double f0 = j["result"]["start_merit"].get<double>();
    const double f1 = j["result"]["merit"].get<double>();
    EXPECT_LT(f1, 1e-4 * f0);
    EXPECT_TRUE(j["result"]["converged"].get<bool>());

    const RunResult cpu = optimize({kBiegung, "--json", "--quiet"});
    ASSERT_EQ(cpu.code, mko::cli::kExitOk);
    const json jc = json::parse(cpu.out);
    EXPECT_NEAR(f0, jc["result"]["start_merit"].get<double>(), cpu_gpu_merit_bound(f0, kBiegung));
    EXPECT_NEAR(shape_factor(doubles(j["runs"][0]["x"])), shape_factor(doubles(jc["runs"][0]["x"])),
                mko::test::kDocumentedValue);

    // Neuladen auf der GPU: bitgleich
    const RunResult again = optimize({out_file, "--gpu", "--max-iterations", "0", "--json"});
    ASSERT_EQ(again.code, mko::cli::kExitOk) << again.err;
    EXPECT_TRUE(same_bits(json::parse(again.out)["result"]["start_merit"].get<double>(), f1));
    // auf der CPU: Rundungsunterschiede der Strahlresiduen
    const FileMerit reloaded = file_merit(out_file);
    EXPECT_NEAR(reloaded.merit, f1, cpu_gpu_merit_bound(f1, out_file));
    std::cout << "[          ] GPU-Endmerit " << f1 << ", CPU beim Neuladen " << reloaded.merit
              << ", |Δ| = " << std::abs(reloaded.merit - f1)
              << ", Schranke = " << cpu_gpu_merit_bound(f1, out_file) << '\n';
    std::cout << "[          ] Startmerit GPU " << f0 << ", CPU "
              << jc["result"]["start_merit"].get<double>() << '\n';
    std::cout << "[          ] Formfaktor GPU " << shape_factor(doubles(j["runs"][0]["x"]))
              << ", CPU " << shape_factor(doubles(jc["runs"][0]["x"])) << ", Zyklen GPU "
              << j["result"]["iterations"] << ", CPU " << jc["result"]["iterations"] << '\n';
    EXPECT_EQ(reloaded.failed, 0);

    // Multistart auf der GPU deterministisch
    const std::vector<std::string> ms{kBiegung, "--gpu", "--json",         "--quiet",
                                      "--runs", "3",     "--perturbation", "0.1"};
    const RunResult m1 = optimize(ms);
    const RunResult m2 = optimize(ms);
    ASSERT_EQ(m1.code, mko::cli::kExitOk) << m1.err;
    EXPECT_EQ(m1.out, m2.out);
}

// Weiches Strahlversagen auf der GPU (M4-T2): versagensfrei wie auf der CPU, Ausgabe
// deterministisch.
TEST(CliOptimizeGpu, SoftFailureStartBecomesFailureFree) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const std::vector<std::string> args{kSystems + "weiches_versagen_start.json", "--gpu", "--json",
                                        "--quiet"};
    const RunResult r = optimize(args);
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["result"]["start_failed_rays"], 27);
    EXPECT_EQ(j["result"]["failed_rays"], 0);
    EXPECT_EQ(j["result"]["hard_failed_rays"], 0);
    EXPECT_GE(j["result"]["penalty_rounds"].get<int>(), 2);
    std::cout << "[          ] GPU: Runden " << j["result"]["penalty_rounds"] << ", w_fail "
              << j["result"]["fail_weight"] << ", Zyklen " << j["result"]["iterations"]
              << ", Merit " << j["result"]["merit"] << '\n';
    EXPECT_EQ(optimize(args).out, r.out);
}

TEST(CliOptimizeGpu, InvalidDevice) {
    const int count = mko::cuda::device_count();
    if (count == 0) {
        GTEST_SKIP() << "kein CUDA-Gerät";
    }
    const RunResult r = optimize({kBiegung, "--gpu", "--device", std::to_string(count)});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("Gerät existiert nicht"), std::string::npos) << r.err;
}
#endif
