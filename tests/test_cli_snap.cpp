// Unterbefehl `mikroptim snap` (M6-T5): kontinuierlich optimieren (`mikroptim optimize`), dann
// einrasten; nach dem Einrasten nur Katalog-Gläser in der Ergebnisdatei, Merit-Verschlechterung
// berichtet, Ergebnisdatei lädt (auch mit `mikroptim trace`) und liefert die berichtete Merit
// bitgleich; Ergebnis gleich der Bibliothek (snap_glasses); Text, Fehlerfälle. Im CUDA-Build
// zusätzlich --gpu (übersprungen ohne Gerät).

#include "cli/optimize_command.hpp"
#include "cli/snap_command.hpp"
#include "cli/trace_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "temp_paths.hpp"
#include "tolerances.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const std::string kSystems = std::string(MKO_SOURCE_DIR) + "/data/systems/";
const std::string kAchromat = kSystems + "achromat_glasvariablen.json";
const std::string kOhneGlas = kSystems + "einzellinse_bk7_biegung.json";

struct RunResult {
    int code = -1;
    std::string out;
    std::string err;
};

using Command = int (*)(std::span<const std::string_view>, std::ostream&, std::ostream&);

RunResult run(Command command, const std::vector<std::string>& args) {
    const std::vector<std::string_view> v(args.begin(), args.end());
    std::ostringstream out;
    std::ostringstream err;
    RunResult r;
    r.code = command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

RunResult snap(const std::vector<std::string>& args) {
    return run(mko::cli::run_snap_command, args);
}

// Eigenes temporäres Verzeichnis pro Test (ctest startet Tests parallel), im Destruktor gelöscht.
class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_(mko::test::temp_dir() / ("mikroptim_cli_snap_" + name)) {
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

private:
    fs::path path_;
};

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Einstellungen wie `mikroptim snap` mit Standardoptionen und `iterations` Zyklen.
mko::GlassSnapOptions cli_options(const mko::SystemDescription& system, int iterations) {
    mko::GlassSnapOptions o;
    o.lm.max_iterations = iterations;
    o.lm.jacobian.extra = mko::concat_extra_residuals(
        {mko::make_paraxial_residuals(mko::make_paraxial_constraints(system)),
         mko::make_ghost_penalty_residuals(mko::make_ghost_penalty(system))});
    return o;
}

// Merit (ohne Zyklus) des Systems in `file` an seinem eigenen Variablenvektor, wie im LM.
double file_merit(const std::string& file) {
    const mko::LoadedSystem loaded = mko::load_system(fs::path(file));
    const mko::SystemBatchHost base = mko::make_system_batch(loaded, 1);
    const mko::VariableSet set(loaded.system, base.topology());
    mko::LmOptions o = cli_options(loaded.system, 0).lm;
    o.max_iterations = 0;
    return mko::optimize_lm_cpu(set, base, set.extract(base, 0), o).runs.at(0).history.at(0);
}

// Kontinuierliches Optimum des Achromaten über `mikroptim optimize` (Datei in `dir`).
std::string continuous_file(const TempDir& dir) {
    const std::string file = dir.file("kontinuierlich.json");
    const RunResult r =
        run(mko::cli::run_optimize_command,
            {kAchromat, "-o", file, "--max-iterations", "5000", "--json", "--quiet"});
    EXPECT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_TRUE(j["result"]["converged"].get<bool>()) << j["result"]["status"];
    return file;
}

} // namespace

// Abnahme über die CLI: nach dem Einrasten nur Katalog-Gläser, Verschlechterung berichtet, die
// Ergebnisdatei lädt mit `mikroptim trace` und liefert die berichtete Merit bitgleich.
TEST(CliSnap, OptimizeThenSnapReproducesMerit) {
    const TempDir dir("reload");
    const std::string cont = continuous_file(dir);
    const std::string out_file = dir.file("eingerastet.json");
    const RunResult r = snap({cont, "-o", out_file, "--json", "--max-iterations", "500"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_EQ(r.err, "");
    const json j = json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.snap/1");
    EXPECT_EQ(j["backend"], "cpu");
    EXPECT_EQ(j["output"], out_file);
    ASSERT_EQ(j["elements"].size(), 2U);
    EXPECT_EQ(j["total_combinations"], 9);
    EXPECT_FALSE(j["truncated"].get<bool>());
    ASSERT_EQ(j["runs"].size(), 9U);
    const json& res = j["result"];
    const int best = j["best_run"].get<int>();
    const double merit = res["merit"].get<double>();
    const double cont_merit = res["continuous_merit"].get<double>();
    EXPECT_TRUE(same_bits(merit, j["runs"][best]["merit"].get<double>()));
    EXPECT_TRUE(same_bits(res["degradation"].get<double>(), merit - cont_merit));
    EXPECT_GE(merit, cont_merit * (1.0 - mko::test::kGlassSnapMeritSlack));
    EXPECT_EQ(res["failed_rays"], 0);
    // bester Lauf = kleinste Merit der Liste
    for (const json& run : j["runs"]) {
        EXPECT_GE(run["merit"].get<double>(), merit);
    }

    // Gleich der Bibliothek
    const mko::LoadedSystem loaded = mko::load_system(fs::path(cont));
    const mko::GlassSnapResult lib = mko::snap_glasses(loaded, mko::make_glass_hull(loaded.catalog),
                                                       cli_options(loaded.system, 500));
    EXPECT_EQ(lib.best, best);
    EXPECT_TRUE(same_bits(lib.best_run().run.merit, merit));
    EXPECT_TRUE(same_bits(lib.continuous_merit, cont_merit));

    // Ergebnisdatei: Katalog-Gläser, keine Glasvariablen, übrige Abschnitte unverändert
    const mko::SystemDescription written = mko::read_system_file(fs::path(out_file));
    for (std::size_t e = 0; e < 2; ++e) {
        const int s = j["elements"][e]["surface"].get<int>();
        const mko::MaterialSpec& m = written.surfaces[static_cast<std::size_t>(s)].material;
        EXPECT_EQ(m.kind, mko::MaterialKind::Catalog);
        EXPECT_EQ(m.name, res["glasses"][e]["glass"].get<std::string>());
        EXPECT_EQ(m.name, j["runs"][best]["glasses"][e].get<std::string>());
    }
    for (const mko::Surface& s : written.surfaces) {
        EXPECT_NE(s.material.kind, mko::MaterialKind::Buchdahl);
    }
    ASSERT_TRUE(written.variables.has_value());
    for (const mko::VariableSpec& v : written.variables->items) {
        EXPECT_NE(v.type, mko::VariableType::Glass);
    }
    EXPECT_EQ(written.constraints, loaded.system.constraints);

    // Neu geladen: bitgleiche Merit
    EXPECT_TRUE(same_bits(file_merit(out_file), merit)) << file_merit(out_file) << " " << merit;

    // mikroptim trace lädt die Ergebnisdatei; RMS nachher = RMS von trace
    const RunResult t = run(mko::cli::run_trace_command, {out_file, "--json"});
    ASSERT_EQ(t.code, mko::cli::kExitOk) << t.err;
    const json jt = json::parse(t.out);
    ASSERT_EQ(jt["wavefront"].size(), j["wavefront"]["result"].size());
    for (std::size_t i = 0; i < jt["wavefront"].size(); ++i) {
        EXPECT_TRUE(same_bits(jt["wavefront"][i]["rms_waves"].get<double>(),
                              j["wavefront"]["result"][i]["rms_waves"].get<double>()));
    }
    // ... und die Datei des kontinuierlichen Systems die RMS vorher (Buchdahl-Material)
    const RunResult tc = run(mko::cli::run_trace_command, {cont, "--json"});
    ASSERT_EQ(tc.code, mko::cli::kExitOk) << tc.err;
    const json jc = json::parse(tc.out);
    for (std::size_t i = 0; i < jc["wavefront"].size(); ++i) {
        EXPECT_TRUE(same_bits(jc["wavefront"][i]["rms_waves"].get<double>(),
                              j["wavefront"]["start"][i]["rms_waves"].get<double>()));
    }

    // Erneutes Einrasten der Ergebnisdatei: keine Glasvariablen mehr
    const RunResult again = snap({out_file});
    EXPECT_EQ(again.code, mko::cli::kExitError);
    EXPECT_NE(again.err.find("keine Glasvariablen"), std::string::npos) << again.err;
}

TEST(CliSnap, TextOutputAndLimit) {
    const RunResult r =
        snap({kAchromat, "--max-iterations", "20", "--neighbors", "2", "--max-combinations", "3"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    for (const char* s : {"Glas-Einrasten: Achromat mit zwei Glasvariablen", "Kandidaten",
                          "Fläche 1:", "Fläche 2:", "Kombinationen (Läufe)",
                          "3 von 4 (kleinste Summe d²)", "bester", "Merit kontinuierlich",
                          "Verschlechterung relativ", "Variablen (ohne Glas)", "Wellenfront"}) {
        EXPECT_NE(r.out.find(s), std::string::npos) << s << "\n" << r.out;
    }
}

TEST(CliSnap, Errors) {
    const TempDir dir("errors");
    EXPECT_EQ(snap({}).code, mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "--neighbors", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "--max-combinations", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "--max-iterations", "-1"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "--unbekannt"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "--device", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "--hard-failure", "--fail-weight", "10"}).code,
              mko::cli::kExitUsage);
    EXPECT_EQ(snap({kAchromat, "-o", kAchromat}).code, mko::cli::kExitUsage);
    const RunResult help = snap({"--help"});
    EXPECT_EQ(help.code, mko::cli::kExitOk);
    EXPECT_NE(help.out.find("mikroptim snap"), std::string::npos);
    const RunResult missing = snap({kSystems + "gibt_es_nicht.json"});
    EXPECT_EQ(missing.code, mko::cli::kExitError);
    const RunResult plain = snap({kOhneGlas});
    EXPECT_EQ(plain.code, mko::cli::kExitError);
    EXPECT_NE(plain.err.find("keine Glasvariablen"), std::string::npos) << plain.err;
    EXPECT_EQ(plain.out, "");
}

#ifndef MIKROPTIM_HAS_CUDA
TEST(CliSnap, GpuUnavailableWithoutCuda) {
    const RunResult r = snap({kAchromat, "--gpu"});
    EXPECT_EQ(r.code, mko::cli::kExitNoGpu);
    EXPECT_NE(r.err.find("ohne CUDA gebaut"), std::string::npos);
}
#else
// GPU gegen CPU: dieselbe Auswahl, Merit bis auf die Rundung der Reoptimierung.
TEST(CliSnapGpu, MatchesCpuSelection) {
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const TempDir dir("gpu");
    const std::string cont = continuous_file(dir);
    const std::string out_file = dir.file("eingerastet_gpu.json");
    const RunResult g = snap({cont, "--json", "--gpu", "--max-iterations", "500", "-o", out_file});
    ASSERT_EQ(g.code, mko::cli::kExitOk) << g.err;
    const RunResult c = snap({cont, "--json", "--max-iterations", "500"});
    ASSERT_EQ(c.code, mko::cli::kExitOk) << c.err;
    const json jg = json::parse(g.out);
    const json jc = json::parse(c.out);
    EXPECT_EQ(jg["backend"], "gpu");
    EXPECT_EQ(jg["best_run"], jc["best_run"]);
    EXPECT_EQ(jg["result"]["glasses"], jc["result"]["glasses"]);
    const double mg = jg["result"]["merit"].get<double>();
    const double mc = jc["result"]["merit"].get<double>();
    EXPECT_LE(std::abs(mg / mc - 1.0), mko::test::kLmReturnRelative);
    // Ergebnisdatei der GPU: Merit (CPU) bis auf die Rundung gleich der berichteten
    EXPECT_LE(std::abs(file_merit(out_file) / mg - 1.0), mko::test::kLmReturnRelative);
}
#endif
