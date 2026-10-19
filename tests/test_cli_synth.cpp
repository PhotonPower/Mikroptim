// Unterbefehl `mikroptim synth` (M6-T3): Kommandozeile, Kurzlauf mit JSON-Ausgabe, Protokolldatei
// und Ergebnisdatei, Textausgabe (docs/cli.md).

#include "cli/synth_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "residual_test_systems.hpp"
#include "temp_paths.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace mko::test {
namespace {

namespace fs = std::filesystem;

struct SynthCliResult {
    int code = -1;
    std::string out;
    std::string err;
};

SynthCliResult synth(const std::vector<std::string>& args) {
    const std::vector<std::string_view> v(args.begin(), args.end());
    std::ostringstream out;
    std::ostringstream err;
    SynthCliResult r;
    r.code = cli::run_synth_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

class SynthTempDir {
public:
    explicit SynthTempDir(const std::string& name)
        : path_(mko::test::temp_dir() / ("mikroptim_cli_synth_" + name)) {
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_);
    }
    ~SynthTempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    SynthTempDir(const SynthTempDir&) = delete;
    SynthTempDir& operator=(const SynthTempDir&) = delete;
    std::string file(const std::string& name) const { return (path_ / name).string(); }

private:
    fs::path path_;
};

const std::string kStart = kResidualSystems + "einzellinse_homotopie_start.json";

std::string read_file(const std::string& f) {
    std::ifstream in(f, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

TEST(CliSynth, HelpAndUsageErrors) {
    const SynthCliResult h = synth({"--help"});
    EXPECT_EQ(h.code, cli::kExitOk);
    EXPECT_NE(h.out.find("--target-na"), std::string::npos);
    EXPECT_EQ(synth({}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--bogus"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--device", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--target-na", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--step", "0.02,0.002"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "-o", kStart}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, kStart}).code, cli::kExitUsage);
    const SynthCliResult missing = synth({kResidualSystems + "gibt_es_nicht.json"});
    EXPECT_EQ(missing.code, cli::kExitError);
    EXPECT_NE(missing.err.find("Fehler"), std::string::npos);
}

// Kurzlauf wie HomotopySmoke.ShortRunCpu: JSON auf stdout gleich der Protokolldatei, Ergebnis
// als Systemdatei am Endpunkt (Pupillenradius f · NA).
TEST(CliSynth, ShortRunJsonLogAndResult) {
    SynthTempDir dir("json");
    const std::string result = dir.file("ergebnis.json");
    const std::string log = dir.file("protokoll.json");
    const SynthCliResult r = synth({kStart, "--target-na", "0.31", "--merit-factor", "10",
                                    "--max-elements", "2", "--max-iterations", "40",
                                    "--element-stages", "2", "-o", result, "--log", log, "--json"});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.synth/1");
    EXPECT_EQ(j["backend"]["gpu"], false);
    EXPECT_EQ(j["options"]["target_na"], 0.31);
    EXPECT_EQ(j["result"]["status"], "max_elements");
    EXPECT_EQ(j["result"]["insertions"], 1);
    EXPECT_EQ(j["result"]["elements"], 2);
    EXPECT_EQ(j["result"]["point"]["aperture"], 0.31);
    EXPECT_GT(j["result"]["steps"].size(), 10U);
    EXPECT_EQ(j["output"], result);
    EXPECT_EQ(nlohmann::json::parse(read_file(log)), j);
    const LoadedSystem l = load_system(result);
    EXPECT_DOUBLE_EQ(l.system.surfaces[0].semi_diameter, 3.1);
    EXPECT_EQ(l.system.surfaces.size(), 6U); // Meniskus eingefügt
}

TEST(CliSynth, TextOutput) {
    const SynthCliResult r =
        synth({kStart, "--max-elements", "1", "--max-iterations", "40", "--merit-factor", "10"});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Schritt  Art"), std::string::npos);
    EXPECT_NE(r.out.find("penalty"), std::string::npos);
    EXPECT_NE(r.out.find("goal"), std::string::npos);
    EXPECT_NE(r.out.find("Status: max_elements"), std::string::npos) << r.out;
    const SynthCliResult q = synth({kStart, "--max-elements", "1", "--max-iterations", "40",
                                    "--merit-factor", "10", "--quiet"});
    ASSERT_EQ(q.code, cli::kExitOk);
    EXPECT_EQ(q.out.find("Schritt  Art"), std::string::npos);
    EXPECT_NE(q.out.find("Status: max_elements"), std::string::npos);
}

} // namespace
} // namespace mko::test
