// Unterbefehl `mikroptim sensitivity` (M10-T3): JSON gegen run_sensitivity_analysis und
// check_manufacturability (bitgleich), Rangfolge, Standardsatz, nur Herstellbarkeit, Text und
// Fehlerfälle.

#include "cli/cli_util.hpp"
#include "cli/sensitivity_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/tolerance/manufacturability.hpp"
#include "mikroptim/tolerance/sensitivity.hpp"
#include "tolerance_test_systems.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using nlohmann::json;

namespace {

const std::string kEinzellinse = std::string(MKO_SOURCE_DIR) + "/data/systems/einzellinse_bk7.json";

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
    r.code = mko::cli::run_sensitivity_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

} // namespace

TEST(CliSensitivity, JsonMatchesLibrary) {
    const RunResult r = run({mko::test::kToleranceSystemFile, "--json"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.sensitivity/1");
    EXPECT_EQ(j["settings"]["set"], "file");
    const mko::LoadedSystem loaded = mko::load_system(mko::test::kToleranceSystemFile);
    const mko::SensitivityResult ref = mko::run_sensitivity_analysis(
        loaded.system, mko::make_system_batch(loaded, 1), mko::SensitivitySettings{});
    EXPECT_EQ(j["nominal"]["q_waves2"].get<double>(), ref.q0);
    ASSERT_EQ(j["tolerances"].size(), ref.tolerances.size());
    std::size_t c = 0;
    for (const json& t : j["tolerances"]) {
        for (const json& comp : t["components"]) {
            ASSERT_LT(c, ref.components.size());
            EXPECT_EQ(comp["q_plus"].get<double>(), ref.components[c].q_plus);
            EXPECT_EQ(comp["jacobian_quadratic"].get<double>(),
                      ref.components[c].jacobian_quadratic);
            ++c;
        }
    }
    EXPECT_EQ(c, ref.components.size());
    // Rangfolge nach größtem ΔQ absteigend
    const json& rank = j["ranking"];
    ASSERT_EQ(rank.size(), ref.tolerances.size());
    for (std::size_t i = 0; i + 1 < rank.size(); ++i) {
        EXPECT_GE(
            j["tolerances"][rank[i].get<std::size_t>()]["worst_delta_q_waves2"].get<double>(),
            j["tolerances"][rank[i + 1].get<std::size_t>()]["worst_delta_q_waves2"].get<double>());
    }
    const mko::ManufacturabilityReport m = mko::check_manufacturability(loaded.system);
    EXPECT_EQ(j["manufacturability"]["checks"].size(), m.checks.size());
    EXPECT_EQ(j["manufacturability"]["warnings"].get<int>(), m.warnings);
}

TEST(CliSensitivity, DefaultSetOptionsAndManufacturabilityOnly) {
    // ohne Toleranzsatz in der Datei: Standardsatz
    RunResult r = run({kEinzellinse, "--json", "--decenter", "0.02", "--index", "0"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    json j = json::parse(r.out);
    EXPECT_EQ(j["settings"]["set"], "default");
    bool decenter = false;
    for (const json& t : j["tolerances"]) {
        EXPECT_NE(t["type"], "index");
        if (t["type"] == "element_decenter") {
            EXPECT_EQ(t["limit"].get<double>(), 0.02);
            decenter = true;
        }
    }
    EXPECT_TRUE(decenter);
    // nur Herstellbarkeit, eigene Grenzen
    r = run(
        {mko::test::kToleranceSystemFile, "--json", "--manufacturability-only", "--min-edge", "5"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    j = json::parse(r.out);
    EXPECT_TRUE(j["tolerances"].is_null());
    EXPECT_EQ(j["manufacturability"]["limits"]["min_edge_thickness_mm"].get<double>(), 5.0);
    EXPECT_GE(j["manufacturability"]["warnings"].get<int>(), 1);
    // Text
    r = run({mko::test::kToleranceSystemFile, "--set", "default", "--top", "3"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Mikroptim-Standard"), std::string::npos);
    EXPECT_NE(r.out.find("Herstellbarkeit"), std::string::npos);
    EXPECT_NE(r.out.find("(3 von"), std::string::npos);
}

TEST(CliSensitivity, Errors) {
    EXPECT_EQ(run({}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--set", "beides"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--set", "file", "--decenter", "0.1"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--step", "0"}).code, mko::cli::kExitUsage);
    EXPECT_EQ(run({kEinzellinse, "--device", "1"}).code, mko::cli::kExitUsage);
    // Datei ohne Toleranzsatz mit --set file
    RunResult r = run({kEinzellinse, "--set", "file"});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("keine Toleranzen"), std::string::npos);
    // Optionen des Standardsatzes bei einer Datei mit Toleranzsatz ohne --set default
    r = run({mko::test::kToleranceSystemFile, "--radius", "0.01"});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_EQ(run({kEinzellinse, "--help"}).code, mko::cli::kExitOk);
#ifndef MIKROPTIM_HAS_CUDA
    EXPECT_EQ(run({kEinzellinse, "--gpu"}).code, mko::cli::kExitNoGpu);
#endif
}
