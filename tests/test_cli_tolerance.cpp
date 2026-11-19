// Unterbefehl `mikroptim tolerance` (M8-T4): JSON-Ausgabe gegen run_tolerance_analysis
// (bitgleich), Histogramm und Werte pro Stichprobe als CSV, Textausgabe, Fehlerfälle und --gpu
// (im CUDA-Build GPU gegen CPU, übersprungen ohne Gerät; sonst Exit-Code 3).

#include "cli/cli_util.hpp"
#include "cli/tolerance_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "temp_paths.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
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
    r.code = mko::cli::run_tolerance_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& file) {
    std::ifstream in(file);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        rows.push_back(split(line));
    }
    return rows;
}

mko::ToleranceResult reference(int samples, std::uint64_t seed) {
    const mko::LoadedSystem loaded = mko::load_system(mko::test::kToleranceSystemFile);
    mko::ToleranceSettings s;
    s.samples = samples;
    s.seed = seed;
    return mko::run_tolerance_analysis(loaded.system, mko::make_system_batch(loaded, 1), s);
}

void expect_stats(const json& j, const mko::DistributionStats& s) {
    EXPECT_EQ(j["count"].get<int>(), s.count);
    EXPECT_EQ(j["mean"].get<double>(), s.mean);
    EXPECT_EQ(j["std"].get<double>(), s.std);
    EXPECT_EQ(j["min"].get<double>(), s.min);
    EXPECT_EQ(j["max"].get<double>(), s.max);
    EXPECT_EQ(j["p50"].get<double>(), s.p50);
    EXPECT_EQ(j["p90"].get<double>(), s.p90);
    EXPECT_EQ(j["p95"].get<double>(), s.p95);
    EXPECT_EQ(j["p99"].get<double>(), s.p99);
}

} // namespace

TEST(CliTolerance, JsonMatchesLibrary) {
    const RunResult r =
        run({mko::test::kToleranceSystemFile, "--json", "--samples", "60", "--seed", "3"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    const json j = json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.tolerance/1");
    EXPECT_EQ(j["backend"], "cpu");
    EXPECT_EQ(j["settings"]["samples"], 60);
    EXPECT_EQ(j["settings"]["seed"], 3);
    EXPECT_EQ(j["settings"]["compensator"], "working_distance");
    EXPECT_EQ(j["tolerances"].size(), 18u);
    EXPECT_EQ(j["tolerances"][17]["type"], "element_tilt");
    EXPECT_EQ(j["tolerances"][17]["element"], json::array({1, 3}));
    EXPECT_EQ(j["draws"], 26);
    const mko::ToleranceResult ref = reference(60, 3);
    EXPECT_EQ(j["nominal"]["rms_waves"].get<double>(), ref.nominal.rms_composite);
    EXPECT_EQ(j["nominal"]["strehl_min"].get<double>(), ref.nominal.strehl_min);
    expect_stats(j["rms_waves"], ref.rms_composite);
    expect_stats(j["strehl_min"], ref.strehl_min);
    expect_stats(j["compensation_mm"], ref.compensation);
    EXPECT_EQ(j["yield"].get<double>(), ref.yield);
    EXPECT_EQ(j["failed_samples"], 0);
    ASSERT_EQ(j["tuples"].size(), 6u);
    for (std::size_t t = 0; t < 6; ++t) {
        expect_stats(j["tuples"][t]["rms_waves"], ref.rms[t]);
        expect_stats(j["tuples"][t]["strehl"], ref.strehl[t]);
        EXPECT_EQ(j["tuples"][t]["weight"].get<double>(), ref.tuple_weight[t]);
    }
    EXPECT_EQ(j["tuples"][4]["field_angle_deg"], 1.0);
    EXPECT_EQ(j["tuples"][4]["wavelength_nm"], 546.07);
}

TEST(CliTolerance, CsvFilesAndText) {
    const auto dir = mko::test::temp_dir();
    const std::filesystem::path hist = dir / "mikroptim_test_tolerance_hist.csv";
    const std::filesystem::path samples = dir / "mikroptim_test_tolerance_samples.csv";
    const RunResult r = run({mko::test::kToleranceSystemFile, "--samples", "40", "--histogram",
                             hist.string(), "--bins", "8", "--samples-file", samples.string(),
                             "--compensator", "none", "--threshold", "0.5"});
    ASSERT_EQ(r.code, mko::cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Toleranzsatz: 8 Einträge, 18 Toleranzen, 26 Zufallsgrößen"),
              std::string::npos)
        << r.out;
    EXPECT_NE(r.out.find("Kompensator: keiner"), std::string::npos);
    EXPECT_NE(r.out.find("Ausbeute (Strehl >= 0.50 in allen Tupeln)"), std::string::npos);
    // Histogramm: je 8 Klassen für strehl_min und rms_waves, Summe = Stichproben
    const auto h = read_csv(hist);
    ASSERT_EQ(h.size(), 17u);
    EXPECT_EQ(h[0][0], "quantity");
    int strehl = 0;
    int rms = 0;
    for (std::size_t i = 1; i < h.size(); ++i) {
        (h[i][0] == "strehl_min" ? strehl : rms) += std::stoi(h[i][4]);
    }
    EXPECT_EQ(strehl, 40);
    EXPECT_EQ(rms, 40);
    // Stichproben: Kopf + 40 Zeilen, Werte wie die Bibliothek (17 signifikante Stellen)
    const auto s = read_csv(samples);
    ASSERT_EQ(s.size(), 41u);
    ASSERT_EQ(s[0].size(), 7u + 2u * 6u);
    mko::ToleranceSettings settings;
    settings.samples = 40;
    settings.compensator = mko::CompensatorType::None;
    const mko::LoadedSystem loaded = mko::load_system(mko::test::kToleranceSystemFile);
    const mko::ToleranceResult ref =
        mko::run_tolerance_analysis(loaded.system, mko::make_system_batch(loaded, 1), settings);
    for (std::size_t i = 0; i < 40; ++i) {
        EXPECT_EQ(std::stod(s[i + 1][1]), ref.results[i].rms_composite) << i;
        EXPECT_EQ(std::stod(s[i + 1][2]), ref.results[i].strehl_min) << i;
        EXPECT_EQ(s[i + 1][6], "none");
    }
}

TEST(CliTolerance, Errors) {
    RunResult r = run({kEinzellinse});
    EXPECT_EQ(r.code, mko::cli::kExitError);
    EXPECT_NE(r.err.find("keine Toleranzen"), std::string::npos) << r.err;
    r = run({mko::test::kToleranceSystemFile, "--samples", "0"});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    r = run({mko::test::kToleranceSystemFile, "--bins", "4"});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    r = run({mko::test::kToleranceSystemFile, "--compensator", "focus"});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    r = run({mko::test::kToleranceSystemFile, "--device", "0"});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    r = run({mko::test::kToleranceSystemFile, "--wobble"});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    r = run({});
    EXPECT_EQ(r.code, mko::cli::kExitUsage);
    r = run({"--help"});
    EXPECT_EQ(r.code, mko::cli::kExitOk);
    EXPECT_NE(r.out.find("Aufruf: mikroptim tolerance"), std::string::npos);
    r = run({(std::string(MKO_SOURCE_DIR) + "/data/systems/gibt_es_nicht.json")});
    EXPECT_EQ(r.code, mko::cli::kExitError);
}

TEST(CliTolerance, Gpu) {
#ifdef MIKROPTIM_HAS_CUDA
    if (mko::cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    const RunResult g = run({mko::test::kToleranceSystemFile, "--json", "--samples", "30",
                             "--compensator", "none", "--gpu"});
    ASSERT_EQ(g.code, mko::cli::kExitOk) << g.err;
    const RunResult c = run(
        {mko::test::kToleranceSystemFile, "--json", "--samples", "30", "--compensator", "none"});
    ASSERT_EQ(c.code, mko::cli::kExitOk) << c.err;
    const json jg = json::parse(g.out);
    const json jc = json::parse(c.out);
    EXPECT_EQ(jg["backend"], "gpu");
    for (const char* key : {"mean", "p50", "p99"}) {
        EXPECT_NEAR(jg["rms_waves"][key].get<double>(), jc["rms_waves"][key].get<double>(),
                    mko::test::kToleranceCpuGpuQuality);
        EXPECT_NEAR(jg["strehl_min"][key].get<double>(), jc["strehl_min"][key].get<double>(),
                    mko::test::kToleranceCpuGpuQuality);
    }
#else
    const RunResult r = run({mko::test::kToleranceSystemFile, "--gpu"});
    EXPECT_EQ(r.code, mko::cli::kExitNoGpu);
    EXPECT_NE(r.err.find("ohne CUDA gebaut"), std::string::npos);
#endif
}
