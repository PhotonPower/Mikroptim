// Warnung bei Systemdateien ohne Feld reference_sphere (ADR-0017, Entscheidung des Maintainers
// vom 2026-11-19): Die Befehle, deren Merit oder Wellenfront an der Referenzsphäre hängt (trace,
// optimize, synth, insert, snap, tolerance, sensitivity), warnen einmal auf stderr. Die Warnung
// entfällt mit --no-reference-warning, mit --reference-sphere und bei gesetztem Feld; stdout
// (Text und JSON) ist mit und ohne Warnung bitgleich (bis auf Laufzeitangaben). eval und
// aberrations hängen nicht an der alten Konvention und warnen nicht.

#include "cli/aberrations_command.hpp"
#include "cli/cli_util.hpp"
#include "cli/eval_command.hpp"
#include "cli/insert_command.hpp"
#include "cli/optimize_command.hpp"
#include "cli/sensitivity_command.hpp"
#include "cli/snap_command.hpp"
#include "cli/synth_command.hpp"
#include "cli/tolerance_command.hpp"
#include "cli/trace_command.hpp"
#include "temp_paths.hpp"
#include "tolerance_test_systems.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace cli = mko::cli;
namespace fs = std::filesystem;

using Command = int (*)(std::span<const std::string_view>, std::ostream&, std::ostream&);

const std::string kSystems = std::string(MKO_SOURCE_DIR) + "/data/systems/";
const std::string kEinzellinse = kSystems + "einzellinse_bk7.json";

struct RunResult {
    int code = -1;
    std::string out;
    std::string err;
};

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

std::vector<std::string> with(std::vector<std::string> args, const std::string& extra) {
    args.push_back(extra);
    return args;
}

std::size_t count(const std::string& text, const std::string& what) {
    std::size_t n = 0;
    for (std::size_t p = text.find(what); p != std::string::npos; p = text.find(what, p + 1)) {
        ++n;
    }
    return n;
}

// Entfernt Laufzeitangaben (Schlüssel elapsed_seconds, seconds) aus einer JSON-Ausgabe.
void strip_times(nlohmann::json& j) {
    if (j.is_object()) {
        j.erase("elapsed_seconds");
        j.erase("seconds");
        for (auto& [key, value] : j.items()) {
            strip_times(value);
        }
    } else if (j.is_array()) {
        for (nlohmann::json& value : j) {
            strip_times(value);
        }
    }
}

// stdout zweier Läufe: JSON ohne Laufzeiten, sonst der Text.
void expect_same_stdout(const RunResult& a, const RunResult& b, bool json) {
    if (!json) {
        EXPECT_EQ(a.out, b.out);
        return;
    }
    nlohmann::json ja = nlohmann::json::parse(a.out);
    nlohmann::json jb = nlohmann::json::parse(b.out);
    strip_times(ja);
    strip_times(jb);
    EXPECT_EQ(ja.dump(), jb.dump());
}

struct Case {
    const char* name;
    Command command;
    std::vector<std::string> args;
    bool json;
};

std::vector<Case> warning_cases(const fs::path& dir) {
    const std::string tol = mko::test::kToleranceSystemFile;
    return {
        {"trace", cli::run_trace_command, {kEinzellinse}, false},
        {"trace", cli::run_trace_command, {kEinzellinse, "--json"}, true},
        {"optimize",
         cli::run_optimize_command,
         {kSystems + "einzellinse_bk7_biegung.json", "--max-iterations", "3", "--quiet"},
         false},
        {"optimize",
         cli::run_optimize_command,
         {kSystems + "einzellinse_bk7_biegung.json", "--max-iterations", "3", "--json"},
         true},
        {"synth",
         cli::run_synth_command,
         {kSystems + "einzellinse_homotopie_start.json", "--target-na", "0.305", "--merit-factor",
          "10", "--max-elements", "1", "--max-iterations", "20", "--json"},
         true},
        {"insert",
         cli::run_insert_command,
         {kSystems + "einzellinse_geisterplatten_exakt.json", "--meniscus", "0,2.5,-0.1,N-BK7",
          "-o", (dir / "eingefuegt.json").string(), "--json"},
         true},
        {"snap",
         cli::run_snap_command,
         {kSystems + "achromat_glasvariablen.json", "--max-iterations", "20", "--neighbors", "2",
          "--max-combinations", "3", "--json"},
         true},
        {"tolerance", cli::run_tolerance_command, {tol, "--samples", "10", "--json"}, true},
        {"sensitivity", cli::run_sensitivity_command, {tol, "--json"}, true},
    };
}

std::string expected_line(const char* name) {
    return "mikroptim " + std::string(name) +
           ": Warnung: " + std::string(cli::kReferenceSphereWarning);
}

// Ohne Feld: Warnung genau einmal; mit --no-reference-warning keine, stdout bitgleich.
TEST(CliReferenceWarning, WarnsOnceAndOptionSuppresses) {
    const fs::path dir = mko::test::temp_dir() / "reference_warning";
    fs::create_directories(dir);
    for (const Case& c : warning_cases(dir)) {
        SCOPED_TRACE(std::string(c.name) + (c.json ? " --json" : ""));
        const RunResult plain = run(c.command, c.args);
        ASSERT_EQ(plain.code, cli::kExitOk) << plain.err;
        EXPECT_EQ(count(plain.err, expected_line(c.name)), 1U) << plain.err;
        EXPECT_EQ(count(plain.out, std::string(cli::kReferenceSphereWarning)), 0U);
        const RunResult quiet =
            run(c.command, with(c.args, std::string(cli::kNoReferenceWarningOption)));
        ASSERT_EQ(quiet.code, cli::kExitOk) << quiet.err;
        EXPECT_EQ(count(quiet.err, "reference_sphere"), 0U) << quiet.err;
        expect_same_stdout(plain, quiet, c.json);
    }
}

// --reference-sphere (trace, optimize, synth) setzt das Feld: keine Warnung; mit last_surface
// ist stdout bitgleich zum Lauf ohne Feld (dieselbe Konvention).
TEST(CliReferenceWarning, ExplicitOptionSuppresses) {
    const fs::path dir = mko::test::temp_dir() / "reference_warning_option";
    fs::create_directories(dir);
    for (const Case& c : warning_cases(dir)) {
        const std::string_view name = c.name;
        if (name != "trace" && name != "optimize" && name != "synth") {
            continue;
        }
        SCOPED_TRACE(std::string(c.name) + (c.json ? " --json" : ""));
        const RunResult plain = run(c.command, c.args);
        ASSERT_EQ(plain.code, cli::kExitOk) << plain.err;
        const RunResult last =
            run(c.command, with(with(c.args, "--reference-sphere"), "last_surface"));
        ASSERT_EQ(last.code, cli::kExitOk) << last.err;
        EXPECT_EQ(count(last.err, "reference_sphere"), 0U) << last.err;
        expect_same_stdout(plain, last, c.json);
        const RunResult infinite =
            run(c.command, with(with(c.args, "--reference-sphere"), "image_infinite"));
        ASSERT_EQ(infinite.code, cli::kExitOk) << infinite.err;
        EXPECT_EQ(count(infinite.err, "reference_sphere"), 0U) << infinite.err;
    }
}

// Feld in der Datei: keine Warnung; mit last_surface dieselbe Wellenfront wie ohne Feld.
TEST(CliReferenceWarning, FieldInFileSuppresses) {
    std::ifstream in(kEinzellinse);
    nlohmann::json j = nlohmann::json::parse(in);
    j["reference_sphere"] = "last_surface";
    for (auto& catalog : j["glass_catalogs"]) {
        // Einträge relativ zur Repository-Wurzel (Suche aufwärts), im Temp-Verzeichnis absolut
        catalog =
            (fs::path(MKO_SOURCE_DIR) / catalog.get<std::string>()).lexically_normal().string();
    }
    const fs::path file = mko::test::temp_dir() / "reference_warning_feld.json";
    std::ofstream(file) << j.dump(2);
    const RunResult with_field = run(cli::run_trace_command, {file.string(), "--json"});
    ASSERT_EQ(with_field.code, cli::kExitOk) << with_field.err;
    EXPECT_TRUE(with_field.err.empty()) << with_field.err;
    const RunResult without = run(cli::run_trace_command, {kEinzellinse, "--json"});
    ASSERT_EQ(without.code, cli::kExitOk) << without.err;
    nlohmann::json a = nlohmann::json::parse(with_field.out);
    nlohmann::json b = nlohmann::json::parse(without.out);
    a.erase("file");
    b.erase("file");
    EXPECT_EQ(a.dump(), b.dump());
}

// Befehle ohne Bezug zur alten Konvention warnen nicht: eval (Sphäre durch die
// Austrittspupille), aberrations (paraxial), sensitivity --manufacturability-only (kein Trace).
TEST(CliReferenceWarning, CommandsWithoutMeritDoNotWarn) {
    const RunResult eval = run(cli::run_eval_command, {kEinzellinse, "--json"});
    ASSERT_EQ(eval.code, cli::kExitOk) << eval.err;
    EXPECT_EQ(count(eval.err, "reference_sphere"), 0U) << eval.err;
    const RunResult ab = run(cli::run_aberrations_command, {kEinzellinse, "--json"});
    ASSERT_EQ(ab.code, cli::kExitOk) << ab.err;
    EXPECT_EQ(count(ab.err, "reference_sphere"), 0U) << ab.err;
    const RunResult m = run(cli::run_sensitivity_command,
                            {mko::test::kToleranceSystemFile, "--manufacturability-only"});
    ASSERT_EQ(m.code, cli::kExitOk) << m.err;
    EXPECT_EQ(count(m.err, "reference_sphere"), 0U) << m.err;
}

} // namespace
