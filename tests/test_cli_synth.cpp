// Unterbefehl `mikroptim synth` (M6-T3): Kommandozeile, Kurzlauf mit JSON-Ausgabe, Protokolldatei
// und Ergebnisdatei, Textausgabe (docs/cli.md).

#include "cli/synth_command.hpp"
#include "ghost_return_test_systems.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "residual_test_systems.hpp"
#include "temp_paths.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
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
    EXPECT_EQ(synth({kStart, "--jacobian", "xyz"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--jacobian"}).code, cli::kExitUsage);
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

// --sine-condition (M11-T4): Option im Protokoll, Eintrag in der Ergebnisdatei; ungültige Werte
// sind Kommandozeilenfehler.
TEST(CliSynth, SineConditionOption) {
    EXPECT_EQ(synth({kStart, "--sine-condition"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--sine-condition", "-2"}).code, cli::kExitUsage);
    SynthTempDir dir("sine");
    const std::string result = dir.file("ergebnis.json");
    const SynthCliResult r = synth({kStart, "--max-elements", "1", "--max-iterations", "5",
                                    "--sine-condition", "25", "-o", result, "--json"});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    EXPECT_EQ(j["options"]["sine_condition"], 25.0);
    const LoadedSystem l = load_system(result);
    ASSERT_TRUE(l.system.constraints.has_value());
    int found = 0;
    for (const ConstraintSpec& c : l.system.constraints->items) {
        if (c.type == ConstraintType::SineCondition) {
            ++found;
            EXPECT_EQ(c.weight, 25.0);
        }
    }
    EXPECT_EQ(found, 1);
}

// Adjungierte Jacobi (M5-T3): derselbe Kurzlauf mit --jacobian adjoint endet wie mit FD
// (Status, Einfügungen, Endpunkt der Homotopie); das Verfahren steht in den Optionen.
TEST(CliSynth, AdjointJacobian) {
    const std::vector<std::string> args{kStart, "--target-na",      "0.31", "--merit-factor",
                                        "10",   "--max-elements",   "2",    "--max-iterations",
                                        "40",   "--element-stages", "2",    "--json"};
    std::vector<std::string> adjoint = args;
    adjoint.push_back("--jacobian");
    adjoint.push_back("adjoint");
    const SynthCliResult fd = synth(args);
    const SynthCliResult ad = synth(adjoint);
    ASSERT_EQ(fd.code, cli::kExitOk) << fd.err;
    ASSERT_EQ(ad.code, cli::kExitOk) << ad.err;
    const nlohmann::json jf = nlohmann::json::parse(fd.out);
    const nlohmann::json ja = nlohmann::json::parse(ad.out);
    EXPECT_EQ(jf["options"]["jacobian"], "fd");
    EXPECT_EQ(ja["options"]["jacobian"], "adjoint");
    EXPECT_EQ(ja["result"]["status"], jf["result"]["status"]);
    EXPECT_EQ(ja["result"]["insertions"], jf["result"]["insertions"]);
    EXPECT_EQ(ja["result"]["elements"], jf["result"]["elements"]);
    EXPECT_EQ(ja["result"]["point"]["aperture"], jf["result"]["point"]["aperture"]);
    const SynthCliResult t = synth({kStart, "--max-elements", "1", "--max-iterations", "5",
                                    "--quiet", "--jacobian", "adjoint"});
    ASSERT_EQ(t.code, cli::kExitOk) << t.err;
    EXPECT_NE(t.out.find("adjungiert"), std::string::npos) << t.out;
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

// Strahlsuche (M6-T4): Optionen nur mit --beam.
TEST(CliSynth, BeamUsageErrors) {
    EXPECT_EQ(synth({kStart, "--seed", "3"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prefilter"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--beam", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--beam", "2", "--alternatives", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--beam", "2", "--starts", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--beam", "2", "--dot", kStart}).code, cli::kExitUsage);
    EXPECT_NE(synth({"--help"}).out.find("--beam B"), std::string::npos);
}

// Kurzlauf mit --beam 2: JSON mit Stammbaum, Protokolldatei, DOT-Datei, Ergebnis.
TEST(CliSynth, BeamShortRunJsonTreeAndDot) {
    SynthTempDir dir("beam");
    const std::string result = dir.file("ergebnis.json");
    const std::string log = dir.file("protokoll.json");
    const std::string dot = dir.file("stammbaum.dot");
    const SynthCliResult r = synth({kStart, "--target-na",
                                    "0.31", "--merit-factor",
                                    "10",   "--max-elements",
                                    "2",    "--max-iterations",
                                    "40",   "--element-stages",
                                    "2",    "--beam",
                                    "2",    "--seed",
                                    "5",    "-o",
                                    result, "--log",
                                    log,    "--dot",
                                    dot,    "--json"});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    EXPECT_EQ(j["format"], "mikroptim.synth/1");
    EXPECT_EQ(j["options"]["beam"]["width"], 2);
    EXPECT_EQ(j["options"]["beam"]["seed"], 5);
    EXPECT_EQ(j["result"]["status"], "max_elements");
    const nlohmann::json& bs = j["result"]["beam_search"];
    EXPECT_EQ(bs["beam_width"], 2);
    ASSERT_GE(bs["candidates"].size(), 3U);
    EXPECT_EQ(bs["candidates"][0]["operation"], "start");
    EXPECT_EQ(bs["candidates"][0]["status"], "expanded");
    EXPECT_EQ(bs["candidates"][1]["parent"], 0);
    EXPECT_FALSE(j["result"]["steps"][0]["candidate"].is_null());
    EXPECT_EQ(nlohmann::json::parse(read_file(log)), j);
    const std::string d = read_file(dot);
    EXPECT_EQ(d.rfind("digraph stammbaum {", 0), 0U);
    EXPECT_NE(d.find("n0 -> n1;"), std::string::npos);
    const LoadedSystem l = load_system(result);
    EXPECT_EQ(l.system.surfaces.size(), 6U);
}

// --tolerance-aware (M11-T6b): jeder Schritt minimiert F_tol mit dem Standardsatz; Optionen im
// Protokoll, Text und Strahlsuche; Kombinationen ohne --tolerance-aware sind Fehler.
TEST(CliSynth, ToleranceAware) {
    EXPECT_EQ(synth({kStart, "--tolerance-tilt", "remove"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--tolerance-samples", "3"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--tolerance-aware", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--tolerance-aware", "0.5", "--tolerance-seed", "2"}).code,
              cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--tolerance-aware", "0.5", "--tolerance-tilt", "x"}).code,
              cli::kExitUsage);
    EXPECT_NE(synth({"--help"}).out.find("--tolerance-aware"), std::string::npos);
    const std::vector<std::string> base{kStart, "--target-na",    "0.305", "--merit-factor",
                                        "10",   "--max-elements", "1",     "--max-iterations",
                                        "20",   "--json"};
    const SynthCliResult plain = synth(base);
    ASSERT_EQ(plain.code, cli::kExitOk) << plain.err;
    const nlohmann::json jp = nlohmann::json::parse(plain.out);
    EXPECT_TRUE(jp["options"]["tolerance_aware"].is_null());
    // ohne --tolerance-tilt: Standard remove (Entscheidung Maintainer 2026-11-19)
    std::vector<std::string> args = base;
    for (const char* a : {"--tolerance-aware", "0.5"}) {
        args.emplace_back(a);
    }
    const SynthCliResult tol = synth(args);
    ASSERT_EQ(tol.code, cli::kExitOk) << tol.err;
    const nlohmann::json jt = nlohmann::json::parse(tol.out);
    const nlohmann::json& o = jt["options"]["tolerance_aware"];
    EXPECT_EQ(o["alpha"], 0.5);
    EXPECT_EQ(o["scheme"], "plus_minus");
    EXPECT_EQ(o["set"], "default");
    EXPECT_EQ(o["tilt"], "remove");
    // Merit des ersten Schritts am selben Startpunkt ist F_tol (größer als die nominelle Merit
    // des gewöhnlichen Laufs); die Bewertung F_B ist nominal und gleich
    const nlohmann::json& lt = jt["result"]["steps"][0]["lm"];
    const nlohmann::json& lp = jp["result"]["steps"][0]["lm"];
    ASSERT_TRUE(lt.is_object());
    ASSERT_TRUE(lp.is_object());
    EXPECT_GT(lt["start_merit"].get<double>(), lp["start_merit"].get<double>());
    // Strahlsuche: Rangfolge nach F_tol, Text nennt die Option (ausdrücklich keep)
    for (const char* a : {"--tolerance-tilt", "keep"}) {
        args.emplace_back(a);
    }
    args.erase(std::find(args.begin(), args.end(), "--json"));
    for (const char* a : {"--beam", "2", "--quiet"}) {
        args.emplace_back(a);
    }
    const SynthCliResult beam = synth(args);
    ASSERT_EQ(beam.code, cli::kExitOk) << beam.err;
    EXPECT_NE(beam.out.find("Toleranzbewusst"), std::string::npos) << beam.out;
}

TEST(CliSynth, BeamTextOutput) {
    const SynthCliResult r = synth({kStart, "--max-elements", "1", "--max-iterations", "40",
                                    "--merit-factor", "10", "--beam", "2"});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Strahlsuche"), std::string::npos);
    EXPECT_NE(r.out.find("Kand  Gr    Schritt  Art"), std::string::npos);
    EXPECT_NE(r.out.find("Stammbaum"), std::string::npos);
    EXPECT_NE(r.out.find("#0 start"), std::string::npos) << r.out;
    EXPECT_NE(r.out.find("Status: max_elements"), std::string::npos) << r.out;
}

} // namespace
} // namespace mko::test

namespace mko::test {
namespace {

// M11-T1c: Stagnation des LM in synth: Standard aus, --stall W[,T] schaltet ein.
TEST(CliSynth, StallDefaultAndSwitch) {
    const std::vector<std::string> base{kStart, "--max-elements", "1",  "--max-iterations",
                                        "40",   "--merit-factor", "10", "--json"};
    const SynthCliResult d = synth(base);
    ASSERT_EQ(d.code, cli::kExitOk) << d.err;
    const nlohmann::json jd = nlohmann::json::parse(d.out);
    EXPECT_EQ(jd["options"]["stall"]["window"], 0); // Standard aus (seit dem Lauf auf M6-T6)
    EXPECT_EQ(jd["options"]["stall"]["tolerance"], 1e-3);
    std::vector<std::string> off = base;
    off.push_back("--stall");
    off.push_back("0");
    const SynthCliResult o = synth(off);
    ASSERT_EQ(o.code, cli::kExitOk) << o.err;
    EXPECT_EQ(nlohmann::json::parse(o.out)["options"]["stall"]["window"], 0);
    std::vector<std::string> custom = base;
    custom.push_back("--stall");
    custom.push_back("30,1e-4");
    const nlohmann::json jc = nlohmann::json::parse(synth(custom).out);
    EXPECT_EQ(jc["options"]["stall"]["window"], 30);
    EXPECT_EQ(jc["options"]["stall"]["tolerance"], 1e-4);
    EXPECT_EQ(synth({kStart, "--stall", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--stall", "x"}).code, cli::kExitUsage);
    EXPECT_NE(synth({"--help"}).out.find("Standard aus"), std::string::npos);
    EXPECT_NE(synth({"--help"}).out.find("--predictor"), std::string::npos);
    EXPECT_NE(synth({"--help"}).out.find("--split-preselect"), std::string::npos);
    EXPECT_EQ(jd["options"]["split"]["preselect"], false);
    // Farbregel (M11-T5c): Achromat aus der Einzellinse erreicht das Ziel
    const SynthCliResult color =
        synth({std::string(MKO_SOURCE_DIR "/data/systems/einzellinse_achromat_start.json"),
               "--color-rule", "--max-elements", "3", "--json"});
    ASSERT_EQ(color.code, cli::kExitOk) << color.err;
    const nlohmann::json jcol = nlohmann::json::parse(color.out);
    EXPECT_EQ(jcol["result"]["status"], "goal_reached");
    EXPECT_EQ(jcol["options"]["color_rule"], 2.0);
    // Spektrum-Homotopie (M11-T5d): Option im JSON, Fehler bei N < 1
    std::vector<std::string> stages = base;
    stages.push_back("--spectrum-stages");
    stages.push_back("3");
    const SynthCliResult st = synth(stages);
    ASSERT_EQ(st.code, cli::kExitOk) << st.err;
    EXPECT_EQ(nlohmann::json::parse(st.out)["options"]["spectrum_stages"], 3);
    EXPECT_EQ(synth({kStart, "--spectrum-stages", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--spectrum-stages", "2", "--spectrum-continuous"}).code,
              cli::kExitUsage);
    // Prädiktor (M11-T3) nur ohne Strahlsuche
    EXPECT_EQ(synth({kStart, "--predictor", "--beam", "2"}).code, cli::kExitUsage);
}

// Augmented Lagrangian (M11-T2): Option im JSON, Fortschreibung im Protokoll, Standard aus,
// Fehler der Kommandozeile.
TEST(CliSynth, AugmentedLagrangian) {
    const std::vector<std::string> base{kStart, "--max-elements", "1",  "--max-iterations",
                                        "40",   "--merit-factor", "10", "--json"};
    const nlohmann::json off = nlohmann::json::parse(synth(base).out);
    EXPECT_TRUE(off["options"]["augmented_lagrangian"].is_null());
    EXPECT_FALSE(off["result"]["steps"][0]["lm"].contains("shift_change"));
    std::vector<std::string> on = base;
    on.push_back("--augmented-lagrangian");
    on.push_back("3,1e-4,0.01");
    const SynthCliResult r = synth(on);
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    EXPECT_EQ(j["options"]["augmented_lagrangian"]["rounds"], 3);
    EXPECT_EQ(j["options"]["augmented_lagrangian"]["tolerance"], 1e-4);
    EXPECT_EQ(j["options"]["augmented_lagrangian"]["penalty_scale"], 0.01);
    const nlohmann::json& steps = j["result"]["steps"];
    ASSERT_FALSE(steps.empty());
    EXPECT_EQ(steps[0]["kind"], "penalty");
    EXPECT_TRUE(steps[0]["lm"]["shift_change"].is_number());
    EXPECT_EQ(steps[0]["point"]["penalty_scale"], 0.01);
    for (const std::string bad : {"-1", "x", "2,-1", "2,1e-3,0"}) {
        EXPECT_EQ(synth({kStart, "--augmented-lagrangian", bad}).code, cli::kExitUsage) << bad;
    }
    EXPECT_EQ(synth({kStart, "--augmented-lagrangian", "3", "--beam", "2"}).code, cli::kExitUsage);
    EXPECT_NE(synth({"--help"}).out.find("--augmented-lagrangian"), std::string::npos);

    // Neustartzustand (Entscheidung Maintainer 2026-11-19): θ am Endpunkt im Protokoll
    // (augmented_lagrangian.shifts), nicht in der Ergebnisdatei; --shifts-from liest es ein
    EXPECT_FALSE(off.contains("augmented_lagrangian"));
    const nlohmann::json& shifts = j["augmented_lagrangian"]["shifts"];
    ASSERT_TRUE(shifts.is_array());
    EXPECT_TRUE(j["augmented_lagrangian"]["shifts_from"].is_null());
    const SynthTempDir dir("al_neustart");
    const std::string log = dir.file("protokoll.json");
    const std::string out = dir.file("ergebnis.json");
    std::vector<std::string> first = on;
    first.insert(first.end(), {"--log", log, "-o", out});
    ASSERT_EQ(synth(first).code, cli::kExitOk);
    const nlohmann::json logged = nlohmann::json::parse(read_file(log));
    EXPECT_EQ(logged["augmented_lagrangian"]["shifts"], shifts);
    EXPECT_EQ(read_file(out).find("\"shift\""), std::string::npos);
    std::vector<std::string> again = on;
    again.insert(again.end(), {"--shifts-from", log});
    const SynthCliResult r2 = synth(again);
    ASSERT_EQ(r2.code, cli::kExitOk) << r2.err;
    EXPECT_EQ(nlohmann::json::parse(r2.out)["augmented_lagrangian"]["shifts_from"], log);
    EXPECT_EQ(synth({kStart, "--shifts-from", log}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--augmented-lagrangian", "1", "--shifts-from", out}).code,
              cli::kExitError);
}

// Gemeinsame Apertur-/Feld-Homotopie und Feldregel (M6-T8): Optionen im JSON nur mit der
// Option (ohne sie ist das Protokoll wie bisher), Schritte "aperture_field", Fehler der
// Kommandozeile.
TEST(CliSynth, JointApertureFieldAndFieldRule) {
    const std::vector<std::string> base{
        kStart, "--target-na",    "0.31", "--start-field",    "0.5", "--target-field",
        "1",    "--max-elements", "1",    "--max-iterations", "40",  "--merit-factor",
        "10",   "--json"};
    const SynthCliResult off = synth(base);
    ASSERT_EQ(off.code, cli::kExitOk) << off.err;
    const nlohmann::json jo = nlohmann::json::parse(off.out);
    EXPECT_FALSE(jo["options"].contains("joint_aperture_field"));
    EXPECT_FALSE(jo["options"].contains("field_rule"));
    bool aperture = false;
    for (const auto& s : jo["result"]["steps"]) {
        EXPECT_NE(s["kind"], "aperture_field");
        aperture = aperture || s["kind"] == "aperture";
    }
    EXPECT_TRUE(aperture);

    std::vector<std::string> on = base;
    on.insert(on.end(), {"--joint-aperture-field", "--joint-field-exponent", "0.5", "--joint-step",
                         "0.5,0.125,0.5", "--field-rule", "--field-ratio", "3"});
    const SynthCliResult r = synth(on);
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    EXPECT_EQ(j["options"]["joint_aperture_field"]["field_exponent"], 0.5);
    EXPECT_EQ(j["options"]["joint_aperture_field"]["step"]["initial"], 0.5);
    EXPECT_EQ(j["options"]["joint_aperture_field"]["step"]["min"], 0.125);
    EXPECT_EQ(j["options"]["field_rule"]["ratio"], 3.0);
    EXPECT_EQ(j["options"]["field_rule"]["min_rms_waves"], 0.05);
    EXPECT_TRUE(j["options"]["field_rule"]["glass"].is_null());
    int joint_steps = 0;
    for (const auto& s : j["result"]["steps"]) {
        EXPECT_NE(s["kind"], "aperture");
        EXPECT_NE(s["kind"], "field");
        joint_steps += s["kind"] == "aperture_field" ? 1 : 0;
    }
    EXPECT_GT(joint_steps, 0);
    if (j["result"]["status"] == "goal_reached" || j["result"]["status"] == "goal_not_reached") {
        EXPECT_EQ(j["result"]["point"]["field"], 1.0);
        EXPECT_EQ(j["result"]["point"]["aperture"], 0.31);
    }

    const SynthCliResult text =
        synth({kStart, "--target-na", "0.31", "--start-field", "0.5", "--target-field", "1",
               "--max-elements", "1", "--max-iterations", "40", "--merit-factor", "10",
               "--joint-aperture-field", "--field-rule", "--quiet"});
    ASSERT_EQ(text.code, cli::kExitOk) << text.err;
    EXPECT_NE(text.out.find("Apertur und Feld gemeinsam"), std::string::npos);
    EXPECT_NE(text.out.find("Feldregel"), std::string::npos);

    EXPECT_FALSE(jo["options"].contains("insertion_improve"));
    std::vector<std::string> improve = base;
    improve.insert(improve.end(), {"--insertion-improve", "3"});
    const SynthCliResult ri = synth(improve);
    ASSERT_EQ(ri.code, cli::kExitOk) << ri.err;
    EXPECT_EQ(nlohmann::json::parse(ri.out)["options"]["insertion_improve"]["attempts"], 3);
    EXPECT_EQ(synth({kStart, "--insertion-improve", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--insertion-improve", "2", "--beam", "2"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--joint-aperture-field", "--beam", "2"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--joint-field-exponent", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--joint-step", "0.1,0.01"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--field-ratio", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--field-min-rms", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--field-glass", "AIR"}).code, cli::kExitUsage);
    const SynthCliResult unknown = synth({kStart, "--field-glass", "GIBT-ES-NICHT"});
    EXPECT_EQ(unknown.code, cli::kExitError);
    EXPECT_NE(unknown.err.find("--field-glass"), std::string::npos);
    EXPECT_NE(synth({"--help"}).out.find("--joint-aperture-field"), std::string::npos);
    EXPECT_NE(synth({"--help"}).out.find("--field-rule"), std::string::npos);
}

// Rückbau und Elementbudget (M6-T9): Optionen nur mit Option im Protokoll; mit --prune und
// Mindestzahl 0 wird die Einzellinse versucht und verworfen (notwendig); Elementbudget setzt die
// höchste Elementzahl; Fehler der Kommandozeile.
TEST(CliSynth, PruneAndElementBudget) {
    // F_max = (0,36 · 2)² ≈ 0,52 Wellen² wie der Kurzlauf, Ziel 2 Wellen: die Einzellinse erreicht
    // das Ziel, der Abschluss fügt nichts ein
    const std::vector<std::string> base{kStart, "--max-iterations", "40",   "--rms-target",
                                        "2",    "--merit-factor",   "0.36", "--json"};
    const SynthCliResult off = synth(base);
    ASSERT_EQ(off.code, cli::kExitOk) << off.err;
    const nlohmann::json jo = nlohmann::json::parse(off.out);
    EXPECT_FALSE(jo["options"].contains("prune"));
    EXPECT_FALSE(jo["result"].contains("removals"));
    EXPECT_FALSE(jo["result"].contains("removal_attempts"));

    std::vector<std::string> on = base;
    on.insert(on.end(),
              {"--prune", "--prune-min-elements", "0", "--prune-stages", "2", "--prune-attempts",
               "1", "--prune-factor", "0.5", "--prune-tolerance", "0.1"});
    const SynthCliResult r = synth(on);
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    const nlohmann::json& po = j["options"]["prune"];
    EXPECT_EQ(po["after_phases"], true);
    EXPECT_TRUE(po["element_budget"].is_null());
    EXPECT_EQ(po["stages"], 2);
    EXPECT_EQ(po["attempts"], 1);
    EXPECT_EQ(po["merit_factor"], 0.5);
    EXPECT_EQ(po["tolerance"], 0.1);
    EXPECT_EQ(po["min_elements"], 0);
    EXPECT_EQ(j["result"]["removal_attempts"], 1);
    EXPECT_EQ(j["result"]["removals"], 0);
    EXPECT_EQ(j["result"]["elements"], 1);
    int stages = 0;
    int notes = 0;
    for (const auto& s : j["result"]["steps"]) {
        stages += s["kind"] == "removal_stage" ? 1 : 0;
        if (s["kind"] == "removal") {
            ++notes;
            EXPECT_FALSE(s["accepted"].get<bool>());
        }
    }
    EXPECT_EQ(stages, 2);
    EXPECT_EQ(notes, 1);

    // Elementbudget 1: Einfügen nötig, kein Element entfernbar (Mindestzahl 1) → max_elements
    std::vector<std::string> budget = base;
    budget.insert(budget.end(), {"--target-na", "0.31", "--element-budget", "1"});
    const SynthCliResult rb = synth(budget);
    ASSERT_EQ(rb.code, cli::kExitOk) << rb.err;
    const nlohmann::json jb = nlohmann::json::parse(rb.out);
    EXPECT_EQ(jb["options"]["max_elements"], 1);
    EXPECT_EQ(jb["options"]["prune"]["element_budget"], 1);
    EXPECT_EQ(jb["options"]["prune"]["after_phases"], false);

    const SynthCliResult text = synth({kStart, "--max-iterations", "40", "--merit-factor", "10",
                                       "--prune", "--element-budget", "3", "--quiet"});
    ASSERT_EQ(text.code, cli::kExitOk) << text.err;
    EXPECT_NE(text.out.find("Rückbau"), std::string::npos) << text.out;
    EXPECT_NE(text.out.find("am Elementbudget 3"), std::string::npos) << text.out;

    EXPECT_EQ(synth({kStart, "--prune", "--prune-stages", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prune", "--prune-attempts", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prune", "--prune-factor", "1.5"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prune", "--prune-tolerance", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prune", "--prune-min-elements", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prune-attempts", "2"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--element-budget", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--element-budget", "3", "--max-elements", "3"}).code,
              cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--prune", "--beam", "2"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--element-budget", "3", "--beam", "2"}).code, cli::kExitUsage);
    const std::string help = synth({"--help"}).out;
    EXPECT_NE(help.find("--prune"), std::string::npos);
    EXPECT_NE(help.find("--element-budget"), std::string::npos);
}

TEST(CliSynth, StartFromReal) {
    // M6-T10: Rückkehr zu Geisterplatten vor dem Lauf (reales System ohne Platten)
    const SynthTempDir dir("start_from_real");
    const std::string in = dir.file("real.json");
    write_ghost_return_real_system(fs::path(in), true);
    const std::vector<std::string> base{
        in,    "--max-iterations", "5",  "--max-steps", "4", "--rms-target",
        "100", "--merit-factor",   "10", "--json"};
    // ohne Option: kein Eintrag im Protokoll, keine Platten
    const std::string plain_out = dir.file("ohne.json");
    std::vector<std::string> plain = base;
    plain.insert(plain.end(), {"-o", plain_out});
    const SynthCliResult off = synth(plain);
    ASSERT_EQ(off.code, cli::kExitOk) << off.err;
    const nlohmann::json jo = nlohmann::json::parse(off.out);
    EXPECT_FALSE(jo["options"].contains("start_from_real"));
    EXPECT_TRUE(load_system(plain_out).system.ghost_plates.empty());

    const std::string out = dir.file("mit.json");
    std::vector<std::string> on = base;
    on.insert(on.end(), {"--start-from-real", "--ghost-glass", "N-BK7", "--ghost-glass", "CAF2",
                         "--ghost-order", "4", "--ghost-weight", "20", "--ghost-positions",
                         "aperture,front", "-o", out});
    const SynthCliResult r = synth(on);
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const nlohmann::json j = nlohmann::json::parse(r.out);
    const nlohmann::json& g = j["options"]["start_from_real"];
    ASSERT_FALSE(g.is_null());
    EXPECT_EQ(g["order"], 4);
    EXPECT_EQ(g["glasses"], nlohmann::json({"N-BK7", "CAF2"}));
    ASSERT_GE(g["positions"].size(), 1u);
    EXPECT_EQ(g["positions"][0]["location"], "aperture");
    EXPECT_LE(g["relative_change"].get<double>(), kInsertionMeritRelative);
    const SystemDescription d = load_system(out).system;
    EXPECT_EQ(d.ghost_plates.size(), 2 * g["positions"].size());
    for (const GhostPlate& p : d.ghost_plates) {
        EXPECT_EQ(p.coefficients.size(), 4u);
    }
    bool penalty = false;
    for (const ConstraintSpec& c : d.constraints->items) {
        if (c.type == ConstraintType::GhostPenalty) {
            penalty = true;
        }
    }
    EXPECT_TRUE(penalty);

    // Textausgabe
    std::vector<std::string> text{in,
                                  "--max-iterations",
                                  "5",
                                  "--max-steps",
                                  "2",
                                  "--quiet",
                                  "--start-from-real",
                                  "--ghost-glass",
                                  "N-BK7"};
    const SynthCliResult t = synth(text);
    ASSERT_EQ(t.code, cli::kExitOk) << t.err;
    EXPECT_NE(t.out.find("Rückkehr zu Geisterplatten"), std::string::npos) << t.out;

    // Fehler
    EXPECT_EQ(synth({in, "--start-from-real"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({in, "--ghost-glass", "N-BK7"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({in, "--ghost-positions", "stop"}).code, cli::kExitUsage);
    EXPECT_EQ(
        synth({in, "--start-from-real", "--ghost-glass", "N-BK7", "--ghost-positions", "mitte"})
            .code,
        cli::kExitUsage);
    EXPECT_EQ(synth({in, "--start-from-real", "--ghost-glass", "N-BK7", "--ghost-order", "0"}).code,
              cli::kExitUsage);
    EXPECT_EQ(synth({in, "--start-from-real", "--ghost-glass", "GIBTESNICHT"}).code,
              cli::kExitError);
    // Startdatei mit Geisterplatten
    EXPECT_EQ(
        synth({kStart, "--start-from-real", "--ghost-glass", "N-BK7", "--max-steps", "1"}).code,
        cli::kExitError);
    const std::string help = synth({"--help"}).out;
    EXPECT_NE(help.find("--start-from-real"), std::string::npos);
}

// M6-T11: Optionen der Farbstufe. Ohne Optionen kein Eintrag color_rule_ext; Rang nach dem
// spektralen Rest und Startradien des Glieds sind seit M6-T11b Standard (im Protokoll unter
// split, mit --no-split-spectrum/--no-split-member-start aus); mit Optionen im JSON und im
// Textkopf; Fehlbedienung Exit 2. Die Einzellinse bei 480/644 nm erreicht das Ziel weiter.
TEST(CliSynth, ColorStageOptions) {
    const std::string achromat = MKO_SOURCE_DIR "/data/systems/einzellinse_achromat_start.json";
    EXPECT_EQ(synth({achromat, "--color-module", "triplet"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({achromat, "--color-module"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({achromat, "--color-module-always"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({achromat, "--color-module-anomalous"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({achromat, "--color-chromatic", "0"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({achromat, "--color-module", "strong", "--color-module-incidence", "95"}).code,
              cli::kExitUsage);
    const std::string help = synth({"--help"}).out;
    for (const char* o :
         {"--split-spectrum", "--split-member-start", "--no-split-spectrum",
          "--no-split-member-start", "--color-chromatic", "--color-module", "--color-module-always",
          "--color-module-anomalous", "--color-module-incidence"}) {
        EXPECT_NE(help.find(o), std::string::npos) << o;
    }

    const SynthCliResult plain = synth({achromat, "--color-rule", "--max-elements", "3", "--json"});
    ASSERT_EQ(plain.code, cli::kExitOk) << plain.err;
    const nlohmann::json jp = nlohmann::json::parse(plain.out);
    EXPECT_FALSE(jp["options"].contains("color_rule_ext"));
    // M6-T11b: Standard
    EXPECT_EQ(jp["options"]["split"]["preselect"], false); // die Farbregel wählt selbst vor
    EXPECT_EQ(jp["options"]["split"]["spectrum_rank"], nlohmann::json::array({480.0, 644.0}));
    EXPECT_EQ(jp["options"]["split"]["member_start"], true);
    EXPECT_EQ(jp["result"]["status"], "goal_reached");

    const SynthCliResult off = synth({achromat, "--color-rule", "--no-split-spectrum",
                                      "--no-split-member-start", "--max-elements", "3", "--json"});
    ASSERT_EQ(off.code, cli::kExitOk) << off.err;
    const nlohmann::json jo = nlohmann::json::parse(off.out);
    EXPECT_TRUE(jo["options"]["split"]["spectrum_rank"].is_null());
    EXPECT_EQ(jo["options"]["split"]["member_start"], false);
    EXPECT_EQ(jo["result"]["status"], "goal_reached");
    // M6-T11 (Bericht, Abschnitt 2): der Zweig „Startradien“ gewinnt, RMS kleiner als ohne
    EXPECT_LT(jp["result"]["goal"]["max_rms_waves"].get<double>(),
              jo["result"]["goal"]["max_rms_waves"].get<double>());

    const SynthCliResult with = synth(
        {achromat, "--split-spectrum", "--split-member-start", "--color-chromatic", "0.5",
         "--color-module", "doublet", "--color-module-anomalous", "--max-elements", "3", "--json"});
    ASSERT_EQ(with.code, cli::kExitOk) << with.err;
    const nlohmann::json jw = nlohmann::json::parse(with.out);
    EXPECT_EQ(jw["options"]["color_rule"], 2.0); // --color-chromatic setzt --color-rule
    EXPECT_EQ(jw["options"]["split"]["preselect"], true);
    EXPECT_EQ(jw["options"]["split"]["spectrum_rank"], nlohmann::json::array({480.0, 644.0}));
    EXPECT_EQ(jw["options"]["split"]["member_start"], true);
    const nlohmann::json& ext = jw["options"]["color_rule_ext"];
    EXPECT_EQ(ext["chromatic_share"], 0.5);
    EXPECT_EQ(ext["module"], "doublet");
    EXPECT_EQ(ext["always"], false);
    EXPECT_EQ(ext["anomalous"], true);
    EXPECT_EQ(ext["incidence_deg"], 40.0);
    EXPECT_EQ(jw["result"]["status"], "goal_reached");

    const SynthCliResult text = synth(
        {achromat, "--color-module", "strong", "--color-module-always", "--max-elements", "1"});
    ASSERT_EQ(text.code, cli::kExitOk) << text.err;
    EXPECT_NE(text.out.find("Farbregel (M6-T11)"), std::string::npos) << text.out;
    EXPECT_NE(text.out.find("starkes Kittglied (40.0°) statt Teilen"), std::string::npos)
        << text.out;

    // M6-T13: --color-variants setzt --color-rule, nur mit Option im JSON und im Textkopf; an der
    // Einzellinse (keine weitere Variante) Ergebnis wie ohne Option
    EXPECT_NE(help.find("--color-variants"), std::string::npos);
    const SynthCliResult var = synth(
        {achromat, "--color-variants", "--insertion-improve", "4", "--max-elements", "3", "--json"});
    ASSERT_EQ(var.code, cli::kExitOk) << var.err;
    const nlohmann::json jv = nlohmann::json::parse(var.out);
    EXPECT_EQ(jv["options"]["color_rule_ext"]["variants"], true);
    const SynthCliResult novar =
        synth({achromat, "--color-rule", "--insertion-improve", "4", "--max-elements", "3", "--json"});
    ASSERT_EQ(novar.code, cli::kExitOk) << novar.err;
    const nlohmann::json jn = nlohmann::json::parse(novar.out);
    EXPECT_FALSE(jn["options"].contains("color_rule_ext"));
    EXPECT_EQ(jv["result"]["status"], jn["result"]["status"]);
    EXPECT_EQ(jv["result"]["goal"]["max_rms_waves"], jn["result"]["goal"]["max_rms_waves"]);
    const SynthCliResult vtext = synth({achromat, "--color-variants", "--max-elements", "1"});
    ASSERT_EQ(vtext.code, cli::kExitOk) << vtext.err;
    EXPECT_NE(vtext.out.find("Varianten: teilbare Linsen"), std::string::npos) << vtext.out;

    // M6-T13: --grow-margin Q in (0, 1], nur mit Option im JSON
    EXPECT_EQ(synth({achromat, "--grow-margin", "1.5"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({achromat, "--grow-margin", "0"}).code, cli::kExitUsage);
    EXPECT_NE(help.find("--grow-margin"), std::string::npos);
    EXPECT_FALSE(jn["options"].contains("grow_margin"));
    const SynthCliResult grow =
        synth({achromat, "--color-rule", "--grow-margin", "0.5", "--max-elements", "3", "--json"});
    ASSERT_EQ(grow.code, cli::kExitOk) << grow.err;
    EXPECT_EQ(nlohmann::json::parse(grow.out)["options"]["grow_margin"], 0.5);
}

// M6-T12: erweitertes Glas-Wörterbuch. Ohne Option kein Eintrag ghost_dictionary; mit
// --ghost-dictionary-extra 0 (nichts ausgewählt) Ergebnis und Ergebnisdatei bitgleich zum Lauf
// ohne Option; mit Option Platten je Lage dazu (Katalog basis.json: kein Flint, also nur CaF₂ als
// anomales Glas), im JSON und im Textkopf; Fehlbedienung Exit 2, unbekanntes Glas Exit 1.
TEST(CliSynth, GhostDictionaryExtra) {
    const SynthTempDir dir("ghost_dictionary");
    const std::vector<std::string> base{kStart, "--target-na", "0.31", "--merit-factor", "10",
                                        "--max-elements", "2", "--max-iterations", "40",
                                        "--element-stages", "2", "--json"};
    const auto run = [&](std::vector<std::string> extra, const std::string& out) {
        std::vector<std::string> a = base;
        a.insert(a.end(), extra.begin(), extra.end());
        a.insert(a.end(), {"-o", out});
        return synth(a);
    };
    const std::string plain_out = dir.file("ohne.json");
    const SynthCliResult plain = run({}, plain_out);
    ASSERT_EQ(plain.code, cli::kExitOk) << plain.err;
    const nlohmann::json jp = nlohmann::json::parse(plain.out);
    EXPECT_FALSE(jp["options"].contains("ghost_dictionary"));

    const std::string zero_out = dir.file("null.json");
    const SynthCliResult zero = run({"--ghost-dictionary-extra", "0"}, zero_out);
    ASSERT_EQ(zero.code, cli::kExitOk) << zero.err;
    nlohmann::json jz = nlohmann::json::parse(zero.out);
    EXPECT_EQ(jz["options"]["ghost_dictionary"]["added_plates"], 0);
    jz["options"].erase("ghost_dictionary");
    jz["output"] = jp["output"];
    // Laufzeiten ausgenommen
    const auto no_seconds = [](nlohmann::json j) {
        j["result"].erase("seconds");
        for (nlohmann::json& st : j["result"]["steps"]) {
            st.erase("seconds");
        }
        return j;
    };
    EXPECT_EQ(no_seconds(jz), no_seconds(jp));
    EXPECT_EQ(read_file(zero_out), read_file(plain_out));

    const std::string out = dir.file("mit.json");
    const SynthCliResult with = run({"--ghost-dictionary-extra", "2"}, out);
    ASSERT_EQ(with.code, cli::kExitOk) << with.err;
    const nlohmann::json j = nlohmann::json::parse(with.out);
    const nlohmann::json& g = j["options"]["ghost_dictionary"];
    EXPECT_EQ(g["extra"], 2);
    ASSERT_EQ(g["glasses"].size(), 1u);
    EXPECT_EQ(g["glasses"][0]["name"], "CAF2");
    EXPECT_EQ(g["glasses"][0]["class"], "anomalous");
    EXPECT_EQ(g["glasses"][0]["present"], false);
    EXPECT_EQ(g["positions"], 2);
    EXPECT_EQ(g["added_plates"], 2);
    const SystemDescription start = load_system(kStart).system;
    const SystemDescription d = load_system(out).system;
    EXPECT_EQ(d.ghost_plates.size(), start.ghost_plates.size() + 2);

    const SynthCliResult text = synth({kStart, "--max-steps", "1", "--max-iterations", "5",
                                       "--ghost-dictionary-glass", "F_SILICA"});
    ASSERT_EQ(text.code, cli::kExitOk) << text.err;
    EXPECT_NE(text.out.find("Glas-Wörterbuch (M6-T12)"), std::string::npos) << text.out;
    EXPECT_NE(text.out.find("F_SILICA (Kron"), std::string::npos) << text.out;

    EXPECT_EQ(synth({kStart, "--ghost-dictionary-extra", "-1"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--ghost-dictionary-extra"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--ghost-dictionary-glass", "AIR"}).code, cli::kExitUsage);
    EXPECT_EQ(synth({kStart, "--ghost-dictionary-glass", "GIBTESNICHT", "--max-steps", "1"}).code,
              cli::kExitError);
    EXPECT_NE(synth({"--help"}).out.find("--ghost-dictionary-extra"), std::string::npos);
}

} // namespace
} // namespace mko::test
