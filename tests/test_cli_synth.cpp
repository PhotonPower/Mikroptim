// Unterbefehl `mikroptim synth` (M6-T3): Kommandozeile, Kurzlauf mit JSON-Ausgabe, Protokolldatei
// und Ergebnisdatei, Textausgabe (docs/cli.md).

#include "cli/synth_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "residual_test_systems.hpp"
#include "temp_paths.hpp"

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

} // namespace
} // namespace mko::test
