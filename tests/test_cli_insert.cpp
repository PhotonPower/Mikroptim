// Unterbefehl `mikroptim insert` (M6-T2, M10-T4): Operator A und B und die Entwurfsoperatoren auf
// Systemdateien, Ergebnisdatei gleich der Bibliothek (insert_meniscus, split_lens, free_glass,
// design_operators.hpp), Merit unverändert,
// Glaskataloge vom Verzeichnis der Ergebnisdatei aus auffindbar, Fehlerfälle.

#include "cli/insert_command.hpp"
#include "insertion_test_systems.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/design_operators.hpp"
#include "temp_paths.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace mko::test {
namespace {

namespace fs = std::filesystem;

struct CliResult {
    int code = -1;
    std::string out;
    std::string err;
};

CliResult insert(const std::vector<std::string>& args) {
    const std::vector<std::string_view> v(args.begin(), args.end());
    std::ostringstream out;
    std::ostringstream err;
    CliResult r;
    r.code = cli::run_insert_command(v, out, err);
    r.out = out.str();
    r.err = err.str();
    return r;
}

// Eigenes temporäres Verzeichnis pro Test, im Destruktor gelöscht.
class InsertTempDir {
public:
    explicit InsertTempDir(const std::string& name)
        : path_(mko::test::temp_dir() / ("mikroptim_cli_insert_" + name)) {
        std::error_code ec;
        fs::remove_all(path_, ec);
        fs::create_directories(path_);
    }
    ~InsertTempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    InsertTempDir(const InsertTempDir&) = delete;
    InsertTempDir& operator=(const InsertTempDir&) = delete;
    std::string file(const std::string& name) const { return (path_ / name).string(); }

private:
    fs::path path_;
};

const std::string kGeister = kResidualSystems + "einzellinse_geisterplatten_exakt.json";

TEST(CliInsert, MeniscusMatchesLibrary) {
    const InsertTempDir dir("meniscus");
    const std::string out = dir.file("neu.json");
    const CliResult r = insert({kGeister, "--meniscus", "0,2.5,-0.1,N-BK7", "-o", out});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Operator A"), std::string::npos);
    EXPECT_NE(r.out.find("relative Änderung"), std::string::npos);
    // Ergebnisdatei lädt aus dem Temp-Verzeichnis (Kataloge umgeschrieben) und gleicht der
    // Bibliothek bis auf die Katalogpfade
    const LoadedSystem written = load_system(out);
    const SystemDescription d =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    SystemDescription expected =
        insert_meniscus(d, meniscus(0, 2.5, -0.1, catalog_glass("N-BK7"))).system;
    expected.glass_catalogs = written.system.glass_catalogs;
    EXPECT_EQ(written.system, expected);
    // system_merit löst Kataloge relativ zu data/systems auf; die Ergebnisdatei nennt sie
    // relativ zu ihrem Temp-Verzeichnis, daher die ursprünglichen Pfade einsetzen
    SystemDescription written_here = written.system;
    written_here.glass_catalogs = d.glass_catalogs;
    EXPECT_LE(relative_change(system_merit(d), system_merit(written_here)),
              kInsertionMeritRelative);
}

TEST(CliInsert, SplitWithFreeGlass) {
    const InsertTempDir dir("split");
    const std::string out = dir.file("neu.json");
    const CliResult r =
        insert({kGeister, "--split", "1,-0.02,0.4", "--free-glass", "--no-variables", "-o", out});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Operator B"), std::string::npos);
    const LoadedSystem written = load_system(out);
    const SystemDescription d =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    LensSplit s = lens_split(1, -0.02, 0.4);
    s.curvature_variable.enabled = false;
    SystemDescription expected = split_lens(d, s).system;
    free_glass(expected, 2);
    expected.glass_catalogs = written.system.glass_catalogs;
    EXPECT_EQ(written.system, expected);
}

// Mechanische Bedingungen (M4-T3) werden mit umnummeriert übernommen.
TEST(CliInsert, MechanicalConstraintsCarriedOver) {
    const InsertTempDir dir("mechanik");
    const std::string out = dir.file("neu.json");
    const std::string in = kResidualSystems + "zweilinser_mechanik.json";
    const CliResult r = insert({in, "--meniscus", "2,0.5,-0.04,N-BK7", "-o", out});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const LoadedSystem written = load_system(out);
    const SystemDescription d = load_residual_system("zweilinser_mechanik.json").system;
    SystemDescription expected =
        insert_meniscus(d, meniscus(2, 0.5, -0.04, catalog_glass("N-BK7"))).system;
    expected.glass_catalogs = written.system.glass_catalogs;
    EXPECT_EQ(written.system, expected);
    EXPECT_EQ(make_mechanical_constraints(written.system).size(),
              make_mechanical_constraints(d).size() + 2);
    // system_merit löst Kataloge relativ zu data/systems auf; die Ergebnisdatei nennt sie
    // relativ zu ihrem Temp-Verzeichnis, daher die ursprünglichen Pfade einsetzen
    SystemDescription written_here = written.system;
    written_here.glass_catalogs = d.glass_catalogs;
    EXPECT_LE(relative_change(system_merit(d), system_merit(written_here)),
              kInsertionMeritRelative);
}

TEST(CliInsert, ConstantGlassAndThickness) {
    const InsertTempDir dir("constant");
    const std::string out = dir.file("neu.json");
    const CliResult r =
        insert({kGeister, "--meniscus", "2,5,0.04,n=1.8", "--thickness", "0.5", "-o", out});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    const SystemDescription w = read_system_file(out);
    EXPECT_EQ(w.surfaces[3].material.kind, MaterialKind::Constant);
    EXPECT_EQ(w.surfaces[3].material.n, 1.8);
    EXPECT_EQ(w.surfaces[3].thickness, 0.5);
}

TEST(CliInsert, Errors) {
    const InsertTempDir dir("errors");
    const std::string out = dir.file("neu.json");
    EXPECT_EQ(insert({}).code, cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "-o", out}).code, cli::kExitUsage); // kein Operator
    EXPECT_EQ(insert({kGeister, "--meniscus", "0,1,0", "-o", out}).code, cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--split", "1,0", "--meniscus", "0,1,0,N-BK7", "-o", out}).code,
              cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--split", "1,0"}).code, cli::kExitUsage); // ohne -o
    EXPECT_EQ(insert({kGeister, "--meniscus", "0,1,0,N-BK7", "--free-glass", "-o", out}).code,
              cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--split", "1,0", "--thickness", "1", "-o", out}).code,
              cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--split", "1,x", "-o", out}).code, cli::kExitUsage);
    // inhaltliche Fehler: Glas nach Fläche 1, unbekanntes Glas, Datei fehlt
    EXPECT_EQ(insert({kGeister, "--meniscus", "1,1,0,N-BK7", "-o", out}).code, cli::kExitError);
    EXPECT_EQ(insert({kGeister, "--meniscus", "0,1,0,UNBEKANNT", "-o", out}).code, cli::kExitError);
    EXPECT_EQ(insert({kResidualSystems + "fehlt.json", "--split", "1,0", "-o", out}).code,
              cli::kExitError);
    EXPECT_FALSE(fs::exists(out));
    // Eingabe nie überschreiben
    EXPECT_EQ(insert({kGeister, "--split", "1,0", "-o", kGeister}).code, cli::kExitUsage);
    const CliResult help = insert({"--help"});
    EXPECT_EQ(help.code, cli::kExitOk);
    EXPECT_NE(help.out.find("--meniscus"), std::string::npos);
    EXPECT_NE(help.out.find("--aplanatic-shell"), std::string::npos);
}

// Operatoren der Entwurfssteuerung (M10-T4): Ergebnisdatei gleich der Bibliothek, Bericht als
// JSON (mikroptim.insert/1) mit Merit vorher und nachher.
TEST(CliInsert, DesignOperatorsMatchLibrary) {
    const InsertTempDir dir("design");
    const SystemDescription d =
        load_residual_system("einzellinse_geisterplatten_exakt.json").system;
    const LoadedSystem loaded = load_system(kGeister);
    struct Case {
        std::vector<std::string> args;
        SystemDescription expected;
        const char* op;
        bool merit_unchanged;
    };
    std::vector<Case> cases;
    {
        GhostDictionaryPlacement p;
        p.after_surface = 2;
        p.distance = 3.0;
        p.glasses = {catalog_glass("N-BK7"), catalog_glass("CAF2")};
        p.order = 5;
        p.penalty_weight = 20.0;
        cases.push_back({{"--ghost-dictionary", "2,3", "--ghost-glass", "N-BK7", "--ghost-glass",
                          "CAF2", "--ghost-order", "5", "--ghost-weight", "20"},
                         place_ghost_dictionary(d, p).system,
                         "ghost_dictionary",
                         true});
    }
    {
        AplanaticShell sh;
        sh.after_surface = 2;
        sh.position = 2.0;
        sh.thickness = 1.0;
        sh.glass = catalog_glass("N-BK7");
        cases.push_back({{"--aplanatic-shell", "2,2,1,N-BK7"},
                         insert_aplanatic_shell(d, loaded.catalog, sh).insertion.system,
                         "aplanatic_shell",
                         false});
    }
    {
        NullDoublet n;
        n.after_surface = 2;
        n.position = 5.0;
        n.curvature = -0.02;
        n.front_glass = catalog_glass("N-BK7");
        n.back_glass = catalog_glass("CAF2");
        cases.push_back({{"--null-doublet", "2,5,-0.02,N-BK7,CAF2"},
                         insert_null_doublet(d, n).system,
                         "null_doublet",
                         true});
    }
    cases.push_back({{"--collapse", "1,0.5"}, collapse_element(d, 1, 0.5), "collapse", false});
    {
        SystemDescription e = d;
        replace_glass(e, 1, catalog_glass("CAF2"));
        cases.push_back({{"--replace-glass", "1,CAF2"}, e, "replace_glass", false});
    }
    for (std::size_t k = 0; k < cases.size(); ++k) {
        SCOPED_TRACE(cases[k].op);
        const std::string out = dir.file("neu" + std::to_string(k) + ".json");
        std::vector<std::string> args = {kGeister, "-o", out, "--json"};
        args.insert(args.end(), cases[k].args.begin(), cases[k].args.end());
        const CliResult r = insert(args);
        ASSERT_EQ(r.code, cli::kExitOk) << r.err;
        EXPECT_NE(r.out.find("\"format\": \"mikroptim.insert/1\""), std::string::npos);
        EXPECT_NE(r.out.find(std::string("\"operator\": \"") + cases[k].op + "\""),
                  std::string::npos);
        const LoadedSystem written = load_system(out);
        SystemDescription expected = cases[k].expected;
        expected.glass_catalogs = written.system.glass_catalogs;
        EXPECT_EQ(written.system, expected);
        if (cases[k].merit_unchanged) {
            SystemDescription here = written.system;
            here.glass_catalogs = d.glass_catalogs;
            EXPECT_LE(relative_change(system_merit(d), system_merit(here)),
                      kInsertionMeritRelative);
        }
    }
    // Rückbau des Null-Kittglieds über --remove ergibt das Ausgangssystem (Dateien)
    const std::string doublet = dir.file("neu2.json");
    const std::string back = dir.file("zurueck.json");
    const CliResult r = insert({doublet, "--remove", "3", "-o", back});
    ASSERT_EQ(r.code, cli::kExitOk) << r.err;
    EXPECT_NE(r.out.find("Element entfernen"), std::string::npos);
    const SystemDescription w = load_system(back).system;
    EXPECT_EQ(w.surfaces.size(), d.surfaces.size());
    EXPECT_NEAR(w.surfaces[2].thickness, d.surfaces[2].thickness,
                kRemovalRoundtripRelative * d.surfaces[2].thickness);
    // Fehler: Platten ohne Glas, Optionen ohne ihren Operator, Glasblock fehlt
    EXPECT_EQ(insert({kGeister, "--ghost-dictionary", "2,3", "-o", back}).code, cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--remove", "1", "--ghost-order", "4", "-o", back}).code,
              cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--remove", "1", "--no-refocus", "-o", back}).code,
              cli::kExitUsage);
    EXPECT_EQ(insert({kGeister, "--remove", "2", "-o", back}).code, cli::kExitError);
}

} // namespace
} // namespace mko::test
