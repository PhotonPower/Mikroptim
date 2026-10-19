// Unterbefehl `mikroptim insert` (M6-T2): Operator A und B auf Systemdateien, Ergebnisdatei
// gleich der Bibliothek (insert_meniscus, split_lens, free_glass), Merit unverändert,
// Glaskataloge vom Verzeichnis der Ergebnisdatei aus auffindbar, Fehlerfälle.

#include "cli/insert_command.hpp"
#include "insertion_test_systems.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
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
        : path_(fs::temp_directory_path() / ("mikroptim_cli_insert_" + name)) {
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
    EXPECT_LE(relative_change(system_merit(d), system_merit(written.system)),
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
    EXPECT_LE(relative_change(system_merit(d), system_merit(written.system)),
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
}

} // namespace
} // namespace mko::test
