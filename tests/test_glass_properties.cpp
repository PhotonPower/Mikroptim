// Glaskennzahlen und Bereichsfilter (M10-T5): geschlossene Form an einem Polynomglas, Datenblatt
// N-BK7, Filter gegen Bruteforce auf den Testkatalogen und die CLI `mikroptim glasses --list`.

#include "cli/cli_util.hpp"
#include "cli/glasses_command.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_properties.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using mko::test::kGlassDatasheetAbbe;
using mko::test::kGlassDatasheetPartial;
using mko::test::kGlassPropertiesFormula;

namespace {

mko::GlassCatalog test_catalogs() {
    mko::GlassCatalog cat =
        mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/synthetisch.json");
    cat.merge(mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json"));
    return cat;
}

} // namespace

TEST(GlassProperties, ClosedFormOfPolynomialGlass) {
    // n² = a + b/λ² (λ in µm): Brechzahlen an den Linien unabhängig ausgerechnet
    mko::Glass g;
    g.name = "TEST";
    g.model = mko::GlassModel::Polynomial;
    g.coefficients = {2.25, 0.012};
    g.exponents = {0.0, -2.0};
    g.range_nm = {400.0, 700.0};
    const auto n = [](double nm) {
        const double l = nm / 1000.0;
        return std::sqrt(2.25 + 0.012 / (l * l));
    };
    const double nd = n(587.5618);
    const double nf = n(486.1327);
    const double nc = n(656.2725);
    const double ng = n(435.8343);
    const double vd = (nd - 1.0) / (nf - nc);
    const double pgf = (ng - nf) / (nf - nc);
    const mko::GlassProperties p = mko::glass_properties(g);
    EXPECT_NEAR(p.n_d, nd, kGlassPropertiesFormula * nd);
    EXPECT_NEAR(p.v_d, vd, kGlassPropertiesFormula * vd);
    EXPECT_NEAR(p.p_gF, pgf, kGlassPropertiesFormula * pgf);
    EXPECT_NEAR(p.dp_gF, pgf - (0.6438 - 0.001682 * vd), kGlassPropertiesFormula);
    // Bereich ohne g-Linie
    g.range_nm = {450.0, 700.0};
    EXPECT_THROW(static_cast<void>(mko::glass_properties(g)), mko::GlassError);
}

TEST(GlassProperties, MatchesDatasheetOfNBK7) {
    // Schott-Datenblatt N-BK7: n_d 1,51680, ν_d 64,17, P_g,F 0,5349, ΔP_g,F −0,0009
    const mko::GlassCatalog cat =
        mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json");
    const mko::GlassProperties p = mko::glass_properties(cat.at("N-BK7"));
    EXPECT_NEAR(p.n_d, 1.5168, 1e-5);
    EXPECT_NEAR(p.v_d, 64.17, kGlassDatasheetAbbe);
    EXPECT_NEAR(p.p_gF, 0.5349, kGlassDatasheetPartial);
    EXPECT_NEAR(p.dp_gF, -0.0009, kGlassDatasheetPartial);
}

TEST(GlassProperties, FilterMatchesBruteForce) {
    const mko::GlassCatalog cat = test_catalogs();
    std::vector<std::string> skipped;
    const std::vector<mko::GlassProperties> all = mko::list_glass_properties(cat, {}, &skipped);
    EXPECT_EQ(skipped, (std::vector<std::string>{"SYN-UV", "SYN-IR"}));
    ASSERT_EQ(all.size() + skipped.size(), cat.size());

    std::mt19937 rng(20261119);
    std::uniform_real_distribution<double> nd(1.40, 1.85);
    std::uniform_real_distribution<double> vd(20.0, 95.0);
    std::uniform_real_distribution<double> pgf(0.50, 0.62);
    std::uniform_real_distribution<double> dp(-0.02, 0.04);
    std::uniform_int_distribution<int> pick(0, 3);
    std::size_t nonempty = 0;
    for (int trial = 0; trial < 400; ++trial) {
        mko::GlassPropertyFilter f;
        auto range = [&](std::uniform_real_distribution<double>& d) {
            mko::ValueRange r;
            const int k = pick(rng); // 0 offen unten, 1 offen oben, 2/3 beidseitig
            double a = d(rng);
            double b = d(rng);
            if (a > b) {
                std::swap(a, b);
            }
            if (k != 0) {
                r.min = a;
            }
            if (k != 1) {
                r.max = b;
            }
            return r;
        };
        if (pick(rng) != 0) {
            f.n_d = range(nd);
        }
        if (pick(rng) != 0) {
            f.v_d = range(vd);
        }
        if (pick(rng) == 0) {
            f.p_gF = range(pgf);
        }
        if (pick(rng) == 0) {
            f.dp_gF = range(dp);
        }
        // Bruteforce: Brechzahlen direkt aus dem Glas, Bedingungen ausgeschrieben
        std::vector<std::string> expected;
        for (const mko::Glass& g : cat.glasses()) {
            if (g.range_nm[0] > 435.8343 || g.range_nm[1] < 656.2725) {
                continue;
            }
            const double n_d = g.index(587.5618);
            const double n_f = g.index(486.1327);
            const double n_c = g.index(656.2725);
            const double n_g = g.index(435.8343);
            const double v = (n_d - 1.0) / (n_f - n_c);
            const double p = (n_g - n_f) / (n_f - n_c);
            const double dpv = p - 0.6438 + 0.001682 * v;
            const auto in = [](const std::optional<mko::ValueRange>& r, double x) {
                return !r || ((!r->min || x >= *r->min) && (!r->max || x <= *r->max));
            };
            if (in(f.n_d, n_d) && in(f.v_d, v) && in(f.p_gF, p) && in(f.dp_gF, dpv)) {
                expected.push_back(g.name);
            }
        }
        std::vector<std::string> got;
        for (const mko::GlassProperties& p : mko::list_glass_properties(cat, f)) {
            got.push_back(p.name);
        }
        EXPECT_EQ(got, expected) << "Versuch " << trial;
        nonempty += expected.empty() ? 0 : 1;
    }
    EXPECT_GT(nonempty, 50u); // die Stichprobe trifft auch nicht leere Filter
}

TEST(CliGlasses, ListWithRangeFilters) {
    auto run = [](std::vector<std::string> a, std::string& out, std::string& err) {
        std::vector<std::string_view> v(a.begin(), a.end());
        std::ostringstream o;
        std::ostringstream e;
        const int code = mko::cli::run_glasses_command(v, o, e);
        out = o.str();
        err = e.str();
        return code;
    };
    const std::string cat = MKO_SOURCE_DIR "/data/glass/synthetisch.json";
    const std::string basis = MKO_SOURCE_DIR "/data/glass/basis.json";
    std::string out;
    std::string err;
    ASSERT_EQ(run({cat, basis, "--list", "--vd", "60,", "--json"}, out, err), mko::cli::kExitOk)
        << err;
    const auto j = nlohmann::json::parse(out);
    EXPECT_EQ(j["mode"], "list");
    EXPECT_EQ(j["skipped"].size(), 2u);
    mko::GlassPropertyFilter f;
    f.v_d = mko::ValueRange{60.0, std::nullopt};
    const auto expected = mko::list_glass_properties(test_catalogs(), f);
    ASSERT_EQ(j["list"].size(), expected.size());
    ASSERT_EQ(j["count"], expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(j["list"][i]["name"], expected[i].name);
        EXPECT_EQ(j["list"][i]["v_d"].get<double>(), expected[i].v_d);
        EXPECT_GE(expected[i].v_d, 60.0);
    }
    ASSERT_EQ(run({basis, "--list"}, out, err), mko::cli::kExitOk) << err;
    EXPECT_NE(out.find("N-BK7"), std::string::npos);
    // ohne --list filtern die Bereiche die Kandidaten der Vorauswahl (M11-T5g)
    EXPECT_EQ(run({cat, "--nd", "1.5,1.6"}, out, err), mko::cli::kExitOk) << err;
    EXPECT_EQ(run({cat, "--list", "--nd", "1.5"}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--list", "--nd", "1.5,x"}, out, err), mko::cli::kExitUsage);
}
