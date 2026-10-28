// Algebraische Glasvorauswahl (M11-T5): Gleichungen, Orakel dünner Achromat, konstruiertes
// Tripel, feste Gläser und eine paraxiale Probe an dünnen Gliedern aus Katalog-Gläsern.

#include "cli/glasses_command.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/insertion.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using mko::test::kGlassPreselectionAchromatRatio;
using mko::test::kGlassPreselectionAlgebra;

namespace {

mko::GlassDispersion glass(const char* name, double n0, double e1, double e2, double e3) {
    return mko::GlassDispersion{name, n0, {e1, e2, e3}};
}

// Erfüllt das Ergebnis die Bedingungen Σ w φ = φ und Σ w φ η_j = 0 (j ≤ k)?
void expect_conditions(const mko::PreselectedGlasses& c, const std::vector<mko::GlassDispersion>& g,
                       const mko::GlassPreselectionOptions& o) {
    const std::size_t m = c.glasses.size();
    const std::vector<double> w = o.weights.empty() ? std::vector<double>(m, 1.0) : o.weights;
    double total = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        total += w[i] * c.powers[i];
        scale += std::abs(w[i] * c.powers[i]);
    }
    EXPECT_NEAR(total, o.power, kGlassPreselectionAlgebra * scale);
    for (int j = 0; j < o.order; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            s += w[i] * c.powers[i] *
                 g[static_cast<std::size_t>(c.glasses[i])].eta[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(s, 0.0, kGlassPreselectionAlgebra * scale) << "Ordnung " << j + 1;
    }
}

} // namespace

// Paare gegen den dünnen Achromat (achromat_split_curvature mit ν = 1/η): Anteil der vorderen
// Linse φ₁/φ = (c₁ − c_s)/(c₁ − c₂).
TEST(GlassPreselection, PairMatchesThinAchromat) {
    const std::vector<mko::GlassDispersion> g{glass("A", 1.52, -0.034, 0.002, 0.0),
                                              glass("B", 1.72, -0.090, 0.009, 0.0)};
    mko::GlassPreselectionOptions o;
    o.power = 0.02;
    const std::vector<mko::PreselectedGlasses> r = mko::preselect_glasses(g, o);
    ASSERT_EQ(r.size(), 1U);
    expect_conditions(r[0], g, o);
    const double c1 = 0.03;
    const double c2 = -0.01;
    const double cs = mko::achromat_split_curvature(c1, c2, 1.0 / g[0].eta[0], 1.0 / g[1].eta[0]);
    EXPECT_NEAR(r[0].powers[0] / o.power, (c1 - cs) / (c1 - c2), kGlassPreselectionAlgebra);
    // sekundäres Spektrum und Summe der Brechkräfte
    const double expect_res =
        std::abs(r[0].powers[0] * g[0].eta[1] + r[0].powers[1] * g[1].eta[1]) / o.power;
    EXPECT_NEAR(r[0].residual, expect_res, 1e-15);
    EXPECT_NEAR(r[0].power_sum, (std::abs(r[0].powers[0]) + std::abs(r[0].powers[1])) / o.power,
                1e-15);
}

// Konstruiertes Tripel mit bekannter Lösung φ = (1,5; −1; 0,5) und verschwindendem tertiärem
// Spektrum steht unter Störgläsern an erster Stelle.
TEST(GlassPreselection, FindsConstructedApochromaticTriple) {
    std::vector<mko::GlassDispersion> g{
        glass("D1", 1.60, -0.070, 0.0040, 0.0010), glass("A", 1.50, -0.040, 0.0020, 0.00010),
        glass("D2", 1.55, -0.050, 0.0030, 0.0020), glass("B", 1.70, -0.110, 0.0060, 0.00035),
        glass("D3", 1.65, -0.080, 0.0080, 0.0030), glass("C", 1.45, -0.100, 0.0060, 0.00040)};
    mko::GlassPreselectionOptions o;
    o.order = 2;
    o.max_results = 50;
    const std::vector<mko::PreselectedGlasses> r = mko::preselect_glasses(g, o);
    ASSERT_FALSE(r.empty());
    const std::vector<int> expected{1, 3, 5}; // A, B, C
    EXPECT_EQ(r[0].glasses, expected);
    EXPECT_NEAR(r[0].powers[0], 1.5, kGlassPreselectionAlgebra);
    EXPECT_NEAR(r[0].powers[1], -1.0, kGlassPreselectionAlgebra);
    EXPECT_NEAR(r[0].powers[2], 0.5, kGlassPreselectionAlgebra);
    EXPECT_LT(r[0].residual, 1e-15);
    EXPECT_NEAR(r[0].power_sum, 3.0, kGlassPreselectionAlgebra);
    for (const mko::PreselectedGlasses& c : r) {
        expect_conditions(c, g, o);
    }
    for (std::size_t i = 1; i < r.size(); ++i) {
        EXPECT_LE(r[i - 1].residual, r[i].residual);
    }
}

// Gewichte, feste Gläser, Filter und Fehler.
TEST(GlassPreselection, WeightsFixedFilterAndErrors) {
    std::vector<mko::GlassDispersion> g;
    for (int i = 0; i < 12; ++i) {
        const double e1 = -0.03 - 0.008 * i;
        g.push_back(glass("G", 1.45 + 0.03 * i, e1, 0.03 * e1 * e1 * (1 + 0.1 * i), 0.0));
    }
    mko::GlassPreselectionOptions o;
    o.order = 2;
    o.weights = {1.0, 0.8, 0.6};
    o.fixed = {4};
    o.max_power_sum = 1e9;
    o.max_results = 1000;
    const std::vector<mko::PreselectedGlasses> r = mko::preselect_glasses(g, o);
    EXPECT_EQ(r.size(), 55U); // C(11, 2), alle lösbar
    for (const mko::PreselectedGlasses& c : r) {
        EXPECT_EQ(c.glasses[0], 4);
        expect_conditions(c, g, o);
    }
    o.max_power_sum = 5.0;
    for (const mko::PreselectedGlasses& c : mko::preselect_glasses(g, o)) {
        EXPECT_LE(c.power_sum, 5.0);
    }
    // gleiche Dispersion: singulär, kein Ergebnis
    const std::vector<mko::GlassDispersion> same{glass("X", 1.5, -0.05, 0.002, 0.0),
                                                 glass("Y", 1.6, -0.05, 0.003, 0.0)};
    EXPECT_TRUE(mko::preselect_glasses(same, {}).empty());
    mko::GlassPreselectionOptions bad;
    bad.order = 3;
    EXPECT_THROW((void)mko::preselect_glasses(g, bad), std::invalid_argument);
    bad = {};
    bad.fixed = {1, 1};
    EXPECT_THROW((void)mko::preselect_glasses(g, bad), std::invalid_argument);
    bad = {};
    bad.weights = {1.0};
    EXPECT_THROW((void)mko::preselect_glasses(g, bad), std::invalid_argument);
}

// Paraxiale Probe mit Katalog-Gläsern: Das beste Paar bzw. Tripel als dünne gekittete Glieder
// (Krümmungsdifferenzen φ_i/(n₀,i − 1), Dicke 10⁻⁴ mm) hat einen deutlich kleineren
// chromatischen Fokusverlauf als eine Einzellinse gleicher Brechkraft.
TEST(GlassPreselection, ThinGroupsAreParaxiallyCorrected) {
    mko::GlassCatalog cat =
        mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/synthetisch.json");
    cat.merge(mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json"));
    std::vector<std::string> skipped;
    const std::vector<mko::GlassDispersion> g = mko::catalog_dispersions(cat, {}, &skipped);
    ASSERT_GE(g.size(), 10U);
    const double phi = 0.01; // f = 100 mm
    const std::vector<double> lines{435.83, 486.13, 546.07, 587.56, 656.27};

    // Spanne der hinteren Schnittweiten über die Linien für dünne Glieder (Gläser, φ_i)
    const auto spread = [&](const std::vector<std::string>& names,
                            const std::vector<double>& powers) {
        std::ostringstream s;
        s << R"({"format": "mikroptim.system/1", "name": "Probe", "units": "mm", "glass_catalogs": [],)"
          << R"("spectrum": {"wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56},)"
          << R"("source": {"type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0]},)"
          << R"("surfaces": [{"type": "stop", "semi_diameter": 1.0, "thickness": 1.0, "material": "AIR"})";
        double c = 0.02;
        for (std::size_t i = 0; i < names.size(); ++i) {
            s << R"(, {"type": "sphere", "curvature": )" << c
              << R"(, "thickness": 0.0001, "material": ")" << names[i]
              << R"(", "semi_diameter": 2.0})";
            c -= powers[i] / (cat.at(names[i]).index(587.56) - 1.0);
        }
        s << R"(, {"type": "sphere", "curvature": )" << c
          << R"(, "thickness": 100.0, "material": "AIR", "semi_diameter": 2.0}]})";
        const mko::SystemDescription sys = mko::read_system_json(s.str());
        double lo = 1e300;
        double hi = -1e300;
        for (double l : lines) {
            const double b = mko::compute_paraxial(sys, cat, l).back_focal_distance;
            lo = std::min(lo, b);
            hi = std::max(hi, b);
        }
        return hi - lo;
    };

    const double single = spread({"N-BK7"}, {phi});
    mko::GlassPreselectionOptions o;
    o.power = phi;
    o.max_power_sum = 6.0;
    const std::vector<mko::PreselectedGlasses> pairs = mko::preselect_glasses(g, o);
    ASSERT_FALSE(pairs.empty());
    const auto names = [&](const mko::PreselectedGlasses& c) {
        std::vector<std::string> n;
        for (int i : c.glasses) {
            n.push_back(g[static_cast<std::size_t>(i)].name);
        }
        return n;
    };
    const double achromat = spread(names(pairs[0]), pairs[0].powers);
    o.order = 2;
    const std::vector<mko::PreselectedGlasses> triples = mko::preselect_glasses(g, o);
    ASSERT_FALSE(triples.empty());
    const double apochromat = spread(names(triples[0]), triples[0].powers);
    std::printf("Spanne g … C: Einzellinse %.3e mm, Achromat %.3e mm, Apochromat %.3e mm\n", single,
                achromat, apochromat);
    EXPECT_LT(achromat, kGlassPreselectionAchromatRatio * single);
    EXPECT_LT(apochromat, achromat);
}

// Kommandozeile `mikroptim glasses`: Text, JSON, festes Glas, Fehler.
TEST(CliGlasses, TextJsonFixedAndErrors) {
    const auto run = [](std::vector<std::string> a, std::string& out, std::string& err) {
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
    ASSERT_EQ(run({cat, basis, "--top", "3"}, out, err), mko::cli::kExitOk) << err;
    EXPECT_NE(out.find("Achromat (Paare)"), std::string::npos);
    ASSERT_EQ(
        run({cat, basis, "--order", "2", "--fixed", "N-BK7", "--top", "4", "--json"}, out, err),
        mko::cli::kExitOk)
        << err;
    const nlohmann::json j = nlohmann::json::parse(out);
    EXPECT_EQ(j["format"], "mikroptim.glasses/1");
    EXPECT_EQ(j["order"], 2);
    ASSERT_FALSE(j["results"].empty());
    EXPECT_LE(j["results"].size(), 4U);
    for (const auto& r : j["results"]) {
        EXPECT_EQ(r["glasses"][0], "N-BK7");
        EXPECT_EQ(r["glasses"].size(), 3U);
    }
    const std::string sys = MKO_SOURCE_DIR "/data/systems/testsystem_mehrlinser.json";
    EXPECT_EQ(run({"--system", sys, "--top", "2"}, out, err), mko::cli::kExitOk) << err;
    EXPECT_EQ(run({}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--order", "3"}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--fixed", "GIBT-ES-NICHT"}, out, err), mko::cli::kExitError);
    EXPECT_NE(err.find("GIBT-ES-NICHT"), std::string::npos);
}
