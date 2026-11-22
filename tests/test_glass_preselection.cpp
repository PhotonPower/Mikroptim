// Algebraische Glasvorauswahl (M11-T5): Gleichungen, Orakel dünner Achromat, konstruiertes
// Tripel, feste Gläser und eine paraxiale Probe an dünnen Gliedern aus Katalog-Gläsern; Filter
// der Kandidaten und Startradien des ganzen Glieds (M11-T5g). Bruteforce-Abnahme gegen den
// Raytrace: test_glass_preselection_raytrace.cpp.

#include "cli/glasses_command.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/dispersion.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/glass/glass_properties.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/insertion.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using mko::test::kGlassPreselectionAchromatRatio;
using mko::test::kGlassPreselectionAlgebra;
using mko::test::kThinMemberSeidel;

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

// ---------------------------------------------------------------------------------------------
// M11-T5g: Startradien des ganzen Glieds und Filter der Kandidaten
// ---------------------------------------------------------------------------------------------

namespace {

mko::GlassCatalog test_catalog() {
    mko::GlassCatalog cat =
        mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/synthetisch.json");
    cat.merge(mko::GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json"));
    return cat;
}

// Dünnes Glied (Dicken 0) hinter der Blende (Radius a) mit den Krümmungen c und Gläsern names,
// λ_ref = 546,074 nm (Bezug des Buchdahl-Fits), Feldwinkel 1°.
mko::SystemDescription thin_member_system(const std::vector<std::string>& names,
                                          const std::vector<double>& c, double a) {
    std::ostringstream s;
    s.precision(17);
    s << R"({"format": "mikroptim.system/1", "name": "Glied", "units": "mm", "glass_catalogs": [],)"
      << R"("spectrum": {"wavelengths_nm": [546.074], "weights": [1.0], "reference_nm": 546.074},)"
      << R"("source": {"type": "collimated", "field_angles_deg": [1.0], "field_weights": [1.0]},)"
      << R"("surfaces": [{"type": "stop", "semi_diameter": )" << a
      << R"(, "thickness": 0.0, "material": "AIR"})";
    for (std::size_t i = 0; i < c.size(); ++i) {
        const bool last = i + 1 == c.size();
        s << R"(, {"type": "sphere", "curvature": )" << c[i] << R"(, "thickness": )"
          << (last ? 100.0 : 0.0) << R"(, "material": ")" << (last ? "AIR" : names[i])
          << R"(", "semi_diameter": )" << 2.0 * a << "}";
    }
    s << "]}";
    return mko::read_system_json(s.str());
}

} // namespace

// Einzellinse: Biegung kleinster sphärischer Aberration für kollimiertes Licht nach Welford,
// B = 2(n² − 1)/(n + 2) (C = −1); S_I hat dort keine reelle Nullstelle. Komafrei:
// B = (2n + 1)(n − 1)/(n + 1).
TEST(GlassPreselection, SingleLensBendingMatchesClassicalFormula) {
    for (const double n : {1.5, 1.7, 1.9}) {
        const std::vector<double> idx{n};
        const std::vector<double> k{0.01};
        const mko::ThinMemberShape sa = mko::thin_member_shape(idx, k);
        ASSERT_EQ(sa.curvatures.size(), 2U);
        const auto shape = [](const mko::ThinMemberShape& m) {
            return (m.curvatures[0] + m.curvatures[1]) / (m.curvatures[0] - m.curvatures[1]);
        };
        EXPECT_FALSE(sa.spherical_zero);
        EXPECT_NEAR(shape(sa), 2.0 * (n * n - 1.0) / (n + 2.0), kGlassPreselectionAlgebra);
        EXPECT_NEAR((n - 1.0) * (sa.curvatures[0] - sa.curvatures[1]), 0.01,
                    kGlassPreselectionAlgebra * 0.01);
        const mko::ThinMemberShape coma =
            mko::thin_member_shape(idx, k, mko::MemberBending::ZeroComa);
        EXPECT_NEAR(shape(coma), (2.0 * n + 1.0) * (n - 1.0) / (n + 1.0),
                    kGlassPreselectionAlgebra);
        EXPECT_NEAR(coma.coma, 0.0, kGlassPreselectionAlgebra * 1e-4);
    }
    // ungültige Eingaben
    const std::vector<double> one{1.5};
    const std::vector<double> two{0.01, 0.02};
    const std::vector<double> bad_n{1.0};
    EXPECT_THROW((void)mko::thin_member_shape(one, two), std::invalid_argument);
    EXPECT_THROW((void)mko::thin_member_shape(bad_n, one), std::invalid_argument);
    EXPECT_THROW((void)mko::thin_member_seidel(one, one), std::invalid_argument);
}

// Bestes Paar und Tripel aus Katalog-Gläsern als dünnes Glied mit den Startkrümmungen beider
// Biegungen: paraxiale Brechkraft gleich φ, Seidel-Summen S_I, S_II des Aberrationsberichts
// (Flächenbeiträge, numerik.md §14) gleich den Formeln (y = a, H aus dem Bericht); bei der
// Biegung „spherical“ mit reeller Lösung S_I = 0, bei „coma“ S_II = 0.
TEST(GlassPreselection, MemberStartRadiiMatchPowerAndSeidelSums) {
    const mko::GlassCatalog cat = test_catalog();
    const std::vector<mko::GlassDispersion> g = mko::catalog_dispersions(cat);
    const double phi = 0.01;
    const double a = 2.0;
    for (const int order : {1, 2}) {
        mko::GlassPreselectionOptions o;
        o.order = order;
        o.power = phi;
        o.max_power_sum = 6.0;
        o.max_results = 3;
        const std::vector<mko::PreselectedGlasses> r = mko::preselect_glasses(g, o);
        ASSERT_FALSE(r.empty());
        for (const mko::PreselectedGlasses& c : r) {
            std::vector<std::string> names;
            std::vector<double> n;
            for (int i : c.glasses) {
                names.push_back(g[static_cast<std::size_t>(i)].name);
                n.push_back(cat.at(names.back()).index(546.074));
            }
            for (const mko::MemberBending b :
                 {mko::MemberBending::MinSpherical, mko::MemberBending::ZeroComa}) {
                const mko::ThinMemberShape m = mko::thin_member_shape(n, c.powers, b);
                ASSERT_EQ(m.curvatures.size(), n.size() + 1);
                const mko::SystemDescription sys = thin_member_system(names, m.curvatures, a);
                EXPECT_NEAR(mko::compute_paraxial(sys, cat, 546.074).power, phi,
                            kGlassPreselectionAlgebra * phi);
                const mko::AberrationReport rep = mko::compute_aberrations(sys, cat);
                // Maßstab: größter Flächenbeitrag
                double s1 = 0.0;
                double s2 = 0.0;
                for (const mko::SurfaceAberration& sa : rep.surfaces) {
                    s1 = std::max(s1, std::abs(sa.S[0]));
                    s2 = std::max(s2, std::abs(sa.S[1]));
                }
                const double y4 = a * a * a * a;
                const double y2h = a * a * rep.lagrange;
                EXPECT_NEAR(rep.S[0], y4 * m.spherical, kThinMemberSeidel * s1);
                EXPECT_NEAR(rep.S[1], y2h * m.coma, kThinMemberSeidel * s2);
                if (b == mko::MemberBending::ZeroComa) {
                    EXPECT_NEAR(rep.S[1], 0.0, kThinMemberSeidel * s2);
                } else if (m.spherical_zero) {
                    EXPECT_NEAR(rep.S[0], 0.0, kThinMemberSeidel * s1);
                }
                // Koeffizienten der Parabel bzw. Geraden gegen thin_member_seidel
                std::vector<double> shifted = m.curvatures;
                for (double& x : shifted) {
                    x += 0.003;
                }
                const auto& [qa, qb, qd] = m.spherical_coefficients;
                const auto& [ls, lt] = m.coma_coefficients;
                const double c1 = shifted[0];
                const std::array<double, 2> direct = mko::thin_member_seidel(n, shifted);
                EXPECT_NEAR(direct[0], qa * c1 * c1 + qb * c1 + qd, kThinMemberSeidel * s1 / y4);
                EXPECT_NEAR(direct[1], ls * c1 + lt, kThinMemberSeidel * s2 / std::abs(y2h));
                std::printf("%s %-9s S_I %+.3e  S_II %+.3e  c =", order == 1 ? "Paar  " : "Tripel",
                            b == mko::MemberBending::ZeroComa ? "coma" : "spherical", rep.S[0],
                            rep.S[1]);
                for (double x : m.curvatures) {
                    std::printf(" %+.5f", x);
                }
                for (const std::string& name : names) {
                    std::printf(" %s", name.c_str());
                }
                std::printf("\n");
            }
        }
    }
}

// Glied im konvergenten Bündel (M6-T11, incoming_slope): vor dem Glied eine dünne Linse N-BK7
// mit Brechkraft K₀ (Dicken 0, an der Blende), also Randstrahlneigung u₁/y = −K₀ am Glied.
// Seidel-Summen der Flächen des Glieds aus dem Aberrationsbericht = thin_member_seidel(n, c, −K₀)
// mal a⁴ bzw. a²H; Biegung S_I = 0 mit dieser Neigung; ohne Neigung bitgleich wie bisher.
TEST(GlassPreselection, MemberShapeWithIncomingSlope) {
    const mko::GlassCatalog cat = test_catalog();
    const std::vector<mko::GlassDispersion> g = mko::catalog_dispersions(cat);
    mko::GlassPreselectionOptions o;
    o.power = 0.01;
    o.max_power_sum = 6.0;
    o.max_results = 1;
    const std::vector<mko::PreselectedGlasses> r = mko::preselect_glasses(g, o);
    ASSERT_EQ(r.size(), 1U);
    std::vector<std::string> names;
    std::vector<double> n;
    for (int i : r[0].glasses) {
        names.push_back(g[static_cast<std::size_t>(i)].name);
        n.push_back(cat.at(names.back()).index(546.074));
    }
    const double a = 2.0;
    const double k0 = 0.02;
    const double n_a = cat.at("N-BK7").index(546.074);
    const double c_a = k0 / (2.0 * (n_a - 1.0)); // bikonvex, Brechkraft K₀
    const mko::ThinMemberShape m =
        mko::thin_member_shape(n, r[0].powers, mko::MemberBending::MinSpherical, -k0);
    const mko::ThinMemberShape m0 = mko::thin_member_shape(n, r[0].powers);
    const mko::ThinMemberShape m0x =
        mko::thin_member_shape(n, r[0].powers, mko::MemberBending::MinSpherical, 0.0);
    EXPECT_EQ(m0.curvatures, m0x.curvatures);
    EXPECT_NE(m.curvatures, m0.curvatures);
    // System: Blende, Linse A (N-BK7, Dicke 0), Luft 0, Glied, Bildabstand 100 mm
    std::vector<std::string> all{"N-BK7", "AIR"};
    std::vector<double> c{c_a, -c_a};
    for (std::size_t i = 0; i < m.curvatures.size(); ++i) {
        all.push_back(i < names.size() ? names[i] : "AIR");
        c.push_back(m.curvatures[i]);
    }
    std::ostringstream s;
    s.precision(17);
    s << R"({"format": "mikroptim.system/1", "name": "Glied", "units": "mm", "glass_catalogs": [],)"
      << R"("spectrum": {"wavelengths_nm": [546.074], "weights": [1.0], "reference_nm": 546.074},)"
      << R"("source": {"type": "collimated", "field_angles_deg": [1.0], "field_weights": [1.0]},)"
      << R"("surfaces": [{"type": "stop", "semi_diameter": )" << a
      << R"(, "thickness": 0.0, "material": "AIR"})";
    for (std::size_t i = 0; i < c.size(); ++i) {
        const bool last = i + 1 == c.size();
        s << R"(, {"type": "sphere", "curvature": )" << c[i] << R"(, "thickness": )"
          << (last ? 100.0 : 0.0) << R"(, "material": ")" << all[i] << R"(", "semi_diameter": )"
          << 2.0 * a << "}";
    }
    s << "]}";
    const mko::SystemDescription sys = mko::read_system_json(s.str());
    const mko::AberrationReport rep = mko::compute_aberrations(sys, cat);
    ASSERT_NEAR(rep.surfaces[3].u_before / rep.surfaces[3].y, -k0, 1e-15);
    double s1 = 0.0;
    double s2 = 0.0;
    double scale1 = 0.0;
    double scale2 = 0.0;
    for (std::size_t i = 3; i < rep.surfaces.size(); ++i) {
        s1 += rep.surfaces[i].S[0];
        s2 += rep.surfaces[i].S[1];
        scale1 = std::max(scale1, std::abs(rep.surfaces[i].S[0]));
        scale2 = std::max(scale2, std::abs(rep.surfaces[i].S[1]));
    }
    const double y4 = a * a * a * a;
    const double y2h = a * a * rep.lagrange;
    EXPECT_NEAR(s1, y4 * m.spherical, kThinMemberSeidel * scale1);
    EXPECT_NEAR(s2, y2h * m.coma, kThinMemberSeidel * scale2);
    const std::array<double, 2> direct = mko::thin_member_seidel(n, m.curvatures, -k0);
    EXPECT_EQ(direct[0], m.spherical);
    if (m.spherical_zero) {
        EXPECT_NEAR(s1, 0.0, kThinMemberSeidel * scale1);
    }
    std::printf("Glied mit u1/y = %.3f/mm: S_I %+.3e (Bericht) %+.3e (Formel), S_II %+.3e/%+.3e\n",
                -k0, s1, y4 * m.spherical, s2, y2h * m.coma);
}

// Filter der Kandidaten: Vorauswahl auf den gefilterten Gläsern = ungefilterte Vorauswahl,
// danach Kombinationen mit einem Glas außerhalb des Filters gestrichen (Bereiche wie
// list_glass_properties, Namenslisten, festes Glas bleibt auch außerhalb des Filters).
TEST(GlassPreselection, CandidateFilterEqualsPostFiltering) {
    const mko::GlassCatalog cat = test_catalog();
    const std::vector<mko::GlassDispersion> all = mko::catalog_dispersions(cat);
    const auto names_of = [](const std::vector<mko::GlassDispersion>& g,
                             const mko::PreselectedGlasses& c) {
        std::vector<std::string> n;
        for (int i : c.glasses) {
            n.push_back(g[static_cast<std::size_t>(i)].name);
        }
        return n;
    };
    const auto index_of = [](const std::vector<mko::GlassDispersion>& list,
                             const std::string& name) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            if (list[i].name == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    std::vector<mko::GlassCandidateFilter> filters(4);
    filters[0].properties.v_d = mko::ValueRange{40.0, std::nullopt};
    filters[1].properties.n_d = mko::ValueRange{1.5, 1.75};
    filters[1].properties.dp_gF = mko::ValueRange{std::nullopt, 0.01};
    filters[2].only = {"N-BK7", "CAF2", "SYN-04", "SYN-12", "SYN-20", "SYN-31"};
    filters[3].exclude = {"CAF2", "SYN-04"};
    filters[3].properties.p_gF = mko::ValueRange{0.53, 0.6};
    int nonempty = 0;
    for (const int order : {1, 2}) {
        for (const bool fixed : {false, true}) {
            for (const mko::GlassCandidateFilter& f : filters) {
                const std::vector<std::string> keep =
                    fixed ? std::vector<std::string>{"SYN-30"} : std::vector<std::string>{};
                const std::vector<mko::GlassDispersion> g =
                    mko::filter_dispersions(all, cat, f, keep);
                const auto allowed = [&](const std::string& name) {
                    if (std::find(keep.begin(), keep.end(), name) != keep.end()) {
                        return true;
                    }
                    if (std::find(f.exclude.begin(), f.exclude.end(), name) != f.exclude.end() ||
                        (!f.only.empty() &&
                         std::find(f.only.begin(), f.only.end(), name) == f.only.end())) {
                        return false;
                    }
                    return f.properties.matches(mko::glass_properties(cat.at(name)));
                };
                std::size_t expect_count = 0;
                for (const mko::GlassDispersion& d : all) {
                    expect_count += allowed(d.name) ? 1U : 0U;
                }
                ASSERT_EQ(g.size(), expect_count);
                mko::GlassPreselectionOptions o;
                o.order = order;
                o.max_results = 100000;
                mko::GlassPreselectionOptions of = o;
                if (fixed) {
                    o.fixed = {index_of(all, "SYN-30")};
                    of.fixed = {index_of(g, "SYN-30")};
                }
                const std::vector<mko::PreselectedGlasses> full = mko::preselect_glasses(all, o);
                const std::vector<mko::PreselectedGlasses> filtered = mko::preselect_glasses(g, of);
                std::vector<std::vector<std::string>> expect;
                for (const mko::PreselectedGlasses& c : full) {
                    const std::vector<std::string> n = names_of(all, c);
                    if (std::all_of(n.begin(), n.end(), allowed)) {
                        expect.push_back(n);
                    }
                }
                ASSERT_EQ(filtered.size(), expect.size());
                nonempty += expect.empty() ? 0 : 1;
                std::printf("Ordnung %d, festes Glas %d: %zu Kandidaten, %zu Kombinationen\n",
                            order, fixed ? 1 : 0, g.size(), expect.size());
                for (std::size_t i = 0; i < expect.size(); ++i) {
                    EXPECT_EQ(names_of(g, filtered[i]), expect[i]) << "Rang " << i;
                }
            }
        }
    }
    // leer nur bei Tripeln ohne festes Glas mit den Filtern 1 und 3 (alle Σ|φ|/φ > 10)
    EXPECT_GE(nonempty, 14);
}

// Spektraler Rest: gleich der Streuung der Brechkraft des dünnen Glieds K(λ) = Σ (n_i(λ) − 1)
// φ_i/(n₀,i − 1) über die Wellenlängen (Buchdahl-Brechzahl, unabhängig ausgewertet), relativ zu
// φ; Rangfolge danach. Bei Tripeln bleibt nur ω³, die Rangfolge ist die des Koeffizienten.
TEST(GlassPreselection, SpectralResidualMatchesThinMemberPowerSpread) {
    const mko::GlassCatalog cat = test_catalog();
    const std::vector<mko::GlassDispersion> g = mko::catalog_dispersions(cat);
    const std::vector<double> lines{435.835, 486.133, 546.074, 587.562, 656.273};
    for (const int order : {1, 2}) {
        mko::GlassPreselectionOptions o;
        o.order = order;
        o.power = 0.01;
        o.max_results = 30;
        o.wavelengths_nm = lines;
        const std::vector<mko::PreselectedGlasses> r = mko::preselect_glasses(g, o);
        ASSERT_EQ(r.size(), 30U);
        for (std::size_t q = 0; q < r.size(); ++q) {
            const mko::PreselectedGlasses& c = r[q];
            std::vector<double> k;
            for (double l : lines) {
                const double w = mko::buchdahl_omega(l * 1e-3);
                double sum = 0.0;
                for (std::size_t i = 0; i < c.glasses.size(); ++i) {
                    const mko::GlassDispersion& d = g[static_cast<std::size_t>(c.glasses[i])];
                    const double n = d.n0 + (d.n0 - 1.0) * (d.eta[0] * w + d.eta[1] * w * w +
                                                            d.eta[2] * w * w * w);
                    sum += (n - 1.0) * c.powers[i] / (d.n0 - 1.0);
                }
                k.push_back(sum / o.power);
            }
            double mean = 0.0;
            for (double x : k) {
                mean += x;
            }
            mean /= static_cast<double>(k.size());
            double var = 0.0;
            for (double x : k) {
                var += (x - mean) * (x - mean);
            }
            const double spread = std::sqrt(var / static_cast<double>(k.size()));
            EXPECT_NEAR(c.spectral_residual, spread, kGlassPreselectionAlgebra * c.power_sum);
            if (q > 0) {
                EXPECT_LE(r[q - 1].spectral_residual, c.spectral_residual);
            }
        }
        if (order == 2) {
            mko::GlassPreselectionOptions plain = o;
            plain.wavelengths_nm.clear();
            const std::vector<mko::PreselectedGlasses> p = mko::preselect_glasses(g, plain);
            for (std::size_t q = 0; q < r.size(); ++q) {
                EXPECT_EQ(p[q].glasses, r[q].glasses);
                EXPECT_TRUE(std::isnan(p[q].spectral_residual));
            }
        }
    }
    mko::GlassPreselectionOptions bad;
    bad.wavelengths_nm = {546.0};
    EXPECT_THROW((void)mko::preselect_glasses(g, bad), std::invalid_argument);
    bad.wavelengths_nm = {546.0, 100.0};
    EXPECT_THROW((void)mko::preselect_glasses(g, bad), std::invalid_argument);
}

// Kommandozeile: Bereichsfilter und Namenslisten in der Vorauswahl, Startkrümmungen.
TEST(CliGlasses, FilterAndStartRadii) {
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
    ASSERT_EQ(
        run({cat, basis, "--vd", "50,", "--fixed", "SYN-30", "--top", "50", "--json"}, out, err),
        mko::cli::kExitOk)
        << err;
    nlohmann::json j = nlohmann::json::parse(out);
    EXPECT_LT(j["glasses"].get<int>(), j["catalog_glasses"].get<int>());
    EXPECT_EQ(j["bending"], "spherical");
    const mko::GlassCatalog c = test_catalog();
    ASSERT_FALSE(j["results"].empty());
    for (const auto& r : j["results"]) {
        EXPECT_EQ(r["glasses"][0], "SYN-30");
        EXPECT_GE(mko::glass_properties(c.at(r["glasses"][1].get<std::string>())).v_d, 50.0);
        ASSERT_EQ(r["curvatures"].size(), 3U);
        // Brechkraft des dünnen Glieds aus den Krümmungen mit n₀
        double k = 0.0;
        for (std::size_t i = 0; i < 2; ++i) {
            k += (r["dispersion"][i]["n0"].get<double>() - 1.0) *
                 (r["curvatures"][i].get<double>() - r["curvatures"][i + 1].get<double>());
        }
        EXPECT_NEAR(k, 1.0, kGlassPreselectionAlgebra);
    }
    ASSERT_EQ(
        run({cat, basis, "--only", "N-BK7,SYN-04,CAF2", "--exclude", "CAF2", "--json"}, out, err),
        mko::cli::kExitOk)
        << err;
    j = nlohmann::json::parse(out);
    EXPECT_EQ(j["glasses"], 2);
    ASSERT_EQ(run({cat, basis, "--top", "2", "--radii", "--bending", "coma"}, out, err),
              mko::cli::kExitOk)
        << err;
    EXPECT_NE(out.find("Krümmungen"), std::string::npos);
    // Gewichte ≠ 1: getrennte Elemente, keine Startkrümmungen
    ASSERT_EQ(run({cat, basis, "--top", "2", "--weights", "1,0.8", "--json"}, out, err),
              mko::cli::kExitOk)
        << err;
    j = nlohmann::json::parse(out);
    EXPECT_FALSE(j["results"][0].contains("curvatures"));
    // Rang nach dem spektralen Rest: Linien explizit bzw. aus der Systemdatei
    ASSERT_EQ(run({cat, basis, "--wavelengths", "486.133,587.562,656.273", "--top", "3", "--json"},
                  out, err),
              mko::cli::kExitOk)
        << err;
    j = nlohmann::json::parse(out);
    EXPECT_EQ(j["wavelengths_nm"].size(), 3U);
    EXPECT_LE(j["results"][0]["spectral_residual"].get<double>(),
              j["results"][1]["spectral_residual"].get<double>());
    const std::string sys = MKO_SOURCE_DIR "/data/systems/einzellinse_achromat_start.json";
    ASSERT_EQ(run({"--system", sys, "--spectrum", "--top", "3"}, out, err), mko::cli::kExitOk)
        << err;
    EXPECT_NE(out.find("Rest (Spektrum)"), std::string::npos);
    EXPECT_EQ(run({cat, "--spectrum"}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--wavelengths", "500"}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--list", "--only", "N-BK7"}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--bending", "x"}, out, err), mko::cli::kExitUsage);
    EXPECT_EQ(run({cat, "--only", "A,,B"}, out, err), mko::cli::kExitUsage);
}
