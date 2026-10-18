// Kleinste-Quadrate-Fit des Buchdahl-Modells an Sellmeier (M1-T3, docs/numerik.md, Abschnitt 10).
//
// Termzahl: "3 Koeffizienten" der Abnahme sind die Dispersionskoeffizienten nu1, nu2, nu3
// (plus n0), "2 Koeffizienten" sind nu1, nu2 (plus n0, nu3 = 0).

#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/dispersion.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <ios>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

using mko::BuchdahlFit;
using mko::BuchdahlFitNodes;
using mko::BuchdahlFitOptions;
using mko::Glass;
using mko::GlassCatalog;
using mko::GlassError;

namespace {

const GlassCatalog& basis() {
    static const GlassCatalog cat =
        GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json");
    return cat;
}

std::string sci(double x) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os.precision(3);
    os << std::scientific << x;
    return os.str();
}

// Erwartet GlassError, dessen Meldung alle `fragments` enthält.
template <class F> void expect_error(F&& f, std::initializer_list<const char*> fragments) {
    try {
        f();
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const GlassError& e) {
        const std::string msg = e.what();
        for (const char* frag : fragments) {
            EXPECT_NE(msg.find(frag), std::string::npos) << "fehlt: " << frag << "\n" << msg;
        }
    }
}

// Brechzahlfunktion, die exakt ein Buchdahl-Polynom ist (analytisches Orakel).
struct BuchdahlPolynomial {
    double n0, nu1, nu2, nu3;
    double operator()(double lambda_nm) const {
        return mko::buchdahl_index(n0, nu1, nu2, nu3, mko::nm_to_um(lambda_nm));
    }
};

// Koeffizienten in der Größenordnung von N-BK7 bzw. CaF2 (Ergebnis des 3-Term-Fits).
constexpr BuchdahlPolynomial kPoly3{1.518724, -0.05146, 0.00528, -0.000575};
constexpr BuchdahlPolynomial kPoly3b{1.434941, -0.02924, 0.00513, 0.00715};

void expect_exact(const BuchdahlFit& fit, const BuchdahlPolynomial& p) {
    const double tol = mko::test::kBuchdahlFitExact;
    EXPECT_NEAR(fit.n0, p.n0, tol);
    EXPECT_NEAR(fit.nu[0], p.nu1, tol);
    EXPECT_NEAR(fit.nu[1], p.nu2, tol);
    EXPECT_NEAR(fit.nu[2], p.nu3, tol);
    EXPECT_LE(fit.max_abs_error, tol);
    EXPECT_LE(fit.rms_error, fit.max_abs_error);
}

} // namespace

// Abnahme M1-T3: maximaler Fitfehler mit 3 Dispersionstermen < 1e-5 für alle Gläser aus
// basis.json; der Fehler mit 2 Termen wird protokolliert (kein Grenzwert).
TEST(BuchdahlFit, AcceptanceBasisCatalog) {
    ASSERT_FALSE(basis().empty());
    std::printf("Buchdahl-Fit 400-700 nm, 301 Tschebyschow-Knoten, Prüfgitter 0,1 nm\n");
    std::printf("%-10s %10s %11s %11s %11s | %9s %9s | %9s %9s | Äquidistant: %9s %9s\n", "Glas",
                "n0", "nu1", "nu2", "nu3", "max3", "rms3", "max2", "rms2", "max3", "max2");
    for (const Glass& g : basis().glasses()) {
        const BuchdahlFit f3 = mko::fit_buchdahl(g, 3);
        const BuchdahlFit f2 = mko::fit_buchdahl(g, 2);
        BuchdahlFitOptions eq;
        eq.nodes = BuchdahlFitNodes::Equidistant;
        const BuchdahlFit e3 = mko::fit_buchdahl(g, 3, eq);
        const BuchdahlFit e2 = mko::fit_buchdahl(g, 2, eq);

        std::printf("%-10s %10.7f %11.4e %11.4e %11.4e | %9.2e %9.2e | %9.2e %9.2e | "
                    "%22.2e %9.2e\n",
                    g.name.c_str(), f3.n0, f3.nu[0], f3.nu[1], f3.nu[2], f3.max_abs_error,
                    f3.rms_error, f2.max_abs_error, f2.rms_error, e3.max_abs_error,
                    e2.max_abs_error);
        RecordProperty(g.name + "_max_error_3", sci(f3.max_abs_error));
        RecordProperty(g.name + "_max_error_2", sci(f2.max_abs_error));
        RecordProperty(g.name + "_rms_error_2", sci(f2.rms_error));

        EXPECT_EQ(f3.dispersion_terms, 3);
        EXPECT_EQ(f2.dispersion_terms, 2);
        EXPECT_EQ(f2.nu[2], 0.0);
        EXPECT_LT(f3.max_abs_error, mko::test::kBuchdahlFit3) << g.name;
        // Plausibilität: Das größere Modell passt nicht schlechter; n0 ~ Brechzahl bei 546 nm.
        EXPECT_GE(f2.rms_error, f3.rms_error) << g.name;
        EXPECT_GE(f2.max_abs_error, f3.max_abs_error) << g.name;
        EXPECT_NEAR(f3.n0, g.index(546.074), mko::test::kBuchdahlFit3) << g.name;
        // Normale Dispersion: n fällt mit lambda, also nu1 = dn/dw < 0.
        EXPECT_LT(f3.nu[0], 0.0) << g.name;
        EXPECT_LT(f2.nu[0], 0.0) << g.name;
    }
}

// Analytisches Orakel: Ist n(lambda) exakt ein Buchdahl-Polynom, liefert der Fit dessen
// Koeffizienten (für beide Knotenverteilungen und auch im interpolierenden Grenzfall).
TEST(BuchdahlFit, RecoversExactPolynomial) {
    for (const BuchdahlPolynomial& p : {kPoly3, kPoly3b}) {
        expect_exact(mko::fit_buchdahl(p, 3), p);
        BuchdahlFitOptions eq;
        eq.nodes = BuchdahlFitNodes::Equidistant;
        expect_exact(mko::fit_buchdahl(p, 3, eq), p);
        BuchdahlFitOptions interp; // 4 Stützstellen für 4 Unbekannte
        interp.samples = 4;
        expect_exact(mko::fit_buchdahl(p, 3, interp), p);
        BuchdahlFitOptions wide; // anderer Bereich
        wide.lambda_min_nm = 300.0;
        wide.lambda_max_nm = 1100.0;
        wide.samples = 57;
        expect_exact(mko::fit_buchdahl(p, 3, wide), p);
    }
    // 2 Terme an ein quadratisches Polynom; 3 Terme an dasselbe liefern nu3 = 0.
    const BuchdahlPolynomial quad{1.6, -0.08, 0.012, 0.0};
    expect_exact(mko::fit_buchdahl(quad, 2), quad);
    expect_exact(mko::fit_buchdahl(quad, 3), quad);
}

// Roundtrip: Fit -> MaterialSpec -> make_index_table weicht auf dem Prüfgitter höchstens um
// max_abs_error von der Sellmeier-Brechzahl ab; das Maximum wird genau erreicht, weil
// material_index dieselbe Formel auswertet.
TEST(BuchdahlFit, RoundtripThroughIndexTable) {
    for (const Glass& g : basis().glasses()) {
        for (const int terms : {2, 3}) {
            const BuchdahlFit fit = mko::fit_buchdahl(g, terms);
            const mko::MaterialSpec m = mko::to_buchdahl_material(fit);
            EXPECT_EQ(m.kind, mko::MaterialKind::Buchdahl);
            EXPECT_EQ(m.n0, fit.n0);
            EXPECT_EQ(m.nu, fit.nu);

            const std::vector<double> grid = mko::buchdahl_check_grid_nm(fit.options);
            const std::vector<mko::MaterialSpec> media = {m};
            const mko::IndexTable t = mko::make_index_table(media, grid, basis());
            double max_err = 0.0;
            for (int l = 0; l < t.wavelengths; ++l) {
                const double lambda_nm = grid[static_cast<std::size_t>(l)];
                EXPECT_EQ(t.at(0, l), fit.index(lambda_nm));
                const double err = std::fabs(t.at(0, l) - g.index(lambda_nm));
                EXPECT_LE(err, fit.max_abs_error) << g.name << " bei " << lambda_nm << " nm";
                max_err = std::fmax(max_err, err);
            }
            EXPECT_EQ(max_err, fit.max_abs_error) << g.name;
        }
    }
}

TEST(BuchdahlFit, NodesAndCheckGrid) {
    BuchdahlFitOptions o;
    const std::vector<double> cheb = mko::buchdahl_fit_nodes_nm(o);
    ASSERT_EQ(cheb.size(), 301u);
    EXPECT_GT(cheb.front(), 400.0);
    EXPECT_LT(cheb.back(), 700.0);
    EXPECT_NEAR(cheb[150], 550.0, mko::test::kLengthFp64); // mittlerer Knoten bei ungerader Zahl
    for (std::size_t i = 1; i < cheb.size(); ++i) {
        EXPECT_LT(cheb[i - 1], cheb[i]);
        EXPECT_NEAR(cheb[i - 1] + cheb[cheb.size() - i], 1100.0, mko::test::kLengthFp64);
    }

    o.nodes = BuchdahlFitNodes::Equidistant;
    const std::vector<double> eq = mko::buchdahl_fit_nodes_nm(o);
    ASSERT_EQ(eq.size(), 301u);
    EXPECT_EQ(eq.front(), 400.0);
    EXPECT_EQ(eq.back(), 700.0);
    EXPECT_NEAR(eq[1] - eq[0], 1.0, mko::test::kLengthFp64);

    const std::vector<double> grid = mko::buchdahl_check_grid_nm(o);
    ASSERT_EQ(grid.size(), 3001u);
    EXPECT_EQ(grid.front(), 400.0);
    EXPECT_EQ(grid.back(), 700.0);
    EXPECT_NEAR(grid[1] - grid[0], 0.1, mko::test::kLengthFp64);
}

TEST(BuchdahlFit, Errors) {
    Glass narrow = basis().at("N-BK7");
    narrow.name = "SCHMAL";
    narrow.range_nm = {450.0, 2500.0};
    expect_error([&] { (void)mko::fit_buchdahl(narrow, 3); },
                 {"SCHMAL", "Fitbereich [400, 700]", "[450, 2500]"});
    BuchdahlFitOptions upper;
    upper.lambda_max_nm = 2600.0;
    expect_error([&] { (void)mko::fit_buchdahl(narrow, 3, upper); }, {"SCHMAL", "2600"});

    const Glass& bk7 = basis().at("N-BK7");
    for (const int terms : {0, 1, 4}) {
        expect_error([&] { (void)mko::fit_buchdahl(bk7, terms); },
                     {"N-BK7", "2 oder 3 Dispersionsterme"});
    }

    BuchdahlFitOptions few;
    few.samples = 3; // 3 Terme brauchen 4 Stützstellen
    expect_error([&] { (void)mko::fit_buchdahl(bk7, 3, few); }, {"3 Stützstellen", "4 Unbekannte"});
    EXPECT_NO_THROW((void)mko::fit_buchdahl(bk7, 2, few));
    few.samples = 1;
    expect_error([&] { (void)mko::fit_buchdahl(kPoly3, 2, few); }, {"mindestens 2 Stützstellen"});

    BuchdahlFitOptions coarse;
    coarse.check_samples = 1;
    expect_error([&] { (void)mko::fit_buchdahl(kPoly3, 3, coarse); }, {"Prüfgitter"});

    BuchdahlFitOptions empty;
    empty.lambda_min_nm = 700.0;
    empty.lambda_max_nm = 400.0;
    expect_error([&] { (void)mko::fit_buchdahl(kPoly3, 3, empty); }, {"ungültiger Bereich"});
    BuchdahlFitOptions nan;
    nan.lambda_min_nm = std::numeric_limits<double>::quiet_NaN();
    expect_error([&] { (void)mko::fit_buchdahl(kPoly3, 3, nan); }, {"ungültiger Bereich"});

    BuchdahlFitOptions deep_uv; // Buchdahl-Koordinate erst oberhalb 146,074 nm definiert
    deep_uv.lambda_min_nm = 140.0;
    expect_error([&] { (void)mko::fit_buchdahl(kPoly3, 3, deep_uv); }, {"140 nm", "146,074"});

    expect_error(
        [&] {
            (void)mko::fit_buchdahl(
                [](double lambda_nm) {
                    return lambda_nm > 600.0 ? std::numeric_limits<double>::infinity() : 1.5;
                },
                3);
        },
        {"nicht endlich"});
    expect_error([&] { (void)mko::fit_buchdahl(mko::IndexFunction{}, 3); },
                 {"keine Brechzahlfunktion"});
}
