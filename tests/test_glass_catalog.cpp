// Glaskatalog `mikroptim.glass/1`: Laden, Validierung, Brechzahlen (M1-T2).

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/dispersion.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

using mko::GlassCatalog;
using mko::GlassError;
using mko::test::kIndexCatalog;

namespace {

constexpr double kLambdaD = 587.56; // nm, He d-Linie

const char* basis_path() {
    return MKO_SOURCE_DIR "/data/glass/basis.json";
}

// Gültiges Glas als JSON-Objekt; `extra` wird vor der schließenden Klammer eingefügt.
std::string glass_json(const std::string& name, const std::string& extra = "") {
    return R"({"name": ")" + name +
           R"(", "model": "sellmeier", "B": [1.0, 0.2, 1.0], "C": [0.006, 0.02, 100.0],)"
           R"( "range_nm": [300, 2500])" +
           extra + "}";
}

std::string catalog_json(const std::string& glasses,
                         const std::string& format = "mikroptim.glass/1") {
    return R"({"format": ")" + format + R"(", "glasses": [)" + glasses + "]}";
}

// Erwartet GlassError, dessen Meldung `fragment` enthält.
void expect_load_error(const std::string& text, const std::string& fragment) {
    try {
        (void)GlassCatalog::from_string(text, "test.json");
        ADD_FAILURE() << "keine Ausnahme, erwartet: " << fragment;
    } catch (const GlassError& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(fragment), std::string::npos) << "Meldung: " << msg;
        EXPECT_NE(msg.find("test.json"), std::string::npos) << "Meldung: " << msg;
    }
}

} // namespace

// Abnahme M1-T2: n_d für N-BK7, CaF2 und Quarzglas, jeweils +-1e-5.
TEST(GlassCatalog, AcceptanceRefractiveIndexD) {
    const GlassCatalog cat = GlassCatalog::from_file(basis_path());
    struct Case {
        const char* name;
        double n_d;
    };
    for (const Case& c :
         {Case{"N-BK7", 1.51680}, Case{"CAF2", 1.43385}, Case{"F_SILICA", 1.45846}}) {
        ASSERT_TRUE(cat.contains(c.name)) << c.name;
        const double n = cat.at(c.name).index(kLambdaD);
        EXPECT_NEAR(n, c.n_d, kIndexCatalog) << c.name;
        std::printf("[ n_d ] %-9s n = %.8f  Soll = %.5f  Abweichung = %+.2e\n", c.name, n, c.n_d,
                    n - c.n_d);
    }
}

// Jedes Glas mit Kontrollwert reference.n_d muss ihn treffen.
TEST(GlassCatalog, AllReferenceValuesMatch) {
    const GlassCatalog cat = GlassCatalog::from_file(basis_path());
    EXPECT_EQ(cat.size(), 3u);
    EXPECT_TRUE(cat.warnings().empty());
    int checked = 0;
    for (const mko::Glass& g : cat.glasses()) {
        if (g.reference_n_d) {
            EXPECT_NEAR(g.index(kLambdaD), *g.reference_n_d, kIndexCatalog) << g.name;
            ++checked;
        }
    }
    EXPECT_EQ(checked, 3);
}

TEST(GlassCatalog, FieldsAreRead) {
    const GlassCatalog cat = GlassCatalog::from_file(basis_path());
    const mko::Glass& g = cat.at("N-BK7");
    EXPECT_EQ(g.B[0], 1.03961212);
    EXPECT_EQ(g.C[2], 103.560653);
    EXPECT_EQ(g.range_nm[0], 300.0);
    EXPECT_EQ(g.range_nm[1], 2500.0);
    ASSERT_TRUE(g.reference_n_d.has_value());
    EXPECT_EQ(*g.reference_n_d, 1.5168);
    EXPECT_EQ(g.source, "Schott-Datenblatt N-BK7");
    EXPECT_EQ(cat.find("n-bk7"), nullptr); // Namen sind exakt
    EXPECT_FALSE(cat.contains("N-SF6"));
    EXPECT_THROW((void)cat.at("N-SF6"), GlassError);
}

// Abnahme M1-T2: Wellenlänge außerhalb von range_nm wird gemeldet.
TEST(GlassCatalog, OutOfRangeIsReported) {
    const GlassCatalog cat = GlassCatalog::from_file(basis_path());
    const mko::Glass& bk7 = cat.at("N-BK7"); // range_nm = [300, 2500]
    EXPECT_NO_THROW((void)bk7.index(300.0));
    EXPECT_NO_THROW((void)bk7.index(2500.0));
    EXPECT_FALSE(bk7.in_range(299.9));
    EXPECT_FALSE(bk7.in_range(std::nan("")));
    for (const double lambda : {299.9, 2500.1, 200.0, 10000.0}) {
        try {
            (void)bk7.index(lambda);
            ADD_FAILURE() << "keine Ausnahme bei " << lambda << " nm";
        } catch (const GlassError& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("N-BK7"), std::string::npos) << msg;
            EXPECT_NE(msg.find("außerhalb"), std::string::npos) << msg;
            EXPECT_NE(msg.find("[300, 2500]"), std::string::npos) << msg;
        }
    }
}

TEST(GlassCatalog, MinimalCatalogAndWarnings) {
    const GlassCatalog cat = GlassCatalog::from_string(
        catalog_json(glass_json("A", R"(, "transmission": {"thickness_mm": 10}, "foo": 1)") + "," +
                     glass_json("B", R"(, "reference": {"n_d": 1.5, "n_e": 1.51})")));
    EXPECT_EQ(cat.size(), 2u);
    EXPECT_FALSE(cat.at("A").reference_n_d.has_value());
    EXPECT_EQ(*cat.at("B").reference_n_d, 1.5);
    ASSERT_EQ(cat.warnings().size(), 2u);
    EXPECT_NE(cat.warnings()[0].find("glasses[0].foo"), std::string::npos);
    EXPECT_NE(cat.warnings()[1].find("glasses[1].reference.n_e"), std::string::npos);
}

TEST(GlassCatalog, Merge) {
    GlassCatalog a = GlassCatalog::from_file(basis_path());
    const GlassCatalog b = GlassCatalog::from_string(catalog_json(glass_json("X")));
    a.merge(b);
    EXPECT_EQ(a.size(), 4u);
    EXPECT_TRUE(a.contains("X"));
    EXPECT_TRUE(a.contains("CAF2"));

    // Doppelter Name: Ausnahme, Katalog unverändert.
    const GlassCatalog c =
        GlassCatalog::from_string(catalog_json(glass_json("Y") + "," + glass_json("N-BK7")));
    EXPECT_THROW(a.merge(c), GlassError);
    EXPECT_EQ(a.size(), 4u);
    EXPECT_FALSE(a.contains("Y"));

    mko::Glass g = a.at("X");
    EXPECT_THROW(a.add(g), GlassError);
    g.name = "Z";
    a.add(g);
    EXPECT_TRUE(a.contains("Z"));
}

// Negativtests für das Laden.
TEST(GlassCatalogErrors, Syntax) {
    expect_load_error("{\"format\": ", "JSON-Syntaxfehler");
}

TEST(GlassCatalogErrors, NotAnObject) {
    expect_load_error("[1, 2]", "Objekt");
}

TEST(GlassCatalogErrors, MissingFormat) {
    expect_load_error(R"({"glasses": []})", "format");
}

TEST(GlassCatalogErrors, UnknownMajorVersion) {
    expect_load_error(catalog_json(glass_json("A"), "mikroptim.glass/2"),
                      "unbekannte Hauptversion '2'");
}

TEST(GlassCatalogErrors, WrongFormatName) {
    expect_load_error(catalog_json(glass_json("A"), "mikroptim.system/1"), "kein Glaskatalog");
}

TEST(GlassCatalogErrors, MissingGlasses) {
    expect_load_error(R"({"format": "mikroptim.glass/1"})", "glasses: Pflichtfeld fehlt");
}

TEST(GlassCatalogErrors, MissingRequiredFields) {
    for (const char* field : {"name", "model", "B", "C", "range_nm"}) {
        // Glas aus Einzelteilen, `field` ausgelassen.
        std::string g = "{";
        const std::pair<const char*, const char*> parts[] = {
            {"name", R"("name": "A")"},
            {"model", R"("model": "sellmeier")"},
            {"B", R"("B": [1.0, 0.2, 1.0])"},
            {"C", R"("C": [0.006, 0.02, 100.0])"},
            {"range_nm", R"("range_nm": [300, 2500])"}};
        bool first = true;
        for (const auto& [key, text] : parts) {
            if (std::string(key) == field) {
                continue;
            }
            g += (first ? "" : ", ") + std::string(text);
            first = false;
        }
        g += "}";
        const std::string other = glass_json("OK1") + "," + glass_json("OK2") + ",";
        expect_load_error(catalog_json(other + g),
                          std::string("glasses[2].") + field + ": Pflichtfeld fehlt");
    }
}

TEST(GlassCatalogErrors, WrongTypes) {
    expect_load_error(catalog_json(R"({"name": 7})"), "glasses[0].name: Zeichenkette erwartet");
    expect_load_error(R"({"format": "mikroptim.glass/1", "glasses": {}})", "glasses: Feld");
    expect_load_error(catalog_json("42"), "glasses[0]: Objekt erwartet");
}

TEST(GlassCatalogErrors, WrongArrayLength) {
    // B und C verschieden lang (ab M9-T4 sind 1 … 8 Terme erlaubt, ADR-0014)
    const std::string g = R"({"name": "A", "model": "sellmeier", "B": [1.0, 0.2],)"
                          R"( "C": [0.006, 0.02, 100.0], "range_nm": [300, 2500]})";
    expect_load_error(catalog_json(g), "glasses[0].C: B und C müssen gleich viele Zahlen haben");
    const std::string nine = "[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]";
    expect_load_error(catalog_json(R"({"name": "A", "model": "sellmeier", "B": )" + nine +
                                   R"(, "C": )" + nine + R"(, "range_nm": [300, 2500]})"),
                      "glasses[0].B: Feld mit 1 bis 8 Zahlen erwartet");
    expect_load_error(catalog_json(R"({"name": "A", "model": "sellmeier", "B": [],)"
                                   R"( "C": [], "range_nm": [300, 2500]})"),
                      "glasses[0].B: Feld mit 1 bis 8 Zahlen erwartet");
}

TEST(GlassCatalogErrors, NonNumericCoefficient) {
    const std::string g = R"({"name": "A", "model": "sellmeier", "B": [1.0, 0.2, 1.0],)"
                          R"( "C": [0.006, "x", 100.0], "range_nm": [300, 2500]})";
    expect_load_error(catalog_json(g), "glasses[0].C[1]: Zahl erwartet");
}

TEST(GlassCatalogErrors, UnknownModel) {
    const std::string g = R"({"name": "A", "model": "schott", "B": [1.0, 0.2, 1.0],)"
                          R"( "C": [0.006, 0.02, 100.0], "range_nm": [300, 2500]})";
    expect_load_error(catalog_json(g),
                      "glasses[0].model: unbekanntes Modell 'schott' (unterstützt: sellmeier, "
                      "polynomial)");
}

TEST(GlassCatalogErrors, InvalidRange) {
    const std::string g = R"({"name": "A", "model": "sellmeier", "B": [1.0, 0.2, 1.0],)"
                          R"( "C": [0.006, 0.02, 100.0], "range_nm": [700, 400]})";
    expect_load_error(catalog_json(g), "glasses[0].range_nm");
}

TEST(GlassCatalogErrors, PoleInsideRange) {
    // C[1] = 0,25 µm^2 -> Pol bei 500 nm
    const std::string g = R"({"name": "A", "model": "sellmeier", "B": [1.0, 0.2, 1.0],)"
                          R"( "C": [0.006, 0.25, 100.0], "range_nm": [300, 2500]})";
    expect_load_error(catalog_json(g), "glasses[0].C[1]: Sellmeier-Pol bei 500 nm");
}

TEST(GlassCatalogErrors, DuplicateName) {
    expect_load_error(catalog_json(glass_json("A") + "," + glass_json("B") + "," + glass_json("A")),
                      "glasses[2].name: doppelter Name 'A' (bereits in glasses[0])");
}

TEST(GlassCatalogErrors, ReservedOrEmptyName) {
    expect_load_error(catalog_json(glass_json("AIR")), "reserviert");
    expect_load_error(catalog_json(glass_json("")), "leer");
}

TEST(GlassCatalogErrors, MissingFile) {
    EXPECT_THROW((void)GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/gibt_es_nicht.json"),
                 GlassError);
}

// --- Glasmodelle ab M9-T4 (ADR-0014) --------------------------------------------------------

// 3-Term-Sellmeier ohne Konstante ist bitgleich zur bisherigen Formel (Ergebnisse unverändert).
TEST(GlassModels, ThreeTermSellmeierBitIdenticalToLegacyFormula) {
    const GlassCatalog cat = GlassCatalog::from_file(basis_path());
    for (const mko::Glass& g : cat.glasses()) {
        ASSERT_EQ(g.model, mko::GlassModel::Sellmeier);
        ASSERT_EQ(g.B.size(), 3u);
        EXPECT_EQ(g.A, 0.0);
        for (int k = 0; k <= 100; ++k) {
            const double nm = 400.0 + 3.0 * k;
            const double legacy = mko::sellmeier_index(g.B.data(), g.C.data(), nm / 1000.0);
            EXPECT_EQ(g.index(nm), legacy) << g.name << " bei " << nm << " nm";
        }
    }
}

TEST(GlassModels, SellmeierWithConstantAndFiveTerms) {
    const std::string g =
        R"({"name": "S5", "model": "sellmeier", "A": 0.05,)"
        R"( "B": [0.9, 0.2, 0.05, 0.01, 0.8], "C": [0.005, 0.02, 0.04, 0.06, 90.0],)"
        R"( "range_nm": [300, 2500]})";
    const GlassCatalog cat = GlassCatalog::from_string(catalog_json(g), "test.json");
    const mko::Glass& s5 = cat.at("S5");
    EXPECT_EQ(s5.B.size(), 5u);
    for (double nm : {350.0, 587.56, 1064.0, 2000.0}) {
        const double l2 = (nm / 1000.0) * (nm / 1000.0);
        const double n2 = 1.0 + 0.05 + 0.9 * l2 / (l2 - 0.005) + 0.2 * l2 / (l2 - 0.02) +
                          0.05 * l2 / (l2 - 0.04) + 0.01 * l2 / (l2 - 0.06) +
                          0.8 * l2 / (l2 - 90.0);
        EXPECT_NEAR(s5.index(nm), std::sqrt(n2), 1e-15) << nm;
    }
}

TEST(GlassModels, PolynomialSchottFormula) {
    // Schott-Formel n^2 = a0 + a1 l^2 + a2 l^-2 + a3 l^-4 + a4 l^-6 + a5 l^-8 (frei gewählt)
    const double a[6] = {2.27, -0.0094, 0.0110, 0.000149, -1.1e-5, 8.0e-7};
    const std::string g =
        R"({"name": "P", "model": "polynomial", "terms": [[2.27, 0], [-0.0094, 2],)"
        R"( [0.0110, -2], [0.000149, -4], [-1.1e-5, -6], [8.0e-7, -8]], "range_nm": [365, 1100]})";
    const GlassCatalog cat = GlassCatalog::from_string(catalog_json(g), "test.json");
    const mko::Glass& p = cat.at("P");
    EXPECT_EQ(p.model, mko::GlassModel::Polynomial);
    for (double nm : {365.0, 435.83, 587.56, 656.27, 1014.0, 1100.0}) {
        const double l = nm / 1000.0;
        const double l2 = l * l;
        const double n2 = a[0] + a[1] * l2 + a[2] / l2 + a[3] / (l2 * l2) + a[4] / (l2 * l2 * l2) +
                          a[5] / (l2 * l2 * l2 * l2);
        EXPECT_NEAR(p.index(nm), std::sqrt(n2), 1e-14) << nm;
    }
}

TEST(GlassModels, PolynomialNonIntegerExponent) {
    // refractiveindex.info erlaubt beliebige Exponenten: n^2 = 2.1 + 0.01 l^-2.5
    const std::string g =
        R"({"name": "Q", "model": "polynomial", "terms": [[2.1, 0], [0.01, -2.5]],)"
        R"( "range_nm": [400, 800]})";
    const GlassCatalog cat = GlassCatalog::from_string(catalog_json(g), "test.json");
    EXPECT_NEAR(cat.at("Q").index(500.0), std::sqrt(2.1 + 0.01 * std::pow(0.5, -2.5)), 1e-15);
    EXPECT_EQ(mko::int_power(0.5, -3), 8.0);
    EXPECT_EQ(mko::int_power(1.5, 0), 1.0);
}

TEST(GlassModelsErrors, PolynomialTerms) {
    expect_load_error(catalog_json(R"({"name": "P", "model": "polynomial", "terms": [],)"
                                   R"( "range_nm": [400, 800]})"),
                      "glasses[0].terms: Feld mit 1 bis 12 Termen");
    expect_load_error(catalog_json(R"({"name": "P", "model": "polynomial", "terms": [[1.0]],)"
                                   R"( "range_nm": [400, 800]})"),
                      "glasses[0].terms[0]: Feld mit genau 2 Zahlen erwartet");
    expect_load_error(catalog_json(R"({"name": "P", "model": "polynomial",)"
                                   R"( "range_nm": [400, 800]})"),
                      "glasses[0].terms: Pflichtfeld fehlt");
}

TEST(GlassModelsErrors, InvalidIndexInsideRange) {
    // n^2 = 2 - 4 l^2 wird ab 707 nm negativ
    expect_load_error(catalog_json(R"({"name": "P", "model": "polynomial",)"
                                   R"( "terms": [[2.0, 0], [-4.0, 2]], "range_nm": [400, 800]})"),
                      "keine gültige Brechzahl");
}
