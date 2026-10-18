// Brechzahltabelle n[Fläche][Wellenlänge] mit allen vier Materialarten (M1-T2).

#include "mikroptim/glass/dispersion.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using mko::GlassCatalog;
using mko::GlassError;
using mko::MaterialKind;
using mko::MaterialSpec;

namespace {

const GlassCatalog& basis() {
    static const GlassCatalog cat =
        GlassCatalog::from_file(MKO_SOURCE_DIR "/data/glass/basis.json");
    return cat;
}

MaterialSpec air() {
    return MaterialSpec{};
}

MaterialSpec catalog_glass(const std::string& name) {
    MaterialSpec m;
    m.kind = MaterialKind::Catalog;
    m.name = name;
    return m;
}

MaterialSpec constant(double n) {
    MaterialSpec m;
    m.kind = MaterialKind::Constant;
    m.n = n;
    return m;
}

MaterialSpec buchdahl(double n0, double nu1, double nu2, double nu3) {
    MaterialSpec m;
    m.kind = MaterialKind::Buchdahl;
    m.n0 = n0;
    m.nu = {nu1, nu2, nu3};
    return m;
}

// Erwartet GlassError, dessen Meldung alle `fragments` enthält.
template <class F>
void expect_error(F&& f, std::initializer_list<const char*> fragments) {
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

} // namespace

TEST(IndexTable, AllMaterialKinds) {
    const std::vector<MaterialSpec> media = {
        air(),                             // 0: Blende in Luft
        catalog_glass("N-BK7"),            // 1
        constant(1.7),                     // 2
        buchdahl(1.5, -0.1, 0.02, -0.003), // 3
        catalog_glass("CAF2"),             // 4
        air(),                             // 5
    };
    const std::vector<double> wl = {486.13, 587.56, 656.27, 546.074};
    const mko::IndexTable t = mko::make_index_table(media, wl, basis());

    ASSERT_EQ(t.surfaces, 6);
    ASSERT_EQ(t.wavelengths, 4);
    ASSERT_EQ(t.n.size(), 24u);
    for (int l = 0; l < t.wavelengths; ++l) {
        const double lambda_um = mko::nm_to_um(wl[l]);
        EXPECT_EQ(t.at(0, l), 1.0);
        EXPECT_EQ(t.at(1, l), basis().at("N-BK7").index(wl[l]));
        EXPECT_EQ(t.at(2, l), 1.7);
        EXPECT_EQ(t.at(3, l), mko::buchdahl_index(1.5, -0.1, 0.02, -0.003, lambda_um));
        EXPECT_EQ(t.at(4, l), basis().at("CAF2").index(wl[l]));
        EXPECT_EQ(t.at(5, l), 1.0);
        // Zeilenweises Layout [s][l]
        for (int s = 0; s < t.surfaces; ++s) {
            EXPECT_EQ(t.at(s, l), t.n[static_cast<std::size_t>(s * t.wavelengths + l)]);
        }
    }
    // Abnahmewert über die Tabelle: N-BK7 bei 587,56 nm
    EXPECT_NEAR(t.at(1, 1), 1.51680, mko::test::kIndexCatalog);
    // Buchdahl bei lambda0 (546,074 nm): n = n0 (bis auf Rundung der nm->µm-Umrechnung)
    EXPECT_NEAR(t.at(3, 3), 1.5, mko::test::kUnitFp64);
    // Normale Dispersion: F > d > C
    EXPECT_GT(t.at(1, 0), t.at(1, 1));
    EXPECT_GT(t.at(1, 1), t.at(1, 2));
}

TEST(IndexTable, Empty) {
    const mko::IndexTable t =
        mko::make_index_table(std::vector<MaterialSpec>{}, std::vector<double>{500.0}, basis());
    EXPECT_EQ(t.surfaces, 0);
    EXPECT_EQ(t.wavelengths, 1);
    EXPECT_TRUE(t.n.empty());
}

TEST(IndexTable, OutOfRangeNamesSurfaceAndGlass) {
    const std::vector<MaterialSpec> media = {air(), air(), catalog_glass("N-BK7")};
    const std::vector<double> wl = {587.56, 250.0};
    expect_error([&] { (void)mko::make_index_table(media, wl, basis()); },
                 {"Fläche 2", "250 nm", "N-BK7", "außerhalb", "[300, 2500]"});
}

TEST(IndexTable, UnknownGlass) {
    const std::vector<MaterialSpec> media = {air(), catalog_glass("N-SF6")};
    expect_error([&] { (void)mko::make_index_table(media, std::vector<double>{587.56}, basis()); },
                 {"Fläche 1", "N-SF6"});
}

TEST(IndexTable, InvalidParameters) {
    expect_error([&] { (void)mko::material_index(constant(-1.0), 500.0, basis()); },
                 {"konstante Brechzahl"});
    expect_error([&] { (void)mko::material_index(air(), 0.0, basis()); },
                 {"ungültige Wellenlänge"});
    expect_error([&] { (void)mko::material_index(buchdahl(1.5, 0.0, 0.0, 0.0), 140.0, basis()); },
                 {"Buchdahl", "140 nm"});
}
