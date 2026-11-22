// Erweitertes Glas-Wörterbuch der Geisterplatten (M6-T12, docs/optimierung.md, Abschnitt 5
// „Erweitertes Wörterbuch“): Auswahl nach der Rangfolge der Glasvorauswahl (Orakel: jedes Paar
// der Vorauswahl einzeln gelöst, Kennzahlen glass_properties), Merit-Invarianz und
// Wiederholbarkeit der Erweiterung, und optische Orakel: eine Flint-Platte des erweiterten
// Wörterbuchs mit der Farbe des dünnen Achromats korrigiert die Farbe der Brechkraft einer
// N-BK7-Einzellinse; die Diagnose führt bei aktiver Flint-Platte zu einem Flint-Element
// (Operator A) bzw. zu einem Kittglied mit dem Flint als Zielglas (Operator B), das die
// Längsfarbe schon beim Einfügen korrigiert.

#include "insertion_optimization.hpp"
#include "insertion_test_systems.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/glass/glass_properties.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/design_operators.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/synth/ghost_dictionary.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace mko::test {
namespace {

constexpr double kNormRadius = 2.5;
// Flintglas mit normaler Teildispersion aus den frei erfundenen Testgläsern (n_d 1,657,
// ν_d 34,5, ΔP_g,F −0,002), ausdrücklich genannt
const std::string kFlint = "SYN-21";

// Einzellinse N-BK7 (f = 50 mm, 480/644 nm, Kataloge basis + synthetisch) mit dem Wörterbuch der
// Zielspezifikation (SYN-36, CaF₂, SYN-30 als Katalogglas, K = 4) 1 mm hinter der Blende,
// eigener Eintrag ghost_penalty.
LoadedSystem singlet_with_dictionary() {
    const LoadedSystem start = load_residual_system("einzellinse_achromat_start.json");
    GhostDictionaryPlacement p;
    p.after_surface = 0;
    p.distance = 1.0;
    p.glasses = {catalog_glass("SYN-36"), catalog_glass("CAF2"), catalog_glass("SYN-30")};
    p.order = 4;
    p.norm_radius = kNormRadius;
    p.penalty_weight = 47.2;
    return prepare_test_system(place_ghost_dictionary(start.system, p).system);
}

std::size_t plate_of(const SystemDescription& d, const std::string& name) {
    for (std::size_t g = 0; g < d.ghost_plates.size(); ++g) {
        if (d.ghost_plates[g].glass.kind == MaterialKind::Catalog &&
            d.ghost_plates[g].glass.name == name) {
            return g;
        }
    }
    throw std::logic_error("keine Platte aus " + name);
}

// Unabhängige Rechnung der Rangfolge: jedes Paar zweier Katalog-Gläser als eigene Zweierliste in
// der Vorauswahl, kleinster spektraler Rest je Glas.
struct RankOracle {
    std::string name;
    double residual = std::numeric_limits<double>::infinity();
    GlassProperties p;
};

std::vector<RankOracle> rank_oracle(const GlassCatalog& catalog,
                                    const GhostDictionaryGlassOptions& o) {
    const std::vector<GlassDispersion> cat = catalog_dispersions(catalog);
    std::vector<RankOracle> out(cat.size());
    for (std::size_t a = 0; a < cat.size(); ++a) {
        out[a].name = cat[a].name;
        for (std::size_t b = 0; b < cat.size(); ++b) {
            if (a == b) {
                continue;
            }
            const std::vector<GlassDispersion> two{cat[a], cat[b]};
            GlassPreselectionOptions po;
            po.max_power_sum = o.max_power_sum;
            po.wavelengths_nm = o.wavelengths_nm;
            for (const PreselectedGlasses& s : preselect_glasses(two, po)) {
                out[a].residual = std::min(out[a].residual, s.spectral_residual);
            }
        }
    }
    std::vector<RankOracle> valid;
    for (RankOracle& r : out) {
        try {
            r.p = glass_properties(catalog.at(r.name));
        } catch (const std::exception&) {
            continue;
        }
        if (std::isfinite(r.residual)) {
            valid.push_back(r);
        }
    }
    return valid;
}

bool is_anomalous(const GlassProperties& p) {
    return std::abs(p.dp_gF) >= 0.01;
}
bool is_flint(const GlassProperties& p) {
    return !is_anomalous(p) && (p.n_d < 1.60 ? p.v_d < 55.0 : p.v_d < 50.0);
}

// Auswahl gegen die unabhängige Rangfolge: abwechselnd das beste Flint- und das beste anomale
// Glas; Gläser, die schon im Wörterbuch stehen (CaF₂ als Katalogglas, SYN-36 und SYN-30 als
// Buchdahl-Fit im Startsystem der Zielspezifikation), sind als vorhanden markiert.
TEST(GhostDictionary, SelectionFollowsPreselectionRank) {
    const LoadedSystem start = load_residual_system("zielspezifikation_start.json");
    GhostDictionaryGlassOptions o;
    o.count = 4;
    const std::vector<GhostDictionaryGlass> sel =
        select_ghost_dictionary_glasses(start.system, start.catalog, o);
    ASSERT_EQ(sel.size(), 4U);

    std::vector<RankOracle> flints;
    std::vector<RankOracle> anomalous;
    for (const RankOracle& r : rank_oracle(start.catalog, o)) {
        if (is_flint(r.p)) {
            flints.push_back(r);
        } else if (is_anomalous(r.p)) {
            anomalous.push_back(r);
        }
    }
    const auto by_residual = [](const RankOracle& a, const RankOracle& b) {
        return a.residual < b.residual;
    };
    std::stable_sort(flints.begin(), flints.end(), by_residual);
    std::stable_sort(anomalous.begin(), anomalous.end(), by_residual);
    ASSERT_GE(flints.size(), 2U);
    ASSERT_GE(anomalous.size(), 2U);
    const std::vector<const RankOracle*> expected{&flints[0], &anomalous[0], &flints[1],
                                                  &anomalous[1]};
    for (std::size_t k = 0; k < sel.size(); ++k) {
        std::printf("%zu: %s (%s, n_d %.4f, v_d %.2f, dP %+.4f, Partner %s, Rest %.3e%s)\n", k,
                    sel[k].name.c_str(), std::string(to_string(sel[k].glass_class)).c_str(),
                    sel[k].n_d, sel[k].v_d, sel[k].dp_gF, sel[k].partner.c_str(),
                    sel[k].spectral_residual, sel[k].present ? ", vorhanden" : "");
        EXPECT_EQ(sel[k].name, expected[k]->name) << k;
        EXPECT_NEAR(sel[k].spectral_residual, expected[k]->residual,
                    1e-12 * std::max(1.0, expected[k]->residual))
            << k;
        EXPECT_TRUE(sel[k].automatic);
        EXPECT_EQ(sel[k].glass_class,
                  k % 2 == 0 ? GhostGlassClass::Flint : GhostGlassClass::Anomalous);
        const bool in_dictionary =
            sel[k].name == "CAF2" || sel[k].name == "SYN-36" || sel[k].name == "SYN-30";
        EXPECT_EQ(sel[k].present, in_dictionary) << sel[k].name;
    }

    // ausdrücklich genannte Gläser zuerst, dann die Rangfolge ohne den genannten
    o.count = 1;
    o.names = {flints[0].name};
    const std::vector<GhostDictionaryGlass> named =
        select_ghost_dictionary_glasses(start.system, start.catalog, o);
    ASSERT_EQ(named.size(), 2U);
    EXPECT_EQ(named[0].name, flints[0].name);
    EXPECT_FALSE(named[0].automatic);
    EXPECT_EQ(named[1].name, flints[1].name);
    o.names = {"CAF2"};
    EXPECT_TRUE(select_ghost_dictionary_glasses(start.system, start.catalog, o)[0].present);

    // Fehler
    o.names = {"GIBTESNICHT"};
    EXPECT_THROW((void)select_ghost_dictionary_glasses(start.system, start.catalog, o),
                 std::invalid_argument);
    o.names = {flints[0].name, flints[0].name};
    EXPECT_THROW((void)select_ghost_dictionary_glasses(start.system, start.catalog, o),
                 std::invalid_argument);
    o.names.clear();
    o.count = -1;
    EXPECT_THROW((void)select_ghost_dictionary_glasses(start.system, start.catalog, o),
                 std::invalid_argument);
}

// Erweiterung: eine Platte je neuem Glas und Lage mit Ordnung und r_g der ersten Platte,
// Variablen und Strafe decken sie ab, Merit unverändert (auch mit aktiven Platten), wiederholter
// Aufruf ändert nichts, ohne Platten bzw. ohne Gläser unverändert.
TEST(GhostDictionary, ExtensionKeepsMeritAndIsIdempotent) {
    LoadedSystem l = singlet_with_dictionary();
    SystemDescription d = l.system;
    d.ghost_plates[0].coefficients = {2e-4, -1e-4, 5e-5, 0.0};
    d.ghost_plates[2].coefficients = {-1e-4, 0.0, 3e-5, 0.0};
    GhostDictionaryGlassOptions o;
    o.count = 4;
    const GhostDictionaryExtension e = extend_ghost_dictionaries(d, l.catalog, o);
    ASSERT_EQ(e.glasses.size(), 4U);
    std::vector<std::string> added;
    for (const GhostDictionaryGlass& g : e.glasses) {
        if (!g.present) {
            added.push_back(g.name);
        }
    }
    ASSERT_GE(added.size(), 2U);
    EXPECT_EQ(e.positions, 1);
    EXPECT_EQ(e.added_plates, static_cast<int>(added.size()));
    ASSERT_EQ(e.system.ghost_plates.size(), 3 + added.size());
    for (std::size_t g = 0; g < 3; ++g) {
        EXPECT_EQ(e.system.ghost_plates[g], d.ghost_plates[g]);
    }
    for (std::size_t g = 3; g < e.system.ghost_plates.size(); ++g) {
        const GhostPlate& p = e.system.ghost_plates[g];
        EXPECT_EQ(p.glass.name, added[g - 3]);
        EXPECT_EQ(p.after_surface, 0);
        EXPECT_EQ(p.distance, 1.0);
        EXPECT_EQ(p.norm_radius, kNormRadius);
        EXPECT_EQ(p.coefficients, std::vector<double>(4, 0.0));
    }
    const int plates = static_cast<int>(e.system.ghost_plates.size());
    const LoadedSystem le = prepare_test_system(e.system);
    const SystemBatchHost b = make_system_batch(le, 1);
    EXPECT_EQ(b.ghost_plates(), plates);
    EXPECT_EQ(make_ghost_penalty(le.system).size(), plates * 4);
    const VariableSet set(le.system, b.topology());
    int coeffs = 0;
    for (int k = 0; k < set.size(); ++k) {
        coeffs += set[k].type == VariableType::GhostCoefficient ? 1 : 0;
    }
    EXPECT_EQ(coeffs, plates * 4);
    const double before = system_merit(d);
    const double after = system_merit(e.system);
    std::printf("Merit vor/nach der Erweiterung: %.15e / %.15e\n", before, after);
    EXPECT_LE(relative_change(before, after), kInsertionMeritRelative);

    const GhostDictionaryExtension again = extend_ghost_dictionaries(e.system, l.catalog, o);
    EXPECT_EQ(again.added_plates, 0);
    EXPECT_EQ(again.system, e.system);
    for (const GhostDictionaryGlass& g : again.glasses) {
        EXPECT_TRUE(g.present) << g.name;
    }

    o.count = 0;
    EXPECT_EQ(extend_ghost_dictionaries(d, l.catalog, o).system, d);
    o.count = 2;
    const LoadedSystem plain = load_residual_system("einzellinse_achromat_start.json");
    const GhostDictionaryExtension none = extend_ghost_dictionaries(plain.system, plain.catalog, o);
    EXPECT_EQ(none.system, plain.system);
    EXPECT_EQ(none.added_plates, 0);
}

// Farbe der Brechkraft δφ = φ(λ_kurz) − φ(λ_lang) aus dem paraxialen Trace (mit Platten).
double power_color(const SystemDescription& d, const GlassCatalog& catalog) {
    return compute_paraxial(d, catalog, d.spectrum.wavelengths_nm.front()).power -
           compute_paraxial(d, catalog, d.spectrum.wavelengths_nm.back()).power;
}

// a₁ einer Platte (nur Ordnung 1: t₂ = 2 a₁/r_g², φ_G = −4 (n_G − 1) a₁/r_g², numerik.md §6) mit
// der Farbe δφ_G = target.
double plate_a1_for_color(const MaterialSpec& glass, const SystemDescription& d,
                          const GlassCatalog& catalog, double target) {
    const double dn = material_index(glass, d.spectrum.wavelengths_nm.front(), catalog) -
                      material_index(glass, d.spectrum.wavelengths_nm.back(), catalog);
    return -target * kNormRadius * kNormRadius / (4.0 * dn);
}

// δφ der dünnen Linse (n_kurz − n_lang)(c₁ − c₂).
double lens_color(const SystemDescription& d, const GlassCatalog& catalog) {
    const MaterialSpec& m = d.surfaces[1].material;
    return (material_index(m, d.spectrum.wavelengths_nm.front(), catalog) -
            material_index(m, d.spectrum.wavelengths_nm.back(), catalog)) *
           (d.surfaces[1].curvature - d.surfaces[2].curvature);
}

GhostDictionaryExtension extend_with_flint(const LoadedSystem& l) {
    GhostDictionaryGlassOptions o;
    o.names = {kFlint};
    GhostDictionaryExtension e = extend_ghost_dictionaries(l.system, l.catalog, o);
    EXPECT_EQ(e.glasses.size(), 1U);
    EXPECT_EQ(e.glasses.at(0).glass_class, GhostGlassClass::Flint);
    EXPECT_EQ(e.added_plates, 1);
    return e;
}

// Orakel dünner Achromat: Die Flint-Platte des erweiterten Wörterbuchs mit δφ_G = −δφ_L (im
// kollimierten Bündel gleiche Randstrahlhöhe wie an der Linse) macht die Brechkraft der
// Einzellinse bis auf die Dicken- und Abstandsterme farbfrei: |δφ| sinkt um mehr als den
// Faktor 10. Die Platte zerstreut (a₁ > 0, φ_G < 0).
TEST(GhostDictionary, FlintPlateCorrectsColorOfSinglet) {
    const LoadedSystem l = singlet_with_dictionary();
    const GhostDictionaryExtension e = extend_with_flint(l);
    SystemDescription d = e.system;
    const std::size_t f = plate_of(d, kFlint);
    const double before = power_color(d, l.catalog);
    d.ghost_plates[f].coefficients[0] =
        plate_a1_for_color(d.ghost_plates[f].glass, d, l.catalog, -lens_color(d, l.catalog));
    const double a1 = d.ghost_plates[f].coefficients[0];
    const double after = power_color(d, l.catalog);
    std::printf("Flint-Platte %s, a1 = %.4e mm: delta phi %.4e -> %.4e 1/mm, phi(ref) %.4e\n",
                kFlint.c_str(), a1, before, after, compute_paraxial(d, l.catalog).power);
    EXPECT_GT(a1, 0.0);
    EXPECT_LT(compute_paraxial(d, l.catalog).power, compute_paraxial(e.system, l.catalog).power);
    EXPECT_LT(std::abs(after), kAchromatColorFactor * std::abs(before));
}

GhostDiagnosis diagnose_at(const SystemDescription& d) {
    const LoadedSystem l = prepare_test_system(d);
    const SystemBatchHost b = make_system_batch(l, 1);
    return diagnose_ghost_plates(l.system, l.catalog, b, 0, make_ghost_penalty(l.system), {});
}

// Diagnose und Operatoren mit dem Zusatzglas: (a) nur die Flint-Platte aktiv → Operator A mit
// dem Flint als Glas des Meniskus (Flint-Element); (b) CaF₂ sammelnd und Flint zerstreuend
// (Brechkraft bei λ_ref zusammen 0, Farbe −δφ_L) → Operator B an der Linse mit dem Flint als
// Zielglas; das Kittglied aus Linsenglas und Zielglas korrigiert mit den Krümmungen des dünnen
// Achromats die Längsfarbe der Einzellinse um mehr als den Faktor 10 (Orakel paraxiale
// Schnittweite).
TEST(GhostDictionary, FlintPlateLeadsToFlintElementOrDoublet) {
    const LoadedSystem l = singlet_with_dictionary();
    const GhostDictionaryExtension e = extend_with_flint(l);
    const std::size_t fi = plate_of(e.system, kFlint);
    const std::size_t ci = plate_of(e.system, "CAF2");
    const double color = -lens_color(e.system, l.catalog);

    // (a) Flint allein
    SystemDescription a = e.system;
    a.ghost_plates[fi].coefficients[0] =
        plate_a1_for_color(a.ghost_plates[fi].glass, a, l.catalog, color);
    const GhostDiagnosis da = diagnose_at(a);
    ASSERT_EQ(da.recommendation, GhostOperator::A) << da.recommendation_text;
    const MeniscusPlan mp = meniscus_from_diagnosis(a, da);
    EXPECT_EQ(mp.insertion.glass.kind, MaterialKind::Catalog);
    EXPECT_EQ(mp.insertion.glass.name, kFlint);
    EXPECT_EQ(mp.plate, static_cast<int>(fi));

    // (b) CaF₂ + Flint: Σ (n_g,ref − 1) a_g = 0 und −4/r_g² Σ δn_g a_g = color
    SystemDescription b = e.system;
    const double ref = b.spectrum.reference_nm;
    const double s = b.spectrum.wavelengths_nm.front();
    const double lg = b.spectrum.wavelengths_nm.back();
    const auto idx = [&](std::size_t g, double lam) {
        return material_index(b.ghost_plates[g].glass, lam, l.catalog);
    };
    const double kc = idx(ci, ref) - 1.0;
    const double kf = idx(fi, ref) - 1.0;
    const double dc = idx(ci, s) - idx(ci, lg);
    const double df = idx(fi, s) - idx(fi, lg);
    const double a_f = -color * kNormRadius * kNormRadius / (4.0 * (df - dc * kf / kc));
    const double a_c = -a_f * kf / kc;
    b.ghost_plates[fi].coefficients[0] = a_f;
    b.ghost_plates[ci].coefficients[0] = a_c;
    EXPECT_LT(std::abs(power_color(b, l.catalog)), kAchromatColorFactor *
                                                       std::abs(power_color(e.system, l.catalog)));
    const GhostDiagnosis db = diagnose_at(b);
    ASSERT_EQ(db.recommendation, GhostOperator::B) << db.recommendation_text;
    SplitInitOptions so;
    so.start_at_target = true;
    const SplitPlan sp = split_from_diagnosis(b, l.catalog, db, so);
    EXPECT_EQ(sp.split.lens_surface, 1);
    EXPECT_EQ(sp.target_glass.kind, MaterialKind::Catalog);
    EXPECT_EQ(sp.target_glass.name, kFlint);
    EXPECT_EQ(sp.plate, static_cast<int>(fi));

    // Kittglied aus Linsenglas und Zielglas ohne aktive Platten. Orakel dünner Achromat mit den
    // relativen Dispersionen V_i = (n_i,ref − 1)/(n_i,kurz − n_i,lang) der beiden Teile:
    // φ₁ = φ V₁/(V₁ − V₂), φ₂ = −φ V₂/(V₁ − V₂); c₁ bleibt, c_s und c₂ folgen daraus. Die
    // Längsfarbe sinkt um mehr als den Faktor 10.
    InsertionResult ins = split_lens(e.system, sp.split);
    free_glass(ins.system, sp.free_surface, sp.target_glass);
    ASSERT_EQ(sp.free_surface, 2);
    std::vector<Surface>& sf = ins.system.surfaces;
    const auto v_of = [&](const MaterialSpec& m) {
        return (material_index(m, ref, l.catalog) - 1.0) /
               (material_index(m, s, l.catalog) - material_index(m, lg, l.catalog));
    };
    const double n1 = material_index(sf[1].material, ref, l.catalog);
    const double n2 = material_index(sf[2].material, ref, l.catalog);
    const double phi = (n1 - 1.0) * (e.system.surfaces[1].curvature - e.system.surfaces[2].curvature);
    const double v1 = v_of(sf[1].material);
    const double v2 = v_of(sf[2].material);
    sf[2].curvature = sf[1].curvature - phi * v1 / (v1 - v2) / (n1 - 1.0);
    sf[3].curvature = sf[2].curvature + phi * v2 / (v1 - v2) / (n2 - 1.0);
    const double c_lens = color_focal_shift(e.system, l.catalog);
    const double c_doublet = color_focal_shift(ins.system, l.catalog);
    std::printf("Kittglied N-BK7 + %s (c_s der Diagnose %.5f/mm, Achromat %.5f/mm): Längsfarbe "
                "%.4e -> %.4e mm, f %.3f -> %.3f mm\n",
                kFlint.c_str(), sp.split.curvature, sf[2].curvature, c_lens, c_doublet,
                compute_paraxial(e.system, l.catalog).focal_length,
                compute_paraxial(ins.system, l.catalog).focal_length);
    EXPECT_LT(std::abs(c_doublet), kAchromatColorFactor * std::abs(c_lens));
}

} // namespace
} // namespace mko::test
