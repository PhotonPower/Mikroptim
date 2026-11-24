// Farbstufe der Synthese (M6-T11, docs/optimierung.md, Abschnitt 7 „Farbstufe“): Operator B und
// Farbregel mit Rang nach dem spektralen Rest und den Startradien des Glieds, Farbbedarf aus den
// chromatischen Seidel-Termen (Kriterium d) und das Kittglied-Modul der Farbregel. Orakel:
// Glasvorauswahl direkt (M11-T5g), thin_member_shape, paraxiale Längsfarbe des dünnen Achromats,
// Merit-Invarianz des Null-Kittglieds, Einfallswinkel des starken Kittglieds.

#include "insertion_optimization.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/glass/glass_properties.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/optim/design_operators.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/synth/optical_homotopy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

using ColorModule = OpticalHomotopyOptions::ColorModule;

LoadedSystem achromat_start() {
    return load_residual_system("einzellinse_achromat_start.json");
}

const std::vector<double> kFullSpectrum{435.83, 486.13, 546.07, 587.56, 656.27};

int glass_index(const std::vector<GlassDispersion>& g, const std::string& name) {
    const auto it =
        std::find_if(g.begin(), g.end(), [&](const GlassDispersion& d) { return d.name == name; });
    return it == g.end() ? -1 : static_cast<int>(it - g.begin());
}

// Operator B der Farbregel: mit Spektrum Partner und Anteile wie die Glasvorauswahl mit festem
// Linsenglas und Rang nach dem spektralen Rest; ohne Spektrum wie bisher (Koeffizient).
TEST(ColorStage, SplitForColorRanksBySpectralResidual) {
    const LoadedSystem start = achromat_start();
    const std::vector<GlassDispersion> g = catalog_dispersions(start.catalog);
    const int bk7 = glass_index(g, "N-BK7");
    ASSERT_GE(bk7, 0);
    SplitInitOptions o;
    GlassPreselectionOptions po;
    po.fixed = {bk7};
    po.max_power_sum = o.preselect_max_power_sum;
    po.max_results = 1;

    const SplitPlan coef = split_for_color(start.system, start.catalog, 1, o);
    const std::vector<PreselectedGlasses> by_coef = preselect_glasses(g, po);
    ASSERT_EQ(by_coef.size(), 1U);
    EXPECT_EQ(coef.target_glass.name, g[static_cast<std::size_t>(by_coef[0].glasses[1])].name);
    EXPECT_TRUE(std::isnan(coef.spectral_residual));

    o.preselect_wavelengths_nm = kFullSpectrum;
    po.wavelengths_nm = kFullSpectrum;
    const SplitPlan spec = split_for_color(start.system, start.catalog, 1, o);
    const std::vector<PreselectedGlasses> by_spec = preselect_glasses(g, po);
    ASSERT_EQ(by_spec.size(), 1U);
    ASSERT_TRUE(spec.preselected);
    EXPECT_EQ(spec.target_glass.name, g[static_cast<std::size_t>(by_spec[0].glasses[1])].name);
    EXPECT_EQ(spec.spectral_residual, by_spec[0].spectral_residual);
    EXPECT_EQ(spec.front_power_fraction, by_spec[0].powers[0]);
    EXPECT_NE(spec.reason.find("spektraler Rest"), std::string::npos) << spec.reason;
    std::printf("Partner nach Koeffizient: %s, nach spektralem Rest: %s (Rest %.3e)\n",
                coef.target_glass.name.c_str(), spec.target_glass.name.c_str(),
                spec.spectral_residual);
}

// Startradien des Glieds als zusätzlicher Zweig von Operator B: Krümmungen gleich
// thin_member_shape mit den Anteilen der Vorauswahl bei gleicher dünner Brechkraft, und (Orakel)
// der dünne Achromat korrigiert die paraxiale Längsfarbe der Einzellinse um mehr als den
// Faktor 10 schon beim Einfügen, ohne Optimierung.
TEST(ColorStage, MemberStartIsThinAchromatAndCorrectsLongitudinalColor) {
    const LoadedSystem start = achromat_start();
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.split.member_start = true;
    o.split.preselect_wavelengths_nm = start.system.spectrum.wavelengths_nm;
    OpticalHomotopyModel m(start, o);
    const HomotopyPoint p = m.start_point();
    const HomotopyNeed need = m.diagnose(m.start_system(), p);
    ASSERT_EQ(need.recommendation, GhostOperator::B) << need.text;
    ASSERT_EQ(need.split_lens, 1);
    const HomotopyInsertion ins = m.insert(m.start_system(), p, need);
    ASSERT_EQ(ins.branches.size(), 2U) << ins.text;
    EXPECT_NE(ins.text.find("Startradien des Glieds"), std::string::npos) << ins.text;

    // Orakel der Krümmungen
    const SystemDescription realized = m.realize(m.start_system(), p);
    const SplitPlan plan = split_for_color(realized, start.catalog, 1, o.split);
    const double lref = realized.spectrum.reference_nm;
    const double n_lens = material_index(start.system.surfaces[1].material, lref, start.catalog);
    const double n_target = material_index(plan.target_glass, lref, start.catalog);
    const double phi =
        (n_lens - 1.0) * (start.system.surfaces[1].curvature - start.system.surfaces[2].curvature);
    const double k1 = plan.front_power_fraction * phi;
    const std::array<double, 2> n{n_lens, n_target};
    const std::array<double, 2> k{k1, phi - k1};
    const ThinMemberShape shape = thin_member_shape(n, k);
    const SystemDescription& member = ins.branches[1];
    ASSERT_EQ(member.surfaces.size(), 4U);
    for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(member.surfaces[static_cast<std::size_t>(1 + j)].curvature,
                    shape.curvatures[static_cast<std::size_t>(j)], 1e-15)
            << "Fläche " << 1 + j;
    }
    // dünne Brechkraft erhalten
    const double k_member =
        (n_lens - 1.0) * (member.surfaces[1].curvature - member.surfaces[2].curvature) +
        (n_target - 1.0) * (member.surfaces[2].curvature - member.surfaces[3].curvature);
    EXPECT_NEAR(k_member, phi, 1e-15);

    // Orakel: paraxiale Längsfarbe (Schnittweite 480 gegen 644 nm)
    const double c_lens = color_focal_shift(start.system, start.catalog);
    const double c_member = color_focal_shift(member, start.catalog);
    const double c_plain = color_focal_shift(ins.branches[0], start.catalog);
    std::printf("Längsfarbe [mm]: Einzellinse %.4e, Zweig wie bisher %.4e, Startradien %.4e\n",
                c_lens, c_plain, c_member);
    EXPECT_LT(std::abs(c_member), kAchromatColorFactor * std::abs(c_lens));

    // ohne Option: ein Zweig wie bisher, gleiches System wie Zweig 0
    OpticalHomotopyOptions plain = o;
    plain.split.member_start = false;
    OpticalHomotopyModel mp(start, plain);
    const HomotopyInsertion ip = mp.insert(mp.start_system(), p, need);
    ASSERT_EQ(ip.branches.size(), 1U);
    EXPECT_EQ(ip.branches[0], ins.branches[0]);
}

// Farbbedarf aus den chromatischen Seidel-Termen (Kriterium d): Schwelle exakt bei σ_col /
// größter RMS (Orakel: σ_col aus dem Aberrationsbericht nach der Formel in
// optical_homotopy.hpp, RMS aus evaluate_wavefront); (a) und (b) abgeschaltet.
TEST(ColorStage, ChromaticCriterionThreshold) {
    const LoadedSystem start = achromat_start();
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.color_ratio = 1e9;
    o.color_seidel = 1e9;
    OpticalHomotopyModel off(start, o);
    const HomotopyPoint p = off.start_point();
    EXPECT_EQ(off.diagnose(off.start_system(), p).recommendation, GhostOperator::None);

    const SystemDescription realized = off.realize(off.start_system(), p);
    AberrationOptions ao;
    ao.real_rays = false;
    const AberrationReport rep = compute_aberrations(realized, start.catalog, ao);
    double zmin = 0.0;
    double zmax = 0.0;
    for (const FocalShift& f : rep.focal_shifts) {
        if (f.nm >= 480.0 - 0.01 && f.nm <= 644.0 + 0.01) {
            zmin = std::min(zmin, f.shift);
            zmax = std::max(zmax, f.shift);
        }
    }
    const double wl = (zmax - zmin) * p.aperture * p.aperture / (2.0 * 546.07e-6);
    const SeidelWaves w = rep.waves();
    const double sigma =
        std::sqrt(std::pow(wl / (4.0 * std::sqrt(3.0)), 2) + std::pow(w.dW111 / 4.0, 2) +
                  std::pow(w.dW040 / (2.0 * std::sqrt(180.0)), 2));
    const WavefrontResult wf = evaluate_wavefront(off.load(realized));
    double worst = 0.0;
    for (int f = 0; f < wf.fields; ++f) {
        for (int kk = 0; kk < wf.wavelengths; ++kk) {
            worst = std::max(worst, wf.at(f, kk).rms_waves);
        }
    }
    ASSERT_GT(sigma, 0.0);
    const double ratio = sigma / worst;
    std::printf("sigma_col %.4f Wellen, größter RMS %.4f Wellen, Verhältnis %.4f\n", sigma, worst,
                ratio);
    OpticalHomotopyOptions below = o;
    below.color_chromatic_share = ratio * (1.0 - 1e-6);
    OpticalHomotopyModel mb(start, below);
    const HomotopyNeed nb = mb.diagnose(mb.start_system(), p);
    EXPECT_EQ(nb.recommendation, GhostOperator::B) << nb.text;
    EXPECT_NE(nb.text.find("chromatischer Anteil"), std::string::npos) << nb.text;
    EXPECT_NE(nb.text.find("sigma_col"), std::string::npos) << nb.text;
    OpticalHomotopyOptions above = o;
    above.color_chromatic_share = ratio * (1.0 + 1e-6);
    OpticalHomotopyModel ma(start, above);
    EXPECT_EQ(ma.diagnose(ma.start_system(), p).recommendation, GhostOperator::None);
}

// Kittglied-Modul (Null-Kittglied): mit color_module_always statt Teilen im einzigen inneren
// Luftspalt (Blende → Linse), Gläser = bestes Paar der Vorauswahl über den Katalog, Merit
// unverändert, innere Homotopie über beide Dicken, Dickenvariable beider Teile; Teilen bleibt
// Variante.
TEST(ColorStage, NullDoubletModuleKeepsMeritAndGrows) {
    const LoadedSystem start = achromat_start();
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.color_module = ColorModule::Doublet;
    o.color_module_always = true;
    o.split.preselect_wavelengths_nm = start.system.spectrum.wavelengths_nm;
    OpticalHomotopyModel m(start, o);
    const HomotopyPoint p = m.start_point();
    const HomotopyNeed need = m.diagnose(m.start_system(), p);
    ASSERT_EQ(need.recommendation, GhostOperator::B) << need.text;
    EXPECT_EQ(need.module_gap, 0);
    EXPECT_EQ(need.split_lens, -1);
    EXPECT_NE(need.text.find("Null-Kittglied"), std::string::npos) << need.text;
    const HomotopyInsertion ins = m.insert(m.start_system(), p, need);
    ASSERT_TRUE(ins.doublet);
    ASSERT_EQ(ins.branches.size(), 1U);
    EXPECT_EQ(ins.new_surfaces, 3);
    EXPECT_EQ(ins.first_surface, 1);
    EXPECT_EQ(ins.stages.size(), static_cast<std::size_t>(o.element_stages));
    const SystemDescription& sys = ins.branches[0];
    ASSERT_EQ(sys.surfaces.size(), m.start_system().surfaces.size() + 3);

    // Orakel Gläser: bestes Paar der Vorauswahl (spektraler Rest über 480/644 nm)
    const std::vector<GlassDispersion> g = catalog_dispersions(start.catalog);
    GlassPreselectionOptions po;
    po.max_power_sum = o.split.preselect_max_power_sum;
    po.max_results = 1;
    po.wavelengths_nm = o.split.preselect_wavelengths_nm;
    const std::vector<PreselectedGlasses> best = preselect_glasses(g, po);
    ASSERT_EQ(best.size(), 1U);
    EXPECT_EQ(sys.surfaces[1].material.name, g[static_cast<std::size_t>(best[0].glasses[0])].name);
    EXPECT_NE(ins.text.find(g[static_cast<std::size_t>(best[0].glasses[1])].name),
              std::string::npos)
        << ins.text;
    // beide Gläser fest aus der Vorauswahl (keine Glasvariable bei Dicke 0)
    EXPECT_EQ(sys.surfaces[2].material.kind, MaterialKind::Catalog);
    EXPECT_EQ(sys.surfaces[2].material.name, g[static_cast<std::size_t>(best[0].glasses[1])].name);

    // Merit unverändert (Dicke 0, drei Flächen gleicher Krümmung)
    const double before = system_merit(m.start_system());
    const double after = system_merit(sys);
    const double rel = relative_change(before, after);
    std::printf("Null-Kittglied: F %.12e -> %.12e, relativ %.2e\n", before, after, rel);
    EXPECT_LE(rel, kInsertionMeritRelative);

    // innere Homotopie: beide Teile Dicke t, Spalt dahinter um 2t kürzer
    SystemDescription staged = sys;
    const double te = o.element_thickness; // letzte Stufe der inneren Homotopie
    m.set_stage(staged, ins, te);
    EXPECT_EQ(staged.surfaces[1].thickness, te);
    EXPECT_EQ(staged.surfaces[2].thickness, te);
    EXPECT_DOUBLE_EQ(staged.surfaces[3].thickness, std::max(ins.gap - 2.0 * te, ins.min_gap));
    m.finish_insertion(staged, ins);
    bool both = false;
    for (const VariableSpec& v : staged.variables->items) {
        if (v.type == VariableType::Thickness && v.surfaces == std::vector<int>{1, 2}) {
            both = true;
            EXPECT_EQ(v.min, o.element_thickness);
            EXPECT_EQ(v.max, o.element_max_thickness);
        }
    }
    EXPECT_TRUE(both);

    // Varianten: weitere Spalte (keine) und Teilen der Linse
    const std::vector<HomotopyNeed> variants = m.insertion_options(m.start_system(), p, need, 4);
    ASSERT_EQ(variants.size(), 2U);
    EXPECT_EQ(variants[1].module_gap, -1);
    EXPECT_EQ(variants[1].split_lens, 1);
}

// Modul nur, wenn keine Linse teilbar ist (ohne color_module_always): Linse zu dünn für zwei Teile
// (untere Schranke der Dicke) → ohne Modul kein Bedarf, mit Modul Null-Kittglied; teilbare
// Linse → Operator B wie bisher, das Modul nur als Variante. Ohne Modul-Option bleibt die
// Empfehlung bitgleich zur bisherigen.
TEST(ColorStage, ModuleOnlyWhenNoLensCanBeSplit) {
    LoadedSystem thin = achromat_start();
    VariableSpec t;
    t.type = VariableType::Thickness;
    t.selection = SurfaceSelection::One;
    t.surfaces = {1};
    t.min = 3.0; // Dicke 4 < 2 · 3
    t.max = 10.0;
    thin.system.variables->items.push_back(t);
    OpticalHomotopyOptions o;
    o.color_rule = true;
    OpticalHomotopyModel plain(thin, o);
    EXPECT_EQ(plain.diagnose(plain.start_system(), plain.start_point()).recommendation,
              GhostOperator::None);
    OpticalHomotopyOptions om = o;
    om.color_module = ColorModule::Doublet;
    OpticalHomotopyModel mm(thin, om);
    const HomotopyNeed nm = mm.diagnose(mm.start_system(), mm.start_point());
    ASSERT_EQ(nm.recommendation, GhostOperator::B) << nm.text;
    EXPECT_EQ(nm.module_gap, 0);
    EXPECT_NE(nm.text.find("keine Linse teilbar"), std::string::npos) << nm.text;
    // Spalt mit Dickenschranke über der halben Dicke: beide Teilstücke wären zu dünn, kein Modul
    // (vorher Abbruch des Laders im Lauf F7)
    LoadedSystem bounded = thin;
    VariableSpec g;
    g.type = VariableType::Thickness;
    g.selection = SurfaceSelection::One;
    g.surfaces = {0};
    g.min = 1.5; // Spalt 2 mm, Hälfte 1 mm
    g.max = 10.0;
    bounded.system.variables->items.push_back(g);
    OpticalHomotopyModel mbnd(bounded, om);
    EXPECT_EQ(mbnd.diagnose(mbnd.start_system(), mbnd.start_point()).recommendation,
              GhostOperator::None);

    const LoadedSystem start = achromat_start();
    OpticalHomotopyModel a(start, o);
    OpticalHomotopyModel b(start, om);
    const HomotopyNeed na = a.diagnose(a.start_system(), a.start_point());
    const HomotopyNeed nb = b.diagnose(b.start_system(), b.start_point());
    EXPECT_EQ(nb.split_lens, na.split_lens);
    EXPECT_EQ(nb.module_gap, -1);
    EXPECT_EQ(nb.text, na.text);
    EXPECT_TRUE(na.module_candidates.empty());
    EXPECT_EQ(nb.module_candidates, std::vector<int>{0});
    const std::vector<HomotopyNeed> v =
        b.insertion_options(b.start_system(), b.start_point(), nb, 4);
    ASSERT_EQ(v.size(), 2U);
    EXPECT_EQ(v[0].split_lens, 1);
    EXPECT_EQ(v[1].module_gap, 0);
    EXPECT_EQ(a.insertion_options(a.start_system(), a.start_point(), na, 4).size(), 1U);
}

// Glasregel anomalous_pgf: mindestens ein Glas des Moduls mit |ΔP_g,F| ≥ 0,01 (Kennzahlen aus
// dem Katalog, glass_properties).
TEST(ColorStage, AnomalousModuleGlass) {
    const LoadedSystem start = achromat_start();
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.color_module = ColorModule::Doublet;
    o.color_module_always = true;
    o.color_module_anomalous = true;
    OpticalHomotopyModel m(start, o);
    const HomotopyNeed need = m.diagnose(m.start_system(), m.start_point());
    ASSERT_EQ(need.module_gap, 0) << need.text;
    const HomotopyInsertion ins = m.insert(m.start_system(), m.start_point(), need);
    const std::string front = ins.branches[0].surfaces[1].material.name;
    // Glas des hinteren Teils: Name aus dem Text „Gläser A + B (...)“
    const std::size_t at = ins.text.find("Gläser ") + std::string("Gläser ").size();
    const std::size_t plus = ins.text.find(" + ", at);
    ASSERT_NE(plus, std::string::npos) << ins.text;
    EXPECT_EQ(ins.text.substr(at, plus - at), front);
    const std::size_t end = ins.text.find(" (", plus);
    const std::string back = ins.text.substr(plus + 3, end - plus - 3);
    const double da = std::abs(glass_properties(start.catalog.at(front)).dp_gF);
    const double db = std::abs(glass_properties(start.catalog.at(back)).dp_gF);
    std::printf("anomales Modul: %s (|dPgF| %.4f) + %s (%.4f)\n", front.c_str(), da, back.c_str(),
                db);
    EXPECT_GE(std::max(da, db), 0.01);
    EXPECT_NE(ins.text.find("anomalous_pgf"), std::string::npos);
}

// Starkes Kittglied: Krümmung so, dass der paraxiale Randstrahl in der Spaltmitte die Fläche
// unter 40° trifft (Orakel: Einfallswinkel aus Strahlhöhe und Steigung des Aberrationsberichts,
// unabhängig gerechnet); Merit unverändert. Paarregel merte_pair.
TEST(ColorStage, StrongDoubletIncidence) {
    const LoadedSystem start = achromat_start();
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.color_module = ColorModule::StrongDoublet;
    o.color_module_always = true;
    o.meniscus.curvature_variables.min = -1.2;
    o.meniscus.curvature_variables.max = 1.2;
    // 40° braucht bei y = 2,5 mm |c| = sin 40°/2,5 = 0,26/mm > 0,9/6 (Halbkugelgrad der Linse,
    // Halbdurchmesser 6 mm): nicht erreichbar, kein Modul
    OpticalHomotopyModel m40(start, o);
    EXPECT_EQ(m40.diagnose(m40.start_system(), m40.start_point()).module_gap, -1);
    o.color_module_incidence_deg = 20.0;
    OpticalHomotopyModel m(start, o);
    const HomotopyPoint p = m.start_point();
    const HomotopyNeed need = m.diagnose(m.start_system(), p);
    ASSERT_EQ(need.module_gap, 0) << need.text;
    const HomotopyInsertion ins = m.insert(m.start_system(), p, need);
    const SystemDescription& sys = ins.branches[0];
    const double c = sys.surfaces[1].curvature;
    EXPECT_EQ(sys.surfaces[2].curvature, c);
    EXPECT_EQ(sys.surfaces[3].curvature, c);
    // Orakel: Randstrahl nach der Blende (kollimiert: Steigung 0, Höhe = Pupillenradius) in der
    // Spaltmitte; Einfallswinkel an der Kugel mit Scheitel dort, exakt (Schnittpunkt)
    const SystemDescription realized = m.realize(m.start_system(), p);
    AberrationOptions ao;
    ao.real_rays = false;
    ao.focal_shift = false;
    const AberrationReport rep = compute_aberrations(realized, start.catalog, ao);
    const double u = rep.surfaces[0].u_after;
    const double y = rep.surfaces[0].y + 0.5 * realized.surfaces[0].thickness * u;
    // Strahl parallel zur Achse (u = 0): sin I = |y c|
    ASSERT_NEAR(u, 0.0, 1e-15);
    const double incidence = std::asin(std::abs(y * c)) * 180.0 / std::acos(-1.0);
    EXPECT_NEAR(incidence, 20.0, 1e-9);
    EXPECT_NE(ins.text.find("Einfallswinkel 20"), std::string::npos) << ins.text;
    EXPECT_NE(ins.text.find("merte_pair"), std::string::npos) << ins.text;
    const double rel = relative_change(system_merit(m.start_system()), system_merit(sys));
    EXPECT_LE(rel, kInsertionMeritRelative);
}

// Dreilinser N-BK7 (frei erfunden, Kataloge wie die Einzellinse, 480/644 nm): Linsen ab den
// Flächen 1, 3 und 5; die mittlere hat eine Dickenschranke über der halben Dicke und lässt sich
// nicht teilen. Mit `plates` das Wörterbuch der Zielspezifikation (SYN-36, CaF₂, SYN-30, K = 4)
// 1 mm hinter der Blende, die SYN-36-Platte mit einem Koeffizienten vierter Ordnung (Bedarf an
// monochromatischer Korrektur für die Geisterdiagnose).
LoadedSystem three_lenses(bool plates) {
    SystemDescription d = achromat_start().system;
    Surface lens = d.surfaces[1];
    Surface back = d.surfaces[2];
    back.thickness = 2.0;
    std::vector<Surface> s{d.surfaces[0]};
    for (const double c : {0.0193, 0.012, 0.008}) {
        lens.curvature = c;
        back.curvature = -c;
        s.push_back(lens);
        s.push_back(back);
    }
    s.back().thickness = 30.0;
    d.surfaces = s;
    VariableSpec t;
    t.type = VariableType::Thickness;
    t.selection = SurfaceSelection::One;
    t.surfaces = {3};
    t.min = 3.0; // Dicke 4 < 2 · 3: nicht teilbar
    t.max = 10.0;
    d.variables->items.push_back(t);
    if (plates) {
        GhostDictionaryPlacement p;
        p.after_surface = 0;
        p.distance = 1.0;
        p.glasses = {catalog_glass("SYN-36"), catalog_glass("CAF2"), catalog_glass("SYN-30")};
        p.order = 4;
        p.norm_radius = 2.5;
        p.penalty_weight = 47.2;
        d = place_ghost_dictionary(d, p).system;
        d.ghost_plates[0].coefficients[1] = 2e-3;
    }
    return prepare_test_system(d);
}

// Varianten der Farbregel (M6-T13, color_variants). Orakel: die Fläche, an der eine Variante
// tatsächlich einfügt (insert). Ohne Option fällt die Variante „Linse 3“ (nicht teilbar) auf die
// Empfehlung zurück – dieselbe Einfügung zweimal (Befund der Läufe F3, M6-T11). Mit der Option
// nur teilbare Linsen ohne Rückfall, jede Variante an einer anderen Stelle, dann die Spalte des
// Moduls und die Empfehlung der Geisterdiagnose (= diagnose ohne Farbregel). Ohne die Option
// bleiben die Varianten wie bisher.
TEST(ColorStage, ColorVariantsAreDistinctInsertions) {
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.color_module = ColorModule::Doublet;
    const LoadedSystem l = three_lenses(false);
    OpticalHomotopyModel m(l, o);
    const HomotopyPoint p = m.start_point();
    const HomotopyNeed need = m.diagnose(m.start_system(), p);
    ASSERT_EQ(need.recommendation, GhostOperator::B) << need.text;
    ASSERT_GE(need.split_lens, 0) << need.text;
    ASSERT_NE(std::find(need.split_candidates.begin(), need.split_candidates.end(), 3),
              need.split_candidates.end());
    const auto where = [&](OpticalHomotopyModel& model, const HomotopyNeed& n) {
        const HomotopyInsertion ins = model.insert(model.start_system(), p, n);
        return std::to_string(ins.doublet ? 1 : 0) + ":" + std::to_string(ins.first_surface);
    };
    // ohne Option: Variante „Linse 3“ fügt an der empfohlenen Linse ein
    const std::vector<HomotopyNeed> old = m.insertion_options(m.start_system(), p, need, 8);
    const std::string rec = where(m, need);
    int duplicates = 0;
    for (std::size_t k = 1; k < old.size(); ++k) {
        duplicates += where(m, old[k]) == rec ? 1 : 0;
    }
    EXPECT_GE(duplicates, 1);

    OpticalHomotopyOptions ov = o;
    ov.color_variants = true;
    OpticalHomotopyModel mv(l, ov);
    const HomotopyNeed nv = mv.diagnose(mv.start_system(), p);
    EXPECT_EQ(nv.text, need.text); // die Empfehlung selbst bleibt
    const std::vector<HomotopyNeed> v = mv.insertion_options(mv.start_system(), p, nv, 8);
    std::vector<std::string> places;
    int lenses = 0;
    int modules = 0;
    for (const HomotopyNeed& n : v) {
        places.push_back(where(mv, n));
        std::printf("Variante %s: %s\n", places.back().c_str(), n.text.c_str());
        EXPECT_NE(n.split_lens, 3);
        lenses += n.split_lens >= 0 ? 1 : 0;
        modules += n.module_gap >= 0 ? 1 : 0;
    }
    EXPECT_EQ(lenses, 2);                        // Linsen 1 und 5
    EXPECT_EQ(modules, static_cast<int>(nv.module_candidates.size()));
    std::vector<std::string> sorted = places;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::unique(sorted.begin(), sorted.end()), sorted.end()) << "doppelte Einfügung";
    // höchstens max
    EXPECT_EQ(mv.insertion_options(mv.start_system(), p, nv, 2).size(), 2U);
    // Empfehlung einer am Zustand nicht teilbaren Linse (am verworfenen Versuch entstanden):
    // insert bricht ab (Orakel); als Variante entfällt sie, die übrigen bleiben
    HomotopyNeed thin = nv;
    thin.split_lens = 3;
    thin.split_candidates = {3};
    EXPECT_THROW((void)mv.insert(mv.start_system(), p, thin), std::invalid_argument);
    const std::vector<HomotopyNeed> vt = mv.insertion_options(mv.start_system(), p, thin, 8);
    EXPECT_TRUE(vt.empty() || vt.front().split_lens != 3);
    for (const HomotopyNeed& n : vt) {
        EXPECT_NE(n.split_lens, 3);
    }
    EXPECT_EQ(vt.size(), static_cast<std::size_t>(nv.module_candidates.size()));

    // mit aktiver Geisterplatte: Empfehlung der Geisterdiagnose als weitere Variante (Orakel:
    // diagnose desselben Systems ohne Farbregel)
    const LoadedSystem lp = three_lenses(true);
    OpticalHomotopyModel mp(lp, ov);
    const HomotopyNeed np = mp.diagnose(mp.start_system(), p);
    ASSERT_GE(np.split_lens, 0) << np.text;
    OpticalHomotopyOptions og;
    og.color_module = ColorModule::Doublet;
    OpticalHomotopyModel mg(lp, og);
    const HomotopyNeed ghost = mg.diagnose(mg.start_system(), p);
    ASSERT_NE(ghost.recommendation, GhostOperator::None) << ghost.text;
    const std::vector<HomotopyNeed> vp = mp.insertion_options(mp.start_system(), p, np, 8);
    ASSERT_FALSE(vp.empty());
    EXPECT_EQ(vp.back().recommendation, ghost.recommendation);
    EXPECT_EQ(vp.back().text, "Variante (Geisterdiagnose): " + ghost.text);
    EXPECT_EQ(vp.back().split_lens, -1);
    EXPECT_EQ(vp.back().module_gap, -1);
}

// Querfarbe (M6-T13, color_module_lateral): Dreilinser mit Feld, Querfarbe überwiegt die
// Längsfarbe (Orakel: Seidel-Terme des Aberrationsberichts, |ΔW111|/4 > |ΔW020|/(2√3)). Dann
// empfiehlt die Farbregel das Modul zuerst, Spalte absteigend nach dem mittleren |y·ȳ| der
// Spaltflächen (Orakel: Strahldaten des Berichts); ohne Option Operator B wie bisher.
TEST(ColorStage, LateralColorModuleOrderedByChiefRayHeight) {
    SystemDescription d = three_lenses(false).system;
    d.source.field_angles_deg = {0.0, 20.0};
    d.source.field_weights = {1.0, 1.0};
    const LoadedSystem l = prepare_test_system(d);
    OpticalHomotopyOptions o;
    o.color_rule = true;
    o.color_module = ColorModule::Doublet;
    OpticalHomotopyModel plain(l, o);
    const HomotopyPoint p = plain.start_point();
    const HomotopyNeed np = plain.diagnose(plain.start_system(), p);
    ASSERT_EQ(np.recommendation, GhostOperator::B) << np.text;
    EXPECT_GE(np.split_lens, 0);
    EXPECT_EQ(np.module_gap, -1);

    const SystemDescription realized = plain.realize(plain.start_system(), p);
    AberrationOptions ao;
    ao.real_rays = false;
    ao.focal_shift = false;
    const AberrationReport rep = compute_aberrations(realized, l.catalog, ao);
    const SeidelWaves w = rep.waves();
    std::printf("dW111/4 = %.4f, dW020/(2 sqrt 3) = %.4f\n", std::abs(w.dW111) / 4.0,
                std::abs(w.dW020) / (2.0 * std::sqrt(3.0)));
    ASSERT_GT(std::abs(w.dW111) / 4.0, std::abs(w.dW020) / (2.0 * std::sqrt(3.0)));

    OpticalHomotopyOptions ol = o;
    ol.color_module_lateral = true;
    OpticalHomotopyModel lat(l, ol);
    const HomotopyNeed nl = lat.diagnose(lat.start_system(), p);
    ASSERT_GE(nl.module_gap, 0) << nl.text;
    EXPECT_EQ(nl.split_lens, -1);
    EXPECT_NE(nl.text.find("Querfarbe"), std::string::npos) << nl.text;
    // Orakel der Reihenfolge: dieselben Spalte wie ohne Option, nach |y·ȳ| absteigend
    std::vector<int> expected = np.module_candidates;
    const auto key = [&](int s) {
        const SurfaceAberration& a = rep.surfaces.at(static_cast<std::size_t>(s));
        const SurfaceAberration& b = rep.surfaces.at(static_cast<std::size_t>(s + 1));
        return 0.5 * (std::abs(a.y * a.ybar) + std::abs(b.y * b.ybar));
    };
    std::stable_sort(expected.begin(), expected.end(),
                     [&](int a, int b) { return key(a) > key(b); });
    EXPECT_EQ(nl.module_candidates, expected);
    EXPECT_EQ(nl.module_gap, expected.front());
    EXPECT_NE(np.module_candidates, expected); // Reihenfolge nach |y| ist eine andere
}

} // namespace
} // namespace mko::test
