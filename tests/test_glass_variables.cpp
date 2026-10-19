// Glas als Variable (M6-T1): Schema (Variablentyp glass, Nebenbedingung glass_hull),
// Umwandlung der Startgläser in Buchdahl-Materialien, Abbildung Vektor ↔ Batch (Brechzahlen bei
// allen Wellenlängen und λ_ref), Strafe außerhalb der Glas-Hülle mit Gradient gegen finite
// Differenzen.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/dispersion.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {
namespace {

namespace fs = std::filesystem;
const fs::path kSystems = fs::path(MKO_SOURCE_DIR) / "data" / "systems";
const std::string kGlassDir = MKO_SOURCE_DIR "/data/glass/";
constexpr double kInf = std::numeric_limits<double>::infinity();

// Basissystem (Flächen 0 … 4): Blende, Linse aus N-BK7, Luft, Platte mit konstantem n = 1,6,
// Luft; VARS und CONS werden ersetzt.
const std::string kBase = R"({
  "format": "mikroptim.system/1",
  "name": "Glasvariablentest",
  "units": "mm",
  "glass_catalogs": ["data/glass/basis.json"],
  "spectrum": { "wavelengths_nm": [486.13, 587.56, 656.27], "weights": [1.0, 1.0, 1.0], "reference_nm": 546.07 },
  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
  "target": { "focal_length": 40.0 },
  "surfaces": [
    { "type": "stop", "thickness": 2.0, "material": "AIR", "semi_diameter": 3.0 },
    { "type": "sphere", "curvature": 0.04, "thickness": 4.0, "material": "N-BK7", "semi_diameter": 6.0 },
    { "type": "sphere", "curvature": -0.01, "thickness": 1.0, "material": "AIR", "semi_diameter": 6.0 },
    { "type": "plane", "thickness": 3.0, "material": { "model": "constant", "n": 1.6 }, "semi_diameter": 6.0 },
    { "type": "plane", "thickness": 30.0, "material": "AIR", "semi_diameter": 6.0 }
  ]VARSCONS
}
)";

std::string text(const std::string& vars, const std::string& cons = "") {
    std::string s = kBase;
    s.replace(s.find("VARS"), 4,
              vars.empty() ? std::string() : ",\n  \"variables\": { \"items\": [" + vars + "] }");
    s.replace(s.find("CONS"), 4,
              cons.empty() ? std::string() : ",\n  \"constraints\": { \"items\": [" + cons + "] }");
    return s;
}

SystemDescription read(const std::string& vars, const std::string& cons = "") {
    return read_system_json(text(vars, cons));
}

void expect_format_error(const std::string& json, const std::string& path,
                         const std::string& message) {
    try {
        (void)read_system_json(json);
        FAIL() << "keine FormatError, erwartet " << path;
    } catch (const FormatError& e) {
        EXPECT_EQ(e.json_path(), path) << e.what();
        EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
    }
}

// Geladenes System nach der Umwandlung der Glasvariablen.
struct Converted {
    LoadedSystem loaded;
    std::vector<GlassConversion> conversions;
};

Converted convert(const SystemDescription& d) {
    const LoadedSystem raw = prepare_system(d, kSystems);
    SystemDescription c = raw.system;
    Converted out;
    out.conversions = convert_glass_variables(c, raw.catalog);
    out.loaded = prepare_system(c, kSystems);
    return out;
}

// ---- Schema --------------------------------------------------------------------------------

TEST(GlassVariables, SchemaResolvesThreeComponents) {
    const SystemDescription s = read(R"({ "type": "glass", "surface": 3 })");
    const std::vector<ResolvedVariable> r = resolve_variables(s);
    ASSERT_EQ(r.size(), 3u);
    const double typical[3] = {kDefaultTypicalGlassIndex, kDefaultTypicalGlassNu1,
                               kDefaultTypicalGlassNu2};
    for (int c = 0; c < 3; ++c) {
        const ResolvedVariable& v = r[static_cast<std::size_t>(c)];
        EXPECT_EQ(v.type, VariableType::Glass);
        EXPECT_EQ(v.surface, 3);
        EXPECT_EQ(v.component, c);
        EXPECT_EQ(v.lower, -kInf);
        EXPECT_EQ(v.upper, kInf);
        EXPECT_EQ(v.typical, typical[c]);
        EXPECT_EQ(default_typical(VariableType::Glass, c), typical[c]);
    }
    // konstantes Material: n₀ = n, ν = 0
    EXPECT_EQ(variable_value(s, r[0]), 1.6);
    EXPECT_EQ(variable_value(s, r[1]), 0.0);
    // Katalogglas: erst umwandeln
    const SystemDescription cat = read(R"({ "type": "glass", "surface": 1 })");
    EXPECT_THROW((void)variable_value(cat, resolve_variables(cat)[0]), std::invalid_argument);

    // "all": alle Flächen mit Glas danach (1 und 3)
    const std::vector<ResolvedVariable> all =
        resolve_variables(read(R"({ "type": "glass", "surfaces": "all" })"));
    ASSERT_EQ(all.size(), 6u);
    EXPECT_EQ(all[0].surface, 1);
    EXPECT_EQ(all[3].surface, 3);

    // Schreiben → Lesen erhält den Eintrag
    const SystemDescription back = read_system_json(write_system_json(s));
    EXPECT_EQ(back.variables, s.variables);
}

TEST(GlassVariables, SchemaNegative) {
    expect_format_error(text(R"({ "type": "glass", "surface": 2 })"), "variables.items[0].surface",
                        "Glasvariable verlangt ein Glas");
    expect_format_error(text(R"({ "type": "glass", "surface": 1, "min": 1.4 })"),
                        "variables.items[0].min", "bei glass nicht erlaubt");
    expect_format_error(text(R"({ "type": "glass", "surface": 1, "max": 1.4 })"),
                        "variables.items[0].max", "bei glass nicht erlaubt");
    expect_format_error(text(R"({ "type": "glass", "surface": 1, "typical": 0.1 })"),
                        "variables.items[0].typical", "bei glass nicht erlaubt");
    expect_format_error(
        text(R"({ "type": "glass", "surfaces": "all" }, { "type": "glass", "surface": 3 })"),
        "variables.items[1].surface", "Variable doppelt");
    expect_format_error(text(R"({ "type": "glass", "surface": 9 })"), "variables.items[0].surface",
                        "existiert nicht");
    expect_format_error(text(R"({ "type": "glass", "plate": 0 })"), "variables.items[0].plate",
                        "nur bei ghost_coefficient");
}

TEST(GlassVariables, GlassHullConstraintSchema) {
    const SystemDescription s =
        read(R"({ "type": "glass", "surface": 1 })",
             R"({ "type": "glass_hull", "weight": 500000.0, "margin": 0.01 })");
    const ConstraintSpec& c = s.constraints->items.at(0);
    EXPECT_EQ(c.type, ConstraintType::GlassHull);
    EXPECT_EQ(c.weight_or_default(), 5e5);
    EXPECT_EQ(c.margin_or_default(), 0.01);
    EXPECT_FALSE(is_paraxial(ConstraintType::GlassHull));
    EXPECT_FALSE(constraint_target(s, ConstraintType::GlassHull).has_value());
    EXPECT_FALSE(constraint_bound(s, c).has_value());
    EXPECT_EQ(to_string(ConstraintType::GlassHull), "glass_hull");
    EXPECT_EQ(constraint_type_from_string("glass_hull"), ConstraintType::GlassHull);
    // nicht paraxial: make_paraxial_constraints übergeht den Eintrag
    EXPECT_EQ(make_paraxial_constraints(s).size(), 0);
    // Roundtrip, Standardwerte
    const std::string written = write_system_json(s);
    EXPECT_EQ(read_system_json(written), s);
    EXPECT_EQ(write_system_json(read_system_json(written)), written);
    const SystemDescription d = read("", R"({ "type": "glass_hull" })");
    EXPECT_EQ(d.constraints->items[0].weight_or_default(), kDefaultGlassHullWeight);
    EXPECT_EQ(d.constraints->items[0].margin_or_default(), 0.0);

    expect_format_error(text("", R"({ "type": "glass_hull", "mode": "max" })"),
                        "constraints.items[0].mode", "kein");
    expect_format_error(text("", R"({ "type": "glass_hull", "margin": -0.1 })"),
                        "constraints.items[0].margin", "mindestens 0");
    expect_format_error(text("", R"({ "type": "glass_hull", "weight": -1 })"),
                        "constraints.items[0].weight", "mindestens 0");
    expect_format_error(text("", R"({ "type": "focal_length", "margin": 0.1 })"),
                        "constraints.items[0].margin", "nur bei glass_hull");
    expect_format_error(text("", R"({ "type": "glass_hull" }, { "type": "glass_hull" })"),
                        "constraints.items[1].type", "doppelt");
    expect_format_error(text("", R"({ "type": "glass_hull", "plate": 0 })"),
                        "constraints.items[0].plate", "nur bei ghost_penalty");
    expect_format_error(text("", R"({ "type": "glass_hull", "smoothing": 1e-6 })"),
                        "constraints.items[0].smoothing", "nur bei ghost_penalty");
    expect_format_error(text("", R"({ "type": "glass_huelle" })"), "constraints.items[0].type",
                        "unbekannter Typ");
}

// ---- Umwandlung ----------------------------------------------------------------------------

TEST(GlassVariables, ConvertCatalogConstantAndBuchdahl) {
    SystemDescription d = read(R"({ "type": "glass", "surfaces": "all" })");
    const LoadedSystem raw = prepare_system(d, kSystems);
    const std::vector<GlassConversion> conv = convert_glass_variables(d, raw.catalog);
    ASSERT_EQ(conv.size(), 2u);
    // N-BK7: 3-Term-Fit über 400–700 nm, Koeffizienten exakt die des Fits
    const BuchdahlFit fit = fit_buchdahl(raw.catalog.at("N-BK7"), 3);
    EXPECT_EQ(conv[0].surface, 1);
    EXPECT_EQ(conv[0].from, MaterialKind::Catalog);
    EXPECT_EQ(conv[0].name, "N-BK7");
    EXPECT_EQ(conv[0].max_fit_error, fit.max_abs_error);
    const MaterialSpec& m = d.surfaces[1].material;
    EXPECT_EQ(m.kind, MaterialKind::Buchdahl);
    EXPECT_EQ(m.n0, fit.n0);
    EXPECT_EQ(m.nu, fit.nu);
    EXPECT_NE(m.nu[2], 0.0); // ν₃ des Fits bleibt (fest)
    // konstantes Material
    EXPECT_EQ(conv[1].from, MaterialKind::Constant);
    EXPECT_EQ(d.surfaces[3].material.kind, MaterialKind::Buchdahl);
    EXPECT_EQ(d.surfaces[3].material.n0, 1.6);
    EXPECT_EQ(d.surfaces[3].material.nu, (std::array<double, 3>{0.0, 0.0, 0.0}));
    // zweite Umwandlung ändert nichts (Buchdahl bleibt)
    const SystemDescription once = d;
    const std::vector<GlassConversion> again = convert_glass_variables(d, raw.catalog);
    EXPECT_EQ(again[0].from, MaterialKind::Buchdahl);
    EXPECT_EQ(d, once);
    // ohne Glasvariablen: nichts
    SystemDescription none = read(R"({ "type": "curvature", "surface": 1 })");
    EXPECT_TRUE(convert_glass_variables(none, raw.catalog).empty());
    EXPECT_EQ(none.surfaces[1].material.kind, MaterialKind::Catalog);
    // unbekanntes Glas
    SystemDescription unknown = read(R"({ "type": "glass", "surface": 1 })");
    EXPECT_THROW(convert_glass_variables(unknown, GlassCatalog{}), GlassError);
}

TEST(GlassVariables, ConvertedIndicesMatchCatalogWithinFit) {
    const SystemDescription d = read(R"({ "type": "glass", "surface": 1 })");
    const LoadedSystem raw = prepare_system(d, kSystems);
    const Converted c = convert(d);
    const SystemBatchHost a = make_system_batch(raw, 1);
    const SystemBatchHost b = make_system_batch(c.loaded, 1);
    double worst = 0.0;
    for (int l = 0; l < a.wavelengths(); ++l) {
        worst = std::max(worst, std::abs(a.index(0, 1, l) - b.index(0, 1, l)));
    }
    worst = std::max(worst, std::abs(a.paraxial_index(0, 1) - b.paraxial_index(0, 1)));
    std::printf("N-BK7 als Buchdahl-Material: max |Δn| = %.2e\n", worst);
    EXPECT_LT(worst, kBuchdahlFit3);
    // Koeffizienten im Batch (nur Buchdahl-Medien, sonst 0)
    EXPECT_EQ(b.buchdahl_coeff(0, 1, 0), c.loaded.system.surfaces[1].material.n0);
    EXPECT_EQ(b.buchdahl_coeff(0, 1, 3), c.loaded.system.surfaces[1].material.nu[2]);
    EXPECT_EQ(a.buchdahl_coeff(0, 1, 0), 0.0);
    EXPECT_EQ(b.buchdahl_coeff(0, 3, 0), 0.0);
}

// ---- Abbildung Vektor ↔ Batch --------------------------------------------------------------

TEST(GlassVariables, VariableSetRequiresBuchdahl) {
    const SystemDescription d = read(R"({ "type": "glass", "surface": 1 })");
    const LoadedSystem raw = prepare_system(d, kSystems);
    const SystemBatchHost b = make_system_batch(raw, 1);
    EXPECT_THROW(VariableSet(raw.system, b.topology()), std::invalid_argument);
}

TEST(GlassVariables, VariableSetRoundtripAndIndices) {
    const Converted c = convert(read(
        R"({ "type": "curvature", "surface": 1 }, { "type": "glass", "surface": 1 },
           { "type": "glass", "surface": 3 })"));
    const SystemDescription& sys = c.loaded.system;
    const SystemBatchHost base = make_system_batch(c.loaded, 2);
    const VariableSet set(sys, base.topology());
    ASSERT_EQ(set.size(), 7);
    EXPECT_EQ(set[1].name, "n0[1]");
    EXPECT_EQ(set[2].name, "nu1[1]");
    EXPECT_EQ(set[3].name, "nu2[1]");
    EXPECT_EQ(set[6].name, "nu2[3]");
    EXPECT_EQ(set[2].glass_component, 1);
    EXPECT_EQ(set[2].batch_surface, 1);
    EXPECT_EQ(set[0].glass_component, -1);
    EXPECT_EQ(set.typical()[3], kDefaultTypicalGlassNu2);
    EXPECT_EQ(set.lower()[1], -kInf);

    // extract aus Batch und Datei: die Koeffizienten des Materials
    const std::vector<double> x0 = set.extract(base, 0);
    EXPECT_EQ(x0, set.extract(sys));
    EXPECT_EQ(x0[1], sys.surfaces[1].material.n0);
    EXPECT_EQ(x0[2], sys.surfaces[1].material.nu[0]);
    EXPECT_EQ(x0[3], sys.surfaces[1].material.nu[1]);
    EXPECT_EQ(x0[4], 1.6);

    // apply: neue Koordinaten (unrealistisch, das ist hier egal)
    std::vector<double> x = x0;
    x[1] = 1.75;
    x[2] = -0.12;
    x[3] = 0.031;
    x[4] = 1.45;
    x[5] = -0.03;
    x[6] = 0.002;
    SystemBatchHost b = base;
    set.apply(b, 1, x);
    EXPECT_EQ(set.extract(b, 1), x);  // bitgleicher Roundtrip
    EXPECT_EQ(set.extract(b, 0), x0); // Variante 0 unverändert
    const double nu3 = sys.surfaces[1].material.nu[2];
    for (int l = 0; l < b.wavelengths(); ++l) {
        EXPECT_EQ(b.index(1, 1, l),
                  buchdahl_index(1.75, -0.12, 0.031, nu3, nm_to_um(b.wavelength_nm(l))));
        EXPECT_EQ(b.index(1, 3, l),
                  buchdahl_index(1.45, -0.03, 0.002, 0.0, nm_to_um(b.wavelength_nm(l))));
        EXPECT_EQ(b.index(1, 2, l), base.index(1, 2, l)); // Luft bleibt
    }
    EXPECT_EQ(b.paraxial_index(1, 1), buchdahl_index(1.75, -0.12, 0.031, nu3, nm_to_um(546.07)));
    EXPECT_EQ(b.buchdahl_coeff(1, 1, 3), nu3); // ν₃ bleibt

    // Batch nach apply ist bitgleich zum Batch aus der geänderten Datei
    SystemDescription changed = sys;
    set.apply_to_description(changed, x);
    EXPECT_EQ(changed.surfaces[1].material.n0, 1.75);
    EXPECT_EQ(changed.surfaces[1].material.nu[2], nu3);
    EXPECT_EQ(set.extract(changed), x);
    const SystemBatchHost rebuilt = make_system_batch(prepare_system(changed, kSystems), 2);
    SystemBatchHost both = base;
    set.apply(both, 0, x);
    set.apply(both, 1, x);
    EXPECT_EQ(both.index_data().size(), rebuilt.index_data().size());
    EXPECT_TRUE(both == rebuilt);

    // Datei, deren Glas kein Buchdahl-Material ist, passt nicht
    SystemDescription wrong = sys;
    wrong.surfaces[1].material.kind = MaterialKind::Catalog;
    wrong.surfaces[1].material.name = "N-BK7";
    EXPECT_THROW(set.apply_to_description(wrong, x), std::invalid_argument);
    // Wellenlänge, bei der die Buchdahl-Koordinate nicht definiert ist
    SystemBatchHost bad = base;
    bad.wavelength_nm(0) = 100.0;
    const SystemBatchHost before = bad;
    EXPECT_THROW(set.apply(bad, 0, x), std::invalid_argument);
    EXPECT_TRUE(bad == before); // nichts geschrieben
}

TEST(GlassVariables, MonochromaticExtractIsExact) {
    // Eine Wellenlänge (587,56 nm), λ_ref = 546,07 nm nicht darunter: Die Brechzahlen allein
    // bestimmen (n₀, ν₁, ν₂) nicht; extract liest buchdahl_coeff.
    SystemDescription d = read(R"({ "type": "glass", "surface": 1 })");
    d.spectrum.wavelengths_nm = {587.56};
    d.spectrum.weights = {1.0};
    const Converted c = convert(d);
    const SystemBatchHost base = make_system_batch(c.loaded, 1);
    const VariableSet set(c.loaded.system, base.topology());
    std::vector<double> x = set.extract(base, 0);
    x[1] += 1e-3;
    x[2] *= 1.1;
    SystemBatchHost b = base;
    set.apply(b, 0, x);
    EXPECT_EQ(set.extract(b, 0), x);
    EXPECT_EQ(b.paraxial_index(0, 1),
              buchdahl_index(x[0], x[1], x[2], b.buchdahl_coeff(0, 1, 3), nm_to_um(546.07)));
}

TEST(GlassVariables, GhostPlatesShiftTableSurface) {
    SystemDescription d = read_system_file(kSystems / "einzellinse_geisterplatte.json");
    // feste Sollbrennweite: sonst hingen die Feldziele des neu gebauten Batches von der
    // paraxialen Brennweite mit dem geänderten Glas ab (während der Optimierung bleiben sie fest)
    d.target = Target{};
    d.target->focal_length = 100.0;
    Variables v;
    VariableSpec g;
    g.type = VariableType::Glass;
    g.selection = SurfaceSelection::One;
    g.surfaces = {1};
    v.items.push_back(g);
    d.variables = v;
    const Converted c = convert(d);
    const SystemBatchHost base = make_system_batch(c.loaded, 1);
    const VariableSet set(c.loaded.system, base.topology());
    EXPECT_EQ(set[0].batch_surface, 3); // zwei Geisterplatten nach der Blende
    std::vector<double> x = set.extract(base, 0);
    x[0] = 1.62;
    SystemBatchHost b = base;
    set.apply(b, 0, x);
    SystemDescription changed = c.loaded.system;
    set.apply_to_description(changed, x);
    EXPECT_TRUE(b == make_system_batch(prepare_system(changed, kSystems), 1));
}

TEST(GlassVariables, PerturbationUsesGlassScale) {
    const Converted c = convert(read(R"({ "type": "glass", "surface": 1 })"));
    const SystemBatchHost base = make_system_batch(c.loaded, 1);
    const VariableSet set(c.loaded.system, base.topology());
    const std::vector<double> x0 = set.extract(base, 0);
    PerturbationOptions p;
    p.glass = {0.0, 0.0};
    EXPECT_EQ(perturbed_vector(set, x0, p, 3), x0);
    p.glass = {0.01, 0.0};
    const std::vector<double> y = perturbed_vector(set, x0, p, 3);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NE(y[k], x0[k]);
        EXPECT_LE(std::abs(y[k] - x0[k]), 0.01 * std::abs(x0[k]));
    }
}

// ---- Strafe außerhalb der Hülle ------------------------------------------------------------

GlassHull synthetic_hull() {
    return make_glass_hull(GlassCatalog::from_file(kGlassDir + "synthetisch.json"));
}

GlassHullPenalty manual_penalty(double weight, double margin, GlassHull hull) {
    GlassHullPenalty p;
    p.hull = std::move(hull);
    p.weight = weight;
    p.margin = margin;
    p.elements = {{1, 1}};
    return p;
}

TEST(GlassVariables, PenaltyFromFile) {
    const Converted c = convert(
        read(R"({ "type": "glass", "surface": 1 })", R"({ "type": "glass_hull", "weight": 1e6 })"));
    const SystemBatchHost base = make_system_batch(c.loaded, 1);
    const VariableSet set(c.loaded.system, base.topology());
    const GlassHullPenalty p = make_glass_hull_penalty(c.loaded.system, set, synthetic_hull());
    EXPECT_EQ(p.item, 0);
    EXPECT_EQ(p.weight, 1e6);
    EXPECT_EQ(p.margin, 0.0);
    ASSERT_EQ(p.size(), 1);
    EXPECT_EQ(p.residual_count(), 3);
    EXPECT_EQ(p.elements[0], (GlassHullElement{1, 1}));
    check_glass_hull_penalty(p, base.topology());

    // Zusatzresiduen: N-BK7 liegt außerhalb der Hülle der Testgläser? Wert gegen die Funktion
    const ExtraResiduals extra = make_glass_hull_residuals(p);
    EXPECT_EQ(extra.count, 3);
    std::vector<double> out(3);
    extra.evaluate(base, 0, out);
    const Point3 r = glass_hull_residual(p, batch_glass_coordinates(base, 0, 1));
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_EQ(out[k], r[k]);
    }
    std::vector<double> wrong(2);
    EXPECT_THROW(glass_hull_residuals(p, base, 0, wrong), std::invalid_argument);

    // ohne Eintrag glass_hull: leer
    const Converted none = convert(read(R"({ "type": "glass", "surface": 1 })"));
    const VariableSet set2(none.loaded.system, base.topology());
    const GlassHullPenalty empty =
        make_glass_hull_penalty(none.loaded.system, set2, synthetic_hull());
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(make_glass_hull_residuals(empty).count, 0);
    // Sicherheitsabstand mit ebener Hülle (basis.json: drei Gläser) wird abgelehnt
    const Converted margin = convert(read(R"({ "type": "glass", "surface": 1 })",
                                          R"({ "type": "glass_hull", "margin": 0.01 })"));
    EXPECT_THROW(
        make_glass_hull_penalty(margin.loaded.system, set,
                                make_glass_hull(GlassCatalog::from_file(kGlassDir + "basis.json"))),
        std::invalid_argument);
    // Prüfung gegen die Topologie
    GlassHullPenalty bad = p;
    bad.elements[0].batch_surface = 99;
    EXPECT_THROW(check_glass_hull_penalty(bad, base.topology()), std::invalid_argument);
    bad = p;
    bad.weight = -1.0;
    EXPECT_THROW(check_glass_hull_penalty(bad, base.topology()), std::invalid_argument);
}

TEST(GlassVariables, PenaltyResidualInsideAndOutside) {
    const GlassHull hull = synthetic_hull();
    const GlassHullPenalty p = manual_penalty(4.0, 0.0, hull);
    // innen (ein inneres Glas der Hülle): 0
    const Point3 inner = hull.points()[5].coordinates; // SYN-06, kein Eckpunkt
    EXPECT_EQ(glass_hull_excess(p, inner), 0.0);
    EXPECT_EQ(glass_hull_penalty(p, inner), 0.0);
    EXPECT_EQ(glass_hull_residual(p, inner), (Point3{0.0, 0.0, 0.0}));
    // außen: r = √w (u − P(u)), |r|² = w d²
    const Point3 out{2.1, -0.02, 0.01};
    const HullProjection proj = hull.project(out);
    ASSERT_GT(proj.distance, 0.1);
    const Point3 r = glass_hull_residual(p, out);
    const Point3 u = hull.normalize(out);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(r[k], 2.0 * (u[k] - proj.closest[k]), kHullDistance);
    }
    EXPECT_NEAR(glass_hull_penalty(p, out), 4.0 * proj.distance * proj.distance,
                kHullDistance * glass_hull_penalty(p, out));
    EXPECT_NEAR(r[0] * r[0] + r[1] * r[1] + r[2] * r[2], glass_hull_penalty(p, out),
                kHullDistance * glass_hull_penalty(p, out));
}

// Gradient des Strafbeitrags w max(0, φ)² nach den Rohkoordinaten gegen zentrale Differenzen
// (Schritt relativ zur Skala der Koordinate).
void expect_penalty_gradient(const GlassHullPenalty& p, const Point3& x) {
    const Point3 g = glass_hull_penalty_gradient(p, x);
    for (std::size_t k = 0; k < 3; ++k) {
        const double h = 1e-6 * p.hull.scale()[k];
        Point3 a = x;
        Point3 b = x;
        a[k] += h;
        b[k] -= h;
        const double fd = (glass_hull_penalty(p, a) - glass_hull_penalty(p, b)) / (2.0 * h);
        EXPECT_NEAR(g[k], fd, kGlassPenaltyGradientFd * (std::abs(fd) + 1.0 / p.hull.scale()[k]))
            << "Komponente " << k;
    }
}

TEST(GlassVariables, PenaltyGradientMatchesFiniteDifferences) {
    const GlassHull hull = synthetic_hull();
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u(-1.5, 1.5);
    // außerhalb der dreidimensionalen Hülle
    const GlassHullPenalty p = manual_penalty(3.0, 0.0, hull);
    int outside = 0;
    for (int i = 0; i < 400 && outside < 100; ++i) {
        const Point3 x = hull.denormalize({u(rng), u(rng), u(rng)});
        if (hull.distance(x) < 0.05) {
            continue;
        }
        expect_penalty_gradient(p, x);
        ++outside;
    }
    EXPECT_EQ(outside, 100);
    // Sicherheitsabstand: im Band innen vor einer Facette (Schwerpunkt − t·Normale, t < δ)
    const GlassHullPenalty pm = manual_penalty(3.0, 0.05, hull);
    const ConvexHull3& h3 = hull.hull();
    for (std::size_t f = 0; f < h3.triangles.size(); ++f) {
        Point3 c{};
        for (const int k : h3.triangles[f]) {
            for (std::size_t j = 0; j < 3; ++j) {
                c[j] += h3.vertices[static_cast<std::size_t>(k)][j] / 3.0;
            }
        }
        const Point3& n = h3.planes[f].normal;
        const double t = 0.01;
        const Point3 x = hull.denormalize({c[0] - t * n[0], c[1] - t * n[1], c[2] - t * n[2]});
        const HullProjection proj = hull.project(x);
        ASSERT_TRUE(proj.inside);
        // nächste Randebene ist diese Facette (bei kleinem t), Überschreitung δ − t
        if (std::abs(proj.signed_distance + t) < kHullDistance) {
            EXPECT_NEAR(glass_hull_excess(pm, x), 0.05 - t, kHullDistance);
        }
        expect_penalty_gradient(pm, x);
        EXPECT_EQ(glass_hull_excess(p, x), 0.0); // ohne δ innen keine Strafe
    }
    // ebene Hülle (basis.json): außerhalb des Dreiecks
    const GlassHull planar = make_glass_hull(GlassCatalog::from_file(kGlassDir + "basis.json"));
    const GlassHullPenalty pp = manual_penalty(2.0, 0.0, planar);
    for (int i = 0; i < 50; ++i) {
        const Point3 x = planar.denormalize({u(rng), u(rng), u(rng)});
        if (planar.distance(x) < 0.05) {
            continue;
        }
        expect_penalty_gradient(pp, x);
    }
}

} // namespace
} // namespace mko::test
