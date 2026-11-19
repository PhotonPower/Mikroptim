// Sinusbedingung als Residuum (M11-T4): OSC (Conrady) aus realen Zonenstrahlen des Achsbündels
// gegen analytische Orakel (zur Blende konzentrische Systeme: OSC = 0 exakt in jeder Zone),
// gegen den Aberrationsbericht (M10-T1b) und die Seidel-Koma bei kleiner Apertur, Auflösung der
// Einträge in Residuen, Schema und Fehlerfälle, Wirkung in einer Optimierung.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using mko::ConstraintType;
using mko::FormatError;
using mko::LoadedSystem;
using mko::ParaxialConstraints;
using mko::SystemBatchHost;
using mko::test::kOscThirdOrder;
using mko::test::kSineConditionConsistency;
using mko::test::kSineConditionExact;

namespace {

const std::filesystem::path kSystems = std::filesystem::path(MKO_SOURCE_DIR) / "data" / "systems";

// Kopf eines Systems mit konstanten Brechzahlen (eine Wellenlänge, Achsfeld).
std::string header(const std::string& extra = "") {
    return R"({
  "format": "mikroptim.system/1",
  "name": "Sinusbedingung",
  "units": "mm",
  "glass_catalogs": [],
  "spectrum": { "wavelengths_nm": [587.56], "weights": [1.0], "reference_nm": 587.56 },
  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
)" + extra;
}

// Zahl verlustfrei für den JSON-Text (std::to_string hat nur sechs Nachkommastellen).
std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

std::string material(double n) {
    return n == 1.0 ? std::string("\"AIR\"") : "{ \"model\": \"constant\", \"n\": " + num(n) + " }";
}

// Zur Blende konzentrisches System: Blende (in Luft, n[0] = 1) und Kugelflächen im Abstand z_s von
// der Blende mit Krümmungsmittelpunkt in der Blendenmitte (c_s = −1/z_s), danach Medium n[s].
// Jeder Strahl hat dann den Abstand p vom Mittelpunkt mit n p invariant, und die Austrittspupille
// liegt im Mittelpunkt: (f_real/f′)·(l′ − l′_P)/(L′ − l′_P) = (u′/sin U′)·(sin U′/u′) = 1 exakt in
// jeder Zone (docs/numerik.md, Abschnitt 14), bei jeder Apertur.
LoadedSystem concentric(double a, const std::vector<double>& gaps, const std::vector<double>& n,
                        double image_distance) {
    std::string s = header() + "  \"surfaces\": [\n";
    s += "    { \"type\": \"stop\", \"semi_diameter\": " + num(a) +
         ", \"thickness\": " + num(gaps[0]) + ", \"material\": " + material(n[0]) + " }";
    double z = 0.0;
    for (std::size_t k = 1; k < n.size(); ++k) {
        z += gaps[k - 1];
        const double t = k < gaps.size() ? gaps[k] : image_distance;
        s += ",\n    { \"type\": \"sphere\", \"curvature\": " + num(-1.0 / z) +
             ", \"thickness\": " + num(t) + ", \"material\": " + material(n[k]) +
             ", \"semi_diameter\": " + num(0.99 * z) + " }";
    }
    s += "\n  ]\n}\n";
    return mko::prepare_system(mko::read_system_json(s), kSystems);
}

double osc(const SystemBatchHost& b, double rho) {
    return mko::batch_sine_condition(b, 0, mko::batch_paraxial(b, 0), rho);
}

// System aus data/systems mit skalierter Blende und einem Feld 1° (Lagrange-Invariante ≠ 0).
LoadedSystem scaled_system(const std::string& name, double stop_scale) {
    LoadedSystem l = mko::load_system(kSystems / name);
    l.system.surfaces[0].semi_diameter *= stop_scale;
    l.system.source.field_angles_deg = {0.0, 1.0};
    l.system.source.field_weights = {1.0, 1.0};
    return mko::prepare_system(l.system, kSystems);
}

} // namespace

// Analytisches Orakel: zur Blende konzentrische Systeme sind komafrei, OSC = 0 bis auf Rundung
// in allen Zonen und bei großer Apertur (H bis 0,7 · |R₁|), auch mit Glas im Bildraum (dort
// fiele die Normierung mit f = 1/φ statt f′ = n′/φ um den Faktor n′ daneben).
TEST(SineCondition, ConcentricSystemsSatisfyItExactly) {
    struct Case {
        const char* name;
        double a;
        std::vector<double> gaps;
        std::vector<double> n;
    };
    const std::vector<Case> cases = {
        {"Meniskus in Luft", 4.0, {10.0, 3.0}, {1.0, 1.6, 1.0}},
        {"große Apertur", 7.0, {10.0, 4.0}, {1.0, 1.9, 1.0}},
        {"Bildraum in Glas", 5.0, {10.0}, {1.0, 1.7}},
        {"zwei Gläser", 4.0, {8.0, 2.0, 3.0}, {1.0, 1.75, 1.45, 1.0}},
    };
    for (const Case& c : cases) {
        const LoadedSystem l = concentric(c.a, c.gaps, c.n, 20.0);
        const SystemBatchHost b = mko::make_system_batch(l, 1);
        for (const double rho : {0.25, 0.5, 0.70710678118654752440, 1.0}) {
            const double v = osc(b, rho);
            ASSERT_TRUE(std::isfinite(v)) << c.name << " ρ = " << rho;
            EXPECT_LE(std::abs(v), kSineConditionExact) << c.name << " ρ = " << rho;
        }
        // Der Aberrationsbericht rechnet mit derselben Formel.
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        ASSERT_TRUE(rep.real_rays_valid) << c.name;
        EXPECT_LE(std::abs(rep.osc), kSineConditionExact) << c.name;
        EXPECT_LE(std::abs(rep.osc_zone), kSineConditionExact) << c.name;
    }
}

// Gegenprobe: dieselbe Fläche mit Krümmungsmittelpunkt nicht in der Blende hat OSC ≠ 0.
TEST(SineCondition, ShiftedStopViolatesIt) {
    LoadedSystem l = concentric(4.0, {10.0, 3.0}, {1.0, 1.6, 1.0}, 20.0);
    l.system.surfaces[0].thickness = 6.0; // Blende 4 mm näher an den Flächen
    l = mko::prepare_system(l.system, kSystems);
    const SystemBatchHost b = mko::make_system_batch(l, 1);
    EXPECT_GT(std::abs(osc(b, 1.0)), 1e-3);
}

// Batch-Rechnung (Tabellenflächen, Brechzahlen bei λ_ref) gleich dem Aberrationsbericht
// (Dateiflächen) am Rand (ρ = 1) und in der Zone (ρ = 1/√2).
TEST(SineCondition, MatchesAberrationReport) {
    for (const char* name :
         {"einzellinse_bk7.json", "testsystem_mehrlinser.json", "zweilinser_petzval.json"}) {
        const LoadedSystem l = mko::load_system(kSystems / name);
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        ASSERT_TRUE(rep.real_rays_valid) << name;
        const SystemBatchHost b = mko::make_system_batch(l, 1);
        const double rim = osc(b, 1.0);
        const double zone = osc(b, mko::kZoneRho);
        EXPECT_NE(rim, 0.0) << name;
        EXPECT_LE(std::abs(rim - rep.osc), kSineConditionConsistency * std::abs(rep.osc)) << name;
        EXPECT_LE(std::abs(zone - rep.osc_zone), kSineConditionConsistency * std::abs(rep.osc_zone))
            << name;
    }
}

// Kleine Apertur: OSC → −S_II/(2H) (lineare Koma 3. Ordnung, Vorzeichen und Größe), in der Zone
// die Hälfte.
TEST(SineCondition, SmallApertureMatchesSeidelComa) {
    for (const char* name : {"einzellinse_bk7.json", "testsystem_mehrlinser.json"}) {
        const LoadedSystem l = scaled_system(name, 0.02);
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        const double seidel = -rep.S[1] / (2.0 * rep.lagrange);
        const SystemBatchHost b = mko::make_system_batch(l, 1);
        const double rim = osc(b, 1.0);
        EXPECT_LE(std::abs(rim - seidel), kOscThirdOrder * std::abs(seidel)) << name;
        EXPECT_LE(std::abs(osc(b, mko::kZoneRho) - 0.5 * seidel),
                  2.0 * kOscThirdOrder * std::abs(0.5 * seidel))
            << name;
    }
}

// Ein Eintrag wird zu einem Residuum √w · OSC(ρ_k) pro Zone (Standard: Zone 1/√2 und Rand).
TEST(SineCondition, ConstraintResolvesToOneResidualPerZone) {
    mko::SystemDescription sys = mko::read_system_file(kSystems / "einzellinse_bk7.json");
    mko::ConstraintSpec spec;
    spec.type = ConstraintType::SineCondition;
    spec.weight = 4.0;
    sys.constraints = mko::Constraints{{spec}};
    mko::validate_system(sys);
    ParaxialConstraints pc = mko::make_paraxial_constraints(sys);
    ASSERT_EQ(pc.size(), 2);
    EXPECT_DOUBLE_EQ(pc.items[0].zone, mko::kZoneRho);
    EXPECT_EQ(pc.items[1].zone, 1.0);
    EXPECT_EQ(pc.items[0].item, 0);
    EXPECT_EQ(pc.items[1].item, 0);
    EXPECT_TRUE(pc.needs_trace());

    sys.constraints->items[0].zones = {0.3, 0.6, 0.9};
    pc = mko::make_paraxial_constraints(sys);
    ASSERT_EQ(pc.size(), 3);
    const LoadedSystem l = mko::prepare_system(sys, kSystems);
    const SystemBatchHost b = mko::make_system_batch(l, 1);
    mko::check_paraxial_constraints(pc, b.topology());
    std::vector<double> r(3);
    mko::paraxial_residuals(pc, b, 0, r);
    const double z[3] = {0.3, 0.6, 0.9};
    for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(r[static_cast<std::size_t>(k)], 2.0 * osc(b, z[k])) << k;
    }
    // Zusatzresiduen wie in optimize/synth
    const mko::ExtraResiduals extra = mko::make_paraxial_residuals(pc);
    std::vector<double> e(3);
    mko::evaluate_extra_residuals(extra, b, e);
    EXPECT_EQ(e, r);
}

// Versagt der Zonenstrahl (verfehlt die Fläche), ist OSC NaN und das Residuum 0.
TEST(SineCondition, FailedRayGivesZeroResidual) {
    LoadedSystem l = concentric(4.0, {10.0, 3.0}, {1.0, 1.6, 1.0}, 20.0);
    SystemBatchHost b = mko::make_system_batch(l, 1);
    b.pupil_radius(0) = 12.0; // größer als |R₁| = 10 mm
    EXPECT_TRUE(std::isnan(osc(b, 1.0)));
    mko::ParaxialConstraint c;
    c.type = ConstraintType::SineCondition;
    c.target = 0.0;
    c.zone = 1.0;
    c.weight = 9.0;
    EXPECT_EQ(mko::paraxial_residual(c, b, 0, mko::batch_paraxial(b, 0)), 0.0);
    EXPECT_THROW(mko::batch_sine_condition(b, 0, mko::batch_paraxial(b, 0), 0.0),
                 std::invalid_argument);
    EXPECT_THROW(mko::batch_sine_condition(b, 1, mko::batch_paraxial(b, 0), 0.5),
                 std::out_of_range);
    c.zone = 1.5;
    ParaxialConstraints pc;
    pc.items.push_back(c);
    EXPECT_THROW(mko::check_paraxial_constraints(pc, b.topology()), std::invalid_argument);
}

// Dateiformat: Zonen lesen und schreiben, Standard ohne Feld, Fehler mit JSON-Pfad.
TEST(SineCondition, Schema) {
    const std::string surfaces = R"(  "surfaces": [
    { "type": "stop", "semi_diameter": 2.0, "thickness": 5.0, "material": "AIR" },
    { "type": "sphere", "curvature": 0.05, "thickness": 3.0, "material": { "model": "constant", "n": 1.5 }, "semi_diameter": 5.0 },
    { "type": "sphere", "curvature": -0.05, "thickness": 18.0, "material": "AIR", "semi_diameter": 5.0 }
  ],
)";
    const auto text = [&](const std::string& items, const std::string& source = "") {
        std::string h = header();
        if (!source.empty()) {
            h.replace(h.find("  \"source\""), std::string::npos, source);
        }
        return h + surfaces + "  \"constraints\": { \"items\": [" + items + "] }\n}\n";
    };
    const mko::SystemDescription s = mko::read_system_json(
        text(R"({ "type": "sine_condition", "weight": 100.0, "zones": [0.5, 1.0] })"));
    ASSERT_EQ(s.constraints->items.size(), 1u);
    EXPECT_EQ(s.constraints->items[0].type, ConstraintType::SineCondition);
    EXPECT_EQ(s.constraints->items[0].zones, (std::vector<double>{0.5, 1.0}));
    const mko::SystemDescription back = mko::read_system_json(mko::write_system_json(s));
    EXPECT_EQ(back.constraints, s.constraints);
    const mko::SystemDescription d = mko::read_system_json(text(R"({ "type": "sine_condition" })"));
    EXPECT_TRUE(d.constraints->items[0].zones.empty());
    EXPECT_EQ(d.constraints->items[0].zones_or_default().size(), 2u);

    const auto expect_error = [&](const std::string& t, const std::string& path) {
        try {
            (void)mko::read_system_json(t);
            ADD_FAILURE() << "keine Ausnahme, erwartet Fehler unter " << path;
        } catch (const FormatError& e) {
            EXPECT_EQ(e.json_path(), path) << e.what();
        }
    };
    expect_error(text(R"({ "type": "sine_condition", "zones": [0.0] })"),
                 "constraints.items[0].zones[0]");
    expect_error(text(R"({ "type": "sine_condition", "zones": [0.5, 1.5] })"),
                 "constraints.items[0].zones[1]");
    expect_error(text(R"({ "type": "sine_condition", "zones": [0.5, 0.5] })"),
                 "constraints.items[0].zones[1]");
    expect_error(text(R"({ "type": "sine_condition", "zones": [] })"),
                 "constraints.items[0].zones");
    expect_error(text(R"({ "type": "sine_condition", "mode": "max" })"),
                 "constraints.items[0].mode");
    expect_error(text(R"({ "type": "sine_condition" }, { "type": "sine_condition" })"),
                 "constraints.items[1].type");
    expect_error(
        text(
            R"({ "type": "working_distance", "zones": [0.5] })",
            R"(  "source": { "type": "collimated", "field_angles_deg": [0.0], "field_weights": [1.0] },
  "target": { "working_distance": 18.0 },
)"),
        "constraints.items[0].zones");
    expect_error(
        text(
            R"({ "type": "sine_condition" })",
            R"(  "source": { "type": "point", "point": [0.0, 0.0, -10.0], "numerical_aperture": 0.1 },
)"),
        "constraints.items[0].type");
}

// Wirkung: Biegung einer Einzellinse bei fester Brennweite. Ohne Feldpunkte sieht die Merit die
// Koma nicht; mit dem Residuum sinkt OSC am Rand um mehr als eine Größenordnung.
TEST(SineCondition, OptimizationRemovesOsc) {
    mko::SystemDescription sys = mko::read_system_file(kSystems / "einzellinse_bk7_biegung.json");
    sys.source.field_angles_deg = {0.0};
    sys.source.field_weights = {1.0};
    if (!sys.constraints) {
        sys.constraints = mko::Constraints{};
    }
    mko::ConstraintSpec spec;
    spec.type = ConstraintType::SineCondition;
    spec.weight = 1e6;
    sys.constraints->items.push_back(spec);
    mko::validate_system(sys);
    const LoadedSystem l = mko::prepare_system(sys, kSystems);
    const SystemBatchHost base = mko::make_system_batch(l, 1);
    const mko::VariableSet set(l.system, base.topology());
    const double before = osc(base, 1.0);

    mko::LmOptions o;
    o.max_iterations = 50;
    o.jacobian.extra = mko::make_paraxial_residuals(mko::make_paraxial_constraints(l.system));
    const std::vector<double> x0 = set.extract(base, 0);
    const mko::LmResult res = mko::optimize_lm_cpu(set, base, x0, o);
    SystemBatchHost after = base;
    set.apply(after, 0, res.runs.front().x);
    const double osc_after = osc(after, 1.0);
    EXPECT_LT(std::abs(osc_after), 0.05 * std::abs(before))
        << "vorher " << before << ", nachher " << osc_after;
}
