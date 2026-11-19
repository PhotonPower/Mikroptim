// Empfindlichkeiten (M10-T3), CPU: Störungen bei ±δ gegen Einzelrechnungen gestörter
// Systemdateien nach ADR-0011 (Dezentrierung und Verkippung als Felder der Flächen, Elemente als
// starre Körper mit eigener Formel), Jacobi-Empfindlichkeit gegen finite Differenzen (zwei
// Schrittweiten und die Einzelrechnungen bei ±δ), analytische Orakel (konzentrische Schale um
// ihren Mittelpunkt gekippt, dezentrierte planparallele Platte: keine Wirkung), Standardsatz und
// Determinismus. GPU-Teil: test_cuda_tolerance.cpp.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/tolerance/sensitivity.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <span>
#include <string>
#include <vector>

using mko::SensitivityComponent;
using mko::SensitivityResult;
using mko::SensitivitySettings;
using mko::ToleranceType;
using nlohmann::json;

namespace {

constexpr double kPi = std::numbers::pi;

// Einzelrechnung einer Systemdatei: Q und Wellenfrontreste ẽ_k = √(ŵ W_k / W_ok) e_k / λ_ref
// (Reihenfolge Feld, Wellenlänge, Stützstelle; ohne versagte Strahlen).
struct Single {
    double q = 0.0;
    std::vector<double> e;
};

// Q = Σ ŵ_f ŵ_λ (RMS/λ_ref)² einer einzeln geladenen Systemdatei (Einzelrechnung): eigener Batch
// mit einer Variante, RESIDUALS auf der CPU, Bewertung mit tuple_quality.
Single single_eval(const json& j) {
    const mko::LoadedSystem loaded = mko::test::load_tolerance_json(j);
    const mko::SystemBatchHost batch = mko::make_non_coaxial(mko::make_system_batch(loaded, 1));
    mko::ResidualSettings rs;
    rs.failure = mko::FailureMode::Hard;
    const mko::ResidualLayout layout(batch.topology(), rs);
    const mko::ResidualResult rr = mko::residuals_cpu(batch, 0, layout);
    double wf = 0.0;
    for (int f = 0; f < batch.fields(); ++f) {
        wf += batch.field_weight(f);
    }
    double wl = 0.0;
    for (int l = 0; l < batch.wavelengths(); ++l) {
        wl += batch.wavelength_weight(l);
    }
    const double lref = batch.reference_wavelength_nm * 1e-6;
    Single out;
    for (int f = 0; f < batch.fields(); ++f) {
        const auto n = static_cast<std::size_t>(layout.samples(f));
        std::vector<double> rest(n);
        for (int l = 0; l < batch.wavelengths(); ++l) {
            const std::size_t at = rr.ray_index(0, layout.offset(f, l));
            const mko::TupleQuality t = mko::tuple_quality(
                layout.quadrature(f), std::span<const double>(rr.opd).subspan(at, n),
                std::span<const std::uint8_t>(rr.status).subspan(at, n),
                batch.wavelength_nm(l) * 1e-6, rest);
            EXPECT_EQ(t.failed, 0);
            const double w = (batch.field_weight(f) / wf) * (batch.wavelength_weight(l) / wl);
            for (std::size_t k = 0; k < n; ++k) {
                const double wk = mko::sample(layout.quadrature(f), static_cast<int>(k)).weight;
                out.e.push_back(std::sqrt(w * wk) * rest[k] / lref); // W_ok = 1 ohne Versagen
            }
            const double rms = t.rms / lref;
            out.q += w * (rms * rms);
        }
    }
    return out;
}

double single_q(const json& j) {
    return single_eval(j).q;
}

SensitivityResult analyse(const json& j, double step = mko::kDefaultSensitivityStep,
                          int threads = 0) {
    const mko::LoadedSystem loaded = mko::test::load_tolerance_json(j);
    const mko::SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    SensitivitySettings s;
    s.step = step;
    s.threads = threads;
    return mko::run_sensitivity_analysis(loaded.system, batch, s);
}

// Systemdatei mit der Störung ξ = sign der Komponente c von Toleranz t (Einzelrechnung nach
// ADR-0011; Elementverkippung um P mit eigener Formel: Scheitel P + R(V − P), Verkippung R,
// Dickenänderungen aus dem Axialanteil).
json perturbed(json j, const mko::ResolvedTolerance& t, int c, double sign) {
    j.erase("tolerances");
    json& s = j["surfaces"];
    const double v = sign * t.limit;
    const auto at = [&s](int i) -> json& { return s[static_cast<std::size_t>(i)]; };
    switch (t.type) {
    case ToleranceType::Radius: {
        const double c0 = at(t.surface)["curvature"].get<double>();
        at(t.surface)["curvature"] = t.radius_fringes ? c0 + v : c0 / (1.0 + v);
        break;
    }
    case ToleranceType::Thickness:
        at(t.surface)["thickness"] = at(t.surface)["thickness"].get<double>() + v;
        break;
    case ToleranceType::Index: {
        // nur Buchdahl-Gläser: n₀ + Δn ist dieselbe Brechzahländerung bei allen Wellenlängen
        json& m = at(t.surface)["material"];
        EXPECT_TRUE(m.is_object());
        m["n0"] = m["n0"].get<double>() + v;
        break;
    }
    case ToleranceType::Decenter:
        at(t.surface)["decenter"] = c == 0 ? json::array({v, 0.0}) : json::array({0.0, v});
        break;
    case ToleranceType::Tilt: {
        const double deg = v * 180.0 / kPi;
        at(t.surface)["tilt_deg"] = c == 0 ? json::array({deg, 0.0}) : json::array({0.0, deg});
        break;
    }
    case ToleranceType::ElementDecenter:
        for (int i = t.first; i <= t.last; ++i) {
            at(i)["decenter"] = c == 0 ? json::array({v, 0.0}) : json::array({0.0, v});
        }
        break;
    case ToleranceType::ElementTilt: {
        const double deg = v * 180.0 / kPi;
        std::vector<double> dz;
        const json nominal = s;
        for (int i = t.first; i <= t.last; ++i) {
            double z = 0.0;
            for (int k = 0; k < i; ++k) {
                z += nominal[static_cast<std::size_t>(k)]["thickness"].get<double>();
            }
            const double l = z - t.pivot_z;
            // R_x(α) e_z = (0, −sin α, cos α), R_y(β) e_z = (sin β, 0, cos β)
            if (c == 0) {
                at(i)["decenter"] = json::array({0.0, -l * std::sin(v)});
                at(i)["tilt_deg"] = json::array({deg, 0.0});
                dz.push_back(l * (std::cos(v) - 1.0));
            } else {
                at(i)["decenter"] = json::array({l * std::sin(v), 0.0});
                at(i)["tilt_deg"] = json::array({0.0, deg});
                dz.push_back(l * (std::cos(v) - 1.0));
            }
        }
        for (int i = t.first; i <= t.last; ++i) {
            const double d = dz[static_cast<std::size_t>(i - t.first)];
            at(i - 1)["thickness"] = at(i - 1)["thickness"].get<double>() + d;
            at(i)["thickness"] = at(i)["thickness"].get<double>() - d;
        }
        break;
    }
    default:
        ADD_FAILURE() << "Typ ohne Einzelrechnung";
    }
    return j;
}

// Testsystem mit einem Toleranzsatz aus allen Typen außer Abbe (Brechzahl am Buchdahl-Glas).
// Feste Sollbrennweite (Bildpunkte der Felder unabhängig von der Störung, wie im Batch der
// Analyse).
json all_types_json() {
    json j = mko::test::with_tolerance_items(json::array({
        {{"type", "radius"}, {"surfaces", {1, 2}}, {"relative", 0.002}},
        {{"type", "radius"}, {"surface", 3}, {"fringes", 3.0}},
        {{"type", "thickness"}, {"surfaces", {0, 1, 2}}, {"delta", 0.02}},
        {{"type", "index"}, {"surface", 2}, {"delta", 0.0005}},
        {{"type", "decenter"}, {"surface", 2}, {"delta", 0.005}},
        {{"type", "tilt"}, {"surface", 3}, {"arcmin", 1.0}},
        {{"type", "element_decenter"}, {"elements", "all"}, {"delta", 0.01}},
        {{"type", "element_tilt"}, {"element", {1, 3}}, {"arcmin", 2.0}, {"pivot", "center"}},
    }));
    j["target"] = {{"focal_length", 20.0}};
    return j;
}

// Konzentrische Schale (Flächen 1, 2 mit gemeinsamem Krümmungsmittelpunkt 10 mm hinter Fläche
// 1), danach eine planparallele Platte (Flächen 3, 4) und eine Sammellinse (Flächen 5, 6);
// frei erfunden.
json shell_plate_json() {
    json j = mko::test::tolerance_system_json();
    j["name"] = "Schale, Platte, Linse";
    j["surfaces"] = json::array({
        {{"type", "stop"}, {"semi_diameter", 1.5}, {"thickness", 2.0}, {"material", "AIR"}},
        {{"type", "sphere"},
         {"curvature", 0.1},
         {"thickness", 2.0},
         {"material", "CAF2"},
         {"semi_diameter", 3.0}},
        {{"type", "sphere"},
         {"curvature", 0.125},
         {"thickness", 1.0},
         {"material", "AIR"},
         {"semi_diameter", 3.0}},
        {{"type", "plane"}, {"thickness", 1.5}, {"material", "F_SILICA"}, {"semi_diameter", 3.0}},
        {{"type", "plane"}, {"thickness", 1.0}, {"material", "AIR"}, {"semi_diameter", 3.0}},
        {{"type", "sphere"},
         {"curvature", 0.08},
         {"thickness", 2.5},
         {"material", "CAF2"},
         {"semi_diameter", 3.0}},
        {{"type", "sphere"},
         {"curvature", -0.08},
         {"thickness", 15.0},
         {"material", "AIR"},
         {"semi_diameter", 3.0}},
    });
    j["tolerances"] = {
        {"distribution", "uniform"},
        {"items",
         json::array({
             // Drehpunkt im gemeinsamen Mittelpunkt: 10 mm hinter dem Scheitel von 1
             {{"type", "element_tilt"}, {"element", {1, 2}}, {"deg", 0.5}, {"pivot", 10.0}},
             {{"type", "element_decenter"}, {"element", {3, 4}}, {"delta", 0.05}},
             // Kontrolle: dieselbe Schale um ihre Mitte gekippt wirkt
             {{"type", "element_tilt"}, {"element", {5, 6}}, {"deg", 0.5}},
         })}};
    return j;
}

bool same(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

} // namespace

// Abnahme (1): Q bei ±δ jeder Komponente gegen die Einzelrechnung der gestörten Systemdatei
// (ADR-0011: decenter, tilt_deg an den Flächen; Elemente als starrer Körper).
TEST(Sensitivity, PerturbationsMatchSingleSystemFiles) {
    const json j = all_types_json();
    const SensitivityResult r = analyse(j);
    ASSERT_EQ(r.tolerances.size(), 11u);
    ASSERT_EQ(r.components.size(), 15u);
    EXPECT_EQ(r.variants, 1 + 4 * 15);
    EXPECT_EQ(r.failed_rays, 0);
    const double q0 = single_q([&] {
        json n = j;
        n.erase("tolerances");
        return n;
    }());
    EXPECT_LE(std::abs(r.q0 - q0), mko::test::kSensitivitySingleSystem * q0);
    double worst = 0.0;
    double worst_element = 0.0;
    for (const SensitivityComponent& c : r.components) {
        const mko::ResolvedTolerance& t = r.tolerances[static_cast<std::size_t>(c.tolerance)];
        EXPECT_EQ(c.failed_rays, 0);
        for (const double sign : {1.0, -1.0}) {
            const double q = single_q(perturbed(j, t, c.component, sign));
            const double got = sign > 0.0 ? c.q_plus : c.q_minus;
            const double dev = std::abs(got - q) / q;
            worst = std::max(worst, dev);
            if (t.type == ToleranceType::ElementTilt || t.type == ToleranceType::ElementDecenter) {
                worst_element = std::max(worst_element, dev);
            }
            EXPECT_LE(dev, mko::test::kSensitivitySingleSystem)
                << to_string(t.type) << " Komponente " << c.component << " Vorzeichen " << sign;
        }
        // die Störung wirkt (keine Toleranz dieses Satzes ist wirkungslos)
        EXPECT_GT(std::max(std::abs(c.q_plus - r.q0), std::abs(c.q_minus - r.q0)), 1e-6 * r.q0)
            << to_string(t.type);
    }
    std::printf("[ Messung ] Q0 = %.6e Wellen², Einzelrechnung: größte relative Abweichung "
                "%.3e (Elemente %.3e)\n",
                r.q0, worst, worst_element);
}

// Abnahme (2): Jacobi-Empfindlichkeit gegen finite Differenzen. (a) Die Jacobi-Spalten mit
// h = 0,01 und h = 0,005 stimmen bis auf O(h²) überein. (b) Gegen die Sekante aus den
// Einzelrechnungen der gestörten Dateien bei ±δ (ADR-0011): s = (ẽ(+δ) − ẽ(−δ))/2 aus zwei
// unabhängig geladenen Systemen; ‖j‖² gegen ‖s‖² und g = 2 ẽ₀ᵀ j gegen 2 ẽ₀ᵀ s (Unterschied
// O(δ²) relativ). (c) g gegen die zentrale Differenz der direkten Werte (Q₊ − Q₋)/2.
TEST(Sensitivity, JacobianAgreesWithFiniteDifferences) {
    const json j = all_types_json();
    const SensitivityResult a = analyse(j, 0.01);
    const SensitivityResult b = analyse(j, 0.005);
    ASSERT_EQ(a.components.size(), b.components.size());
    const Single nominal = single_eval([&] {
        json n = j;
        n.erase("tolerances");
        return n;
    }());
    ASSERT_EQ(nominal.e.size(), static_cast<std::size_t>(a.residuals));
    double step_dev = 0.0;
    double secant_dev = 0.0;
    double linear_dev = 0.0;
    double gauss_newton_dev = 0.0;
    for (std::size_t k = 0; k < a.components.size(); ++k) {
        const SensitivityComponent& x = a.components[k];
        const SensitivityComponent& y = b.components[k];
        const mko::ResolvedTolerance& t = a.tolerances[static_cast<std::size_t>(x.tolerance)];
        // Skala: Größe der Änderung von Q bei ±δ
        const double scale = std::max({std::abs(x.q_plus - a.q0), std::abs(x.q_minus - a.q0),
                                       std::abs(x.jacobian_linear) + x.jacobian_quadratic});
        // (a)
        const double d1 = std::max(std::abs(x.jacobian_linear - y.jacobian_linear),
                                   std::abs(x.jacobian_quadratic - y.jacobian_quadratic)) /
                          scale;
        step_dev = std::max(step_dev, d1);
        EXPECT_LE(d1, mko::test::kSensitivityJacobianStep) << to_string(t.type) << " " << k;
        // (b)
        const Single plus = single_eval(perturbed(j, t, x.component, 1.0));
        const Single minus = single_eval(perturbed(j, t, x.component, -1.0));
        double ss = 0.0;
        double es = 0.0;
        for (std::size_t i = 0; i < nominal.e.size(); ++i) {
            const double sec = 0.5 * (plus.e[i] - minus.e[i]);
            ss += sec * sec;
            es += nominal.e[i] * sec;
        }
        const double d2 =
            std::max(std::abs(x.jacobian_quadratic - ss), std::abs(x.jacobian_linear - 2.0 * es)) /
            scale;
        secant_dev = std::max(secant_dev, d2);
        EXPECT_LE(d2, mko::test::kSensitivityJacobianSecant) << to_string(t.type) << " " << k;
        // (c)
        const double d3 = std::abs(x.jacobian_linear - x.linear) / scale;
        linear_dev = std::max(linear_dev, d3);
        EXPECT_LE(d3, mko::test::kSensitivityJacobianLinear) << to_string(t.type) << " " << k;
        // Information: Gauß-Newton-Term gegen den direkten quadratischen Term b
        gauss_newton_dev =
            std::max(gauss_newton_dev, std::abs(x.jacobian_quadratic - x.quadratic) / scale);
    }
    std::printf("[ Messung ] Jacobi h = 0,01 gegen 0,005: %.3e; gegen Sekante der Dateien bei "
                "±δ: %.3e; g gegen (Q+ - Q-)/2: %.3e; ‖j‖² gegen b (Krümmung der Reste): %.3e "
                "(relativ zu max |ΔQ|)\n",
                step_dev, secant_dev, linear_dev, gauss_newton_dev);
}

// Analytische Orakel: Eine konzentrische Schale um ihren gemeinsamen Krümmungsmittelpunkt
// gekippt und eine planparallele Platte quer verschoben bleiben dieselben Flächen (bis auf den
// Rand); Q und die Jacobi-Spalte ändern sich nur durch Rundung. Kontrolle: die Linse um ihre
// Mitte gekippt wirkt.
TEST(Sensitivity, InvariantMotionsHaveNoEffect) {
    const SensitivityResult r = analyse(shell_plate_json());
    ASSERT_EQ(r.tolerances.size(), 3u);
    ASSERT_EQ(r.components.size(), 6u);
    double worst = 0.0;
    double worst_j = 0.0;
    for (const SensitivityComponent& c : r.components) {
        const double dq = std::max(std::abs(c.q_plus - r.q0), std::abs(c.q_minus - r.q0)) / r.q0;
        const double jq = c.jacobian_quadratic / r.q0;
        if (c.tolerance < 2) {
            worst = std::max(worst, dq);
            worst_j = std::max(worst_j, jq);
            EXPECT_LE(dq, mko::test::kSensitivityInvariant) << c.tolerance;
            EXPECT_LE(jq, mko::test::kSensitivityInvariant) << c.tolerance;
        } else {
            EXPECT_GT(dq, 1e-3) << "Kontrolle";
            EXPECT_GT(jq, 1e-5) << "Kontrolle";
        }
    }
    std::printf("[ Messung ] invariante Bewegungen: |ΔQ|/Q0 <= %.3e, ‖j‖²/Q0 <= %.3e\n", worst,
                worst_j);
}

// Ohne Feld außerhalb der Achse und ohne Störung der Symmetrie ist Q gerade in der Dezentrierung
// und gleich für x und y (Drehsymmetrie): linearer Term 0, beide Komponenten gleich.
TEST(Sensitivity, AxialFieldIsSymmetric) {
    json j = all_types_json();
    j["source"]["field_angles_deg"] = json::array({0.0});
    j["source"]["field_weights"] = json::array({1.0});
    const SensitivityResult r = analyse(j);
    for (std::size_t k = 0; k + 1 < r.components.size(); ++k) {
        const SensitivityComponent& x = r.components[k];
        const mko::ResolvedTolerance& t = r.tolerances[static_cast<std::size_t>(x.tolerance)];
        if (t.type != ToleranceType::ElementDecenter || x.component != 0) {
            continue;
        }
        const SensitivityComponent& y = r.components[k + 1];
        const double scale = x.quadratic;
        EXPECT_GT(scale, 0.0);
        EXPECT_LE(std::abs(x.linear), mko::test::kSensitivityInvariant * r.q0);
        EXPECT_LE(std::abs(x.quadratic - y.quadratic), mko::test::kSensitivityInvariant * r.q0);
        EXPECT_LE(std::abs(x.jacobian_quadratic - y.jacobian_quadratic),
                  mko::test::kSensitivityInvariant * r.q0);
    }
}

TEST(Sensitivity, DefaultSetCoversSystem) {
    const mko::LoadedSystem loaded = mko::load_system(mko::test::kToleranceSystemFile);
    mko::SystemDescription sys = loaded.system;
    sys.tolerances = mko::default_sensitivity_tolerances(sys);
    const std::vector<mko::ResolvedTolerance> res = mko::resolve_tolerances(sys);
    // ein Element (1–3), drei gekrümmte Flächen, Dicken 0, 1, 2, Gläser 1, 2
    ASSERT_EQ(res.size(), 2u + 3u + 3u + 2u);
    EXPECT_EQ(res[0].type, ToleranceType::ElementDecenter);
    EXPECT_EQ(res[0].first, 1);
    EXPECT_EQ(res[0].last, 3);
    EXPECT_DOUBLE_EQ(res[0].limit, 0.01);
    EXPECT_EQ(res[1].type, ToleranceType::ElementTilt);
    EXPECT_DOUBLE_EQ(res[1].limit, kPi / (180.0 * 60.0));
    EXPECT_DOUBLE_EQ(res[1].pivot_z, 0.5 * (2.0 + 2.0 + 2.5 + 1.2));
    EXPECT_EQ(res[2].type, ToleranceType::Radius);
    EXPECT_DOUBLE_EQ(res[2].limit, 1e-3);
    EXPECT_EQ(res[5].type, ToleranceType::Thickness);
    EXPECT_EQ(res[7].surface, 2);
    EXPECT_EQ(res[8].type, ToleranceType::Index);
    EXPECT_DOUBLE_EQ(res[9].limit, 5e-4);
    // Grenze 0 schaltet eine Art ab; ungültige Grenzen
    mko::SensitivityDefaults d;
    d.element_tilt_arcmin = 0.0;
    d.index = 0.0;
    sys.tolerances = mko::default_sensitivity_tolerances(sys, d);
    EXPECT_EQ(mko::resolve_tolerances(sys).size(), 1u + 3u + 3u);
    d.thickness_mm = -1.0;
    EXPECT_THROW((void)mko::default_sensitivity_tolerances(sys, d), std::invalid_argument);
    // die Analyse läuft mit dem Standardsatz
    const SensitivityResult r = mko::run_sensitivity_analysis(
        [&] {
            mko::SystemDescription s = loaded.system;
            s.tolerances = mko::default_sensitivity_tolerances(s);
            return s;
        }(),
        mko::make_system_batch(loaded, 1), SensitivitySettings{});
    EXPECT_EQ(r.components.size(), 4u + 3u + 3u + 2u);
    const std::vector<mko::SensitivitySummary> sum = mko::summarize_sensitivity(r);
    ASSERT_EQ(sum.size(), r.tolerances.size());
    for (const mko::SensitivitySummary& s : sum) {
        EXPECT_TRUE(std::isfinite(s.worst_delta_q));
        EXPECT_NEAR(s.worst_rms * s.worst_rms, r.q0 + s.worst_delta_q, 1e-15);
    }
}

TEST(Sensitivity, DeterministicForAnyThreadCount) {
    const json j = all_types_json();
    const SensitivityResult a = analyse(j, 0.01, 1);
    const SensitivityResult b = analyse(j, 0.01, 7);
    ASSERT_EQ(a.components.size(), b.components.size());
    EXPECT_TRUE(same(a.q0, b.q0));
    for (std::size_t k = 0; k < a.components.size(); ++k) {
        const SensitivityComponent& x = a.components[k];
        const SensitivityComponent& y = b.components[k];
        EXPECT_TRUE(same(x.q_plus, y.q_plus) && same(x.q_minus, y.q_minus) &&
                    same(x.jacobian_linear, y.jacobian_linear) &&
                    same(x.jacobian_quadratic, y.jacobian_quadratic))
            << k;
    }
}

TEST(Sensitivity, RejectsInvalidSettings) {
    const mko::LoadedSystem loaded = mko::load_system(mko::test::kToleranceSystemFile);
    const mko::SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    SensitivitySettings s;
    s.step = 0.0;
    EXPECT_THROW((void)mko::run_sensitivity_analysis(loaded.system, batch, s),
                 std::invalid_argument);
    s.step = 1.5;
    EXPECT_THROW((void)mko::run_sensitivity_analysis(loaded.system, batch, s),
                 std::invalid_argument);
    s.step = 0.01;
    mko::SystemBatchHost two = batch;
    two.resize(2);
    EXPECT_THROW((void)mko::run_sensitivity_analysis(loaded.system, two, s), std::invalid_argument);
    // ohne Toleranzsatz: nur die nominelle Variante
    mko::SystemDescription none = loaded.system;
    none.tolerances.reset();
    const SensitivityResult r = mko::run_sensitivity_analysis(none, batch, s);
    EXPECT_TRUE(r.components.empty());
    EXPECT_EQ(r.variants, 1);
    EXPECT_GT(r.q0, 0.0);
}
