// Geglättete L1-Strafe der Geisterkoeffizienten (M4-T6): Glättung (Wert, Ableitung, Grenzfälle),
// Ordnungsgewichte, Aufbau aus der Systemdatei, Residuen aus dem Batch, Verkettung von
// Zusatzresiduen und Geisterkoeffizienten als Variablen (VariableSet, Variantengenerator);
// docs/optimierung.md, Abschnitte 2 und 3.

#include "ghost_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace mko::test {
namespace {

constexpr double kW = 54.4; // μ · w_k [Wellen²/mm]
constexpr double kEps = 1e-6;

// Richardson-Extrapolation (3 Stufen) der zentralen Differenz von f bei a mit Startschritt h.
template <class F>
double richardson(F&& f, double a, double h) {
    const auto d = [&](double s) { return (f(a + s) - f(a - s)) / (2.0 * s); };
    const double d1 = d(h), d2 = d(h / 2.0), d4 = d(h / 4.0);
    const double r1 = (4.0 * d2 - d1) / 3.0;
    const double r2 = (4.0 * d4 - d2) / 3.0;
    return (16.0 * r2 - r1) / 15.0;
}

// Wert: r(0) = 0, r ungerade, r² = μ w (√(a² + ε²) − ε) ≈ μ w |a| für |a| ≫ ε (relativer
// Unterschied ε/|a|), r ≈ √(μ w/(2ε)) a für |a| ≪ ε.
TEST(GhostPenalty, SmoothingValue) {
    EXPECT_EQ(ghost_penalty_residual(0.0, kW, kEps), 0.0);
    EXPECT_EQ(ghost_penalty_square(0.0, kW, kEps), 0.0);
    for (const double a : {1e-9, 3e-7, 1e-6, 2e-5, 1e-3, 0.1, 5.0}) {
        const double r = ghost_penalty_residual(a, kW, kEps);
        EXPECT_EQ(ghost_penalty_residual(-a, kW, kEps), -r) << a;
        EXPECT_GT(r, 0.0) << a;
        // Quadrat gleich der Pseudo-Huber-Form (direkt, a groß genug gegen Auslöschung)
        const double sq = kW * (std::sqrt(a * a + kEps * kEps) - kEps);
        if (a >= 1e-6) {
            EXPECT_NEAR(r * r, sq, 1e-14 * sq) << a;
        }
        EXPECT_NEAR(ghost_penalty_square(a, kW, kEps), r * r, 4e-15 * r * r) << a;
        // r² ≤ μ w |a|, Unterschied höchstens μ w ε
        EXPECT_LE(r * r, kW * a * (1.0 + 1e-15)) << a;
        EXPECT_GE(r * r, kW * (a - kEps) * (1.0 - 1e-15)) << a;
    }
    // |a| ≫ ε: r² / (μ w |a|) = 1 − ε/|a| + O(ε²/a²)
    for (const double a : {1e-4, 1e-3, 1e-2}) {
        const double r = ghost_penalty_residual(a, kW, kEps);
        EXPECT_NEAR(r * r / (kW * a), 1.0 - kEps / a, 2.0 * (kEps / a) * (kEps / a)) << a;
    }
    // |a| ≪ ε: linear mit Steigung √(μ w/(2ε))
    for (const double a : {1e-12, 1e-10, -1e-9}) {
        const double slope = std::sqrt(kW / (2.0 * kEps));
        EXPECT_NEAR(ghost_penalty_residual(a, kW, kEps), slope * a,
                    std::abs(slope * a) * (a / kEps) * (a / kEps))
            << a;
    }
    // Gewicht 0 schaltet ab, große |a| ohne Überlauf
    EXPECT_EQ(ghost_penalty_residual(1e-3, 0.0, kEps), 0.0);
    EXPECT_TRUE(std::isfinite(ghost_penalty_residual(1e200, kW, kEps)));
    EXPECT_GT(ghost_penalty_residual(1e200, kW, kEps), 0.0);
}

// Ableitung: analytisch gegen Richardson-Extrapolation, glatt durch a = 0 (keine Knickstelle).
TEST(GhostPenalty, DerivativeMatchesFiniteDifferences) {
    const auto r = [](double a) { return ghost_penalty_residual(a, kW, kEps); };
    for (const double a : {0.0, 1e-8, -1e-7, 5e-7, -1e-6, 3e-6, -1e-5, 1e-4, 1e-2}) {
        const double h = 0.05 * std::max(std::abs(a), kEps);
        const double fd = richardson(r, a, h);
        const double an = ghost_penalty_derivative(a, kW, kEps);
        EXPECT_NEAR(an, fd, 1e-8 * std::abs(an)) << a;
    }
    // dr/da(0) = √(μ w/(2ε)): das Gauß-Newton-Modell (dr/da)² = μ w/(2ε) enthält genau die
    // halbe Krümmung von r² bei 0 (d²(r²)/da² = μ w/ε), also die exakte Krümmung von F = Σ r²
    // in der Näherung 2 JᵀJ.
    const double d0 = ghost_penalty_derivative(0.0, kW, kEps);
    EXPECT_NEAR(d0 * d0, kW / (2.0 * kEps), 1e-14 * kW / kEps);
    const auto sq = [](double a) { return ghost_penalty_square(a, kW, kEps); };
    const double h = 1e-3 * kEps;
    const double curvature = (sq(h) - 2.0 * sq(0.0) + sq(-h)) / (h * h);
    EXPECT_NEAR(2.0 * d0 * d0, curvature, 1e-5 * curvature);
}

// Gradient von r²: μ w a/√(a² + ε²) – derselbe wie bei der Form des Plans √(μ w)(a² + ε²)^{1/4}
// (deren Quadrat μ w √(a² + ε²) sich nur um die Konstante μ w ε unterscheidet). Damit hat das
// Problem dasselbe Minimum.
TEST(GhostPenalty, GradientEqualsPlanForm) {
    for (const double a : {-1e-3, -2e-6, -1e-7, 0.0, 4e-7, 1e-6, 1e-5, 0.3}) {
        const double r = ghost_penalty_residual(a, kW, kEps);
        const double grad = 2.0 * r * ghost_penalty_derivative(a, kW, kEps);
        const double plan = kW * a / std::sqrt(a * a + kEps * kEps);
        EXPECT_NEAR(grad, plan, 1e-14 * kW) << a;
        // Form des Plans: r_P = √(μ w)(a² + ε²)^{1/4}, d(r_P²)/da per Richardson
        const auto rp2 = [](double x) { return kW * std::sqrt(x * x + kEps * kEps); };
        EXPECT_NEAR(richardson(rp2, a, 0.05 * std::max(std::abs(a), kEps)), plan, 1e-8 * kW) << a;
    }
}

TEST(GhostPenalty, OrderWeights) {
    EXPECT_EQ(ghost_order_weight(1, 1.0, 2.0), 2.0);
    EXPECT_EQ(ghost_order_weight(2, 1.0, 2.0), 4.0);
    EXPECT_EQ(ghost_order_weight(6, 1.0, 2.0), 64.0);
    EXPECT_EQ(ghost_order_weight(2, 0.5, 3.0), 4.5);
    EXPECT_EQ(ghost_order_weight(0, 0.5, 3.0), 0.5);
}

// Aktivierungsgradient für das Beugungsziel: 2σΔn/(λ√(2k + 1)); das μ der Testsysteme ist
// daraus für σ = λ/14, Δn = 0,5, k = 2 gewählt.
TEST(GhostPenalty, ActivationGradient) {
    const double g = ghost_activation_gradient(1.0 / 14.0, 0.5, 587.56, 2);
    EXPECT_NEAR(g, 2.0 * (1.0 / 14.0) * 0.5 / (587.56e-6 * std::sqrt(5.0)), 1e-12 * g);
    EXPECT_NEAR(g, 54.367, 1e-3);
    const GhostPenalty pen = make_ghost_penalty(ghost_need_problem().loaded.system);
    const GhostPenaltyTerm& t2 = penalty_term(pen, 0, 2);
    EXPECT_NEAR(t2.weight, g, 1e-3 * g);
}

// Aufbau aus der Systemdatei: alle Ordnungen jeder Platte, Gewicht μ w₀ κ^k, ε des Eintrags.
TEST(GhostPenalty, FromSystemFile) {
    const FdProblem p = ghost_exact_problem();
    const GhostPenalty pen = make_ghost_penalty(p.loaded.system);
    ASSERT_EQ(pen.size(), 16);
    int i = 0;
    for (int g = 0; g < 4; ++g) {
        for (int k = 1; k <= 4; ++k) {
            const GhostPenaltyTerm& t = pen.terms[static_cast<std::size_t>(i++)];
            EXPECT_EQ(t.plate, g);
            EXPECT_EQ(t.order, k);
            EXPECT_EQ(t.weight, 13.6 * ghost_order_weight(k, 1.0, 2.0));
            EXPECT_EQ(t.smoothing, 1e-7);
            EXPECT_EQ(t.item, 1);
        }
    }
    check_ghost_penalty(pen, p.base.topology());
    // Ohne Eintrag ghost_penalty: keine Strafe
    SystemDescription d = p.loaded.system;
    d.constraints->items.pop_back();
    EXPECT_TRUE(make_ghost_penalty(d).empty());
}

// Mehrere Einträge mit getrennten Platten, Standardwerte, nicht bestrafte Platte.
TEST(GhostPenalty, PlateSelection) {
    SystemDescription d = ghost_exact_problem().loaded.system;
    ConstraintSpec a;
    a.type = ConstraintType::GhostPenalty;
    a.plate_selection = SurfaceSelection::List;
    a.plates = {2, 0};
    ConstraintSpec b;
    b.type = ConstraintType::GhostPenalty;
    b.plate_selection = SurfaceSelection::One;
    b.plates = {3};
    b.weight = 5.0;
    b.order_weight = 0.5;
    b.order_factor = 3.0;
    b.smoothing = 2e-6;
    d.constraints->items = {a, b};
    const GhostPenalty pen = make_ghost_penalty(d);
    ASSERT_EQ(pen.size(), 12); // Platten 0, 2, 3 (aufsteigend), Platte 1 ohne Strafe
    EXPECT_EQ(pen.terms[0].plate, 0);
    EXPECT_EQ(pen.terms[4].plate, 2);
    EXPECT_EQ(pen.terms[8].plate, 3);
    EXPECT_EQ(pen.terms[1].weight, 4.0); // μ = 1, w₀ = 1, κ = 2, k = 2
    EXPECT_EQ(pen.terms[1].smoothing, kDefaultGhostSmoothing);
    EXPECT_EQ(pen.terms[0].item, 0);
    EXPECT_EQ(pen.terms[9].weight, 5.0 * 0.5 * 9.0);
    EXPECT_EQ(pen.terms[9].smoothing, 2e-6);
    EXPECT_EQ(pen.terms[9].item, 1);
}

// Residuen aus dem Batch: Formel pro Term aus ghost_coeff, Strafbeitrag je Platte, Zusatzresiduen.
TEST(GhostPenalty, ResidualsFromBatch) {
    const FdProblem p = ghost_exact_problem();
    const GhostPenalty pen = make_ghost_penalty(p.loaded.system);
    std::vector<double> r(static_cast<std::size_t>(pen.size()));
    ghost_penalty_residuals(pen, p.base, 0, r);
    double total = 0.0;
    std::vector<double> per_plate(4, 0.0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        const GhostPenaltyTerm& t = pen.terms[i];
        const double a = p.loaded.system.ghost_plates[static_cast<std::size_t>(t.plate)]
                             .coefficients[static_cast<std::size_t>(t.order - 1)];
        EXPECT_EQ(r[i], ghost_penalty_residual(a, t.weight, t.smoothing)) << i;
        total += r[i] * r[i];
        per_plate[static_cast<std::size_t>(t.plate)] += r[i] * r[i];
    }
    EXPECT_EQ(ghost_penalty_merit(pen, p.base, 0), total);
    for (int g = 0; g < 4; ++g) {
        EXPECT_EQ(ghost_penalty_merit(pen, p.base, 0, g), per_plate[static_cast<std::size_t>(g)]);
    }
    const ExtraResiduals extra = make_ghost_penalty_residuals(pen);
    ASSERT_EQ(extra.count, pen.size());
    std::vector<double> e(r.size());
    extra.evaluate(p.base, 0, e);
    EXPECT_EQ(e, r);
    EXPECT_EQ(make_ghost_penalty_residuals({}).count, 0);
    std::vector<double> wrong(3);
    EXPECT_THROW(ghost_penalty_residuals(pen, p.base, 0, wrong), std::invalid_argument);
    EXPECT_THROW(ghost_penalty_residuals(pen, p.base, 1, r), std::out_of_range);
}

TEST(GhostPenalty, CheckAgainstTopology) {
    const FdProblem p = ghost_exact_problem();
    const GhostPenalty ok = make_ghost_penalty(p.loaded.system);
    for (int field = 0; field < 5; ++field) {
        GhostPenalty bad = ok;
        GhostPenaltyTerm& t = bad.terms.back();
        switch (field) {
        case 0:
            t.plate = 4;
            break;
        case 1:
            t.order = 5;
            break;
        case 2:
            t.weight = -1.0;
            break;
        case 3:
            t.smoothing = 0.0;
            break;
        default:
            t.weight = std::numeric_limits<double>::infinity();
            break;
        }
        EXPECT_THROW(check_ghost_penalty(bad, p.base.topology()), std::invalid_argument) << field;
    }
}

// Verkettung: Reihenfolge der Teile, leere Teile entfallen, ein Teil unverändert.
TEST(GhostPenalty, ConcatExtraResiduals) {
    const FdProblem p = ghost_exact_problem();
    ExtraResiduals a;
    a.count = 2;
    a.evaluate = [](const SystemBatchHost&, int v, std::span<double> out) {
        out[0] = 1.0 + v;
        out[1] = 2.0;
    };
    ExtraResiduals b;
    b.count = 1;
    b.evaluate = [](const SystemBatchHost&, int, std::span<double> out) { out[0] = 3.0; };
    const ExtraResiduals c = concat_extra_residuals({a, ExtraResiduals{}, b});
    ASSERT_EQ(c.count, 3);
    std::vector<double> out(3);
    c.evaluate(p.base, 0, out);
    EXPECT_EQ(out, (std::vector<double>{1.0, 2.0, 3.0}));
    EXPECT_EQ(concat_extra_residuals({ExtraResiduals{}, b}).count, 1);
    EXPECT_EQ(concat_extra_residuals({}).count, 0);
    EXPECT_FALSE(concat_extra_residuals({}).evaluate);
    ExtraResiduals broken;
    broken.count = 1;
    EXPECT_THROW(concat_extra_residuals({a, broken}), std::invalid_argument);
    std::vector<double> short_out(2);
    EXPECT_THROW(c.evaluate(p.base, 0, short_out), std::invalid_argument);
    // Paraxiale Nebenbedingungen und Strafe zusammen wie in mikroptim optimize
    const GhostPenalty pen = make_ghost_penalty(p.loaded.system);
    const ExtraResiduals both = ghost_extra(p.loaded.system, pen);
    ASSERT_EQ(both.count, 1 + pen.size());
    std::vector<double> v(static_cast<std::size_t>(both.count));
    both.evaluate(p.base, 0, v);
    std::vector<double> par(1), gh(static_cast<std::size_t>(pen.size()));
    paraxial_only(p.loaded.system).evaluate(p.base, 0, par);
    ghost_penalty_residuals(pen, p.base, 0, gh);
    EXPECT_EQ(v[0], par[0]);
    for (std::size_t i = 0; i < gh.size(); ++i) {
        EXPECT_EQ(v[i + 1], gh[i]);
    }
}

// Geisterkoeffizienten als Variablen: Lage im Batch, Namen, Roundtrip Vektor ↔ Batch ↔
// Beschreibung, Störung im Variantengenerator.
TEST(GhostPenalty, VariableSetGhostCoefficients) {
    const FdProblem p = ghost_exact_problem();
    ASSERT_EQ(p.set.size(), 3 + 16);
    const Variable& v = p.set[3 + 5]; // Platte 1, Ordnung 2
    EXPECT_EQ(v.type, VariableType::GhostCoefficient);
    EXPECT_EQ(v.ghost_plate, 1);
    EXPECT_EQ(v.ghost_order, 2);
    EXPECT_EQ(v.name, "a[1,2]");
    EXPECT_EQ(v.batch_surface, p.base.ghost_surface(1));
    EXPECT_EQ(v.typical, kDefaultTypicalGhostCoefficient);
    EXPECT_EQ(p.x0[3 + 5], 2e-5);
    EXPECT_EQ(p.set.extract(p.loaded.system), p.x0);

    std::vector<double> x = p.x0;
    for (int k = 3; k < p.set.size(); ++k) {
        x[static_cast<std::size_t>(k)] = 1e-3 * k;
    }
    SystemBatchHost b = p.base;
    p.set.apply(b, 0, x);
    EXPECT_EQ(b.ghost_coeff(0, 1, 1), 1e-3 * 8);
    EXPECT_EQ(p.set.extract(b, 0), x);
    SystemDescription d = p.loaded.system;
    p.set.apply_to_description(d, x);
    EXPECT_EQ(d.ghost_plates[1].coefficients[1], 1e-3 * 8);
    EXPECT_EQ(p.set.extract(d), x);
    // Batch aus der geänderten Beschreibung ist bitgleich zum Batch nach apply
    const LoadedSystem reloaded = prepare_system(d, std::filesystem::path(kResidualSystems));
    EXPECT_TRUE(make_system_batch(reloaded, 1) == b);
    // andere Geisterplatten in der Beschreibung werden abgelehnt
    SystemDescription other = p.loaded.system;
    other.ghost_plates[0].coefficients.push_back(0.0);
    EXPECT_THROW(p.set.apply_to_description(other, x), std::invalid_argument);

    // Störung: relativ 1 % wie die übrigen Typen; a = 0 bleibt ungestört
    PerturbationOptions po;
    po.ghost_coefficient = {0.5, 0.0};
    po.seed = 7;
    const std::vector<double> y = perturbed_vector(p.set, p.x0, po, 1);
    for (int k = 3; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        if (p.x0[kk] == 0.0) {
            EXPECT_EQ(y[kk], 0.0) << k;
        } else {
            EXPECT_LE(std::abs(y[kk] - p.x0[kk]), 0.5 * std::abs(p.x0[kk])) << k;
        }
    }
    EXPECT_EQ(&po.scale(VariableType::GhostCoefficient), &po.ghost_coefficient);
}

} // namespace
} // namespace mko::test
