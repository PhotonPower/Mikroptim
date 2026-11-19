// Stichprobenmodell der Monte-Carlo-Tolerierung (M8-T4, docs/optimierung.md, Abschnitt
// „Monte-Carlo-Tolerierung“): Verteilungen der Zufallsgrößen gegen exakte Momente (Gleich-
// verteilung, abgeschnittene Irwin-Hall-Verteilung aus ihrer Dichte), Reproduzierbarkeit pro
// Startwert und Index, Anwendung jeder Toleranzart auf den Batch gegen geschlossene Formeln
// (Krümmung, Dicke, Brechzahl, Abbe-Zahl, Dezentrierung, Verkippung, Starrkörperbewegung eines
// Elements um den Drehpunkt), ξ = 0 bitgleich zur Vorlage, Fehlerfälle.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/tolerance/tolerance_model.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

using mko::SystemBatchHost;
using mko::ToleranceDistribution;
using mko::ToleranceModel;
using nlohmann::json;

namespace {

// Dichte der Irwin-Hall-Verteilung (Summe von 12 Gleichverteilten auf [0, 1]) bei x ∈ [0, 12].
double irwin_hall_density(double x) {
    double sum = 0.0;
    double binom = 1.0; // C(12, k)
    for (int k = 0; k <= 12 && k <= static_cast<int>(std::floor(x)); ++k) {
        sum += (k % 2 == 0 ? 1.0 : -1.0) * binom * std::pow(x - k, 11);
        binom = binom * (12 - k) / (k + 1);
    }
    return sum / 39916800.0; // 11!
}

// Varianz von ξ = z/c, z = Irwin-Hall − 6 abgeschnitten auf |z| <= c (Simpson über die Dichte).
double truncated_irwin_hall_variance(double c) {
    const int n = 20000;
    const double h = 2.0 * c / n;
    double m0 = 0.0;
    double m2 = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double z = -c + i * h;
        const double w = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        const double f = irwin_hall_density(z + 6.0);
        m0 += w * f;
        m2 += w * f * z * z;
    }
    return m2 / m0 / (c * c);
}

struct Moments {
    double mean = 0.0;
    double var = 0.0;
    double m4 = 0.0;
    double min = 1e300;
    double max = -1e300;
};

Moments moments(ToleranceDistribution d, double c, int n) {
    std::mt19937_64 engine(mko::variant_seed(7, 3));
    std::vector<double> x(static_cast<std::size_t>(n));
    Moments m;
    for (double& v : x) {
        v = mko::tolerance_deviate(engine, d, c);
        m.mean += v;
        m.min = std::min(m.min, v);
        m.max = std::max(m.max, v);
    }
    m.mean /= n;
    for (const double v : x) {
        m.var += (v - m.mean) * (v - m.mean);
        m.m4 += std::pow(v - m.mean, 4);
    }
    m.var /= n - 1;
    m.m4 /= n;
    return m;
}

struct Fixture {
    mko::LoadedSystem loaded;
    SystemBatchHost nominal; // nicht koaxial, eine Variante
    ToleranceModel model;
};

Fixture make_fixture(const json& j) {
    Fixture f;
    f.loaded = mko::test::load_tolerance_json(j);
    f.nominal = mko::make_non_coaxial(mko::make_system_batch(f.loaded, 1));
    f.model = ToleranceModel(f.loaded.system, f.nominal.topology());
    return f;
}

// Batch mit der nominellen Variante und ξ = 0 bis auf die Ziehungen `values` ab der ersten
// Ziehung der Toleranz k.
SystemBatchHost apply_single(const Fixture& f, int k, std::vector<double> values) {
    SystemBatchHost b = f.nominal;
    std::vector<double> xi(static_cast<std::size_t>(f.model.draws()), 0.0);
    const int at = f.model.draw_offset(k);
    for (std::size_t i = 0; i < values.size(); ++i) {
        xi[static_cast<std::size_t>(at) + i] = values[i];
    }
    f.model.apply(b, 0, xi);
    return b;
}

bool same_bits(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
            return false;
        }
    }
    return true;
}

bool same_variant_data(const SystemBatchHost& a, const SystemBatchHost& b) {
    return same_bits(a.curvature_data(), b.curvature_data()) &&
           same_bits(a.thickness_data(), b.thickness_data()) &&
           same_bits(a.index_data(), b.index_data()) &&
           same_bits(a.paraxial_index_data(), b.paraxial_index_data()) &&
           same_bits(a.decenter_tilt_data(), b.decenter_tilt_data());
}

} // namespace

TEST(ToleranceModel, UniformDeviateMoments) {
    const int n = 200000;
    const Moments m = moments(ToleranceDistribution::Uniform, 2.0, n);
    EXPECT_GE(m.min, -1.0);
    EXPECT_LE(m.max, 1.0);
    const double var = 1.0 / 3.0;
    EXPECT_LT(std::abs(m.mean), mko::test::kMonteCarloSigmas * std::sqrt(var / n));
    EXPECT_LT(std::abs(m.var - var),
              mko::test::kMonteCarloSigmas * std::sqrt((m.m4 - var * var) / n));
}

TEST(ToleranceModel, TruncatedNormalDeviateMoments) {
    const int n = 200000;
    for (const double c : {0.5, 1.0, 2.0, 3.0}) {
        const Moments m = moments(ToleranceDistribution::Normal, c, n);
        EXPECT_GE(m.min, -1.0) << c;
        EXPECT_LE(m.max, 1.0) << c;
        const double var = truncated_irwin_hall_variance(c);
        EXPECT_LT(std::abs(m.mean), mko::test::kMonteCarloSigmas * std::sqrt(var / n)) << c;
        EXPECT_LT(std::abs(m.var - var),
                  mko::test::kMonteCarloSigmas * std::sqrt((m.m4 - var * var) / n))
            << c << ": " << m.var << " gegen " << var;
    }
    // Ohne Abschneiden (c = 6 ist der Rand der Irwin-Hall-Verteilung) ist die Varianz von z
    // genau 1, also Var(ξ) = 1/36.
    EXPECT_NEAR(truncated_irwin_hall_variance(6.0), 1.0 / 36.0, 1e-9);
    std::mt19937_64 engine(1);
    EXPECT_THROW((void)mko::tolerance_deviate(engine, ToleranceDistribution::Normal, 0.4),
                 std::invalid_argument);
    EXPECT_THROW((void)mko::tolerance_deviate(engine, ToleranceDistribution::Normal, 6.5),
                 std::invalid_argument);
}

TEST(ToleranceModel, DeviatesDependOnlyOnSeedAndIndex) {
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    ASSERT_EQ(f.model.draws(), 26);
    const std::vector<double> a = f.model.deviates(42, 17);
    const std::vector<double> b = f.model.deviates(42, 17);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, f.model.deviates(42, 18));
    EXPECT_NE(a, f.model.deviates(43, 17));
    // Eigene Nachrechnung: ein Generator pro Stichprobe, Ziehungen in Toleranzreihenfolge
    std::mt19937_64 engine(mko::variant_seed(42, 17));
    std::size_t d = 0;
    for (const mko::ResolvedTolerance& t : f.model.tolerances()) {
        for (int k = 0; k < t.draws(); ++k) {
            EXPECT_EQ(a[d++], mko::tolerance_deviate(engine, t.distribution, t.cutoff));
        }
    }
    std::vector<double> wrong(3);
    EXPECT_THROW(f.model.deviates(1, 1, wrong), std::invalid_argument);
}

TEST(ToleranceModel, ZeroDeviatesLeaveBatchBitIdentical) {
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    SystemBatchHost b = f.nominal;
    const std::vector<double> zero(static_cast<std::size_t>(f.model.draws()), 0.0);
    f.model.apply(b, 0, zero);
    EXPECT_TRUE(same_variant_data(b, f.nominal));
    // Negative Zufallsgrößen bei Grenze 0 (−0) ändern ebenfalls kein Bit
    json j = mko::test::tolerance_system_json();
    for (json& item : j["tolerances"]["items"]) {
        for (const char* key : {"delta", "relative", "fringes", "arcmin"}) {
            if (item.contains(key)) {
                item[key] = 0.0;
            }
        }
    }
    const Fixture g = make_fixture(j);
    SystemBatchHost c = g.nominal;
    std::vector<double> minus(static_cast<std::size_t>(g.model.draws()), -0.5);
    g.model.apply(c, 0, minus);
    EXPECT_TRUE(same_variant_data(c, g.nominal));
}

TEST(ToleranceModel, RadiusThicknessIndexAbbe) {
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    const auto& tol = f.model.tolerances();
    const SystemBatchHost& n = f.nominal;
    // Radius in Passen (Toleranz 1 = Fläche 2): c' = c + Δc
    {
        const SystemBatchHost b = apply_single(f, 1, {0.5});
        EXPECT_EQ(b.curvature(0, 2), n.curvature(0, 2) + tol[1].limit * 0.5);
        EXPECT_EQ(b.curvature(0, 1), n.curvature(0, 1));
    }
    // Dicke (Toleranz 4 = Fläche 1): t' = t + δ
    {
        const SystemBatchHost b = apply_single(f, 4, {-0.75});
        EXPECT_EQ(b.thickness(0, 1), n.thickness(0, 1) + 0.02 * -0.75);
        EXPECT_EQ(b.thickness(0, 2), n.thickness(0, 2));
    }
    // Brechzahl (Toleranz 6 = Glas nach Fläche 1): Δn bei allen Wellenlängen und paraxial
    {
        const SystemBatchHost b = apply_single(f, 6, {1.0});
        for (int l = 0; l < n.wavelengths(); ++l) {
            EXPECT_EQ(b.index(0, 1, l), n.index(0, 1, l) + 0.0005) << l;
            EXPECT_EQ(b.index(0, 2, l), n.index(0, 2, l)) << l;
        }
        EXPECT_EQ(b.paraxial_index(0, 1), n.paraxial_index(0, 1) + 0.0005);
    }
    // Abbe-Zahl (Toleranz 9 = Glas nach Fläche 2) zusammen mit Δn (Toleranz 7):
    // ν'_e = ν_e (1 + e) mit ν_e = (n_e − 1)/(n_F − n_C), n_e' = n_e + Δn
    {
        SystemBatchHost b = f.nominal;
        std::vector<double> xi(static_cast<std::size_t>(f.model.draws()), 0.0);
        xi[static_cast<std::size_t>(f.model.draw_offset(7))] = -0.4;
        xi[static_cast<std::size_t>(f.model.draw_offset(9))] = 0.8;
        f.model.apply(b, 0, xi);
        const double dn = 0.0005 * -0.4;
        const double e = 0.005 * 0.8;
        const double nu = (n.paraxial_index(0, 2) - 1.0) / (n.index(0, 2, 0) - n.index(0, 2, 2));
        const double nu_b =
            (b.paraxial_index(0, 2) - 1.0 - dn) / (b.index(0, 2, 0) - b.index(0, 2, 2));
        EXPECT_NEAR(nu_b / (nu * (1.0 + e)), 1.0, mko::test::kUnitFp64);
        EXPECT_EQ(b.paraxial_index(0, 2), n.paraxial_index(0, 2) + dn);
        EXPECT_NEAR(b.index(0, 2, 1), n.index(0, 2, 1) + dn, mko::test::kUnitFp64);
    }
    // Relativer Radius: c' = c/(1 + ε)
    {
        const Fixture g = make_fixture(mko::test::with_tolerance_items(
            json::array({{{"type", "radius"}, {"surface", 3}, {"relative", 0.002}}})));
        const SystemBatchHost b = apply_single(g, 0, {-0.5});
        EXPECT_EQ(b.curvature(0, 3), g.nominal.curvature(0, 3) / (1.0 - 0.001));
    }
}

TEST(ToleranceModel, DecenterAndTiltOfSurfaces) {
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    const double tilt = f.model.tolerances()[14].limit; // 1′
    SystemBatchHost b = f.nominal;
    std::vector<double> xi(static_cast<std::size_t>(f.model.draws()), 0.0);
    xi[static_cast<std::size_t>(f.model.draw_offset(11))] = 0.6;      // Δx Fläche 2
    xi[static_cast<std::size_t>(f.model.draw_offset(11)) + 1] = -0.2; // Δy Fläche 2
    xi[static_cast<std::size_t>(f.model.draw_offset(14))] = 0.5;      // α Fläche 2
    xi[static_cast<std::size_t>(f.model.draw_offset(14)) + 1] = -1.0; // β Fläche 2
    f.model.apply(b, 0, xi);
    EXPECT_EQ(b.decenter_tilt(0, 2, 0), 0.005 * 0.6);
    EXPECT_EQ(b.decenter_tilt(0, 2, 1), 0.005 * -0.2);
    EXPECT_EQ(b.decenter_tilt(0, 2, 2), tilt * 0.5);
    EXPECT_EQ(b.decenter_tilt(0, 2, 3), tilt * -1.0);
    for (const int s : {1, 3}) {
        for (int k = 0; k < 4; ++k) {
            EXPECT_EQ(b.decenter_tilt(0, s, k), 0.0) << s << ' ' << k;
        }
    }
    // Dicken bleiben bei Einzelflächen unverändert (Kippung um den Scheitel)
    EXPECT_TRUE(same_bits(b.thickness_data(), f.nominal.thickness_data()));
}

TEST(ToleranceModel, ElementDecenterMovesAllSurfaces) {
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    const SystemBatchHost b = apply_single(f, 16, {0.3, -0.7});
    for (int s = 1; s <= 3; ++s) {
        EXPECT_EQ(b.decenter_tilt(0, s, 0), 0.01 * 0.3) << s;
        EXPECT_EQ(b.decenter_tilt(0, s, 1), 0.01 * -0.7) << s;
        EXPECT_EQ(b.decenter_tilt(0, s, 2), 0.0) << s;
        EXPECT_EQ(b.decenter_tilt(0, s, 3), 0.0) << s;
    }
    EXPECT_TRUE(same_bits(b.thickness_data(), f.nominal.thickness_data()));
}

TEST(ToleranceModel, ElementTiltRotatesAboutPivot) {
    // Kippung α um die x-Achse durch P = (0, 0, z_P): Scheitel (0, 0, z_s) → (0, −l sin α,
    // z_P + l cos α), l = z_s − z_P; Flächennormale (0, −sin α, cos α), also Kippwinkel α. Die
    // axiale Verschiebung δz = l (cos α − 1) geht in die Dicken davor (+δz) und danach (−δz).
    // Dazu β um die y-Achse getrennt: Scheitel (l sin β, 0, z_P + l cos β).
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    const mko::ResolvedTolerance& t = f.model.tolerances()[17];
    ASSERT_EQ(t.type, mko::ToleranceType::ElementTilt);
    const double zp = t.pivot_z;
    const double z[4] = {0.0, 2.0, 4.5, 5.7};
    for (const bool about_x : {true, false}) {
        const double a = 0.9 * t.limit;
        const SystemBatchHost b =
            about_x ? apply_single(f, 17, {0.9, 0.0}) : apply_single(f, 17, {0.0, 0.9});
        double dz[4] = {0.0, 0.0, 0.0, 0.0};
        for (int s = 1; s <= 3; ++s) {
            const double l = z[s] - zp;
            dz[s] = l * (std::cos(a) - 1.0);
            const double dx = about_x ? 0.0 : l * std::sin(a);
            const double dy = about_x ? -l * std::sin(a) : 0.0;
            EXPECT_NEAR(b.decenter_tilt(0, s, 0), dx, mko::test::kLengthFp64) << s;
            EXPECT_NEAR(b.decenter_tilt(0, s, 1), dy, mko::test::kLengthFp64) << s;
            EXPECT_NEAR(b.decenter_tilt(0, s, 2), about_x ? a : 0.0, mko::test::kUnitFp64) << s;
            EXPECT_NEAR(b.decenter_tilt(0, s, 3), about_x ? 0.0 : a, mko::test::kUnitFp64) << s;
        }
        for (int s = 0; s <= 3; ++s) {
            // Dicke nach Fläche s: +δz der nächsten Fläche, −δz der Fläche s
            const double next = s + 1 <= 3 ? dz[s + 1] : 0.0;
            EXPECT_NEAR(b.thickness(0, s), f.nominal.thickness(0, s) + next - dz[s],
                        mko::test::kLengthFp64)
                << s;
        }
    }
}

TEST(ToleranceModel, MoveSurfaceComposesRotations) {
    // Zwei Kippungen um denselben Drehpunkt hintereinander ergeben bei Drehungen um dieselbe
    // Achse die Summe der Winkel.
    mko::ElementMotion m;
    m.alpha = 0.01;
    m.pivot_z = 3.0;
    const mko::SurfacePlacement p1 = mko::move_surface({}, 5.0, m);
    const mko::SurfacePlacement p2 = mko::move_surface(p1, 5.0, m);
    m.alpha = 0.02;
    const mko::SurfacePlacement q = mko::move_surface({}, 5.0, m);
    EXPECT_NEAR(p2.alpha, q.alpha, mko::test::kUnitFp64);
    EXPECT_NEAR(p2.dy, q.dy, mko::test::kLengthFp64);
    EXPECT_NEAR(p2.dz, q.dz, mko::test::kLengthFp64);
    EXPECT_NEAR(p2.beta, 0.0, mko::test::kUnitFp64);
    // Ohne Drehung reine Verschiebung, bitgleich
    mko::ElementMotion s;
    s.dx = 0.25;
    s.dy = -0.5;
    const mko::SurfacePlacement r = mko::move_surface(p1, 5.0, s);
    EXPECT_EQ(r.dx, p1.dx + 0.25);
    EXPECT_EQ(r.dy, p1.dy - 0.5);
    EXPECT_EQ(r.dz, p1.dz);
    EXPECT_EQ(r.alpha, p1.alpha);
}

TEST(ToleranceModel, WriteSamplesMatchesSingleApplication) {
    const Fixture f = make_fixture(mko::test::tolerance_system_json());
    SystemBatchHost out = f.nominal;
    out.resize(5);
    f.model.write_samples(f.nominal, 0, 99, 1000, out, 1, 4);
    for (int j = 0; j < 4; ++j) {
        SystemBatchHost one = f.nominal;
        f.model.apply(one, 0, f.model.deviates(99, 1000 + static_cast<std::uint64_t>(j)));
        SystemBatchHost part;
        part.assign_variants(out, 1 + j, 1);
        EXPECT_TRUE(same_variant_data(part, one)) << j;
    }
    // Fehler: koaxiale Topologie, anderer Batch, Bereich, Länge
    EXPECT_THROW(ToleranceModel(f.loaded.system, mko::make_system_batch(f.loaded, 1).topology()),
                 std::invalid_argument);
    EXPECT_THROW(f.model.write_samples(f.nominal, 0, 1, 0, out, 3, 3), std::out_of_range);
    EXPECT_THROW(f.model.write_samples(out, 0, 1, 0, out, 0, 1), std::invalid_argument);
    std::vector<double> xi(2);
    EXPECT_THROW(f.model.apply(out, 0, xi), std::invalid_argument);
    SystemBatchHost coax = mko::make_system_batch(f.loaded, 1);
    EXPECT_THROW(f.model.apply(coax, 0, f.model.deviates(1, 1)), std::invalid_argument);
}
