// Variantengenerator (M3-T1): gestörte Varianten (deterministisch, plattformunabhängig) und
// Muster der finiten Differenzen.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using mko::Distribution;
using mko::FdScheme;
using mko::PerturbationOptions;
using mko::SystemBatchHost;
using mko::VariableSet;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!same_bits(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

// Beispieldatei: c[1], c[2] in [−0,05; 0,05], t[1] in [2; 12], w in [50; 150] (x_typ 10).
struct Fixture {
    mko::LoadedSystem loaded = mko::load_system(kSystems + "einzellinse_bk7_variablen.json");
    SystemBatchHost base = mko::make_system_batch(loaded, 1);
    VariableSet set{loaded.system, base.topology()};
    std::vector<double> x0 = set.extract(base, 0); // {0.00967, -0.00967, 6.0, 99.05}
};

// Prüft, dass Variante v von `batch` bis auf die Variablen gleich der Vorlage ist.
void expect_only_variables_differ(const VariableSet& set, const SystemBatchHost& batch, int v,
                                  const SystemBatchHost& base, int base_variant) {
    SystemBatchHost a(batch.topology(), 1);
    a.copy_variant_from(batch, v, 0);
    SystemBatchHost b(base.topology(), 1);
    b.copy_variant_from(base, base_variant, 0);
    set.apply(a, 0, set.extract(b, 0));
    EXPECT_TRUE(a == b) << "Variante " << v;
}

} // namespace

// ---- Zufallszahlen ----------------------------------------------------------------------

// Feste Werte: der Standard legt mt19937_64 bitgenau fest (10.000. Wert bei Startwert 5489),
// SplitMix64 ist eine feste Ganzzahlfunktion.
TEST(VariantGenerator, RandomBuildingBlocksArePortable) {
    std::mt19937_64 engine(5489u);
    engine.discard(9999);
    EXPECT_EQ(engine(), 9981545732273789042ULL);
    EXPECT_EQ(mko::splitmix64(0), 0xE220A8397B1DCDAFULL);
    EXPECT_EQ(mko::variant_seed(42, 0), 0xBDD732262FEB6E95ULL);
    EXPECT_EQ(mko::variant_seed(42, 1), 0x28EFE333B266F103ULL);
    EXPECT_EQ(mko::unit_uniform(0), 0.0);
    EXPECT_EQ(mko::unit_uniform(~0ULL), 1.0 - 0x1.0p-53);
    EXPECT_EQ(mko::unit_uniform(1ULL << 63), 0.5);
    EXPECT_EQ(mko::unit_uniform((1ULL << 11) - 1), 0.0); // die unteren 11 Bit fallen weg
}

// Erwartungswerte aus einer unabhängigen Python-Implementierung (mt19937_64, SplitMix64,
// exakte Bruchrechnung für fma); gleich auf Windows und Linux.
TEST(VariantGenerator, PerturbedValuesArePlatformIndependent) {
    const Fixture f;
    PerturbationOptions opt;
    opt.seed = 42;
    const std::vector<double> y0 = mko::perturbed_vector(f.set, f.x0, opt, 0);
    const std::vector<double> y1 = mko::perturbed_vector(f.set, f.x0, opt, 1);
    EXPECT_TRUE(same_bits(
        y0, {0.00960031256552798, -0.009579233400076301, 6.056423511822777, 98.55214299700086}));
    EXPECT_TRUE(same_bits(
        y1, {0.009694628931969861, -0.009605728055568991, 6.010199414155366, 98.39441906638912}));
    opt.distribution = Distribution::IrwinHall;
    const std::vector<double> z0 = mko::perturbed_vector(f.set, f.x0, opt, 0);
    EXPECT_TRUE(same_bits(
        z0, {0.00963440572889103, -0.00972408266796377, 5.965772408901302, 99.29548709859037}));
}

TEST(VariantGenerator, SameSeedGivesIdenticalBatches) {
    const Fixture f;
    PerturbationOptions opt;
    opt.seed = 7;
    const SystemBatchHost a = mko::make_perturbed_batch(f.set, f.base, 0, f.x0, 50, opt);
    const SystemBatchHost b = mko::make_perturbed_batch(f.set, f.base, 0, f.x0, 50, opt);
    EXPECT_TRUE(a == b);
    ASSERT_EQ(a.variants(), 50);
    // Variante j hängt nicht von der Zahl der Varianten ab
    const SystemBatchHost c = mko::make_perturbed_batch(f.set, f.base, 0, f.x0, 5, opt);
    for (int v = 0; v < 5; ++v) {
        EXPECT_TRUE(same_bits(f.set.extract(a, v), f.set.extract(c, v)));
    }
    // anderer Startwert, andere Varianten; alle Varianten verschieden
    opt.seed = 8;
    const SystemBatchHost d = mko::make_perturbed_batch(f.set, f.base, 0, f.x0, 50, opt);
    for (int v = 0; v < 50; ++v) {
        EXPECT_NE(f.set.extract(a, v), f.set.extract(d, v));
        if (v > 0) {
            EXPECT_NE(f.set.extract(a, v), f.set.extract(a, v - 1));
        }
        expect_only_variables_differ(f.set, a, v, f.base, 0);
    }
    // gemeinsame Felder aus der Vorlage
    EXPECT_EQ(a.reference_wavelength_nm, f.base.reference_wavelength_nm);
    EXPECT_EQ(a.wavelength_nm(0), f.base.wavelength_nm(0));
}

TEST(VariantGenerator, PerturbationScaleAndBounds) {
    const Fixture f;
    PerturbationOptions opt;
    opt.seed = 3;
    opt.curvature = {0.0, 1e-3};       // absolut
    opt.thickness = {0.1, 0.0};        // relativ
    opt.working_distance = {0.0, 0.0}; // fest
    const SystemBatchHost b = mko::make_perturbed_batch(f.set, f.base, 0, f.x0, 200, opt);
    for (int v = 0; v < b.variants(); ++v) {
        const std::vector<double> y = f.set.extract(b, v);
        EXPECT_LE(std::abs(y[0] - f.x0[0]), 1e-3 * (1 + 1e-12));
        EXPECT_LE(std::abs(y[2] - f.x0[2]), 0.6 * (1 + 1e-12));
        EXPECT_EQ(y[3], f.x0[3]);
        EXPECT_TRUE(f.set.feasible(y));
    }
    // große Störung: Projektion auf die Schranken, einige Werte genau auf der Schranke
    opt.curvature = {10.0, 0.0};
    opt.thickness = {10.0, 0.0};
    opt.working_distance = {10.0, 0.0};
    const SystemBatchHost big = mko::make_perturbed_batch(f.set, f.base, 0, f.x0, 100, opt);
    int at_bound = 0;
    for (int v = 0; v < big.variants(); ++v) {
        const std::vector<double> y = f.set.extract(big, v);
        EXPECT_TRUE(f.set.feasible(y));
        for (int k = 0; k < f.set.size(); ++k) {
            at_bound +=
                y[static_cast<std::size_t>(k)] == f.set.lower()[static_cast<std::size_t>(k)] ||
                y[static_cast<std::size_t>(k)] == f.set.upper()[static_cast<std::size_t>(k)];
        }
    }
    EXPECT_GT(at_bound, 100);
}

TEST(VariantGenerator, FirstIsBaseAndWriteIntoExistingBatch) {
    const Fixture f;
    PerturbationOptions opt;
    opt.seed = 11;
    opt.first_is_base = true;
    SystemBatchHost b = f.base;
    b.resize(6);
    // Varianten 2 … 5 aus Variante 0 desselben Batches (Vorlage und Ziel sind ein Objekt)
    mko::write_perturbed_variants(f.set, b, 0, f.x0, opt, b, 2, 4);
    EXPECT_TRUE(same_bits(f.set.extract(b, 2), f.x0));
    EXPECT_TRUE(same_bits(f.set.extract(b, 3), mko::perturbed_vector(f.set, f.x0, opt, 1)));
    for (int v = 2; v < 6; ++v) {
        expect_only_variables_differ(f.set, b, v, f.base, 0);
    }
    // Variante 1 blieb unberührt (leer)
    EXPECT_EQ(b.pupil_radius(1), 0.0);
    EXPECT_THROW(mko::write_perturbed_variants(f.set, b, 0, f.x0, opt, b, 3, 4), std::out_of_range);
    EXPECT_THROW(mko::write_perturbed_variants(f.set, b, 0, f.x0, opt, b, -1, 1),
                 std::out_of_range);
}

// Irwin-Hall mit 12 Summanden: Mittel 0, Varianz 1, begrenzt.
TEST(VariantGenerator, IrwinHallMoments) {
    const Fixture f;
    PerturbationOptions opt;
    opt.seed = 1;
    opt.distribution = Distribution::IrwinHall;
    opt.curvature = {0.0, 1.0}; // ξ direkt, die Schranken ±0,05 stören hier
    opt.thickness = {0.0, 1.0};
    opt.working_distance = {0.0, 1.0};
    double sum = 0.0;
    double sum2 = 0.0;
    double max_abs = 0.0;
    const int n = 20000;
    for (int j = 0; j < n; ++j) {
        const std::vector<double> y = mko::perturbed_vector(f.set, f.x0, opt, j);
        const double xi = y[3] - f.x0[3]; // w: Schranken [50, 150] weit genug
        sum += xi;
        sum2 += xi * xi;
        max_abs = std::max(max_abs, std::abs(xi));
    }
    const double mean = sum / n;
    EXPECT_NEAR(mean, 0.0, 0.03);
    EXPECT_NEAR(sum2 / n - mean * mean, 1.0, 0.05);
    EXPECT_LT(max_abs, 6.0);
}

TEST(VariantGenerator, RejectsInvalidInput) {
    const Fixture f;
    PerturbationOptions opt;
    std::vector<double> x = f.x0;
    x[2] = 1.0; // unter t_min = 2
    EXPECT_THROW((void)mko::perturbed_vector(f.set, x, opt, 0), std::invalid_argument);
    EXPECT_THROW((void)mko::fd_pattern(f.set, x, FdScheme::Central), std::invalid_argument);
    x = f.x0;
    x.pop_back();
    EXPECT_THROW((void)mko::perturbed_vector(f.set, x, opt, 0), std::invalid_argument);
    opt.thickness.relative = -1.0;
    EXPECT_THROW((void)mko::perturbed_vector(f.set, f.x0, opt, 0), std::invalid_argument);
    opt.thickness.relative = 0.0;
    EXPECT_THROW((void)mko::make_perturbed_batch(f.set, f.base, 0, f.x0, -1, opt),
                 std::invalid_argument);
    EXPECT_THROW((void)mko::make_perturbed_batch(f.set, f.base, 1, f.x0, 2, opt),
                 std::out_of_range);
}

// ---- Finite Differenzen ------------------------------------------------------------------

TEST(VariantGenerator, FdStepFactor) {
    EXPECT_NEAR(mko::kFdStepFactor, std::cbrt(DBL_EPSILON), 1e-15 * mko::kFdStepFactor);
    EXPECT_EQ(mko::fd_step(2.0, 1.0), 2.0 * mko::kFdStepFactor);
    EXPECT_EQ(mko::fd_step(-0.001, 0.01), 0.01 * mko::kFdStepFactor);
    EXPECT_EQ(mko::fd_variant_count(FdScheme::Central, 4), 9);
    EXPECT_EQ(mko::fd_variant_count(FdScheme::Forward, 4), 5);
    EXPECT_EQ(mko::fd_plus_variant(FdScheme::Central, 2), 5);
    EXPECT_EQ(mko::fd_minus_variant(2), 6);
    EXPECT_EQ(mko::fd_plus_variant(FdScheme::Forward, 2), 3);
}

TEST(VariantGenerator, CentralPattern) {
    const Fixture f;
    const mko::FdBatch fd = mko::make_fd_batch(f.set, f.base, 0, f.x0, FdScheme::Central);
    const int p = f.set.size();
    ASSERT_EQ(fd.batch.variants(), 2 * p + 1);
    EXPECT_TRUE(same_bits(f.set.extract(fd.batch, 0), f.x0));
    for (int k = 0; k < p; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        const double h = mko::kFdStepFactor * std::max(std::abs(f.x0[kk]), f.set[k].typical);
        EXPECT_EQ(fd.pattern.nominal[kk], h);
        EXPECT_EQ(fd.pattern.stencil[kk], mko::FdStencil::Central);
        EXPECT_EQ(fd.pattern.point_plus[kk], f.x0[kk] + h);
        EXPECT_EQ(fd.pattern.point_minus[kk], f.x0[kk] - h);
        EXPECT_EQ(fd.pattern.step_plus[kk], fd.pattern.point_plus[kk] - f.x0[kk]);
        EXPECT_EQ(fd.pattern.step_minus[kk], fd.pattern.point_minus[kk] - f.x0[kk]);
        EXPECT_NEAR(fd.pattern.step_plus[kk], h,
                    std::abs(f.x0[kk]) * DBL_EPSILON); // Rundung von x₀ + h
        std::vector<double> plus = f.x0;
        plus[kk] = fd.pattern.point_plus[kk];
        std::vector<double> minus = f.x0;
        minus[kk] = fd.pattern.point_minus[kk];
        EXPECT_TRUE(same_bits(f.set.extract(fd.batch, 2 * k + 1), plus)) << k;
        EXPECT_TRUE(same_bits(f.set.extract(fd.batch, 2 * k + 2), minus)) << k;
    }
    for (int v = 0; v < fd.batch.variants(); ++v) {
        expect_only_variables_differ(f.set, fd.batch, v, f.base, 0);
    }
    // c[1] = 0,00967 < x_typ = 0,01: h aus x_typ; t[1] = 6 > x_typ = 1: h aus |x|
    EXPECT_EQ(fd.pattern.nominal[0], 0.01 * mko::kFdStepFactor);
    EXPECT_EQ(fd.pattern.nominal[2], 6.0 * mko::kFdStepFactor);
    EXPECT_EQ(fd.pattern.nominal[3], 99.05 * mko::kFdStepFactor);
}

TEST(VariantGenerator, ForwardPattern) {
    const Fixture f;
    const mko::FdBatch fd = mko::make_fd_batch(f.set, f.base, 0, f.x0, FdScheme::Forward);
    const int p = f.set.size();
    ASSERT_EQ(fd.batch.variants(), p + 1);
    EXPECT_TRUE(fd.pattern.step_minus.empty());
    EXPECT_TRUE(fd.pattern.point_minus.empty());
    EXPECT_TRUE(same_bits(f.set.extract(fd.batch, 0), f.x0));
    for (int k = 0; k < p; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        std::vector<double> plus = f.x0;
        plus[kk] = f.x0[kk] + fd.pattern.nominal[kk];
        EXPECT_TRUE(same_bits(f.set.extract(fd.batch, k + 1), plus)) << k;
    }
}

// Schritte an den Schranken (M3-T2): zentral, wenn x₀ ± h zulässig ist; sonst einseitig mit
// x₀ + σh und x₀ + 2σh zur Seite mit mehr Platz; nur bei sehr engem Bereich gekürzt.
TEST(VariantGenerator, FdStepsAtBounds) {
    using mko::FdStencil;
    const Fixture f;
    std::vector<double> x = f.x0;
    x[0] = 0.05;                                   // c[1] auf der oberen Schranke
    x[1] = -0.05;                                  // c[2] auf der unteren Schranke
    x[2] = 12.0 - 0.5 * 12.0 * mko::kFdStepFactor; // t[1] knapp unter t_max = 12
    const mko::FdPattern c = mko::fd_pattern(f.set, x, FdScheme::Central);
    ASSERT_EQ(c.stencil.size(), x.size());
    EXPECT_EQ(c.stencil[0], FdStencil::OneSided);
    EXPECT_EQ(c.point_plus[0], 0.05 - c.nominal[0]);
    EXPECT_EQ(c.point_minus[0], 0.05 - 2.0 * c.nominal[0]);
    EXPECT_EQ(c.step_plus[0], c.point_plus[0] - 0.05);
    EXPECT_EQ(c.step_minus[0], c.point_minus[0] - 0.05);
    EXPECT_EQ(c.stencil[1], FdStencil::OneSided);
    EXPECT_EQ(c.point_plus[1], -0.05 + c.nominal[1]);
    EXPECT_EQ(c.point_minus[1], -0.05 + 2.0 * c.nominal[1]);
    EXPECT_EQ(c.stencil[2], FdStencil::OneSided);
    EXPECT_EQ(c.point_plus[2], x[2] - c.nominal[2]);
    EXPECT_EQ(c.point_minus[2], x[2] - 2.0 * c.nominal[2]);
    EXPECT_EQ(c.stencil[3], FdStencil::Central);
    for (std::size_t k = 0; k < x.size(); ++k) {
        EXPECT_NE(c.step_plus[k], 0.0);
        EXPECT_NE(c.step_minus[k], 0.0);
        EXPECT_NE(c.step_plus[k], c.step_minus[k]);
    }

    const mko::FdPattern fw = mko::fd_pattern(f.set, x, FdScheme::Forward);
    EXPECT_EQ(fw.point_plus[0], 0.05 - fw.nominal[0]); // Rückwärtsschritt an der oberen Schranke
    EXPECT_LT(fw.step_plus[0], 0.0);
    EXPECT_EQ(fw.point_plus[1], -0.05 + fw.nominal[1]);
    EXPECT_EQ(fw.point_plus[2], x[2] - fw.nominal[2]);

    // Schranken enger als h auf beiden Seiten: Forward nimmt die Seite mit mehr Platz.
    mko::SystemDescription sys = f.loaded.system;
    mko::VariableSpec tight;
    tight.type = mko::VariableType::Thickness;
    tight.selection = mko::SurfaceSelection::One;
    tight.surfaces = {1};
    tight.min = 6.0 - 1e-6;
    tight.max = 6.0 + 2e-6;
    sys.variables = mko::Variables{{tight}};
    const VariableSet tight_set(sys, f.base.topology());
    const std::vector<double> t0 = {6.0};
    const mko::FdPattern tf = mko::fd_pattern(tight_set, t0, FdScheme::Forward);
    EXPECT_EQ(tf.point_plus[0], 6.0 + 2e-6);
    EXPECT_EQ(tf.stencil[0], mko::FdStencil::Forward);
    // Auf der oberen Schranke eines engen Bereichs: Schritt dorthin 0 (einseitige Differenz).
    const std::vector<double> t1 = {6.0 + 2e-6};
    const mko::FdPattern te = mko::fd_pattern(tight_set, t1, FdScheme::Central);
    EXPECT_EQ(te.stencil[0], mko::FdStencil::Truncated);
    EXPECT_EQ(te.step_plus[0], 0.0);
    EXPECT_EQ(te.point_minus[0], 6.0 - 1e-6);
    EXPECT_GT(te.step_plus[0] - te.step_minus[0], 0.0);
    // Zentral: Bereich (3e-6) kürzer als 3h, also gekürzt bis an beide Schranken.
    const mko::FdPattern tc = mko::fd_pattern(tight_set, t0, FdScheme::Central);
    EXPECT_EQ(tc.stencil[0], mko::FdStencil::Truncated);
    EXPECT_EQ(tc.point_plus[0], 6.0 + 2e-6);
    EXPECT_EQ(tc.point_minus[0], 6.0 - 1e-6);
}

// Mehrere Läufe in einem Batch (gebündelter LM): Muster an beliebiger Stelle.
TEST(VariantGenerator, FdPatternsAtOffsets) {
    const Fixture f;
    PerturbationOptions opt;
    opt.seed = 5;
    const std::vector<double> x1 = mko::perturbed_vector(f.set, f.x0, opt, 0);
    const int count = mko::fd_variant_count(FdScheme::Central, f.set.size());
    SystemBatchHost b = f.base;
    b.resize(2 * count);
    const mko::FdPattern p0 =
        mko::write_fd_variants(f.set, f.base, 0, f.x0, FdScheme::Central, b, 0);
    const mko::FdPattern p1 =
        mko::write_fd_variants(f.set, f.base, 0, x1, FdScheme::Central, b, count);
    const mko::FdBatch single = mko::make_fd_batch(f.set, f.base, 0, x1);
    EXPECT_EQ(p1.step_plus, single.pattern.step_plus);
    for (int v = 0; v < count; ++v) {
        EXPECT_TRUE(same_bits(f.set.extract(b, count + v), f.set.extract(single.batch, v)));
    }
    EXPECT_TRUE(same_bits(f.set.extract(b, 0), f.x0));
    EXPECT_EQ(p0.step_plus.size(), static_cast<std::size_t>(f.set.size()));
    EXPECT_THROW(
        (void)mko::write_fd_variants(f.set, f.base, 0, x1, FdScheme::Central, b, count + 1),
        std::out_of_range);
}
