// Zahlentyp df64 (M7-T1) auf der CPU: fehlerfreie Transformationen, Umwandlungen,
// Genauigkeit der Grundoperationen, Wurzel, sin/cos, Vergleiche, Randfälle und Wertebereich,
// Trace in df64 gegen FP64 über dieselben Funktionen wie der Kernel und der Wertebereich der
// Zwischenwerte im Tracer (docs/numerik.md, Abschnitt 12).

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/math/df64.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_table.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using mko::df64;
using mko::PupilMode;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::test::kDf64AddRelative;
using mko::test::kDf64DivRelative;
using mko::test::kDf64MulRelative;
using mko::test::kDf64SqrtRelative;
using mko::test::kDf64Trig;
using mko::test::kDf64VsFp64;

namespace {

constexpr double kU2 = 0x1p-48; // u² für df64 (u = 2⁻²⁴)

// Wert, den der Compiler nicht als Konstante falten kann (sonst warnt MSVC bei absichtlichem
// Über- und Unterlauf oder Division durch 0, C4756/C4723).
double runtime(double x) {
    volatile double v = x;
    return v;
}

double to_d(df64 x) {
    return static_cast<double>(x);
}

bool same_bits(df64 a, df64 b) {
    return std::memcmp(&a, &b, sizeof(df64)) == 0;
}

// Zufällige double mit Betrag 10^[-e, e] und zufälligem Vorzeichen.
double random_value(std::mt19937_64& rng, double decades) {
    std::uniform_real_distribution<double> m(1.0, 10.0);
    std::uniform_real_distribution<double> e(-decades, decades);
    std::bernoulli_distribution sign(0.5);
    const double x = m(rng) * std::pow(10.0, std::floor(e(rng)));
    return sign(rng) ? -x : x;
}

// Normiert: |lo| <= ulp(hi)/2.
bool normalized(df64 x) {
    if (x.hi == 0.0f) {
        return x.lo == 0.0f;
    }
    const float up = std::nextafter(x.hi, std::numeric_limits<float>::infinity());
    const float ulp = std::abs(up - x.hi);
    return std::abs(x.lo) <= 0.5f * ulp;
}

// Größter relativer Fehler einer Operation in Einheiten von u².
struct ErrorStat {
    double max_u2 = 0.0;
    std::size_t violations = 0;
    void add(double got, double ref, double bound) {
        const double err = std::abs(got - ref);
        const double scale = std::abs(ref);
        if (scale > 0.0) {
            max_u2 = std::max(max_u2, err / scale / kU2);
        }
        if (!(err <= bound * scale)) {
            ++violations;
        }
    }
};

} // namespace

TEST(Df64, TypeProperties) {
    static_assert(sizeof(df64) == 8);
    static_assert(std::is_trivially_default_constructible_v<df64>);
    static_assert(std::is_trivially_copyable_v<df64>);
    EXPECT_STREQ(mko::cuda::to_string(mko::cuda::Precision::Fp64), "FP64");
    EXPECT_STREQ(mko::cuda::to_string(mko::cuda::Precision::Df64), "DF64");
    EXPECT_NO_THROW(mko::cuda::check_precision(mko::cuda::Precision::Df64));
    EXPECT_THROW(mko::cuda::check_precision(static_cast<mko::cuda::Precision>(7)),
                 std::invalid_argument);
}

// two_sum, fast_two_sum und two_prod sind exakt (Prüfung in double: die exakte Summe zweier
// float mit Exponentendifferenz <= 20 und das exakte Produkt (48 Bit) passen in 53 Bit).
TEST(Df64, ErrorFreeTransformations) {
    using namespace mko::df64_detail;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<float> m(-1.0f, 1.0f);
    std::uniform_int_distribution<int> e(-10, 10);
    int failures = 0;
    for (int i = 0; i < 200000; ++i) {
        const float a = std::ldexp(m(rng), e(rng));
        const float b = std::ldexp(m(rng), e(rng));
        float s, err;
        two_sum(a, b, s, err);
        failures += (static_cast<double>(s) + static_cast<double>(err) !=
                     static_cast<double>(a) + static_cast<double>(b))
                        ? 1
                        : 0;
        failures += s != a + b ? 1 : 0;
        const float big = std::abs(a) >= std::abs(b) ? a : b;
        const float small = std::abs(a) >= std::abs(b) ? b : a;
        fast_two_sum(big, small, s, err);
        failures += (static_cast<double>(s) + static_cast<double>(err) !=
                     static_cast<double>(big) + static_cast<double>(small))
                        ? 1
                        : 0;
        float p;
        two_prod(a, b, p, err);
        failures += (static_cast<double>(p) + static_cast<double>(err) !=
                     static_cast<double>(a) * static_cast<double>(b))
                        ? 1
                        : 0;
    }
    EXPECT_EQ(failures, 0);
}

// Konstruktion aus double: hi = float(x), lo = float(x − hi), normiert, relativer Fehler
// <= 2⁻⁴⁹; aus Ganzzahlen exakt; Rückwandlung nach double exakt für jedes df64 aus double.
TEST(Df64, Conversions) {
    std::mt19937_64 rng(12);
    double max_rel = 0.0;
    for (int i = 0; i < 100000; ++i) {
        const double x = random_value(rng, 30.0);
        const df64 d(x);
        ASSERT_EQ(d.hi, static_cast<float>(x));
        ASSERT_TRUE(normalized(d)) << x;
        const double back = to_d(d);
        max_rel = std::max(max_rel, std::abs(back - x) / std::abs(x));
        // hi + lo umfasst höchstens 53 Bit: Rückwandlung und erneute Umwandlung sind exakt.
        ASSERT_TRUE(same_bits(df64(back), d));
    }
    EXPECT_LE(max_rel, 0x1p-49);
    std::printf("[ Info     ] df64(double): max. relativer Fehler %.3e (2^-49 = %.3e)\n", max_rel,
                0x1p-49);
    for (const long long i : {0LL, 1LL, -1LL, 7LL, 16777217LL, -16777219LL, 2147483647LL,
                              -2147483647LL - 1, 123456789012345LL}) {
        EXPECT_EQ(to_d(df64(i)), static_cast<double>(i)) << i;
    }
    EXPECT_EQ(to_d(df64(3)), 3.0);
    EXPECT_EQ(df64(0.5f).hi, 0.5f);
    EXPECT_EQ(df64(0.5f).lo, 0.0f);
    // π aus dem double-Literal: Fehler gegen π (double) <= 2⁻⁴⁹ relativ
    const double pi = 3.14159265358979323846;
    EXPECT_LE(std::abs(to_d(df64(pi)) - pi), 0x1p-49 * pi);
    // Wertebereich von float: |x| > FLT_MAX ergibt ±∞, NaN bleibt NaN
    EXPECT_TRUE(std::isinf(df64(runtime(1e39)).hi));
    EXPECT_EQ(df64(runtime(1e39)).lo, 0.0f);
    EXPECT_TRUE(std::isinf(df64(-std::numeric_limits<double>::infinity()).hi));
    EXPECT_TRUE(std::isnan(df64(std::numeric_limits<double>::quiet_NaN()).hi));
    EXPECT_TRUE(std::signbit(df64(-0.0).hi));
}

// Abnahme (Typ): Grundoperationen gegen eine FP64-Referenz innerhalb der Schranken aus der
// Literatur (tests/tolerances.hpp), auch bei Auslöschung (a ≈ −b) und über 16 Dekaden.
TEST(Df64, ArithmeticAccuracy) {
    std::mt19937_64 rng(13);
    std::uniform_real_distribution<double> tiny(-15.0, -1.0);
    ErrorStat add, sub, mul, mulf, div, cancel;
    for (int i = 0; i < 200000; ++i) {
        const df64 a(random_value(rng, 8.0));
        const df64 b(random_value(rng, 8.0));
        const double ad = to_d(a);
        const double bd = to_d(b);
        add.add(to_d(a + b), ad + bd, kDf64AddRelative);
        sub.add(to_d(a - b), ad - bd, kDf64AddRelative);
        mul.add(to_d(a * b), ad * bd, kDf64MulRelative);
        mulf.add(to_d(a * b.hi), ad * static_cast<double>(b.hi), kDf64MulRelative);
        div.add(to_d(a / b), ad / bd, kDf64DivRelative);
        // Auslöschung: c = −a (1 + δ), δ = 10^[-15, -1]; a + c in double exakt (Sterbenz)
        const df64 c(-ad * (1.0 + std::pow(10.0, tiny(rng))));
        cancel.add(to_d(a + c), ad + to_d(c), kDf64AddRelative);
        ASSERT_TRUE(normalized(a + b) && normalized(a * b) && normalized(a / b));
    }
    EXPECT_EQ(add.violations, 0u);
    EXPECT_EQ(sub.violations, 0u);
    EXPECT_EQ(mul.violations, 0u);
    EXPECT_EQ(mulf.violations, 0u);
    EXPECT_EQ(div.violations, 0u);
    EXPECT_EQ(cancel.violations, 0u);
    std::printf("[ Info     ] max. relativer Fehler in u^2 = 2^-48: + %.2f, - %.2f, * %.2f, "
                "*float %.2f, / %.2f, Auslöschung %.2f\n",
                add.max_u2, sub.max_u2, mul.max_u2, mulf.max_u2, div.max_u2, cancel.max_u2);
    // Exakte Fälle
    EXPECT_TRUE(same_bits(df64(3) * df64(7), df64(21)));
    EXPECT_TRUE(same_bits(df64(1) / df64(4), df64(0.25)));
    EXPECT_TRUE(same_bits(df64(1.5) - df64(1.5), df64(0.0)));
    df64 acc = df64(1);
    acc += df64(2);
    acc *= df64(3);
    acc -= df64(1);
    acc /= df64(4);
    EXPECT_EQ(to_d(acc), 2.0);
    EXPECT_EQ(to_d(-df64(2.5)), -2.5);
}

// Wurzel: relativer Fehler <= 25/8 u²; Randfälle wie double.
TEST(Df64, SqrtAccuracyAndEdgeCases) {
    std::mt19937_64 rng(14);
    ErrorStat st;
    for (int i = 0; i < 200000; ++i) {
        const df64 a(std::abs(random_value(rng, 20.0)));
        st.add(to_d(mko::msqrt(a)), std::sqrt(to_d(a)), kDf64SqrtRelative);
        ASSERT_TRUE(normalized(mko::msqrt(a)));
    }
    EXPECT_EQ(st.violations, 0u);
    std::printf("[ Info     ] sqrt: max. relativer Fehler %.2f u^2\n", st.max_u2);
    EXPECT_TRUE(same_bits(mko::msqrt(df64(4)), df64(2)));
    EXPECT_TRUE(same_bits(mko::msqrt(df64(0.0)), df64(0.0)));
    EXPECT_TRUE(std::signbit(mko::msqrt(df64(-0.0)).hi)); // √−0 = −0 wie double
    EXPECT_TRUE(std::isnan(mko::msqrt(df64(-1.0)).hi));
    EXPECT_TRUE(std::isnan(mko::msqrt(df64(-1e-30)).hi));
    EXPECT_TRUE(std::isinf(mko::msqrt(df64(std::numeric_limits<double>::infinity())).hi));
    EXPECT_TRUE(std::isnan(mko::msqrt(df64(std::numeric_limits<double>::quiet_NaN())).hi));
    // Division durch 0 ergibt NaN (dokumentierte Abweichung von double)
    EXPECT_TRUE(std::isnan((df64(1) / df64(runtime(0.0))).hi));
}

// sin und cos: absoluter Fehler <= kDf64Trig gegen FP64 für |x| <= 100, exakt für 0, die
// Azimute der Pupillenquadratur und kleine Feldwinkel; NaN außerhalb des Bereichs.
TEST(Df64, SinCosAccuracy) {
    std::mt19937_64 rng(15);
    std::uniform_real_distribution<double> u(-100.0, 100.0);
    double max_err = 0.0;
    auto check = [&](double xd) {
        const df64 x(xd);
        df64 s, c;
        mko::msincos(x, s, c);
        const double ref_x = to_d(x);
        const double es = std::abs(to_d(s) - std::sin(ref_x));
        const double ec = std::abs(to_d(c) - std::cos(ref_x));
        max_err = std::max({max_err, es, ec});
        EXPECT_LE(es, kDf64Trig) << xd;
        EXPECT_LE(ec, kDf64Trig) << xd;
        EXPECT_TRUE(same_bits(mko::msin(x), s));
        EXPECT_TRUE(same_bits(mko::mcos(x), c));
    };
    for (int i = 0; i < 100000; ++i) {
        check(u(rng));
    }
    for (int k = -64; k <= 64; ++k) {
        check(k * 3.14159265358979323846 / 4.0); // Quadrantengrenzen
    }
    for (int m = 1; m <= 16; ++m) {
        for (int j = 0; j < 2 * m; ++j) {
            check((2 * j + 1) * 3.14159265358979323846 / (2 * m));
        }
    }
    for (const double th : {1e-12, 1e-6, 0.0087, 0.02, 0.0349, 0.3}) {
        check(th);
        check(-th);
    }
    std::printf("[ Info     ] sin/cos: max. absoluter Fehler %.3e\n", max_err);
    df64 s, c;
    mko::msincos(df64(0), s, c);
    EXPECT_TRUE(same_bits(s, df64(0)));
    EXPECT_TRUE(same_bits(c, df64(1)));
    EXPECT_TRUE(std::isnan(mko::msin(df64(2e6)).hi));
    EXPECT_TRUE(std::isnan(mko::mcos(df64(std::numeric_limits<double>::infinity())).hi));
    EXPECT_TRUE(std::isnan(mko::msin(df64(std::numeric_limits<double>::quiet_NaN())).hi));
}

// Vergleiche: lexikographisch (hi, lo) für normierte Werte, NaN wie double.
TEST(Df64, Comparisons) {
    const df64 one(1);
    const df64 above = df64::from_parts(1.0f, 1e-9f);
    const df64 below = df64::from_parts(1.0f, -1e-9f);
    EXPECT_TRUE(below < one && one < above && below < above);
    EXPECT_TRUE(above > one && one >= one && one <= one && above >= below);
    EXPECT_FALSE(one < one);
    EXPECT_TRUE(one == df64(1.0) && one != above);
    EXPECT_TRUE(df64(0.0) == df64(-0.0));
    const df64 nan(std::numeric_limits<double>::quiet_NaN());
    EXPECT_FALSE(nan < one || nan > one || nan <= one || nan >= one || nan == nan);
    EXPECT_TRUE(nan != nan);
    EXPECT_FALSE(nan > df64(0)); // Tracer: !(denom > 0) meldet NaN als Versagen
    std::mt19937_64 rng(16);
    for (int i = 0; i < 10000; ++i) {
        // Abstand 2⁻⁴⁶ relativ: unterscheidbar in df64 (benachbarte double wären es nicht,
        // df64 hat ca. 48 statt 53 Bit).
        const double a = random_value(rng, 10.0);
        const double b = a * (1.0 + 0x1p-46);
        EXPECT_EQ(df64(a) < df64(b), a < b);
        EXPECT_EQ(df64(b) < df64(a), b < a);
        EXPECT_EQ(df64(a) <= df64(b), a <= b);
    }
}

// Wertebereich von float (dokumentiert in numerik.md §12): Ergebnisse über FLT_MAX ≈ 3,4e38
// werden NaN (nicht ±∞: die fehlerfreien Transformationen bilden ∞ − ∞), volle relative
// Genauigkeit nur bis 2⁻¹⁰⁰ ≈ 7,9e-31 (darunter wird lo subnormal).
TEST(Df64, ValueRange) {
    EXPECT_TRUE(std::isnan((df64(runtime(3e38)) * df64(10)).hi));
    EXPECT_TRUE(std::isnan((df64(runtime(2e38)) + df64(2e38)).hi));
    EXPECT_FALSE(df64(runtime(3e38)) * df64(10) > df64(0)); // Prüfungen im Tracer: Versagen
    for (const double x : {1e-30, 0x1p-100, 1e30, 3e38}) {
        const double rel = std::abs(to_d(df64(x)) - x) / x;
        EXPECT_LE(rel, 0x1p-49) << x;
    }
    // Unterhalb von 2⁻¹⁰⁰ geht Genauigkeit verloren (lo subnormal), der Betrag bleibt aber
    // auf ca. 1e-45 genau.
    const double x = 1.2345678901234567e-38;
    EXPECT_GT(std::abs(to_d(df64(x)) - x) / x, 0x1p-49);
    EXPECT_LE(std::abs(to_d(df64(x)) - x), 1e-45);
    EXPECT_EQ(to_d(df64(runtime(1e-30)) * df64(1e-30)), 0.0); // Unterlauf
}

// ---------------------------------------------------------------------------------------------
// Trace in df64 über dieselben Funktionen wie der Kernel (batch_ray_opl mit FP64-Batch und
// Flächentabelle im Rechentyp R, opd_statistics im Rechentyp R).
// ---------------------------------------------------------------------------------------------

namespace {

// Flächentabelle einer Variante im Rechentyp R (wie die Kopie im Shared Memory des Kernels).
template <class R>
struct TypedTable {
    std::vector<R> curvature, thickness, index, ghost_coeff, ghost_index, ghost_radius;
    mko::SurfaceTableView<R> view{};

    explicit TypedTable(const mko::SurfaceTableView<double>& t) {
        const auto conv = [](const double* p, std::size_t n, std::vector<R>& out) {
            out.clear();
            for (std::size_t i = 0; i < n; ++i) {
                out.push_back(R(p[i]));
            }
        };
        const auto S = static_cast<std::size_t>(t.surfaces);
        const auto L = static_cast<std::size_t>(t.wavelengths);
        const auto G = static_cast<std::size_t>(t.ghost_plates);
        const auto K = static_cast<std::size_t>(t.ghost_terms);
        conv(t.curvature, S, curvature);
        conv(t.thickness, S, thickness);
        conv(t.index, S * L, index);
        if (G > 0) {
            conv(t.ghost_coeff, G * K, ghost_coeff);
            conv(t.ghost_index, G * L, ghost_index);
            conv(t.ghost_norm_radius, G, ghost_radius);
        }
        view = mko::SurfaceTableView<R>{t.surfaces,
                                        t.wavelengths,
                                        t.type,
                                        curvature.data(),
                                        thickness.data(),
                                        index.data(),
                                        t.ghost_plates,
                                        t.ghost_terms,
                                        G > 0 ? ghost_coeff.data() : nullptr,
                                        G > 0 ? ghost_index.data() : nullptr,
                                        G > 0 ? ghost_radius.data() : nullptr};
    }
};

template <class R>
mko::PupilQuadrature<R> convert_quadrature(const mko::PupilQuadrature<double>& q) {
    mko::PupilQuadrature<R> r{};
    r.rings = q.rings;
    r.spokes = q.spokes;
    r.mode = q.mode;
    for (int i = 0; i < q.rings; ++i) {
        r.u[i] = R(q.u[i]);
        r.w[i] = R(q.w[i]);
    }
    return r;
}

// OPL, OPD (nach Kolbenabzug, opd_statistics) und Status aller Strahlen der Variante v in der
// Ablage des Layouts, gerechnet im Typ R, als double.
struct TypedRays {
    std::vector<double> opl, opd;
    std::vector<TraceStatus> status;
};

template <class R>
TypedRays typed_rays(const SystemBatchHost& batch, int v, const ResidualLayout& layout) {
    const mko::SystemBatchView<double> view = batch.view();
    const TypedTable<R> table(view.surface_table(v));
    const auto rays = static_cast<std::size_t>(layout.rays_per_variant());
    TypedRays out;
    out.opl.assign(rays, 0.0);
    out.opd.assign(rays, 0.0);
    out.status.assign(rays, TraceStatus::Ok);
    for (int f = 0; f < layout.fields(); ++f) {
        const mko::PupilQuadrature<R> q = convert_quadrature<R>(layout.quadrature(f));
        const int n = mko::sample_count(q);
        const auto center = mko::batch_image_point(view, table.view, v, f);
        std::vector<R> opl(static_cast<std::size_t>(n)), weight(opl.size()), opd(opl.size());
        std::vector<TraceStatus> st(opl.size());
        for (int l = 0; l < layout.wavelengths(); ++l) {
            for (int k = 0; k < n; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                const mko::PupilSample<R> s = mko::sample(q, k);
                weight[ku] = s.weight;
                st[ku] = mko::batch_ray_opl(view, table.view, l, v, f, l, s, center, &opl[ku]);
            }
            mko::opd_statistics(opl.data(), weight.data(), st.data(), n, opd.data());
            const auto o = static_cast<std::size_t>(layout.offset(f, l));
            for (std::size_t k = 0; k < opl.size(); ++k) {
                out.opl[o + k] = static_cast<double>(opl[k]);
                out.opd[o + k] = static_cast<double>(opd[k]);
                out.status[o + k] = st[k];
            }
        }
    }
    return out;
}

// Testsysteme wie in den GPU-Tests (gestörte Varianten, Geisterplatten, Punktquelle, 64
// Flächen); Variantenzahl klein, weil die CPU sie Strahl für Strahl rechnet.
struct NamedBatch {
    std::string name;
    SystemBatchHost batch;
};

std::vector<NamedBatch> cpu_test_batches(int variants) {
    std::vector<NamedBatch> out;
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    out.push_back({"einzellinse_bk7", mko::test::perturbed_batch(lens, variants, 1001)});
    out.push_back(
        {"testsystem_mehrlinser",
         mko::test::perturbed_batch(mko::test::load_residual_system("testsystem_mehrlinser.json"),
                                    variants, 2002)});
    out.push_back({"planplatte_bk7",
                   mko::test::perturbed_batch(
                       mko::test::load_residual_system("planplatte_bk7.json"), variants, 3003)});
    out.push_back(
        {"einzellinse_geisterplatte",
         mko::test::perturbed_batch(
             mko::test::load_residual_system("einzellinse_geisterplatte.json"), variants, 8008)});
    mko::SystemDescription p = lens.system;
    p.source.type = mko::SourceType::Point;
    p.source.field_angles_deg.clear();
    p.source.field_weights.clear();
    p.source.numerical_aperture = 0.05;
    p.source.point = {1.0, 2.0, -300.0};
    p.surfaces.back().thickness = 150.0;
    out.push_back({"punktquelle",
                   mko::test::perturbed_batch(
                       mko::prepare_system(p, std::filesystem::path(mko::test::kResidualSystems)),
                       variants, 4004)});
    out.push_back({"64 Flächen", mko::test::many_surface_batch(std::max(2, variants / 10), 5005)});
    return out;
}

} // namespace

// Abnahme M7-T1 auf der CPU (auch in der CI): Trace, Referenzsphäre und Kolbenabzug in df64
// über dieselben Funktionen wie der Kernel gegen FP64: Status gleich, |ΔOPL| und |ΔOPD|
// <= kDf64VsFp64 für alle gültigen Strahlen. Der Pfad mit R = double ist bitgleich zur
// CPU-Referenz residuals_cpu (die verallgemeinerten Batch-Funktionen ändern FP64 nicht).
TEST(Df64, CpuTraceMatchesFp64) {
    for (const NamedBatch& nb : cpu_test_batches(150)) {
        SCOPED_TRACE(nb.name);
        const ResidualLayout layout(nb.batch.topology());
        double max_opl = 0.0;
        double max_opd = 0.0;
        std::size_t rays = 0;
        std::size_t failed = 0;
        std::size_t status_mismatch = 0;
        for (int v = 0; v < nb.batch.variants(); ++v) {
            const TypedRays fp = typed_rays<double>(nb.batch, v, layout);
            const TypedRays df = typed_rays<df64>(nb.batch, v, layout);
            const mko::ResidualResult ref = mko::residuals_cpu(nb.batch, v, layout);
            ASSERT_EQ(std::memcmp(fp.opd.data(), ref.opd.data(), fp.opd.size() * sizeof(double)),
                      0);
            for (std::size_t i = 0; i < fp.opl.size(); ++i) {
                ++rays;
                if (fp.status[i] != df.status[i]) {
                    ++status_mismatch;
                    continue;
                }
                if (fp.status[i] != TraceStatus::Ok) {
                    ++failed;
                    continue;
                }
                max_opl = std::max(max_opl, std::abs(df.opl[i] - fp.opl[i]));
                max_opd = std::max(max_opd, std::abs(df.opd[i] - fp.opd[i]));
            }
        }
        EXPECT_EQ(status_mismatch, 0u);
        EXPECT_LE(max_opl, kDf64VsFp64);
        EXPECT_LE(max_opd, kDf64VsFp64);
        std::printf("[ Info     ] %-26s %7zu Strahlen, %6zu versagt: max |dOPL| = %.3e mm, "
                    "max |dOPD| = %.3e mm\n",
                    nb.name.c_str(), rays, failed, max_opl, max_opd);
    }
}

// ---------------------------------------------------------------------------------------------
// Wertebereich der Zwischenwerte: Zahlentyp, der jede Operation in FP64 rechnet und den
// größten und kleinsten (≠ 0) Betrag aller Zwischenergebnisse festhält.
// ---------------------------------------------------------------------------------------------

namespace {

struct RangeStats {
    double max_abs = 0.0;
    double min_abs = std::numeric_limits<double>::infinity();
    bool non_finite = false;
    void add(double x) {
        const double a = std::abs(x);
        if (!std::isfinite(a)) {
            non_finite = true;
            return;
        }
        max_abs = std::max(max_abs, a);
        if (a > 0.0) {
            min_abs = std::min(min_abs, a);
        }
    }
    void merge(const RangeStats& o) {
        max_abs = std::max(max_abs, o.max_abs);
        min_abs = std::min(min_abs, o.min_abs);
        non_finite = non_finite || o.non_finite;
    }
};

RangeStats g_range; // Zwischenwerte des laufenden Strahls (Tests laufen in einem Thread)

struct Tracked {
    double v;
    Tracked() = default;
    explicit Tracked(double x) : v(x) { g_range.add(x); }
    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    explicit Tracked(I i) : v(static_cast<double>(i)) {}
    explicit operator double() const { return v; }
};

Tracked operator-(Tracked a) {
    return Tracked(-a.v);
}
Tracked operator+(Tracked a, Tracked b) {
    return Tracked(a.v + b.v);
}
Tracked operator-(Tracked a, Tracked b) {
    return Tracked(a.v - b.v);
}
Tracked operator*(Tracked a, Tracked b) {
    return Tracked(a.v * b.v);
}
Tracked operator/(Tracked a, Tracked b) {
    return Tracked(a.v / b.v);
}
Tracked& operator+=(Tracked& a, Tracked b) {
    return a = a + b;
}
Tracked& operator-=(Tracked& a, Tracked b) {
    return a = a - b;
}
bool operator<(Tracked a, Tracked b) {
    return a.v < b.v;
}
bool operator>(Tracked a, Tracked b) {
    return a.v > b.v;
}
bool operator>=(Tracked a, Tracked b) {
    return a.v >= b.v;
}
Tracked msqrt(Tracked a) {
    return Tracked(std::sqrt(a.v));
}
Tracked msin(Tracked a) {
    return Tracked(std::sin(a.v));
}
Tracked mcos(Tracked a) {
    return Tracked(std::cos(a.v));
}

} // namespace

// Der Wertebereich von float (|x| < 3,4e38, volle df64-Genauigkeit ab 2⁻¹⁰⁰ ≈ 7,9e-31) reicht für
// alle Zwischenwerte des Trace, der Referenzsphäre und der Residuen gültiger Strahlen der
// Testsysteme (gestörte Varianten wie in den GPU-Tests, Geisterplatten, Punktquelle, 64 Flächen).
// Versagte Strahlen werden getrennt ausgewiesen (ihr Status hängt nicht von Werten außerhalb ab,
// solange die Prüfgrößen endlich sind).
TEST(Df64, TracerIntermediateValuesStayInFloatRange) {
    RangeStats valid_all;    // Trace und Referenzsphäre gültiger Strahlen, Eingaben
    RangeStats residual_all; // Kolbenabzug, OPD, Residuen und Quadrate
    RangeStats failed_all;
    for (const NamedBatch& nb : cpu_test_batches(150)) {
        const ResidualLayout layout(nb.batch.topology());
        const mko::SystemBatchView<double> view = nb.batch.view();
        RangeStats valid;
        RangeStats residual;
        RangeStats failed;
        for (int v = 0; v < nb.batch.variants(); ++v) {
            g_range = RangeStats{};
            const TypedTable<Tracked> table(view.surface_table(v));
            RangeStats setup = g_range;
            for (int f = 0; f < layout.fields(); ++f) {
                const mko::PupilQuadrature<Tracked> q =
                    convert_quadrature<Tracked>(layout.quadrature(f));
                const int n = mko::sample_count(q);
                g_range = RangeStats{};
                const auto center = mko::batch_image_point(view, table.view, v, f);
                setup.merge(g_range);
                for (int l = 0; l < layout.wavelengths(); ++l) {
                    std::vector<Tracked> opl(static_cast<std::size_t>(n)), w(opl.size()),
                        opd(opl.size());
                    std::vector<TraceStatus> st(opl.size());
                    for (int k = 0; k < n; ++k) {
                        const auto ku = static_cast<std::size_t>(k);
                        g_range = RangeStats{};
                        const mko::PupilSample<Tracked> s = mko::sample(q, k);
                        w[ku] = s.weight;
                        st[ku] =
                            mko::batch_ray_opl(view, table.view, l, v, f, l, s, center, &opl[ku]);
                        (st[ku] == TraceStatus::Ok ? valid : failed).merge(g_range);
                    }
                    g_range = RangeStats{};
                    mko::opd_statistics(opl.data(), w.data(), st.data(), n, opd.data());
                    const Tracked scale(mko::residual_scale(view, f, l));
                    for (int k = 0; k < n; ++k) {
                        const auto ku = static_cast<std::size_t>(k);
                        const Tracked r = mko::ray_residual(scale, mko::sample(q, k), opd[ku]);
                        (void)(r * r); // Quadrat im Kernel MERIT
                    }
                    residual.merge(g_range);
                }
            }
            valid.merge(setup);
        }
        std::printf("[ Info     ] %-26s Trace: |x| in [%.3e, %.3e], Residuen: |x| in [%.3e, "
                    "%.3e], versagte Strahlen: max %.3e%s\n",
                    nb.name.c_str(), valid.min_abs, valid.max_abs, residual.min_abs,
                    residual.max_abs, failed.max_abs, failed.non_finite ? " (nicht endlich)" : "");
        valid_all.merge(valid);
        residual_all.merge(residual);
        failed_all.merge(failed);
    }
    EXPECT_FALSE(valid_all.non_finite);
    EXPECT_FALSE(residual_all.non_finite);
    // Auch Quadrate und Produkte zweier Zwischenwerte bleiben unter FLT_MAX ≈ 3,4e38.
    EXPECT_LT(valid_all.max_abs, 1e18);
    EXPECT_LT(residual_all.max_abs, 1e18);
    // Kein Zwischenwert des Trace unterhalb des Bereichs voller df64-Genauigkeit. Im Kolbenabzug
    // dürfen kleinere Werte vorkommen: Quadrate von OPD im Bereich des Rundungsrauschens (z. B.
    // Planplatte, OPD ≈ 1e-15 mm); ihr absoluter Fehler (≈ 1e-45) ist bedeutungslos.
    EXPECT_GT(valid_all.min_abs, 0x1p-100);
    std::printf("[ Info     ] alle Systeme: Trace |x| in [%.3e, %.3e], Residuen |x| in [%.3e, "
                "%.3e], versagte Strahlen max %.3e\n",
                valid_all.min_abs, valid_all.max_abs, residual_all.min_abs, residual_all.max_abs,
                failed_all.max_abs);
}
