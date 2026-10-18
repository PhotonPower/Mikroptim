// Zahlentyp df64 auf der GPU (M7-T1, nur mit MIKROPTIM_ENABLE_CUDA): Bitgleichheit der
// df64-Operationen zwischen Host und Gerät, Abnahme df64 gegen FP64 über die TraceEngine
// (Modi RESIDUALS und MERIT) für die Szenarien der Testsuite, Determinismus, Verhalten an der
// Versagensgrenze und Durchsatz (docs/numerik.md, Abschnitt 12; docs/benchmarks.md, "df64").

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/df64_check.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/math/df64.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using mko::BatchTopology;
using mko::df64;
using mko::MeritResult;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::EngineOptions;
using mko::cuda::Precision;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
using mko::test::kDf64VsFp64;
using mko::test::kDf64VsFp64Relative;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool identical(const ResidualResult& a, const ResidualResult& b) {
    return a.variants == b.variants && a.stride == b.stride &&
           same_bytes(a.residuals, b.residuals) && same_bytes(a.opd, b.opd) &&
           same_bytes(a.status, b.status) && same_bytes(a.extrema.values, b.extrema.values);
}

bool identical(const MeritResult& a, const MeritResult& b) {
    return a.variants == b.variants && same_bytes(a.merit, b.merit) &&
           same_bytes(a.failed, b.failed) && same_bytes(a.extrema.values, b.extrema.values);
}

TraceSettings settings_for(Precision p, const ResidualSettings& q = {}) {
    TraceSettings s;
    s.quadrature = q;
    s.opd = true;
    s.extrema = true;
    s.precision = p;
    return s;
}

// Schranke pro Strahl (tests/tolerances.hpp): max(kDf64VsFp64, kDf64VsFp64Relative · |x|), also
// rein absolut 1e-9 mm bis |x| = 10 mm.
double df64_bound(double reference) {
    return std::max(kDf64VsFp64, kDf64VsFp64Relative * std::abs(reference));
}

// Dieselbe Schranke für ein Strahlresiduum in Wellen (M4-T1): r = u · OPD mit
// u = s_{f,l} · √W (residual_units), also max(u · kDf64VsFp64, kDf64VsFp64Relative · |r|).
double df64_residual_bound(double unit, double reference) {
    return mko::test::residual_bound(unit, reference, kDf64VsFp64, kDf64VsFp64Relative);
}

// Vergleich df64 gegen FP64 über alle Strahlen und Varianten eines Batches.
struct Df64Comparison {
    std::size_t rays = 0;
    std::size_t failed = 0;          // Strahlen mit Status ≠ Ok in FP64
    std::size_t status_flips = 0;    // Status in df64 ≠ FP64
    double max_residual = 0.0;       // max |r_df64 − r_FP64| / u [mm], Strahlen mit Status Ok
    double max_opd = 0.0;            // max |OPD_df64 − OPD_FP64| [mm]
    double max_opd_cpu = 0.0;        // max |OPD_df64 − OPD_CPU(FP64)| [mm]
    double max_relative = 0.0;       // max |ΔOPD| / |OPD| der Strahlen mit |ΔOPD| > 1e-10 mm
    double max_height = 0.0;         // max |Δ Strahlhöhe| der Max-Reduktionen [mm]
    double max_cos = 0.0;            // max |Δ cos I|
    double max_sqrt_merit = 0.0;     // max |√F_df64 − √F_FP64| [Wellen]
    double max_merit_relative = 0.0; // max |F_df64 − F_FP64| / F_FP64 (F > 0)
    std::size_t violations = 0;      // Werte außerhalb von df64_bound
    std::size_t above_absolute = 0;  // Strahlen mit |ΔOPD| oder |Δr| > kDf64VsFp64
    std::size_t above_1e10 = 0;      // Strahlen mit |ΔOPD| > 1e-10 mm (zur Information)
};

Df64Comparison compare_df64(const SystemBatchHost& batch, const ResidualSettings& q = {}) {
    Df64Comparison c;
    const ResidualLayout layout(batch.topology(), q);
    TraceEngine engine(0);
    const TraceOutput fp =
        engine.run(batch, TraceMode::Residuals, settings_for(Precision::Fp64, q));
    const TraceOutput df =
        engine.run(batch, TraceMode::Residuals, settings_for(Precision::Df64, q));
    const TraceOutput fpm = engine.run(batch, TraceMode::Merit, settings_for(Precision::Fp64, q));
    const TraceOutput dfm = engine.run(batch, TraceMode::Merit, settings_for(Precision::Df64, q));
    const ResidualResult cpu = mko::residuals_cpu(batch, layout);
    const std::vector<double> units = mko::test::residual_units(batch, layout);
    const ResidualResult& a = fp.residuals;
    const ResidualResult& b = df.residuals;
    const auto check = [&c](double got, double ref) {
        const double d = std::abs(got - ref);
        if (!(d <= df64_bound(ref))) {
            ++c.violations;
        }
        return d;
    };
    for (int v = 0; v < batch.variants(); ++v) {
        double merit_bound_sq = 0.0; // Σ_k Schranke_k² über die Residuen der Variante
        for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
            ++c.rays;
            const double u = units[static_cast<std::size_t>(i)];
            const double rb = df64_residual_bound(u, a.residual(v, i));
            merit_bound_sq += rb * rb;
            const TraceStatus sa = a.status_at(v, i);
            if (sa != b.status_at(v, i)) {
                ++c.status_flips;
                continue;
            }
            if (sa != TraceStatus::Ok) {
                ++c.failed;
                EXPECT_EQ(b.residual(v, i), 0.0);
                continue;
            }
            const double dr_waves = std::abs(b.residual(v, i) - a.residual(v, i));
            if (!(dr_waves <= df64_residual_bound(u, a.residual(v, i)))) {
                ++c.violations;
            }
            const double dr = u > 0.0 ? dr_waves / u : dr_waves; // in mm OPD
            const double dopd = check(b.opd_at(v, i), a.opd_at(v, i));
            const double dcpu = check(b.opd_at(v, i), cpu.opd_at(v, i));
            c.max_residual = std::max(c.max_residual, dr);
            c.max_opd = std::max(c.max_opd, dopd);
            c.max_opd_cpu = std::max(c.max_opd_cpu, dcpu);
            c.above_absolute += (dr > kDf64VsFp64 || dopd > kDf64VsFp64) ? 1 : 0;
            if (dopd > 1e-10) {
                c.max_relative = std::max(c.max_relative, dopd / std::abs(a.opd_at(v, i)));
                if (++c.above_1e10 <= 10) {
                    std::printf("[ Info     ] |dOPD| > 1e-10 mm: v=%d i=%lld, OPD = %.6e mm, "
                                "|dOPD| = %.3e mm (relativ %.2e), Pupillenradius %.3f mm\n",
                                v, static_cast<long long>(i), a.opd_at(v, i), dopd,
                                dopd / std::abs(a.opd_at(v, i)), batch.pupil_radius(v));
                }
            }
        }
        for (int s = 0; s < batch.surfaces(); ++s) {
            c.max_height = std::max(c.max_height,
                                    check(b.extrema.max_height(v, s), a.extrema.max_height(v, s)));
            c.max_cos = std::max(c.max_cos, check(b.extrema.min_cos_incidence(v, s),
                                                  a.extrema.min_cos_incidence(v, s)));
            // Max-Reduktionen aus MERIT und RESIDUALS sind auch in df64 bitgleich.
            EXPECT_EQ(b.extrema.max_height(v, s), dfm.merit.extrema.max_height(v, s));
            EXPECT_EQ(b.extrema.min_cos_incidence(v, s), dfm.merit.extrema.min_cos_incidence(v, s));
        }
        // Merit: |√F_df64 − √F_FP64| <= ‖r_df64 − r_FP64‖₂ <= √(Σ_k Schranke_k²) (Dreiecks-
        // ungleichung der 2-Norm mit der Schranke pro Strahl), solange kein Status kippt.
        const auto vu = static_cast<std::size_t>(v);
        const double fa = fpm.merit.merit[vu];
        const double fb = dfm.merit.merit[vu];
        const double d = std::abs(std::sqrt(fb) - std::sqrt(fa));
        c.max_sqrt_merit = std::max(c.max_sqrt_merit, d);
        if (fa > 0.0) {
            c.max_merit_relative = std::max(c.max_merit_relative, std::abs(fb - fa) / fa);
        }
        if (!(d <= std::sqrt(merit_bound_sq))) {
            ++c.violations;
        }
        EXPECT_EQ(fpm.merit.failed[vu], dfm.merit.failed[vu]);
    }
    return c;
}

void report(const std::string& name, const Df64Comparison& c) {
    std::printf("[ Info     ] %-26s %7zu Strahlen, %6zu versagt, %zu Statuswechsel: max |dr| = "
                "%.2e, |dOPD| = %.2e (CPU %.2e) mm, %zu Strahlen über kDf64VsFp64 (rel. %.1e), "
                "|dh| = %.2e mm, |dcosI| = %.2e, |d sqrt F| = %.2e Wellen, max |dF|/F = %.2e\n",
                name.c_str(), c.rays, c.failed, c.status_flips, c.max_residual, c.max_opd,
                c.max_opd_cpu, c.above_absolute, c.max_relative, c.max_height, c.max_cos,
                c.max_sqrt_merit, c.max_merit_relative);
}

void expect_within(const Df64Comparison& c) {
    EXPECT_EQ(c.status_flips, 0u);
    EXPECT_EQ(c.violations, 0u) << "max |dr| = " << c.max_residual
                                << ", max |dOPD| = " << c.max_opd;
}

// Gerätespeicher eines Laufs im Modus RESIDUALS (wie test_cuda_trace_engine.cpp).
std::size_t residual_run_bytes(const BatchTopology& t, int variants, const ResidualLayout& layout,
                               const TraceSettings& s) {
    mko::OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.residuals = true;
    r.ray_status = true;
    r.opd = s.opd;
    r.merit = false;
    r.surface_max = s.extrema;
    return mko::estimate_device_bytes(t, variants, r) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * sizeof(mko::ResidualFieldPlan));
}

} // namespace

// Die df64-Operationen rechnen auf dem Gerät bitgleich wie auf dem Host: Die fehlerfreien
// Transformationen werden auf dem Gerät nicht durch FMA-Kontraktion verfälscht (Intrinsics
// __fadd_rn/__fmul_rn/__fmaf_rn …), auch nicht in zusammengesetzten Ausdrücken (a · b + a).
TEST(CudaDf64, DeviceMatchesHostBitwise) {
    MKO_REQUIRE_GPU();
    std::mt19937_64 rng(21);
    std::uniform_real_distribution<double> m(1.0, 10.0);
    std::uniform_int_distribution<int> e(-12, 12);
    std::uniform_real_distribution<double> angle(-50.0, 50.0);
    std::bernoulli_distribution sign(0.5);
    const auto value = [&] {
        const double x = m(rng) * std::pow(10.0, e(rng));
        return sign(rng) ? -x : x;
    };
    std::vector<df64> a, b;
    std::vector<double> x;
    for (int i = 0; i < 100000; ++i) {
        // Jede vierte Eingabe im Winkelbereich (sin/cos ohne NaN)
        a.push_back(df64(i % 4 == 0 ? angle(rng) : value()));
        b.push_back(df64(value()));
        x.push_back(value() * (1.0 + 1e-12 * m(rng)));
    }
    a.push_back(df64(0));
    b.push_back(df64(1));
    x.push_back(-0.0);
    std::vector<df64> out;
    std::vector<double> as_double;
    ASSERT_TRUE(mko::cuda::df64_check_ops_gpu(a, b, x, out, as_double));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        df64 host[mko::cuda::kDf64CheckOps];
        double host_double = 0.0;
        mko::cuda::df64_check_ops(a[i], b[i], x[i], host, &host_double);
        for (int j = 0; j < mko::cuda::kDf64CheckOps; ++j) {
            const df64 dev = out[i * mko::cuda::kDf64CheckOps + static_cast<std::size_t>(j)];
            const bool both_nan = std::isnan(dev.hi) && std::isnan(host[j].hi);
            if (!both_nan && std::memcmp(&dev, &host[j], sizeof(df64)) != 0) {
                if (++mismatches <= 10) {
                    ADD_FAILURE() << "Operation " << j << ", Eingabe " << i << ": Gerät (" << dev.hi
                                  << ", " << dev.lo << "), Host (" << host[j].hi << ", "
                                  << host[j].lo << ")";
                }
            }
        }
        if (std::memcmp(&host_double, &as_double[i], sizeof(double)) != 0) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

// Abnahme M7-T1: df64 gegen FP64 für die Szenarien der Testsuite über die TraceEngine (RESIDUALS
// mit OPD und Max-Reduktionen, MERIT mit Max-Reduktionen): Status gleich, |Δr|, |ΔOPD|,
// |Δ Strahlhöhe| und |Δ cos I| <= max(kDf64VsFp64, kDf64VsFp64Relative · |x|) für alle gültigen
// Strahlen, auch gegen die CPU-Referenz in FP64; Merit über die 2-Norm der Schranken pro Strahl.
TEST(CudaDf64, MatchesFp64ForPerturbedVariants) {
    MKO_REQUIRE_GPU();
    struct Case {
        std::string name;
        SystemBatchHost batch;
        ResidualSettings quadrature;
    };
    std::vector<Case> cases;
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    const mko::LoadedSystem multi = mko::test::load_residual_system("testsystem_mehrlinser.json");
    cases.push_back({"einzellinse_bk7", mko::test::perturbed_batch(lens, 1000, 1001), {}});
    cases.push_back({"testsystem_mehrlinser", mko::test::perturbed_batch(multi, 1000, 2002), {}});
    cases.push_back(
        {"testsystem_mehrlinser 6x5", mko::test::perturbed_batch(multi, 1000, 3003), {6, 5}});
    cases.push_back({"planplatte_bk7",
                     mko::test::perturbed_batch(
                         mko::test::load_residual_system("planplatte_bk7.json"), 500, 3030),
                     {}});
    {
        // Geisterplatten mit gestörten Koeffizienten und Gläsern (wie CudaResiduals)
        SystemBatchHost g = mko::test::perturbed_batch(
            mko::test::load_residual_system("einzellinse_geisterplatte.json"), 500, 8008);
        std::mt19937_64 rng(8009);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (int v = 1; v < g.variants(); ++v) {
            for (int p = 0; p < g.ghost_plates(); ++p) {
                for (int k = 0; k < g.ghost_terms(); ++k) {
                    g.ghost_coeff(v, p, k) =
                        g.ghost_coeff(v, p, k) * (1.0 + 0.5 * u(rng)) + 2e-4 * u(rng) / (k + 1);
                }
                const double dn = 0.05 * u(rng);
                for (int l = 0; l < g.wavelengths(); ++l) {
                    g.ghost_index(v, p, l) += dn;
                }
            }
        }
        cases.push_back({"einzellinse_geisterplatte", std::move(g), {}});
    }
    {
        mko::SystemDescription p = lens.system;
        p.source.type = mko::SourceType::Point;
        p.source.field_angles_deg.clear();
        p.source.field_weights.clear();
        p.source.numerical_aperture = 0.05;
        p.source.point = {1.0, 2.0, -300.0};
        p.surfaces.back().thickness = 150.0;
        cases.push_back(
            {"punktquelle",
             mko::test::perturbed_batch(
                 mko::prepare_system(p, std::filesystem::path(mko::test::kResidualSystems)), 200,
                 4004),
             {}});
    }
    cases.push_back({"64 Flächen", mko::test::many_surface_batch(200, 6006), {}});
    for (const Case& cs : cases) {
        SCOPED_TRACE(cs.name);
        ASSERT_NO_THROW(cs.batch.validate_values());
        const Df64Comparison c = compare_df64(cs.batch, cs.quadrature);
        report(cs.name, c);
        expect_within(c);
        RecordProperty(cs.name + "_max_opd_diff", std::to_string(c.max_opd));
    }
}

// Wie MatchesFp64ForPerturbedVariants mit je 10.000 Varianten und anderen Seeds (ca. 3·10⁷
// Strahlen, Laufzeit ca. 1 min); abgeschaltet, Aufruf mit --gtest_also_run_disabled_tests
// --gtest_filter=CudaDf64.DISABLED_StressManySeeds.
TEST(CudaDf64, DISABLED_StressManySeeds) {
    MKO_REQUIRE_GPU();
    const char* files[] = {"einzellinse_bk7.json", "testsystem_mehrlinser.json",
                           "planplatte_bk7.json", "einzellinse_geisterplatte.json"};
    std::uint64_t seed = 90001;
    for (const char* file : files) {
        for (int rep = 0; rep < 2; ++rep) {
            SCOPED_TRACE(file);
            const SystemBatchHost batch =
                mko::test::perturbed_batch(mko::test::load_residual_system(file), 10000, seed++);
            const Df64Comparison c = compare_df64(batch);
            report(file, c);
            expect_within(c);
        }
    }
}

// Verhalten an der Versagensgrenze:Der Pupillenradius wird so gewählt, dass der äußerste
// Strahl des Achsfeldes die zweite Linsenfläche gerade noch trifft bzw. verfehlt (Grenzradius
// a* per Bisektion in FP64 auf der CPU), a = a* (1 + δ). Für |δ| >= 1e-10 stimmen die Status
// von df64, FP64 (GPU) und FP64 (CPU) überein; näher an der Grenze darf der Status in df64
// kippen (Prüfgröße mit relativem Fehler ~u² ≈ 4e-15 statt ~1e-16). Ein kippender Status ist
// bei hartem Versagen ein Sprung in Kolben und Merit (weiches Versagen folgt in M4-T2).
TEST(CudaDf64, StatusAtFailureBoundary) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost probe = mko::make_system_batch(lens, 1);
    const ResidualLayout layout(probe.topology());
    const std::int64_t ray = layout.offset(0, 0) + layout.samples(0) - 1; // äußerster Strahl
    const auto fails = [&](double a) {
        probe.pupil_radius(0) = a;
        return mko::residuals_cpu(probe, 0, layout).status_at(0, ray) != TraceStatus::Ok;
    };
    double lo = probe.pupil_radius(0);
    ASSERT_FALSE(fails(lo));
    double hi = lo;
    while (!fails(hi)) {
        hi *= 2.0;
        ASSERT_LT(hi, 1e6);
    }
    for (int i = 0; i < 200 && std::nextafter(lo, hi) < hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        (fails(mid) ? hi : lo) = mid;
    }
    const double critical = hi;
    const std::vector<double> deltas = {-1e-4, -1e-7, -1e-10, -1e-12, -1e-13, -1e-14, -1e-15, 0.0,
                                        1e-15, 1e-14, 1e-13,  1e-12,  1e-10,  1e-7,   1e-4};
    SystemBatchHost batch = mko::make_system_batch(lens, static_cast<int>(deltas.size()));
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        batch.pupil_radius(static_cast<int>(i)) = critical * (1.0 + deltas[i]);
    }
    TraceEngine engine(0);
    TraceSettings s;
    s.precision = Precision::Fp64;
    const TraceOutput fp = engine.run(batch, TraceMode::Residuals, s);
    s.precision = Precision::Df64;
    const TraceOutput df = engine.run(batch, TraceMode::Residuals, s);
    const ResidualResult cpu = mko::residuals_cpu(batch, layout);
    int flips = 0;
    std::printf("[ Info     ] Grenzradius a* = %.17g mm (Status jenseits: %s)\n", critical,
                mko::trace_status_name(cpu.status_at(static_cast<int>(deltas.size()) - 1, ray)));
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        const int v = static_cast<int>(i);
        const TraceStatus sf = fp.residuals.status_at(v, ray);
        const TraceStatus sd = df.residuals.status_at(v, ray);
        const TraceStatus sc = cpu.status_at(v, ray);
        std::printf("[ Info     ]   delta = %+.0e: CPU %s, GPU FP64 %s, GPU df64 %s\n", deltas[i],
                    mko::trace_status_name(sc), mko::trace_status_name(sf),
                    mko::trace_status_name(sd));
        if (std::abs(deltas[i]) >= 1e-10) {
            EXPECT_EQ(sd, sc) << "delta = " << deltas[i];
            EXPECT_EQ(sf, sc) << "delta = " << deltas[i];
            EXPECT_EQ(sc == TraceStatus::Ok, deltas[i] < 0.0) << "delta = " << deltas[i];
        } else {
            flips += sd != sc ? 1 : 0;
        }
        // Alle übrigen Strahlen der Variante liegen weit von ihrer Grenze: gleicher Status.
        for (std::int64_t k = 0; k < layout.rays_per_variant(); ++k) {
            if (k != ray) {
                EXPECT_EQ(df.residuals.status_at(v, k), fp.residuals.status_at(v, k));
            }
        }
    }
    std::printf("[ Info     ] Statuswechsel df64 gegen FP64 für |delta| < 1e-10: %d\n", flips);
    RecordProperty("boundary_flips", std::to_string(flips));
}

// ADR-0007 für df64: zehn Läufe mit gleichen Eingaben sind bitgleich (RESIDUALS mit OPD und
// Max-Reduktionen, MERIT mit Max-Reduktionen), ebenso über mehrere Tranchen.
TEST(CudaDf64, RepeatedRunsAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 700, 5005);
    const TraceSettings s = settings_for(Precision::Df64);
    TraceEngine engine(0);
    const TraceOutput first_r = engine.run(batch, TraceMode::Residuals, s);
    const TraceOutput first_m = engine.run(batch, TraceMode::Merit, s);
    EXPECT_GT(first_r.residuals.failed_count(), 0u);
    TraceOutput out;
    for (int run = 1; run < 10; ++run) {
        engine.run(batch, TraceMode::Residuals, s, out);
        EXPECT_TRUE(identical(out.residuals, first_r.residuals)) << "Lauf " << run;
        engine.run(batch, TraceMode::Merit, s, out);
        EXPECT_TRUE(identical(out.merit, first_m.merit)) << "Lauf " << run;
    }
    // Vier Tranchen (Grenze für genau 200 Varianten) liefern dieselben Bits.
    const ResidualLayout layout(batch.topology(), s.quadrature);
    EngineOptions limited;
    limited.memory_limit_bytes = residual_run_bytes(batch.topology(), 200, layout, s);
    limited.reserve = mko::MemoryReserve{0, 0.0};
    TraceEngine small(0, limited);
    small.run(batch, TraceMode::Residuals, s, out);
    EXPECT_EQ(out.info.tranches, 4);
    EXPECT_TRUE(identical(out.residuals, first_r.residuals));
}

// MERIT und RESIDUALS bilden auch in df64 dieselben Residuen: F_v gegen die kompensierte
// Summe der Quadrate der (nach double gewandelten) RESIDUALS-Ausgabe.
TEST(CudaDf64, MeritMatchesSumOfResiduals) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, 2002);
    TraceSettings s;
    s.precision = Precision::Df64;
    TraceEngine engine(0);
    const TraceOutput r = engine.run(batch, TraceMode::Residuals, s);
    const TraceOutput m = engine.run(batch, TraceMode::Merit, s);
    double max_rel = 0.0;
    for (int v = 0; v < batch.variants(); ++v) {
        double sum = 0.0;
        double comp = 0.0; // Neumaier
        for (std::int64_t i = 0; i < r.residuals.rays_per_variant; ++i) {
            const double x = r.residuals.residual(v, i) * r.residuals.residual(v, i);
            const double t = sum + x;
            comp += std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
            sum = t;
        }
        sum += comp;
        const double f = m.merit.merit[static_cast<std::size_t>(v)];
        if (sum > 0.0) {
            max_rel = std::max(max_rel, std::abs(f - sum) / sum);
        } else {
            EXPECT_EQ(f, 0.0);
        }
    }
    EXPECT_LE(max_rel, mko::test::kMeritVsResidualSumDf64);
    std::printf("[ Info     ] df64: max |F - Σr²| / Σr² = %.2e\n", max_rel);
}

// Der Zahlentyp ändert weder Speicherbedarf noch Shared Memory; ungültige Werte werden
// abgewiesen.
TEST(CudaDf64, SettingsAndLimits) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 10, 1);
    TraceEngine engine(0);
    TraceSettings fp;
    TraceSettings df;
    df.precision = Precision::Df64;
    for (const TraceMode mode : {TraceMode::Residuals, TraceMode::Merit}) {
        EXPECT_EQ(engine.max_variants(batch.topology(), mode, fp),
                  engine.max_variants(batch.topology(), mode, df));
    }
    for (const bool extrema : {false, true}) {
        EXPECT_EQ(mko::cuda::merit_max_samples(extrema, 3, 8, Precision::Fp64),
                  mko::cuda::merit_max_samples(extrema, 3, 8, Precision::Df64));
    }
    TraceSettings bad;
    bad.precision = static_cast<Precision>(7);
    EXPECT_THROW((void)engine.run(batch, TraceMode::Residuals, bad), std::invalid_argument);
    EXPECT_THROW((void)engine.run(batch, TraceMode::Merit, bad), std::invalid_argument);
    EXPECT_THROW((void)mko::cuda::merit_max_samples(false, 0, 0, bad.precision),
                 std::invalid_argument);
    EXPECT_NO_THROW((void)engine.run(batch, TraceMode::Merit, df));
}

// Durchsatz df64 gegen FP64 über die TraceEngine (mit Upload und Download), Modi RESIDUALS und
// MERIT, 10.000 gestörte Varianten der Testsysteme (mit versagten Strahlen) und zweier Stapel.
// Ergänzung zu bench/trace_throughput (docs/benchmarks.md, Abschnitt "df64"); abgeschaltet,
// Aufruf mit --gtest_also_run_disabled_tests --gtest_filter=CudaDf64.DISABLED_Throughput.
TEST(CudaDf64, DISABLED_Throughput) {
    MKO_REQUIRE_GPU();
    struct Config {
        std::string name;
        SystemBatchHost batch;
    };
    std::vector<Config> configs;
    configs.push_back({"einzellinse_bk7 (S = 3)",
                       mko::test::perturbed_batch(
                           mko::test::load_residual_system("einzellinse_bk7.json"), 10000, 7007)});
    configs.push_back(
        {"testsystem_mehrlinser (S = 8)",
         mko::test::perturbed_batch(mko::test::load_residual_system("testsystem_mehrlinser.json"),
                                    10000, 7007)});
    configs.push_back({"Stapel S = 30", mko::test::many_surface_batch(10000, 7007, 30)});
    configs.push_back({"Stapel S = 64", mko::test::many_surface_batch(10000, 7007, 64)});
    TraceEngine engine(0);
    std::printf("[ Info     ] Gerät: %s\n", engine.device_name().c_str());
    constexpr int kRuns = 5;
    for (const Config& cfg : configs) {
        const ResidualLayout layout(cfg.batch.topology());
        const double rays = static_cast<double>(cfg.batch.variants()) *
                            static_cast<double>(layout.rays_per_variant());
        for (const TraceMode mode : {TraceMode::Residuals, TraceMode::Merit}) {
            double t[2] = {0.0, 0.0};
            for (const Precision p : {Precision::Fp64, Precision::Df64}) {
                TraceSettings s;
                s.precision = p;
                TraceOutput out;
                engine.run(cfg.batch, mode, s, out); // Aufwärmen, Puffer anlegen
                std::vector<double> times;
                for (int i = 0; i < kRuns; ++i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    engine.run(cfg.batch, mode, s, out);
                    times.push_back(
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                            .count());
                }
                std::sort(times.begin(), times.end());
                t[p == Precision::Df64 ? 1 : 0] = times[kRuns / 2]; // Median
            }
            std::printf("[ Info     ] %-30s %-9s %7.0f Strahlen/Var.: FP64 %8.2f ms (%.3e "
                        "Strahlen/s), df64 %8.2f ms (%.3e Strahlen/s), Faktor %.2f\n",
                        cfg.name.c_str(), mko::cuda::to_string(mode),
                        static_cast<double>(layout.rays_per_variant()), t[0] * 1e3, rays / t[0],
                        t[1] * 1e3, rays / t[1], t[0] / t[1]);
            const std::string key =
                cfg.name.substr(0, cfg.name.find(' ')) + "_" + mko::cuda::to_string(mode);
            RecordProperty(key + "_fp64_rays_per_second", std::to_string(rays / t[0]));
            RecordProperty(key + "_df64_rays_per_second", std::to_string(rays / t[1]));
        }
    }
}
