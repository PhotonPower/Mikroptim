// Kernel RESIDUALS (M2-T2, nur mit MIKROPTIM_ENABLE_CUDA): GPU gegen CPU-Referenz
// (residuals_cpu) für zufällig gestörte Varianten, Determinismus, Grenzfälle. Die Läufe gehen
// über die öffentliche API TraceEngine (M2-T4); nur DeviceBufferReuse und InvalidUseThrows
// prüfen den Baustein DeviceResiduals direkt (Pufferverwaltung und eigene Fehlerfälle).

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
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
using mko::PupilMode;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SurfaceKind;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::DeviceResiduals;
using mko::cuda::DeviceSystemBatch;
using mko::cuda::ResidualRunOptions;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
using mko::test::kCpuGpuFp64;
using mko::test::kCpuGpuFp64Relative;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

struct Comparison {
    double max_residual = 0.0;      // max |r_GPU − r_CPU| über alle Strahlen [mm]
    double max_opd = 0.0;           // max |OPD_GPU − OPD_CPU| über alle Strahlen [mm]
    double max_relative = 0.0;      // max |Δ| / |x| der Strahlen mit |Δ| > kCpuGpuFp64
    std::size_t above_absolute = 0; // Strahlen mit |Δr| oder |ΔOPD| > kCpuGpuFp64
    std::size_t violations = 0;     // Strahlen außerhalb von max(kCpuGpuFp64, rel · |x|)
    std::size_t rays = 0;
    std::size_t failed = 0;            // versagte Strahlen
    int variants_with_failures = 0;    // Varianten mit mindestens einem versagten Strahl
    int variants_all_failed_tuple = 0; // Varianten mit einem Tupel ohne gültigen Strahl
};

// Prüft |gpu − cpu| <= max(kCpuGpuFp64, kCpuGpuFp64Relative · |cpu|) (tests/tolerances.hpp)
// und führt die Statistik nach. Rückgabe: |gpu − cpu| (NaN als sehr groß).
double check_value(double gpu, double cpu, Comparison& c, bool& above) {
    double d = std::abs(gpu - cpu);
    if (std::isnan(d)) {
        d = 1e300;
    }
    if (d > kCpuGpuFp64) {
        above = true;
        c.max_relative = std::max(c.max_relative, d / std::abs(cpu));
    }
    if (d > std::max(kCpuGpuFp64, kCpuGpuFp64Relative * std::abs(cpu))) {
        ++c.violations;
    }
    return d;
}

// Vergleicht GPU- und CPU-Ergebnis Strahl für Strahl: Status gleich, Residuen und OPD
// innerhalb der Schranke von check_value, weitere Residuen (stride > rays) auf dem Gerät 0.
Comparison compare(const ResidualResult& gpu, const ResidualResult& cpu,
                   const ResidualLayout& layout) {
    Comparison c;
    EXPECT_EQ(gpu.variants, cpu.variants);
    EXPECT_EQ(gpu.rays_per_variant, cpu.rays_per_variant);
    EXPECT_EQ(gpu.rays_per_variant, layout.rays_per_variant());
    const bool with_opd = !gpu.opd.empty();
    int mismatches = 0;
    for (int v = 0; v < cpu.variants; ++v) {
        bool variant_failed = false;
        bool tuple_failed = false;
        for (int f = 0; f < layout.fields(); ++f) {
            for (int l = 0; l < layout.wavelengths(); ++l) {
                const std::int64_t o = layout.offset(f, l);
                int ok = 0;
                for (int k = 0; k < layout.samples(f); ++k) {
                    const std::int64_t i = o + k;
                    ++c.rays;
                    const TraceStatus sg = gpu.status_at(v, i);
                    const TraceStatus sc = cpu.status_at(v, i);
                    if (sg != sc && ++mismatches <= 10) {
                        ADD_FAILURE() << "Status verschieden: v=" << v << " f=" << f << " l=" << l
                                      << " k=" << k << ": GPU " << static_cast<unsigned>(sg)
                                      << ", CPU " << static_cast<unsigned>(sc);
                    }
                    if (sc == TraceStatus::Ok) {
                        ++ok;
                    } else {
                        ++c.failed;
                        variant_failed = true;
                        if (sg == sc) {
                            EXPECT_EQ(gpu.residual(v, i), 0.0);
                        }
                    }
                    bool above = false;
                    c.max_residual =
                        std::max(c.max_residual,
                                 check_value(gpu.residual(v, i), cpu.residual(v, i), c, above));
                    if (with_opd) {
                        c.max_opd = std::max(
                            c.max_opd, check_value(gpu.opd_at(v, i), cpu.opd_at(v, i), c, above));
                    }
                    if (above) {
                        ++c.above_absolute;
                        std::printf("[ Info     ] |GPU - CPU| > kCpuGpuFp64: v=%d f=%d l=%d k=%d, "
                                    "OPD = %.6e mm, |dr| = %.3e, |dOPD| = %.3e mm\n",
                                    v, f, l, k, cpu.opd_at(v, i),
                                    std::abs(gpu.residual(v, i) - cpu.residual(v, i)),
                                    with_opd ? std::abs(gpu.opd_at(v, i) - cpu.opd_at(v, i)) : 0.0);
                    }
                }
                tuple_failed = tuple_failed || ok == 0;
            }
        }
        for (std::int64_t e = gpu.rays_per_variant; e < gpu.stride; ++e) {
            EXPECT_EQ(gpu.residual(v, e), 0.0) << "weiteres Residuum " << e;
        }
        c.variants_with_failures += variant_failed ? 1 : 0;
        c.variants_all_failed_tuple += tuple_failed ? 1 : 0;
    }
    EXPECT_EQ(mismatches, 0);
    EXPECT_EQ(c.violations, 0u) << "max |dr| = " << c.max_residual
                                << ", max |dOPD| = " << c.max_opd;
    return c;
}

// Bitgleichheit zweier GPU-Ergebnisse.
bool identical(const ResidualResult& a, const ResidualResult& b) {
    const auto same = [](const auto& x, const auto& y) {
        return x.size() == y.size() &&
               (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(*x.data())) == 0);
    };
    return a.variants == b.variants && a.stride == b.stride && same(a.residuals, b.residuals) &&
           same(a.opd, b.opd) && same(a.status, b.status);
}

// Lauf über die TraceEngine auf Gerät 0 (OPD angefordert).
ResidualResult gpu_residuals(const SystemBatchHost& batch, const ResidualSettings& settings = {},
                             std::int64_t extra = 0) {
    TraceEngine engine(0);
    TraceSettings s;
    s.quadrature = settings;
    s.opd = true;
    s.extra_residuals = extra;
    const TraceOutput out = engine.run(batch, TraceMode::Residuals, s);
    EXPECT_EQ(out.info.tranches, 1);
    return out.residuals;
}

// Stapel aus S = kMaxSurfaces Flächen: Blende, 31 schwach gekrümmte Linsen (Gläser n = 1,5
// und 1,7 abwechselnd), letzte Fläche (Luft auf beiden Seiten) mit Arbeitsabstand; zwei
// Felder (Achse, Feldpunkt).
SystemBatchHost many_surface_batch(int variants, std::uint64_t seed) {
    BatchTopology t;
    t.surfaces = mko::kMaxSurfaces;
    t.wavelengths = 2;
    t.fields = 2;
    t.type.assign(static_cast<std::size_t>(t.surfaces), SurfaceKind::Sphere);
    t.type[0] = SurfaceKind::Stop;
    t.pupil_mode = {PupilMode::Axis, PupilMode::Half};
    SystemBatchHost b(t, variants);
    b.wavelength_nm(0) = 486.13;
    b.wavelength_nm(1) = 656.27;
    b.wavelength_weight(0) = b.wavelength_weight(1) = 1.0;
    b.field_weight(0) = b.field_weight(1) = 1.0;
    b.reference_wavelength_nm = 587.56;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int S = t.surfaces;
    for (int v = 0; v < variants; ++v) {
        const double p = v == 0 ? 0.0 : 1.0; // Variante 0 ungestört
        for (int s = 1; s < S; ++s) {
            const bool glass = s % 2 == 1 && s < S - 1; // Medium nach Fläche s
            b.curvature(v, s) = (glass ? 0.0008 : -0.0006) * (1.0 + 0.2 * p * u(rng));
            b.thickness(v, s) = (glass ? 0.8 : 0.4) * (1.0 + 0.2 * p * u(rng));
            const double n = glass ? (s % 4 == 1 ? 1.5 : 1.7) : 1.0;
            for (int l = 0; l < 2; ++l) {
                b.index(v, s, l) = glass ? n + 0.01 * (1 - l) : 1.0;
            }
        }
        for (int l = 0; l < 2; ++l) {
            b.index(v, 0, l) = 1.0;
        }
        b.thickness(v, 0) = 2.0;
        b.thickness(v, S - 1) = 40.0 * (1.0 + 0.1 * p * u(rng));
        b.pupil_radius(v) = 3.0 * (1.0 + 0.3 * p * u(rng));
        b.field_angle(v, 1) = 0.02 * (1.0 + 0.3 * p * u(rng));
        b.field_target(v, 1) = 0.8 * (1.0 + 0.1 * p * u(rng));
    }
    b.validate_values();
    return b;
}

} // namespace

// Abnahme M2-T2: |GPU − CPU| <= kCpuGpuFp64 für alle Strahlen von je 1.000 zufällig gestörten
// Varianten der Testsysteme (Residuen und OPD, Status gleich).
TEST(CudaResiduals, MatchCpuForPerturbedVariants) {
    MKO_REQUIRE_GPU();
    struct Case {
        const char* file;
        std::uint64_t seed;
        ResidualSettings settings;
        std::int64_t extra;
    };
    const Case cases[] = {{"einzellinse_bk7.json", 1001, {}, 0},
                          {"testsystem_mehrlinser.json", 2002, {}, 0},
                          {"testsystem_mehrlinser.json", 3003, {6, 5}, 7}};
    int index = 0;
    for (const Case& cs : cases) {
        SCOPED_TRACE(cs.file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(cs.file);
        const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, cs.seed);
        ASSERT_NO_THROW(batch.validate_values());
        const ResidualLayout layout(batch.topology(), cs.settings);
        const ResidualResult gpu = gpu_residuals(batch, cs.settings, cs.extra);
        EXPECT_EQ(gpu.stride, layout.rays_per_variant() + cs.extra);
        const ResidualResult cpu = mko::residuals_cpu(batch, layout);
        const Comparison c = compare(gpu, cpu, layout);
        EXPECT_EQ(c.rays, 1000u * static_cast<std::size_t>(layout.rays_per_variant()));
        EXPECT_GT(c.failed, 0u) << "Versagenspfad nicht geprüft";
        EXPECT_GT(c.variants_with_failures, 10);
        EXPECT_LT(c.failed, c.rays / 2);
        std::printf("[ Info     ] %s (N=%d, M=%d): %zu Strahlen, %zu versagt (%d Varianten, "
                    "%d mit Tupel ohne gültigen Strahl), max |dr| = %.3e mm, max |dOPD| = %.3e "
                    "mm, %zu Strahlen über kCpuGpuFp64 (max. relativ %.2e)\n",
                    cs.file, cs.settings.rings, cs.settings.spokes, c.rays, c.failed,
                    c.variants_with_failures, c.variants_all_failed_tuple, c.max_residual,
                    c.max_opd, c.above_absolute, c.max_relative);
        const std::string key = "case" + std::to_string(index++);
        RecordProperty(key + "_max_residual_diff", std::to_string(c.max_residual));
        RecordProperty(key + "_max_opd_diff", std::to_string(c.max_opd));
        RecordProperty(key + "_failed_rays", std::to_string(c.failed));
        RecordProperty(key + "_rays_above_kCpuGpuFp64", std::to_string(c.above_absolute));
    }
}

// Punktquelle neben der Achse (volle Pupille), gestörte Varianten.
TEST(CudaResiduals, MatchCpuForPointSource) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    mko::SystemDescription p = lens.system;
    p.source.type = mko::SourceType::Point;
    p.source.field_angles_deg.clear();
    p.source.field_weights.clear();
    p.source.numerical_aperture = 0.05;
    p.source.point = {1.0, 2.0, -300.0};
    p.surfaces.back().thickness = 150.0;
    const mko::LoadedSystem ls =
        mko::prepare_system(p, std::filesystem::path(mko::test::kResidualSystems));
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 200, 4004);
    const ResidualLayout layout(batch.topology());
    ASSERT_EQ(layout.pupil_mode(0), PupilMode::Full);
    const ResidualResult gpu = gpu_residuals(batch);
    const Comparison c = compare(gpu, mko::residuals_cpu(batch, layout), layout);
    std::printf("[ Info     ] Punktquelle: %zu Strahlen, %zu versagt, max |dr| = %.3e mm\n", c.rays,
                c.failed, c.max_residual);
}

// ADR-0007: zehn Läufe mit gleichen Eingaben sind bitgleich (auch mit wiederverwendeten
// Puffern nach einem Lauf anderer Größe und ohne OPD).
TEST(CudaResiduals, RepeatedRunsAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, 5005);
    TraceEngine engine(0);
    TraceSettings s;
    s.opd = true;
    TraceOutput out;
    engine.run(batch, TraceMode::Residuals, s, out);
    const ResidualResult first = out.residuals;
    EXPECT_EQ(first.variants, 1000);
    for (int run = 1; run < 10; ++run) {
        if (run == 5) {
            // Zwischendurch ein kleinerer Lauf ohne OPD in dieselben Puffer und dieselbe Ausgabe
            engine.run(mko::test::perturbed_batch(ls, 10, 1), TraceMode::Residuals, {}, out);
            EXPECT_TRUE(out.residuals.opd.empty());
        }
        engine.run(batch, TraceMode::Residuals, s, out);
        EXPECT_TRUE(identical(out.residuals, first)) << "Lauf " << run;
    }
    // Eine frische Engine liefert dasselbe
    EXPECT_TRUE(identical(gpu_residuals(batch), first));
}

// Baustein DeviceResiduals: Puffer wird über Läufe verschiedener Größe wiederverwendet,
// Gerätezeiger für Folgekernel, Ergebnis wie über die TraceEngine.
TEST(CudaResiduals, DeviceBufferReuse) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 300, 5005);
    const ResidualLayout layout(batch.topology());
    const DeviceSystemBatch device(batch);
    ResidualRunOptions opt;
    opt.opd = true;
    DeviceResiduals out;
    out.run(device, layout, opt);
    const ResidualResult first = out.download();
    EXPECT_TRUE(identical(first, gpu_residuals(batch)));
    EXPECT_EQ(out.variants(), 300);
    EXPECT_NE(out.device_residuals(), nullptr);
    EXPECT_NE(out.device_status(), nullptr);
    EXPECT_NE(out.device_opd(), nullptr);
    const std::size_t capacity = out.capacity_bytes();
    const DeviceSystemBatch small(mko::test::perturbed_batch(ls, 10, 1));
    out.run(small, layout);
    EXPECT_EQ(out.device_opd(), nullptr);
    EXPECT_EQ(out.capacity_bytes(), capacity);
    out.run(device, layout, opt);
    EXPECT_EQ(out.capacity_bytes(), capacity);
    EXPECT_TRUE(identical(out.download(), first));
    EXPECT_TRUE(identical(mko::cuda::run_residuals(device, layout, opt), first));
}

// Grenzfälle: Varianten mit vielen versagten Strahlen und mit Tupeln, deren Strahlen alle
// versagen (Kolben undefiniert → Residuen und OPD 0, Status gesetzt); Nachbarvarianten bleiben
// unberührt.
TEST(CudaResiduals, FailedBlocksAreMasked) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost batch = mko::make_system_batch(ls, 5);
    batch.pupil_radius(1) = 1e4;   // alle Strahlen verfehlen die Linse (R = 103 mm)
    batch.pupil_radius(2) = 110.0; // äußere Ringe verfehlen
    batch.curvature(3, 1) = 0.2;   // R = 5 mm: fast alle Strahlen verfehlen oder reflektieren
    const ResidualLayout layout(batch.topology());
    const ResidualResult gpu = gpu_residuals(batch);
    const ResidualResult cpu = mko::residuals_cpu(batch, layout);
    const Comparison c = compare(gpu, cpu, layout);
    for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
        EXPECT_NE(gpu.status_at(1, i), TraceStatus::Ok);
        EXPECT_EQ(gpu.residual(1, i), 0.0);
        EXPECT_EQ(gpu.opd_at(1, i), 0.0);
        EXPECT_EQ(gpu.status_at(0, i), TraceStatus::Ok);
        EXPECT_EQ(gpu.status_at(4, i), TraceStatus::Ok);
    }
    EXPECT_GE(c.variants_with_failures, 3);
    EXPECT_GE(c.variants_all_failed_tuple, 1);
    std::printf("[ Info     ] Grenzfälle: %zu von %zu Strahlen versagt, max |dr| = %.3e mm\n",
                c.failed, c.rays, c.max_residual);
}

// S = kMaxSurfaces (größte Flächentabelle im Shared Memory).
TEST(CudaResiduals, MaxSurfaces) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost batch = many_surface_batch(200, 6006);
    const ResidualLayout layout(batch.topology());
    const ResidualResult gpu = gpu_residuals(batch);
    const Comparison c = compare(gpu, mko::residuals_cpu(batch, layout), layout);
    EXPECT_LT(c.failed, c.rays);
    std::printf("[ Info     ] S = %d: %zu Strahlen, %zu versagt, max |dr| = %.3e mm, max |dOPD| "
                "= %.3e mm\n",
                batch.surfaces(), c.rays, c.failed, c.max_residual, c.max_opd);
}

// Großer Batch (10.000 Varianten) läuft durch und stimmt mit der CPU überein; grobe
// Durchsatzzahl über die TraceEngine einschließlich Upload und Download (der eigentliche
// Benchmark ist M2-T6).
TEST(CudaResiduals, LargeBatch) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 10000, 7007);
    const ResidualLayout layout(batch.topology());
    TraceEngine engine(0);
    TraceOutput out;
    engine.run(batch, TraceMode::Residuals, {}, out); // Aufwärmen, Puffer anlegen
    constexpr int kRuns = 5;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kRuns; ++i) {
        engine.run(batch, TraceMode::Residuals, {}, out);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / kRuns;
    const ResidualResult& gpu = out.residuals;
    EXPECT_TRUE(gpu.opd.empty());
    EXPECT_EQ(out.info.tranches, 1);
    ASSERT_EQ(gpu.variants, 10000);
    const Comparison c = compare(gpu, mko::residuals_cpu(batch, layout), layout);
    const double rays = 10000.0 * static_cast<double>(layout.rays_per_variant());
    std::printf("[ Info     ] 10.000 Varianten x %lld Strahlen (S = %d): %.3f ms pro Lauf "
                "(TraceEngine mit Upload und Download), %.3e Strahlen/s, max |dr| = %.3e mm\n",
                static_cast<long long>(layout.rays_per_variant()), batch.surfaces(), seconds * 1e3,
                rays / seconds, c.max_residual);
    RecordProperty("rays_per_second", std::to_string(rays / seconds));
}

// Baustein DeviceResiduals: eigene Fehlerfälle (die der TraceEngine prüft
// test_cuda_trace_engine.cpp).
TEST(CudaResiduals, InvalidUseThrows) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost batch = mko::make_system_batch(ls, 2);
    const ResidualLayout layout(batch.topology());
    DeviceResiduals out;
    EXPECT_THROW((void)out.download(), std::logic_error);
    EXPECT_EQ(out.device_residuals(), nullptr);
    const DeviceSystemBatch empty;
    EXPECT_THROW(out.run(empty, layout), std::logic_error);
    const DeviceSystemBatch device(batch);
    ResidualRunOptions bad;
    bad.extra_residuals = -1;
    EXPECT_THROW(out.run(device, layout, bad), std::invalid_argument);
    // Layout eines anderen Systems
    const mko::LoadedSystem other = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const ResidualLayout wrong(mko::make_system_batch(other, 1).topology());
    EXPECT_THROW(out.run(device, wrong), std::invalid_argument);
    // Upload einer Variante, die das Achsfeld verlässt
    batch.field_angle(1, 0) = 0.01;
    EXPECT_THROW(DeviceSystemBatch{batch}, std::invalid_argument);
    // Nach den Fehlern funktioniert der Puffer weiter
    EXPECT_NO_THROW(out.run(device, layout));
    EXPECT_EQ(out.download().variants, 2);
}
