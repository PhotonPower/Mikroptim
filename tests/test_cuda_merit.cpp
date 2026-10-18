// Kernel MERIT (M2-T3, nur mit MIKROPTIM_ENABLE_CUDA): Merit gegen die Summe der
// RESIDUALS-Ausgabe, Determinismus, GPU gegen CPU-Referenz (merit_cpu, surface_extrema_cpu),
// Max-Reduktionen, Grenzfälle, gewichtete Merit gegen die RMS-Werte aus M1-T7 (M4-T1). Fachliche
// Tests laufen über die öffentliche API TraceEngine (M2-T4); nur DeviceBufferReuse und
// InvalidUseThrows prüfen den Baustein DeviceMerit direkt.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using mko::MeritResult;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::DeviceMerit;
using mko::cuda::DeviceSystemBatch;
using mko::cuda::MeritRunOptions;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
using mko::test::kCpuGpuFp64;
using mko::test::kCpuGpuFp64Relative;
using mko::test::kMeritVsResidualSum;
using mko::test::kWeightedMerit;

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

bool identical(const MeritResult& a, const MeritResult& b) {
    return a.variants == b.variants && a.extrema.surfaces == b.extrema.surfaces &&
           same_bytes(a.merit, b.merit) && same_bytes(a.failed, b.failed) &&
           same_bytes(a.extrema.values, b.extrema.values);
}

// Ein Lauf über die TraceEngine auf Gerät 0 (eine Tranche).
TraceOutput engine_run(const SystemBatchHost& batch, TraceMode mode,
                       const ResidualSettings& quadrature = {}, bool extrema = true,
                       std::span<const double> extra_merit = {}) {
    TraceEngine engine(0);
    TraceSettings s;
    s.quadrature = quadrature;
    s.extrema = extrema;
    s.extra_merit = extra_merit;
    TraceOutput out = engine.run(batch, mode, s);
    EXPECT_EQ(out.info.tranches, 1);
    return out;
}

MeritResult gpu_merit(const SystemBatchHost& batch, const ResidualSettings& quadrature = {},
                      bool extrema = true) {
    return engine_run(batch, TraceMode::Merit, quadrature, extrema).merit;
}

ResidualResult gpu_residuals(const SystemBatchHost& batch, const ResidualSettings& quadrature = {},
                             bool extrema = true) {
    return engine_run(batch, TraceMode::Residuals, quadrature, extrema).residuals;
}

// Σ r_k² der Variante v über alle Strahlresiduen in der Ablage (f, l, k), kompensiert
// summiert (Neumaier). Fehler ~ ε · F unabhängig von der Zahl der Summanden; Vergleichswert
// für die Abnahme "Merit = Σ RESIDUALS²".
double compensated_sum_of_squares(const ResidualResult& r, int v) {
    double sum = 0.0;
    double c = 0.0;
    for (std::int64_t i = 0; i < r.rays_per_variant; ++i) {
        const double x = r.residual(v, i) * r.residual(v, i);
        const double t = sum + x;
        c += std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
        sum = t;
    }
    return sum + c;
}

// Dieselbe Summe in der Reihenfolge des Kernels nachgerechnet: pro Tupel (f, l) Thread
// t = 0 … 127 mit fma(r, r, acc) über k = t, t + 128, …, Baumreduktion im Warp
// (__shfl_down_sync, Abstände 16 … 1; Lanes ohne Partner behalten ihren Wert), Warps und
// danach Tupel in fester Reihenfolge. Nur zur Information (bitgleich, solange nvcc die
// Quadratsumme wie angenommen kontrahiert).
double emulate_kernel_sum(const ResidualResult& r, const ResidualLayout& layout, int v) {
    constexpr int kThreads = mko::cuda::kMeritThreads;
    double merit = 0.0;
    for (int f = 0; f < layout.fields(); ++f) {
        for (int l = 0; l < layout.wavelengths(); ++l) {
            std::array<double, kThreads> acc{};
            for (int k = 0; k < layout.samples(f); ++k) {
                const double x = r.residual(v, layout.offset(f, l) + k);
                acc[static_cast<std::size_t>(k % kThreads)] =
                    std::fma(x, x, acc[static_cast<std::size_t>(k % kThreads)]);
            }
            double block = 0.0;
            for (int w = 0; w < kThreads / 32; ++w) {
                std::array<double, 32> lane{};
                std::copy_n(acc.begin() + 32 * w, 32, lane.begin());
                for (int offset = 16; offset > 0; offset /= 2) {
                    std::array<double, 32> next = lane;
                    for (int i = 0; i + offset < 32; ++i) {
                        next[static_cast<std::size_t>(i)] =
                            lane[static_cast<std::size_t>(i)] +
                            lane[static_cast<std::size_t>(i + offset)];
                    }
                    lane = next;
                }
                block += lane[0];
            }
            merit += block;
        }
    }
    return merit;
}

struct MeritComparison {
    double max_relative = 0.0; // max |F_MERIT − Σ r²| / Σ r²
    int bit_identical = 0;     // Varianten, deren Merit bitgleich zur Nachrechnung ist
    int all_failed = 0;        // Varianten ohne gültigen Strahl
    int with_failures = 0;     // Varianten mit versagten Strahlen
    std::int64_t failed = 0;   // versagte Strahlen insgesamt
};

// Abnahme: F_MERIT gegen Σ r² der RESIDUALS-Ausgabe (relativ < kMeritVsResidualSum), versagte
// Strahlen gegen den Status.
MeritComparison compare_with_residuals(const MeritResult& m, const ResidualResult& r,
                                       const ResidualLayout& layout) {
    MeritComparison c;
    EXPECT_EQ(m.variants, r.variants);
    EXPECT_EQ(m.rays_per_variant, r.rays_per_variant);
    int reported = 0;
    for (int v = 0; v < r.variants; ++v) {
        const auto vi = static_cast<std::size_t>(v);
        std::int64_t failed = 0;
        for (std::int64_t i = 0; i < r.rays_per_variant; ++i) {
            failed += r.status_at(v, i) != TraceStatus::Ok ? 1 : 0;
        }
        EXPECT_EQ(m.failed[vi], failed) << "v = " << v;
        c.failed += failed;
        c.with_failures += failed > 0 ? 1 : 0;
        c.all_failed += failed == r.rays_per_variant ? 1 : 0;
        const double reference = compensated_sum_of_squares(r, v);
        if (reference == 0.0) {
            EXPECT_EQ(m.merit[vi], 0.0) << "v = " << v;
            continue;
        }
        const double rel = std::abs(m.merit[vi] - reference) / reference;
        c.max_relative = std::max(c.max_relative, rel);
        if (!(rel < kMeritVsResidualSum) && ++reported <= 10) {
            ADD_FAILURE() << "v = " << v << ": MERIT " << m.merit[vi] << ", Σ r² " << reference
                          << ", relativ " << rel;
        }
        c.bit_identical += m.merit[vi] == emulate_kernel_sum(r, layout, v) ? 1 : 0;
    }
    return c;
}

struct CpuComparison {
    double max_sqrt_merit = 0.0; // max |√F_GPU − √F_CPU| [Wellen]
    double max_height = 0.0;     // max |h_GPU − h_CPU| [mm]
    double max_cos = 0.0;        // max |cos I_GPU − cos I_CPU|
};

// GPU gegen CPU-Referenz. Merit: |√F_GPU − √F_CPU| = | ‖r_GPU‖ − ‖r_CPU‖ | <= ‖r_GPU − r_CPU‖,
// und jedes Residuum liegt innerhalb max(u_k · kCpuGpuFp64, kCpuGpuFp64Relative · |r|) (Abnahme
// M2-T2 für die OPD, mit r = u_k · OPD in Wellen umgerechnet, M4-T1; u_k aus residual_units);
// die Schranke ist die 2-Norm dieser Toleranzen plus Summationsrundung
// (kMeritVsResidualSum · √F). Extrema: max ist 1-Lipschitz, also dieselben Toleranzen wie pro
// Strahl (Höhe absolut bzw. relativ, cos I absolut).
CpuComparison compare_with_cpu(const MeritResult& gpu, const MeritResult& cpu,
                               const ResidualResult& cpu_residuals,
                               const std::vector<double>& units) {
    CpuComparison c;
    EXPECT_EQ(gpu.variants, cpu.variants);
    EXPECT_TRUE(same_bytes(gpu.failed, cpu.failed));
    for (int v = 0; v < cpu.variants; ++v) {
        const auto vi = static_cast<std::size_t>(v);
        double bound_sq = 0.0;
        for (std::int64_t i = 0; i < cpu_residuals.rays_per_variant; ++i) {
            const double t = mko::test::residual_bound(units[static_cast<std::size_t>(i)],
                                                       cpu_residuals.residual(v, i), kCpuGpuFp64,
                                                       kCpuGpuFp64Relative);
            bound_sq += t * t;
        }
        const double d = std::abs(std::sqrt(gpu.merit[vi]) - std::sqrt(cpu.merit[vi]));
        EXPECT_LE(d, std::sqrt(bound_sq) + kMeritVsResidualSum * std::sqrt(cpu.merit[vi]))
            << "v = " << v;
        c.max_sqrt_merit = std::max(c.max_sqrt_merit, d);
        if (!cpu.extrema.empty()) {
            for (int s = 0; s < cpu.extrema.surfaces; ++s) {
                const double h = cpu.extrema.max_height(v, s);
                const double dh = std::abs(gpu.extrema.max_height(v, s) - h);
                const double dc = std::abs(gpu.extrema.min_cos_incidence(v, s) -
                                           cpu.extrema.min_cos_incidence(v, s));
                EXPECT_LE(dh, std::max(kCpuGpuFp64, kCpuGpuFp64Relative * h))
                    << "v = " << v << ", s = " << s;
                EXPECT_LE(dc, kCpuGpuFp64) << "v = " << v << ", s = " << s;
                c.max_height = std::max(c.max_height, dh);
                c.max_cos = std::max(c.max_cos, dc);
            }
        }
    }
    return c;
}

struct Case {
    const char* file;
    std::uint64_t seed;
    ResidualSettings settings;
};

const Case kCases[] = {{"einzellinse_bk7.json", 1001, {}},
                       {"testsystem_mehrlinser.json", 2002, {}},
                       {"testsystem_mehrlinser.json", 3003, {6, 5}}};

} // namespace

// Abnahme M2-T3: Die Merit stimmt mit der Summe der Quadrate der RESIDUALS-Ausgabe überein
// (< 1e-13 relativ), für je 1.000 zufällig gestörte Varianten der Testsysteme. Dazu: Merit mit
// und ohne Max-Reduktionen bitgleich, Max-Reduktionen aus MERIT und RESIDUALS bitgleich.
TEST(CudaMerit, MatchesSumOfResiduals) {
    MKO_REQUIRE_GPU();
    int index = 0;
    for (const Case& cs : kCases) {
        SCOPED_TRACE(cs.file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(cs.file);
        const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, cs.seed);
        const ResidualLayout layout(batch.topology(), cs.settings);
        const ResidualResult r = gpu_residuals(batch, cs.settings);
        const MeritResult m = gpu_merit(batch, cs.settings);
        const MeritComparison c = compare_with_residuals(m, r, layout);
        EXPECT_GT(c.with_failures, 10) << "Versagenspfad nicht geprüft";

        const MeritResult plain = gpu_merit(batch, cs.settings, false);
        EXPECT_TRUE(plain.extrema.empty());
        EXPECT_TRUE(same_bytes(plain.merit, m.merit)) << "Merit hängt von den Extrema ab";
        EXPECT_TRUE(same_bytes(plain.failed, m.failed));
        ASSERT_EQ(r.extrema.surfaces, batch.surfaces());
        EXPECT_TRUE(same_bytes(r.extrema.values, m.extrema.values));

        std::printf("[ Info     ] %s (N=%d, M=%d): 1000 Varianten, max |F - Σr²| / Σr² = %.2e, "
                    "%d bitgleich zur Nachrechnung der Kernel-Reihenfolge, %d mit versagten "
                    "Strahlen (%lld Strahlen), %d ohne gültigen Strahl\n",
                    cs.file, cs.settings.rings, cs.settings.spokes, c.max_relative, c.bit_identical,
                    c.with_failures, static_cast<long long>(c.failed), c.all_failed);
        const std::string key = "case" + std::to_string(index++);
        RecordProperty(key + "_max_relative", std::to_string(c.max_relative));
        RecordProperty(key + "_bit_identical_to_emulation", std::to_string(c.bit_identical));
    }
}

// GPU gegen CPU-Referenz (Merit, versagte Strahlen, Max-Reduktionen), dieselben Varianten.
TEST(CudaMerit, MatchesCpu) {
    MKO_REQUIRE_GPU();
    for (const Case& cs : kCases) {
        SCOPED_TRACE(cs.file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(cs.file);
        const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, cs.seed);
        const ResidualLayout layout(batch.topology(), cs.settings);
        const MeritResult gpu = gpu_merit(batch, cs.settings);
        const MeritResult cpu = mko::merit_cpu(batch, layout, true);
        const CpuComparison c = compare_with_cpu(gpu, cpu, mko::residuals_cpu(batch, layout),
                                                 mko::test::residual_units(batch, layout));
        std::printf("[ Info     ] %s: max |√F_GPU - √F_CPU| = %.3e Wellen, max |dh| = %.3e mm, "
                    "max |d cos I| = %.3e\n",
                    cs.file, c.max_sqrt_merit, c.max_height, c.max_cos);
    }
}

// Geisterplatten (M4-T5): MERIT lädt die Geisterdaten wie RESIDUALS in den dynamischen Shared
// Memory. 1.000 gestörte Varianten von einzellinse_geisterplatte.json (auch Koeffizienten und
// Gläser gestört): Merit gegen Σ RESIDUALS² und CPU, Extrema gegen CPU und bitgleich zu
// RESIDUALS. Die Extrema beziehen sich auf Tabellenflächen (Dateiflächen plus Geisterplatten).
TEST(CudaMerit, GhostPlates) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_geisterplatte.json");
    SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, 10010);
    ASSERT_GT(batch.ghost_plates(), 0);
    std::mt19937_64 rng(10011);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int v = 1; v < batch.variants(); ++v) {
        for (int g = 0; g < batch.ghost_plates(); ++g) {
            for (int k = 0; k < batch.ghost_terms(); ++k) {
                batch.ghost_coeff(v, g, k) =
                    batch.ghost_coeff(v, g, k) * (1.0 + 0.5 * u(rng)) + 2e-4 * u(rng) / (k + 1);
            }
            const double dn = 0.05 * u(rng);
            for (int l = 0; l < batch.wavelengths(); ++l) {
                batch.ghost_index(v, g, l) += dn;
            }
        }
    }
    ASSERT_NO_THROW(batch.validate_values());
    const ResidualLayout layout(batch.topology());
    const MeritResult gpu = gpu_merit(batch);
    const ResidualResult r = gpu_residuals(batch);
    const MeritComparison c = compare_with_residuals(gpu, r, layout);
    EXPECT_TRUE(same_bytes(r.extrema.values, gpu.extrema.values));
    EXPECT_EQ(gpu.extrema.surfaces, batch.surfaces()); // Tabellenflächen
    const MeritResult cpu = mko::merit_cpu(batch, layout, true);
    const CpuComparison d = compare_with_cpu(gpu, cpu, mko::residuals_cpu(batch, layout),
                                             mko::test::residual_units(batch, layout));
    // Die Geisterplatten sind Ebenen (c = 0): cos I = d_z vor dem Phasenschritt, also <= 1
    for (int s = 0; s < batch.surfaces(); ++s) {
        if (batch.type(s) == mko::SurfaceKind::GhostPlate) {
            EXPECT_LE(gpu.extrema.min_cos_incidence(0, s), 1.0);
            EXPECT_GT(gpu.extrema.max_height(0, s), 0.0);
        }
    }
    std::printf("[ Info     ] %s (G = %d, K = %d, S = %d Tabellenflächen): max |F - Σr²| / Σr² = "
                "%.2e, %d mit versagten Strahlen, max |√F_GPU - √F_CPU| = %.3e Wellen, max |dh| = "
                "%.3e mm, max |d cos I| = %.3e\n",
                ls.system.name.c_str(), batch.ghost_plates(), batch.ghost_terms(), batch.surfaces(),
                c.max_relative, c.with_failures, d.max_sqrt_merit, d.max_height, d.max_cos);
}

// Punktquelle neben der Achse (volle Pupille) und S = kMaxSurfaces.
TEST(CudaMerit, PointSourceAndMaxSurfaces) {
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
    for (const SystemBatchHost& batch :
         {mko::test::perturbed_batch(ls, 200, 4004), mko::test::many_surface_batch(200, 6006)}) {
        const ResidualLayout layout(batch.topology());
        const MeritResult gpu = gpu_merit(batch);
        const MeritComparison c = compare_with_residuals(gpu, gpu_residuals(batch), layout);
        const MeritResult cpu = mko::merit_cpu(batch, layout, true);
        const CpuComparison d = compare_with_cpu(gpu, cpu, mko::residuals_cpu(batch, layout),
                                                 mko::test::residual_units(batch, layout));
        std::printf("[ Info     ] S = %d: max |F - Σr²| / Σr² = %.2e, max |√F_GPU - √F_CPU| = "
                    "%.3e Wellen, max |dh| = %.3e mm, max |d cos I| = %.3e\n",
                    batch.surfaces(), c.max_relative, d.max_sqrt_merit, d.max_height, d.max_cos);
    }
}

// Abnahme M2-T3 (ADR-0007): zehn Läufe mit gleichen Eingaben sind bitgleich (Merit, versagte
// Strahlen, Max-Reduktionen), auch mit wiederverwendeten Puffern nach einem Lauf anderer Größe,
// mit einer frischen Engine und für dieselben Varianten an anderer Stelle eines Batches.
TEST(CudaMerit, RepeatedRunsAreBitIdentical) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, 5005);
    TraceEngine engine(0);
    TraceSettings s;
    s.extrema = true;
    TraceOutput out;
    engine.run(batch, TraceMode::Merit, s, out);
    const MeritResult first = out.merit;
    ASSERT_EQ(first.variants, 1000);
    ASSERT_FALSE(first.extrema.empty());
    int identical_runs = 1;
    for (int run = 1; run < 10; ++run) {
        if (run == 5) {
            // Zwischendurch ein kleinerer Lauf ohne Extrema mit derselben Engine und Ausgabe
            engine.run(mko::test::perturbed_batch(ls, 10, 1), TraceMode::Merit, {}, out);
            EXPECT_TRUE(out.merit.extrema.empty());
        }
        engine.run(batch, TraceMode::Merit, s, out);
        const bool same = identical(out.merit, first);
        EXPECT_TRUE(same) << "Lauf " << run;
        identical_runs += same ? 1 : 0;
    }
    EXPECT_TRUE(identical(TraceEngine(0).run(batch, TraceMode::Merit, s).merit, first));

    // Lage im Batch ändert nichts: Varianten 500 … 509 als eigener Batch
    SystemBatchHost part = batch;
    for (int i = 0; i < 10; ++i) {
        part.copy_variant(500 + i, i);
    }
    part.resize(10);
    const MeritResult sub = engine.run(part, TraceMode::Merit, s).merit;
    ASSERT_EQ(sub.variants, 10);
    for (int i = 0; i < 10; ++i) {
        const auto v = static_cast<std::size_t>(500 + i);
        const auto j = static_cast<std::size_t>(i);
        EXPECT_EQ(std::memcmp(&sub.merit[j], &first.merit[v], sizeof(double)), 0) << i;
        EXPECT_EQ(sub.failed[j], first.failed[v]);
        EXPECT_EQ(std::memcmp(&sub.extrema.values[sub.extrema.index(i, 0)],
                              &first.extrema.values[first.extrema.index(500 + i, 0)],
                              2 * sizeof(double) * static_cast<std::size_t>(batch.surfaces())),
                  0);
    }
    RecordProperty("identical_runs", std::to_string(identical_runs));
}

// Baustein DeviceMerit: Puffer wird über Läufe verschiedener Größe wiederverwendet,
// Gerätezeiger für Folgekernel, Variantenbereich, Ergebnis wie über die TraceEngine.
TEST(CudaMerit, DeviceBufferReuse) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 300, 5005);
    const ResidualLayout layout(batch.topology());
    const DeviceSystemBatch device(batch);
    MeritRunOptions opt;
    opt.extrema = true;
    DeviceMerit out;
    out.run(device, layout, opt);
    const MeritResult first = out.download();
    EXPECT_EQ(out.variants(), 300);
    EXPECT_TRUE(out.has_extrema());
    EXPECT_NE(out.device_merit(), nullptr);
    EXPECT_NE(out.device_failed(), nullptr);
    EXPECT_NE(out.device_extrema(), nullptr);
    const std::size_t capacity = out.capacity_bytes();
    // Kleinerer Lauf ohne Extrema in denselben Puffer
    out.run(DeviceSystemBatch(mko::test::perturbed_batch(ls, 10, 1)), layout);
    EXPECT_EQ(out.capacity_bytes(), capacity);
    EXPECT_EQ(out.device_extrema(), nullptr);
    EXPECT_TRUE(out.download().extrema.empty());
    out.run(device, layout, opt);
    EXPECT_TRUE(identical(out.download(), first));
    EXPECT_TRUE(identical(mko::cuda::run_merit(device, layout, opt), first));
    EXPECT_TRUE(identical(gpu_merit(batch), first));
    // Variantenbereich [100, 110) auf dem Gerät mit den Nummern 0 … 9
    DeviceSystemBatch part(batch);
    part.upload(batch, 100, 10);
    const MeritResult sub = mko::cuda::run_merit(part, layout, opt);
    ASSERT_EQ(sub.variants, 10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(std::memcmp(&sub.merit[static_cast<std::size_t>(i)],
                              &first.merit[static_cast<std::size_t>(100 + i)], sizeof(double)),
                  0);
    }
}

// Grenzfälle: Variante ohne gültigen Strahl (Merit 0, alle Strahlen versagt, Extrema neutral),
// teilweise Versagen, Nachbarvarianten unberührt; geschlossene Formeln für Höhe und
// Einfallswinkel an Blende und erster Fläche eines achsparallelen Bündels.
TEST(CudaMerit, FailedRaysAndClosedForm) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem lens = mko::test::load_residual_system("einzellinse_bk7.json");
    SystemBatchHost batch = mko::make_system_batch(lens, 5);
    batch.pupil_radius(1) = 1e4;   // alle Strahlen verfehlen die Linse (R = 103 mm)
    batch.pupil_radius(2) = 110.0; // äußere Ringe verfehlen
    batch.curvature(3, 1) = 0.2;   // R = 5 mm: fast alle Strahlen verfehlen oder reflektieren
    const ResidualLayout layout(batch.topology());
    const MeritResult gpu = gpu_merit(batch);
    const MeritResult cpu = mko::merit_cpu(batch, layout, true);
    compare_with_cpu(gpu, cpu, mko::residuals_cpu(batch, layout),
                     mko::test::residual_units(batch, layout));
    compare_with_residuals(gpu, gpu_residuals(batch), layout);
    EXPECT_EQ(gpu.failed[0], 0);
    EXPECT_EQ(gpu.failed[4], 0);
    EXPECT_EQ(gpu.failed[1], layout.rays_per_variant());
    EXPECT_EQ(gpu.merit[1], 0.0);
    EXPECT_GT(gpu.failed[2], 0);
    EXPECT_GT(gpu.failed[3], 0);
    EXPECT_EQ(gpu.merit[0], gpu.merit[4]); // gleiche Variante, andere Lage
    for (int s = 0; s < batch.surfaces(); ++s) {
        EXPECT_EQ(gpu.extrema.max_height(1, s), 0.0);
        EXPECT_EQ(gpu.extrema.min_cos_incidence(1, s), 1.0);
        EXPECT_EQ(gpu.extrema.max_height(0, s), gpu.extrema.max_height(4, s));
    }
    EXPECT_LT(gpu.extrema.max_height(2, 1), 1.0 / batch.curvature(2, 1));

    // Achsparalleles Bündel (nur Feld 0): h = a·ρ_N an Blende und Fläche 1,
    // cos I = 1 bzw. √(1 − (c h)²).
    mko::SystemDescription d = lens.system;
    d.source.field_angles_deg = {0.0};
    d.source.field_weights = {1.0};
    const SystemBatchHost axis = mko::make_system_batch(
        mko::prepare_system(d, std::filesystem::path(mko::test::kResidualSystems)), 1);
    const ResidualLayout axis_layout(axis.topology());
    const MeritResult a = gpu_merit(axis);
    const mko::PupilQuadrature<double>& q = axis_layout.quadrature(0);
    const double h = axis.pupil_radius(0) * std::sqrt(q.u[q.rings - 1]);
    const double c = axis.curvature(0, 1);
    EXPECT_NEAR(a.extrema.max_height(0, 0), h, mko::test::kLengthFp64);
    EXPECT_EQ(a.extrema.min_cos_incidence(0, 0), 1.0);
    EXPECT_NEAR(a.extrema.max_height(0, 1), h, mko::test::kLengthFp64);
    EXPECT_NEAR(a.extrema.min_cos_incidence(0, 1), std::sqrt(1.0 - c * h * c * h),
                mko::test::kUnitFp64);
}

// Erweiterungspunkt weitere Residuen: F_v = (Σ Teilsummen) + extra_merit[v], also bitgleich zur
// Summe auf dem Host.
TEST(CudaMerit, ExtraMeritIsAddedLast) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 100, 9009);
    const MeritResult base = gpu_merit(batch, {}, false);
    std::vector<double> extra(100);
    for (std::size_t v = 0; v < extra.size(); ++v) {
        extra[v] = 1e-3 * static_cast<double>(v) + 1e-9;
    }
    const MeritResult with = engine_run(batch, TraceMode::Merit, {}, false, extra).merit;
    for (std::size_t v = 0; v < extra.size(); ++v) {
        const double expected = base.merit[v] + extra[v];
        EXPECT_EQ(std::memcmp(&with.merit[v], &expected, sizeof(double)), 0) << v;
    }
}

// Großer Batch (10.000 Varianten) gegen die CPU; grobe Durchsatzzahlen über die TraceEngine
// einschließlich Upload und Download: MERIT (mit und ohne Max-Reduktionen) und RESIDUALS
// (der eigentliche Benchmark ist M2-T6).
TEST(CudaMerit, LargeBatch) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 10000, 7007);
    const ResidualLayout layout(batch.topology());
    TraceEngine engine(0);
    TraceOutput merit_out;
    TraceOutput residual_out;
    constexpr int kRuns = 5;
    const auto time = [&](TraceMode mode, const TraceSettings& s, TraceOutput& out) {
        engine.run(batch, mode, s, out); // Aufwärmen, Puffer anlegen
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kRuns; ++i) {
            engine.run(batch, mode, s, out);
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / kRuns;
    };
    TraceSettings plain;
    TraceSettings with_extrema;
    with_extrema.extrema = true;
    const double t_merit = time(TraceMode::Merit, plain, merit_out);
    const double t_residuals = time(TraceMode::Residuals, plain, residual_out);
    const double t_extrema = time(TraceMode::Merit, with_extrema, merit_out);
    const MeritResult& gpu = merit_out.merit;
    ASSERT_EQ(gpu.variants, 10000);
    const MeritResult cpu = mko::merit_cpu(batch, layout, true);
    const CpuComparison c = compare_with_cpu(gpu, cpu, mko::residuals_cpu(batch, layout),
                                             mko::test::residual_units(batch, layout));
    const MeritComparison r = compare_with_residuals(gpu, residual_out.residuals, layout);
    const double rays = 10000.0 * static_cast<double>(layout.rays_per_variant());
    std::printf(
        "[ Info     ] 10.000 Varianten x %lld Strahlen (S = %d), TraceEngine mit Upload und "
        "Download: MERIT %.3f ms (%.3e Strahlen/s), MERIT + Extrema %.3f ms (%.3e "
        "Strahlen/s), RESIDUALS %.3f ms (%.3e Strahlen/s); max |√F_GPU - √F_CPU| = %.3e "
        "Wellen, max |F - Σr²| / Σr² = %.2e\n",
        static_cast<long long>(layout.rays_per_variant()), batch.surfaces(), t_merit * 1e3,
        rays / t_merit, t_extrema * 1e3, rays / t_extrema, t_residuals * 1e3, rays / t_residuals,
        c.max_sqrt_merit, r.max_relative);
    RecordProperty("merit_rays_per_second", std::to_string(rays / t_merit));
    RecordProperty("merit_extrema_rays_per_second", std::to_string(rays / t_extrema));
    RecordProperty("residuals_rays_per_second", std::to_string(rays / t_residuals));
}

// Baustein DeviceMerit: eigene Fehlerfälle (die der TraceEngine prüft
// test_cuda_trace_engine.cpp).
TEST(CudaMerit, InvalidUseThrows) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 2);
    const ResidualLayout layout(batch.topology());
    DeviceMerit out;
    EXPECT_THROW((void)out.download(), std::logic_error);
    EXPECT_EQ(out.device_merit(), nullptr);
    EXPECT_EQ(out.device_failed(), nullptr);
    const DeviceSystemBatch empty;
    EXPECT_THROW(out.run(empty, layout), std::logic_error);
    const DeviceSystemBatch device(batch);
    // Layout eines anderen Systems
    const mko::LoadedSystem other = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const ResidualLayout wrong(mko::make_system_batch(other, 1).topology());
    EXPECT_THROW(out.run(device, wrong), std::invalid_argument);
    // extra_merit mit falscher Länge
    const std::vector<double> extra(3, 0.0);
    MeritRunOptions bad;
    bad.extra_merit = extra;
    EXPECT_THROW(out.run(device, layout, bad), std::invalid_argument);
    // Zu viele Stützstellen pro Feld für das Shared Memory
    const int limit = mko::cuda::merit_max_samples(true);
    EXPECT_GT(limit, 1000);
    EXPECT_GE(mko::cuda::merit_max_samples(false), limit);
    std::printf("[ Info     ] MERIT: höchstens %d Stützstellen pro Feld (mit Extrema), %d (ohne)\n",
                limit, mko::cuda::merit_max_samples(false));
    const ResidualLayout huge(batch.topology(), ResidualSettings{16, limit / 16 + 1});
    MeritRunOptions opt;
    opt.extrema = true;
    EXPECT_THROW(out.run(device, huge, opt), std::invalid_argument);
    // Genau an der Grenze geht es
    const ResidualLayout at_limit(batch.topology(), ResidualSettings{16, limit / 16});
    EXPECT_NO_THROW(out.run(device, at_limit, opt));
    // Nach den Fehlern funktioniert der Puffer weiter
    EXPECT_NO_THROW(out.run(device, layout));
    EXPECT_EQ(out.download().variants, 2);
}

// Abnahme M4-T1 auf dem Gerät (Rechentyp FP64 und df64): Merit des Kernels MERIT gegen
// Σ_f Σ_λ ŵ_f ŵ_λ (RMS_fλ / λ_ref)²
// (a) mit dem RMS aus der OPD desselben Gerätelaufs (RESIDUALS mit OPD, tuple_wavefronts):
//     < kWeightedMerit relativ; prüft die Gewichte im Kernel (dieselbe OPD auf beiden Seiten);
// (b) mit dem RMS aus evaluate_wavefront (M1-T7, CPU): √F ist die 2-Norm der Residuen
//     s_{f,l} √W_k OPD_k, also |√F_GPU − √F_M1| <= δ · √(Σ_(f,l) s²) (Σ_k W_k = 1 pro Tupel)
//     mit der Schranke δ der OPD (kCpuGpuFp64 bzw. kDf64VsFp64, alle |OPD| < 0,1 mm) plus
//     Summationsrundung kWeightedMerit · √F. Die relative Abweichung wird ausgegeben.
TEST(CudaMerit, EqualsWeightedSumOfWavefrontRms) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    for (const mko::cuda::Precision p : {mko::cuda::Precision::Fp64, mko::cuda::Precision::Df64}) {
        const bool df64 = p == mko::cuda::Precision::Df64;
        const double opd_bound = df64 ? mko::test::kDf64VsFp64 : kCpuGpuFp64;
        double worst_same = 0.0;
        double worst_m1 = 0.0;
        for (const mko::test::WeightedCase& c : mko::test::weighted_cases()) {
            SCOPED_TRACE(c.name + (df64 ? " (df64)" : " (FP64)"));
            const SystemBatchHost batch = mko::make_system_batch(c.system, 1);
            const double lambda_mm = c.system.system.spectrum.reference_nm * 1e-6;
            TraceSettings s;
            s.precision = p;
            s.opd = true;
            const TraceOutput res = engine.run(batch, TraceMode::Residuals, s);
            const TraceOutput mer = engine.run(batch, TraceMode::Merit, s);
            ASSERT_EQ(res.residuals.failed_count(), 0u);
            const double f_gpu = mer.merit.merit[0];

            std::vector<double> rms_gpu;
            for (const mko::TupleWavefront& t :
                 mko::tuple_wavefronts(res.residuals, res.layout, 0)) {
                rms_gpu.push_back(t.rms / lambda_mm);
            }
            const double same = mko::test::weighted_rms_merit(c.system.system, rms_gpu);
            const double rel_same = std::abs(f_gpu - same) / same;
            EXPECT_LT(rel_same, kWeightedMerit) << f_gpu << " vs " << same;

            const mko::WavefrontResult wf = mko::evaluate_wavefront(c.system);
            std::vector<double> rms_m1;
            for (const mko::WavefrontMap& m : wf.maps) {
                rms_m1.push_back(m.rms_waves);
            }
            const double m1 = mko::test::weighted_rms_merit(c.system.system, rms_m1);
            const double bound = opd_bound * mko::test::residual_scale_norm(batch, res.layout) +
                                 kWeightedMerit * std::sqrt(m1);
            EXPECT_LE(std::abs(std::sqrt(f_gpu) - std::sqrt(m1)), bound);
            const double rel_m1 = std::abs(f_gpu - m1) / m1;
            worst_same = std::max(worst_same, rel_same);
            worst_m1 = std::max(worst_m1, rel_m1);
            std::printf("[ Info     ] %s %-46s F = %.15e Wellen², relativ zu Σ ŵ ŵ RMS² (gleicher "
                        "Lauf) %.2e, zu M1-T7 (CPU) %.2e\n",
                        df64 ? "df64" : "FP64", c.name.c_str(), f_gpu, rel_same, rel_m1);
        }
        RecordProperty(std::string(df64 ? "df64" : "fp64") + "_max_relative_same_run",
                       std::to_string(worst_same));
        RecordProperty(std::string(df64 ? "df64" : "fp64") + "_max_relative_m1t7",
                       std::to_string(worst_m1));
    }
}
