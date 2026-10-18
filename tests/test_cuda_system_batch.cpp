// SystemBatch im Gerätespeicher (M2-T1, nur mit MIKROPTIM_ENABLE_CUDA): Upload und Download
// verlustfrei, große Batches, Speicherabfrage, Layoutprüfung per Trace auf der GPU.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_trace_cpu.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/batch_trace_check.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using mko::BatchTopology;
using mko::SurfaceKind;
using mko::SystemBatchHost;
using mko::cuda::DeviceSystemBatch;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

BatchTopology ghost_topology() {
    BatchTopology t;
    t.surfaces = 6;
    t.wavelengths = 3;
    t.fields = 4;
    t.ghost_plates = 2;
    t.ghost_terms = 5;
    t.type = {SurfaceKind::Stop,   SurfaceKind::Sphere,     SurfaceKind::GhostPlate,
              SurfaceKind::Sphere, SurfaceKind::GhostPlate, SurfaceKind::Plane};
    return t;
}

// Füllt alle Felder mit zufälligen Bitmustern (auch NaN mit Nutzlast, ±0, Subnormale, ±inf),
// damit ein Upload/Download-Fehler an jeder Stelle auffällt.
void fill_random_bits(SystemBatchHost& b, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const double specials[] = {-0.0,
                               0.0,
                               std::numeric_limits<double>::denorm_min(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::bit_cast<double>(std::uint64_t{0x7ff8dead0000beefULL})};
    std::size_t n = 0;
    const auto fill = [&](std::span<double> values) {
        for (double& x : values) {
            x = (n++ % 97 < std::size(specials)) ? specials[n % std::size(specials)]
                                                 : std::bit_cast<double>(rng());
        }
    };
    fill(b.ghost_norm_radius_data());
    fill(b.wavelength_nm_data());
    fill(b.wavelength_weight_data());
    fill(b.field_weight_data());
    fill(b.curvature_data());
    fill(b.thickness_data());
    fill(b.index_data());
    fill(b.pupil_radius_data());
    fill(b.field_angle_data());
    fill(b.field_target_data());
    fill(b.ghost_coeff_data());
    fill(b.ghost_index_data());
    b.reference_wavelength_nm = std::bit_cast<double>(rng());
    b.source_point = {std::bit_cast<double>(rng()), -0.0, std::bit_cast<double>(rng())};
    b.source_na = std::bit_cast<double>(rng());
}

// Gestörte Varianten der Einzellinse (Krümmungen, Dicken, Pupillenradius, Feldwinkel).
SystemBatchHost perturbed_lens_batch(int variants, std::uint64_t seed) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_bk7.json");
    SystemBatchHost b = mko::make_system_batch(ls, variants);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int v = 1; v < variants; ++v) {
        b.curvature(v, 1) *= 1.0 + 0.2 * u(rng);
        b.curvature(v, 2) *= 1.0 + 0.2 * u(rng);
        b.thickness(v, 0) += 2.0 * u(rng);
        b.thickness(v, 1) += 1.0 * u(rng);
        b.thickness(v, 2) += 5.0 * u(rng);
        b.pupil_radius(v) = 10.0 + 3.0 * u(rng);
        for (int f = 1; f < b.fields(); ++f) {
            b.field_angle(v, f) *= 1.0 + 0.5 * u(rng);
        }
        for (int l = 0; l < b.wavelengths(); ++l) {
            b.index(v, 1, l) += 0.05 * u(rng);
        }
    }
    return b;
}

} // namespace

TEST(CudaSystemBatch, DeviceMemoryInfo) {
    MKO_REQUIRE_GPU();
    const mko::cuda::DeviceMemoryInfo info = mko::cuda::device_memory_info();
    EXPECT_GT(info.total_bytes, 0u);
    EXPECT_GT(info.free_bytes, 0u);
    EXPECT_LE(info.free_bytes, info.total_bytes);
    EXPECT_THROW(mko::cuda::device_memory_info(mko::cuda::device_count()), mko::cuda::CudaError);
    std::printf("[ Info     ] GPU 0: %.2f GiB frei von %.2f GiB\n",
                static_cast<double>(info.free_bytes) / (1 << 30),
                static_cast<double>(info.total_bytes) / (1 << 30));
}

// Abnahme M2-T1: Upload und Download sind verlustfrei (bitgleich, alle Felder).
TEST(CudaSystemBatch, UploadDownloadIsLossless) {
    MKO_REQUIRE_GPU();
    SystemBatchHost host(ghost_topology(), 37);
    fill_random_bits(host, 12345);
    const DeviceSystemBatch device(host);
    EXPECT_FALSE(device.empty());
    EXPECT_EQ(device.variants(), 37);
    EXPECT_EQ(device.layout().total_bytes, mko::make_batch_layout(host.topology(), 37).total_bytes);
    const SystemBatchHost back = device.download();
    EXPECT_TRUE(back == host);

    // Download in einen bestehenden, größeren Batch an eine versetzte Stelle
    SystemBatchHost big(ghost_topology(), 40);
    device.download(big, 3);
    for (int v = 0; v < 37; ++v) {
        for (int s = 0; s < 6; ++s) {
            ASSERT_EQ(std::bit_cast<std::uint64_t>(big.curvature(v + 3, s)),
                      std::bit_cast<std::uint64_t>(host.curvature(v, s)));
        }
        for (int g = 0; g < 2; ++g) {
            for (int k = 0; k < 5; ++k) {
                ASSERT_EQ(std::bit_cast<std::uint64_t>(big.ghost_coeff(v + 3, g, k)),
                          std::bit_cast<std::uint64_t>(host.ghost_coeff(v, g, k)));
            }
        }
    }
    EXPECT_EQ(big.curvature(0, 1), 0.0);
    EXPECT_EQ(big.curvature(2, 5), 0.0);
}

TEST(CudaSystemBatch, UploadRangeAndReuse) {
    MKO_REQUIRE_GPU();
    SystemBatchHost host(ghost_topology(), 20);
    fill_random_bits(host, 777);
    DeviceSystemBatch device;
    EXPECT_TRUE(device.empty());
    device.upload(host, 5, 10); // Varianten 5 … 14 als 0 … 9
    EXPECT_EQ(device.variants(), 10);
    const std::size_t capacity = device.capacity_bytes();
    SystemBatchHost expected(ghost_topology(), 10);
    for (int v = 0; v < 10; ++v) {
        expected.copy_variant_from(host, v + 5, v);
    }
    for (int g = 0; g < 2; ++g) {
        expected.ghost_norm_radius(g) = host.ghost_norm_radius(g);
    }
    for (int l = 0; l < 3; ++l) {
        expected.wavelength_nm(l) = host.wavelength_nm(l);
        expected.wavelength_weight(l) = host.wavelength_weight(l);
    }
    for (int f = 0; f < 4; ++f) {
        expected.field_weight(f) = host.field_weight(f);
    }
    expected.reference_wavelength_nm = host.reference_wavelength_nm;
    expected.source_point = host.source_point;
    expected.source_na = host.source_na;
    EXPECT_TRUE(device.download() == expected);

    // Kleinerer Upload nutzt den Puffer weiter; Rückschreiben an die Ursprungsstelle
    device.upload(host, 18);
    EXPECT_EQ(device.variants(), 2);
    EXPECT_EQ(device.capacity_bytes(), capacity);
    SystemBatchHost copy = host;
    for (int v = 18; v < 20; ++v) {
        copy.pupil_radius(v) = 0.0;
    }
    device.download(copy, 18);
    EXPECT_TRUE(copy == host);

    // Nur gemeinsame Felder
    device.upload(host, 20, 0);
    EXPECT_EQ(device.variants(), 0);
    EXPECT_EQ(device.download().variants(), 0);

    // Verschieben übergibt den Puffer
    DeviceSystemBatch moved(std::move(device));
    EXPECT_TRUE(device.empty()); // NOLINT(bugprone-use-after-move): definierter Zustand
    EXPECT_FALSE(moved.empty());
    EXPECT_EQ(device.capacity_bytes(), 0u);
    moved.release();
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(moved.capacity_bytes(), 0u);
}

TEST(CudaSystemBatch, ViewHasDevicePointers) {
    MKO_REQUIRE_GPU();
    SystemBatchHost host(ghost_topology(), 3);
    fill_random_bits(host, 1);
    const DeviceSystemBatch device(host);
    const mko::SystemBatchView<double> v = device.view();
    EXPECT_EQ(v.shape.variants, 3);
    EXPECT_EQ(v.shape.surfaces, 6);
    EXPECT_EQ(v.shape.ghost_terms, 5);
    const void* ptrs[] = {
        v.type,         v.ghost_surface, v.ghost_norm_radius, v.wavelength_nm, v.wavelength_weight,
        v.field_weight, v.curvature,     v.thickness,         v.index,         v.pupil_radius,
        v.field_angle,  v.field_target,  v.ghost_coeff,       v.ghost_index};
    for (const void* p : ptrs) {
        EXPECT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % mko::kBatchAlignment, 0u);
    }
    EXPECT_NE(static_cast<const void*>(v.curvature),
              static_cast<const void*>(host.curvature_data().data()));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(v.source_na),
              std::bit_cast<std::uint64_t>(host.source_na));
}

TEST(CudaSystemBatch, Errors) {
    MKO_REQUIRE_GPU();
    const DeviceSystemBatch empty;
    EXPECT_THROW(empty.view(), std::logic_error);
    EXPECT_THROW(empty.download(), std::logic_error);

    SystemBatchHost host(ghost_topology(), 4);
    DeviceSystemBatch device(host);
    EXPECT_THROW(device.upload(host, 5), std::out_of_range);
    EXPECT_THROW(device.upload(host, 2, 3), std::out_of_range);
    EXPECT_TRUE(device.empty() == false); // Prüfung vor jeder Änderung
    SystemBatchHost small(ghost_topology(), 3);
    EXPECT_THROW(device.download(small), std::out_of_range);
    EXPECT_THROW(device.download(host, 1), std::out_of_range);
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_bk7.json");
    SystemBatchHost other = mko::make_system_batch(ls, 4);
    EXPECT_THROW(device.download(other), std::invalid_argument);
}

// Großer Batch (10.000 gestörte Varianten): Speicherabschätzung gegen den freien Speicher,
// Upload und Download bitgleich.
TEST(CudaSystemBatch, LargeBatch) {
    MKO_REQUIRE_GPU();
    constexpr int kVariants = 10000;
    SystemBatchHost host = perturbed_lens_batch(kVariants, 2026);
    mko::OutputRequest out;
    out.rays_per_variant = mko::rays_per_variant(3, 3, 80, 1, 10);
    const int max_v = mko::cuda::max_variants_on_device(host.topology(), out);
    std::printf("[ Info     ] Einzellinse, %lld Strahlen/Variante: max. %d Varianten auf GPU 0, "
                "Batch mit %d Varianten: %zu Byte\n",
                static_cast<long long>(out.rays_per_variant), max_v, kVariants,
                mko::make_batch_layout(host.topology(), kVariants).total_bytes);
    ASSERT_GE(max_v, kVariants);
    const DeviceSystemBatch device(host);
    EXPECT_TRUE(device.download() == host);
}

// Layoutprüfung: trace_ray über die Gerätesicht (globaler Speicher) stimmt für alle Strahlen
// gestörter Varianten mit dem CPU-Trace der Batch-Variante überein (keine Kernel aus M2-T2).
TEST(CudaSystemBatch, GpuTraceMatchesCpu) {
    MKO_REQUIRE_GPU();
    constexpr int kVariants = 64;
    SystemBatchHost host = perturbed_lens_batch(kVariants, 99);
    // Eine Variante mit großer Blende und stark gekrümmter Fläche (Versagen)
    host.pupil_radius(kVariants - 1) = 14.0;
    host.curvature(kVariants - 1, 2) = -0.08;
    const DeviceSystemBatch device(host);
    const auto q = mko::make_pupil_quadrature(8, 8, mko::PupilMode::Full);
    std::vector<mko::Ray<double>> rays;
    std::vector<unsigned> status;
    std::vector<int> surface;
    mko::cuda::trace_batch_gpu(device, q, rays, status, surface);
    const int n = mko::sample_count(q);
    ASSERT_EQ(rays.size(), static_cast<std::size_t>(kVariants) * 3 * 3 * n);

    double max_err = 0.0;
    std::size_t failed = 0;
    for (int v = 0; v < kVariants; ++v) {
        const mko::CpuTraceResult cpu = mko::trace_variant_cpu(host, v, q);
        for (int f = 0; f < 3; ++f) {
            for (int l = 0; l < 3; ++l) {
                for (int k = 0; k < n; ++k) {
                    const std::size_t c = cpu.at(f, l, k);
                    const std::size_t g = ((static_cast<std::size_t>(v) * 3 + f) * 3 + l) * n + k;
                    ASSERT_EQ(static_cast<unsigned>(cpu.status[c]), status[g]);
                    ASSERT_EQ(cpu.failed_surface[c], surface[g]);
                    if (cpu.status[c] != mko::TraceStatus::Ok) {
                        ++failed;
                        continue;
                    }
                    const auto& a = cpu.rays[c];
                    const auto& b = rays[g];
                    const double err = std::max({std::abs(a.p.x - b.p.x), std::abs(a.p.y - b.p.y),
                                                 std::abs(a.p.z - b.p.z), std::abs(a.d.x - b.d.x),
                                                 std::abs(a.d.y - b.d.y), std::abs(a.d.z - b.d.z),
                                                 std::abs(a.opl - b.opl)});
                    ASSERT_LE(err, mko::test::kCpuGpuFp64) << "v=" << v << " f=" << f;
                    max_err = std::max(max_err, err);
                }
            }
        }
    }
    EXPECT_GT(failed, 0u);
    std::printf("[ Info     ] Batch-Trace CPU/GPU: max Abweichung = %.3e (%zu Strahlen, davon %zu "
                "mit gleichem Versagensstatus)\n",
                max_err, rays.size(), failed);
}
