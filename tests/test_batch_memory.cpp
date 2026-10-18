// Speicherabschätzung und Ablage im Gerätepuffer (M2-T1), rein auf dem Host.
// Die erwarteten Byteszahlen sind von Hand aus den Formeln in docs/architektur.md berechnet.

#include "mikroptim/batch/batch_memory.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

using mko::BatchTopology;
using mko::OutputRequest;
using mko::SurfaceKind;

namespace {

// Wie die Einzellinse: S = 3, L = 3, F = 3, keine Geisterplatten.
BatchTopology lens_topology() {
    BatchTopology t;
    t.surfaces = 3;
    t.wavelengths = 3;
    t.fields = 3;
    t.type = {SurfaceKind::Stop, SurfaceKind::Sphere, SurfaceKind::Sphere};
    return t;
}

// S = 6, L = 3, F = 4, G = 2, K = 5.
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

// Zielsystem-Größenordnung: 30 Flächen, 7 Wellenlängen, 5 Felder, 6 Geisterplatten à 8 Terme.
BatchTopology objective_topology() {
    BatchTopology t;
    t.surfaces = 30;
    t.wavelengths = 7;
    t.fields = 5;
    t.ghost_plates = 6;
    t.ghost_terms = 8;
    t.type.assign(30, SurfaceKind::Sphere);
    t.type[0] = SurfaceKind::Stop;
    for (int s : {3, 8, 13, 18, 23, 27}) {
        t.type[static_cast<std::size_t>(s)] = SurfaceKind::GhostPlate;
    }
    return t;
}

} // namespace

TEST(BatchMemory, AlignBytes) {
    EXPECT_EQ(mko::kBatchAlignment, 256u);
    EXPECT_EQ(mko::align_bytes(0), 0u);
    EXPECT_EQ(mko::align_bytes(1), 256u);
    EXPECT_EQ(mko::align_bytes(256), 256u);
    EXPECT_EQ(mko::align_bytes(257), 512u);
}

TEST(BatchMemory, BatchLayoutLens) {
    const mko::SystemBatchLayout b = mko::make_batch_layout(lens_topology(), 10);
    // gemeinsam: type 3 B -> 256, keine Geisterplatten, wavelength_nm/weight je 24 B -> 256,
    // field_weight 24 B -> 256
    EXPECT_EQ(b.type.offset, 0u);
    EXPECT_EQ(b.type.bytes, 3u);
    EXPECT_EQ(b.ghost_surface.offset, 256u);
    EXPECT_EQ(b.ghost_surface.bytes, 0u);
    EXPECT_EQ(b.ghost_norm_radius.offset, 256u);
    EXPECT_EQ(b.wavelength_nm.offset, 256u);
    EXPECT_EQ(b.wavelength_nm.bytes, 24u);
    EXPECT_EQ(b.wavelength_weight.offset, 512u);
    EXPECT_EQ(b.field_weight.offset, 768u);
    // pro Variante: curvature/thickness 10·3·8 = 240 -> 256, index 10·9·8 = 720 -> 768,
    // pupil 80 -> 256, field_angle/target 240 -> 256
    EXPECT_EQ(b.curvature.offset, 1024u);
    EXPECT_EQ(b.curvature.bytes, 240u);
    EXPECT_EQ(b.thickness.offset, 1280u);
    EXPECT_EQ(b.index.offset, 1536u);
    EXPECT_EQ(b.index.bytes, 720u);
    EXPECT_EQ(b.pupil_radius.offset, 2304u);
    EXPECT_EQ(b.pupil_radius.bytes, 80u);
    EXPECT_EQ(b.field_angle.offset, 2560u);
    EXPECT_EQ(b.field_target.offset, 2816u);
    EXPECT_EQ(b.ghost_coeff.offset, 3072u);
    EXPECT_EQ(b.ghost_coeff.bytes, 0u);
    EXPECT_EQ(b.ghost_index.offset, 3072u);
    // Brechzahlen bei λ_ref (M3-T3): paraxial_index 240 -> 256, paraxial_ghost_index 0
    EXPECT_EQ(b.paraxial_index.offset, 3072u);
    EXPECT_EQ(b.paraxial_index.bytes, 240u);
    EXPECT_EQ(b.paraxial_ghost_index.offset, 3328u);
    EXPECT_EQ(b.paraxial_ghost_index.bytes, 0u);
    EXPECT_EQ(b.total_bytes, 3328u);
}

TEST(BatchMemory, BatchLayoutGhostPlates) {
    const mko::SystemBatchLayout b = mko::make_batch_layout(ghost_topology(), 1000);
    // gemeinsam: 6 + 8 + 16 + 24 + 24 + 32 Byte, jeweils -> 256: 1536
    EXPECT_EQ(b.ghost_surface.bytes, 8u);
    EXPECT_EQ(b.ghost_norm_radius.bytes, 16u);
    EXPECT_EQ(b.curvature.offset, 1536u);
    // curvature/thickness 48000 -> 48128, index 144000 -> 144128, pupil 8000 -> 8192,
    // field_angle/target 32000, ghost_coeff 80000 -> 80128, ghost_index 48000 -> 48128
    EXPECT_EQ(b.ghost_coeff.bytes, 80000u);
    EXPECT_EQ(b.ghost_index.bytes, 48000u);
    EXPECT_EQ(b.ghost_index.offset, 442368u - 48128u);
    // paraxial_index 48000 -> 48128, paraxial_ghost_index 16000 -> 16128
    EXPECT_EQ(b.paraxial_index.offset, 442368u);
    EXPECT_EQ(b.paraxial_index.bytes, 48000u);
    EXPECT_EQ(b.paraxial_ghost_index.offset, 442368u + 48128u);
    EXPECT_EQ(b.paraxial_ghost_index.bytes, 16000u);
    EXPECT_EQ(b.total_bytes, 442368u + 48128u + 16128u);
    // Jeder Bereich beginnt ausgerichtet und überlappt nicht mit dem nächsten
    const mko::BufferRegion regions[] = {b.type,
                                         b.ghost_surface,
                                         b.ghost_norm_radius,
                                         b.wavelength_nm,
                                         b.wavelength_weight,
                                         b.field_weight,
                                         b.curvature,
                                         b.thickness,
                                         b.index,
                                         b.pupil_radius,
                                         b.field_angle,
                                         b.field_target,
                                         b.ghost_coeff,
                                         b.ghost_index,
                                         b.paraxial_index,
                                         b.paraxial_ghost_index};
    for (std::size_t i = 0; i < std::size(regions); ++i) {
        EXPECT_EQ(regions[i].offset % mko::kBatchAlignment, 0u);
        if (i + 1 < std::size(regions)) {
            EXPECT_LE(regions[i].offset + regions[i].bytes, regions[i + 1].offset);
        }
    }
}

TEST(BatchMemory, RaysPerVariant) {
    EXPECT_EQ(mko::rays_per_variant(3, 3, 80), 720);
    // ein Achsfeld mit N = 10 Stützstellen, zwei Feldpunkte mit N · M = 80
    EXPECT_EQ(mko::rays_per_variant(3, 3, 80, 1, 10), 510);
    EXPECT_EQ(mko::rays_per_variant(5, 7, 80, 1, 10), 7 * (10 + 4 * 80));
    EXPECT_THROW(mko::rays_per_variant(3, 3, 80, 4, 10), std::invalid_argument);
    EXPECT_THROW(mko::rays_per_variant(3, -1, 80), std::invalid_argument);
}

TEST(BatchMemory, OutputLayoutAndEstimate) {
    OutputRequest out;
    out.rays_per_variant = mko::rays_per_variant(3, 3, 80, 1, 10); // 510
    out.extra_residuals = 5;
    const mko::OutputLayout o = mko::make_output_layout(lens_topology(), 10, out, 100);
    // residuals 10·515·8 = 41200 -> 41216, status 5100 -> 5120, merit 80 -> 256,
    // merit_partial 10·9·8 = 720 -> 768, failed_rays 10·8 = 80 -> 256,
    // failed_rays_partial 10·9·4 = 360 -> 512, surface_max 10·2·3·8 = 480 -> 512,
    // surface_max_partial 10·9·2·3·8 = 4320 -> 4352, extra 1000 -> 1024
    EXPECT_EQ(o.residuals.bytes, 41200u);
    EXPECT_EQ(o.ray_status.offset, 41216u);
    EXPECT_EQ(o.ray_status.bytes, 5100u);
    EXPECT_EQ(o.merit.bytes, 80u);
    EXPECT_EQ(o.merit_partial.bytes, 720u);
    EXPECT_EQ(o.failed_rays.bytes, 80u);
    EXPECT_EQ(o.failed_rays_partial.bytes, 360u);
    EXPECT_EQ(o.surface_max.bytes, 480u);
    EXPECT_EQ(o.surface_max_partial.bytes, 4320u);
    EXPECT_EQ(o.extra.bytes, 1000u);
    EXPECT_EQ(o.total_bytes, 54016u);
    EXPECT_EQ(mko::estimate_device_bytes(lens_topology(), 10, out, 100), 3328u + 54016u);

    // Nicht angeforderte Ausgaben kosten nichts
    OutputRequest none;
    none.rays_per_variant = 510;
    none.residuals = false;
    none.ray_status = false;
    none.merit = false;
    none.surface_max = false;
    EXPECT_EQ(mko::make_output_layout(lens_topology(), 10, none).total_bytes, 0u);
    EXPECT_EQ(mko::estimate_device_bytes(lens_topology(), 10, none), 3328u);

    OutputRequest bad;
    bad.rays_per_variant = -1;
    EXPECT_THROW(mko::make_output_layout(lens_topology(), 1, bad), std::invalid_argument);
    EXPECT_THROW(mko::make_batch_layout(lens_topology(), -1), std::invalid_argument);
    BatchTopology broken = lens_topology();
    broken.type.pop_back();
    EXPECT_THROW(mko::estimate_device_bytes(broken, 1, out), std::invalid_argument);
}

TEST(BatchMemory, EstimateIsMonotone) {
    OutputRequest out;
    out.rays_per_variant = mko::rays_per_variant(5, 7, 80, 1, 10);
    const BatchTopology t = objective_topology();
    std::size_t previous = mko::estimate_device_bytes(t, 0, out);
    for (int v = 1; v <= 3000; ++v) {
        const std::size_t e = mko::estimate_device_bytes(t, v, out);
        ASSERT_GT(e, previous) << "V = " << v; // jede Variante kostet mindestens die Residuen
        previous = e;
    }
    // Nahezu linear: Abweichung von V · (Bytes pro Variante) nur durch Aufrundung pro Feld
    const std::size_t e1 = mko::estimate_device_bytes(t, 1000, out);
    const std::size_t e2 = mko::estimate_device_bytes(t, 2000, out);
    EXPECT_LE(e2, 2 * e1);
    EXPECT_GE(e2 + 64 * mko::kBatchAlignment, 2 * e1);
}

TEST(BatchMemory, UsableBytes) {
    const std::size_t MiB = std::size_t{1} << 20;
    const std::size_t GiB = std::size_t{1} << 30;
    // 1 GiB frei: Reserve max(256 MiB, 5 %) = 256 MiB
    EXPECT_EQ(mko::usable_bytes(GiB), 768 * MiB);
    // 8 GiB frei, Reserve max(256 MiB, 25 %) = 2 GiB (exakt darstellbarer Anteil)
    EXPECT_EQ(mko::usable_bytes(8 * GiB, mko::MemoryReserve{256 * MiB, 0.25}), 6 * GiB);
    // 24 GiB frei, Standardreserve 5 % ≈ 1,2 GiB
    EXPECT_NEAR(static_cast<double>(mko::usable_bytes(24 * GiB)), 0.95 * 24.0 * GiB, 1.0);
    EXPECT_EQ(mko::usable_bytes(100 * MiB), 0u);
    EXPECT_EQ(mko::usable_bytes(GiB, mko::MemoryReserve{0, 0.0}), GiB);
    EXPECT_THROW(mko::usable_bytes(GiB, mko::MemoryReserve{0, 1.5}), std::invalid_argument);
}

TEST(BatchMemory, MaxVariantsRespectsLimit) {
    OutputRequest out;
    out.rays_per_variant = mko::rays_per_variant(5, 7, 80, 1, 10);
    const BatchTopology t = objective_topology();
    const mko::MemoryReserve reserve{};
    const std::size_t MiB = std::size_t{1} << 20;
    for (std::size_t free_bytes : {300 * MiB, 1024 * MiB, 16384 * MiB, 24576 * MiB}) {
        const int v = mko::max_variants(free_bytes, t, out, reserve);
        const std::size_t usable = mko::usable_bytes(free_bytes, reserve);
        ASSERT_GT(v, 0) << free_bytes;
        EXPECT_LE(mko::estimate_device_bytes(t, v, out), usable);
        EXPECT_GT(mko::estimate_device_bytes(t, v + 1, out), usable);
    }
    // Zu wenig Speicher: 0 Varianten
    EXPECT_EQ(mko::max_variants(100 * MiB, t, out, reserve), 0);
    // Zusätzlicher Speicher pro Variante verkleinert den Batch
    EXPECT_LT(mko::max_variants(1024 * MiB, t, out, reserve, 1 << 20),
              mko::max_variants(1024 * MiB, t, out, reserve));
    // Riesiger Speicher: Grenze durch das Grid (V · F · L <= 2^31 − 1)
    const int grid_limit = static_cast<int>(mko::kMaxGridBlocks / (5 * 7));
    EXPECT_EQ(mko::max_variants(std::size_t{1} << 62, t, out, mko::MemoryReserve{0, 0.0}),
              grid_limit);
    BatchTopology broken = t;
    broken.fields = 0;
    EXPECT_THROW(mko::max_variants(1024 * MiB, broken, out), std::invalid_argument);

    // Zahlen für die Doku (RTX 3090 und P100, Zielsystem-Topologie, N = 10, M = 8)
    std::printf("[ Info     ] Zielsystem (S=30, L=7, F=5, G=6, K=8, %lld Strahlen/Variante): "
                "%zu Byte/Variante; max. Varianten bei 16 GiB frei: %d, bei 24 GiB frei: %d\n",
                static_cast<long long>(out.rays_per_variant),
                mko::estimate_device_bytes(t, 1001, out) - mko::estimate_device_bytes(t, 1000, out),
                mko::max_variants(16384 * MiB, t, out), mko::max_variants(24576 * MiB, t, out));
}
