#pragma once

// Werkzeuge für die Tests der Aufzeichnung des Vorwärtsdurchlaufs (M5-T1, CPU und GPU):
// Strahlen eines Batches in der Ablage von ResidualLayout, CPU-Referenz der Aufzeichnung pro
// Strahl (AdjointRecord über batch_ray_opl) in der Ablage des Downloads von
// mko::cuda::DeviceAdjointForward und die Prüfsumme der Aufzeichnung auf dem Host.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace mko::test {

// Ein Strahl eines Batches: Flächentabelle der Variante (alle Wellenlängen), Wellenlänge,
// Startstrahl, Bildpunkt und Index in der Ablage [v · rays + offset(f, l) + k].
struct BatchRay {
    int v = 0;
    int f = 0;
    int l = 0;
    int k = 0;
    std::size_t index = 0;
    SurfaceTableView<double> table{};
    Ray<double> start{};
    Vec3<double> center{};
};

// Ruft fn(ray) für alle Strahlen aller Varianten auf (Reihenfolge der Ablage).
inline void for_each_batch_ray(const SystemBatchHost& batch, const ResidualLayout& layout,
                               const std::function<void(const BatchRay&)>& fn) {
    const SystemBatchView<double> view = batch.view();
    for (int v = 0; v < batch.variants(); ++v) {
        const SurfaceTableView<double> table = view.surface_table(v);
        for (int f = 0; f < layout.fields(); ++f) {
            const PupilQuadrature<double>& q = layout.quadrature(f);
            const Vec3<double> center = batch_image_point(view, table, v, f);
            for (int l = 0; l < layout.wavelengths(); ++l) {
                for (int k = 0; k < layout.samples(f); ++k) {
                    BatchRay r;
                    r.v = v;
                    r.f = f;
                    r.l = l;
                    r.k = k;
                    r.index = static_cast<std::size_t>(v) *
                                  static_cast<std::size_t>(layout.rays_per_variant()) +
                              static_cast<std::size_t>(layout.offset(f, l) + k);
                    r.table = table;
                    r.start = batch_source_ray(view, v, f, l, sample(q, k));
                    r.center = center;
                    fn(r);
                }
            }
        }
    }
}

// Aufzeichnung aller Strahlen eines Batches (CPU-Referenz), Ablage wie AdjointForwardResult.
struct CpuRecords {
    int surfaces = 0;
    std::vector<std::uint8_t> status;
    std::vector<double> opl;
    std::vector<std::int32_t> count;
    std::vector<double> states; // [ray][S][6]
    std::vector<std::uint8_t> branch;
    std::vector<double> image_s;
    std::vector<std::uint8_t> image_branch;
};

// CPU-Referenz: pro Strahl trace_to_reference_sphere mit AdjointRecord<double> (weicher Modus
// nach layout.soft_failure()).
inline CpuRecords cpu_records(const SystemBatchHost& batch, const ResidualLayout& layout) {
    CpuRecords c;
    const int S = batch.surfaces();
    c.surfaces = S;
    const std::size_t rays = static_cast<std::size_t>(batch.variants()) *
                             static_cast<std::size_t>(layout.rays_per_variant());
    const auto s_count = static_cast<std::size_t>(S);
    c.status.assign(rays, 0);
    c.opl.assign(rays, 0.0);
    c.count.assign(rays, 0);
    c.states.assign(rays * s_count * 6, 0.0);
    c.branch.assign(rays * s_count, 0);
    c.image_s.assign(rays, 0.0);
    c.image_branch.assign(rays, 0);
    const bool soft_mode = layout.soft_failure();
    const auto rec = std::make_unique<AdjointRecord<double>>();
    for_each_batch_ray(batch, layout, [&](const BatchRay& r) {
        Ray<double> ray = r.start;
        SoftFailure<double> soft{0.0, TraceStatus::Ok};
        NoSurfaceObserver* const none = nullptr;
        const TraceStatus st = trace_to_reference_sphere(r.table, r.l, ray, r.center, nullptr, none,
                                                         soft_mode ? &soft : nullptr, rec.get());
        const std::size_t i = r.index;
        c.status[i] = static_cast<std::uint8_t>(st);
        c.opl[i] = ray.opl;
        c.count[i] = rec->count;
        for (int s = 0; s < rec->count; ++s) {
            const std::size_t o = (i * s_count + static_cast<std::size_t>(s)) * 6;
            const auto su = static_cast<std::size_t>(s);
            c.states[o + 0] = rec->p[su].x;
            c.states[o + 1] = rec->p[su].y;
            c.states[o + 2] = rec->p[su].z;
            c.states[o + 3] = rec->d[su].x;
            c.states[o + 4] = rec->d[su].y;
            c.states[o + 5] = rec->d[su].z;
            c.branch[i * s_count + su] = rec->branch[su];
        }
        c.image_s[i] = rec->image_s;
        c.image_branch[i] = rec->image_branch;
    });
    return c;
}

// Prüfsumme der Aufzeichnung eines Strahls aus der Ablage [S][6] und [S] (wie der Kernel,
// Definition in include/mikroptim/cuda/adjoint.hpp; hier ohne CUDA-Header nachgebildet).
inline std::uint64_t host_checksum_step(std::uint64_t h, std::uint64_t word) {
    h ^= word;
    h *= 0x100000001b3ull;
    h ^= h >> 29;
    return h;
}

inline std::uint64_t host_record_checksum(const double* states, const std::uint8_t* branch,
                                          int count, double image_s, std::uint8_t image_branch) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (int s = count - 1; s >= 0; --s) {
        for (int i = 0; i < 6; ++i) {
            h = host_checksum_step(h, std::bit_cast<std::uint64_t>(states[s * 6 + i]));
        }
        h = host_checksum_step(h, branch[s]);
    }
    h = host_checksum_step(h, std::bit_cast<std::uint64_t>(image_s));
    h = host_checksum_step(h, image_branch);
    h = host_checksum_step(h, static_cast<std::uint64_t>(count));
    return h;
}

} // namespace mko::test
