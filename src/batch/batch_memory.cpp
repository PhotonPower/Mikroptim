// Ablage im Gerätepuffer und Speicherabschätzung für SystemBatch und Trace-Ausgaben.

#include "mikroptim/batch/batch_memory.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace mko {

namespace {

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

// Sättigende Arithmetik: Überläufe werden zu SIZE_MAX (passt dann sicher nicht in den Speicher).
std::size_t mul_sat(std::size_t a, std::size_t b) {
    if (a != 0 && b > kSizeMax / a) {
        return kSizeMax;
    }
    return a * b;
}
std::size_t add_sat(std::size_t a, std::size_t b) {
    return b > kSizeMax - a ? kSizeMax : a + b;
}
std::size_t align_sat(std::size_t bytes) {
    return bytes > kSizeMax - kBatchAlignment ? kSizeMax : align_bytes(bytes);
}

std::size_t to_size(std::int64_t n) {
    return static_cast<std::size_t>(n);
}

// Hängt einen Bereich mit `bytes` Nutzbytes an das Ende `end` an.
BufferRegion place(std::size_t& end, std::size_t bytes) {
    const BufferRegion r{end, bytes};
    end = add_sat(end, align_sat(bytes));
    return r;
}

void check_variants(int variants) {
    if (variants < 0) {
        throw std::invalid_argument("Speicherabschätzung: negative Variantenzahl");
    }
}

} // namespace

SystemBatchLayout make_batch_layout(const BatchTopology& t, int variants) {
    validate_topology(t);
    check_variants(variants);
    const std::size_t V = to_size(variants);
    const std::size_t S = to_size(t.surfaces);
    const std::size_t L = to_size(t.wavelengths);
    const std::size_t F = to_size(t.fields);
    const std::size_t G = to_size(t.ghost_plates);
    const std::size_t K = to_size(t.ghost_terms);
    constexpr std::size_t d = sizeof(double);

    SystemBatchLayout b;
    std::size_t end = 0;
    b.type = place(end, S * sizeof(SurfaceKind));
    b.ghost_surface = place(end, G * sizeof(int));
    b.ghost_norm_radius = place(end, G * d);
    b.wavelength_nm = place(end, L * d);
    b.wavelength_weight = place(end, L * d);
    b.field_weight = place(end, F * d);
    b.curvature = place(end, mul_sat(V, S * d));
    b.thickness = place(end, mul_sat(V, S * d));
    b.index = place(end, mul_sat(V, S * L * d));
    b.pupil_radius = place(end, mul_sat(V, d));
    b.field_angle = place(end, mul_sat(V, F * d));
    b.field_target = place(end, mul_sat(V, F * d));
    b.ghost_coeff = place(end, mul_sat(V, G * K * d));
    b.ghost_index = place(end, mul_sat(V, G * L * d));
    b.paraxial_index = place(end, mul_sat(V, S * d));
    b.paraxial_ghost_index = place(end, mul_sat(V, G * d));
    b.total_bytes = end;
    return b;
}

std::int64_t rays_per_variant(int fields, int wavelengths, int samples, int axis_fields,
                              int axis_samples) {
    if (fields < 0 || wavelengths < 0 || samples < 0 || axis_fields < 0 || axis_samples < 0 ||
        axis_fields > fields) {
        throw std::invalid_argument("rays_per_variant: ungültige Größen");
    }
    return static_cast<std::int64_t>(wavelengths) *
           (static_cast<std::int64_t>(axis_fields) * axis_samples +
            static_cast<std::int64_t>(fields - axis_fields) * samples);
}

OutputLayout make_output_layout(const BatchTopology& t, int variants, const OutputRequest& out,
                                std::size_t extra_bytes_per_variant) {
    validate_topology(t);
    check_variants(variants);
    if (out.rays_per_variant < 0 || out.extra_residuals < 0) {
        throw std::invalid_argument("Speicherabschätzung: negative Ausgabegröße");
    }
    const std::size_t V = to_size(variants);
    const std::size_t S = to_size(t.surfaces);
    const std::size_t blocks = to_size(t.fields) * to_size(t.wavelengths); // pro Variante
    const std::size_t rays = to_size(out.rays_per_variant);
    const std::size_t residuals = add_sat(rays, to_size(out.extra_residuals));
    constexpr std::size_t d = sizeof(double);

    OutputLayout o;
    std::size_t end = 0;
    o.residuals = place(end, out.residuals ? mul_sat(mul_sat(V, residuals), d) : 0);
    o.ray_status = place(end, out.ray_status ? mul_sat(V, rays) : 0);
    o.opd = place(end, out.opd ? mul_sat(mul_sat(V, rays), d) : 0);
    o.merit = place(end, out.merit ? mul_sat(V, d) : 0);
    o.merit_partial = place(end, out.merit ? mul_sat(mul_sat(V, blocks), d) : 0);
    o.failed_rays = place(end, out.merit ? mul_sat(V, sizeof(std::int64_t)) : 0);
    o.failed_rays_partial =
        place(end, out.merit ? mul_sat(mul_sat(V, blocks), sizeof(std::int32_t)) : 0);
    o.surface_max = place(end, out.surface_max ? mul_sat(V, 2 * S * d) : 0);
    o.surface_max_partial =
        place(end, out.surface_max ? mul_sat(mul_sat(V, blocks), 2 * S * d) : 0);
    o.extra = place(end, mul_sat(V, extra_bytes_per_variant));
    o.total_bytes = end;
    return o;
}

std::size_t estimate_device_bytes(const BatchTopology& t, int variants, const OutputRequest& out,
                                  std::size_t extra_bytes_per_variant) {
    return add_sat(make_batch_layout(t, variants).total_bytes,
                   make_output_layout(t, variants, out, extra_bytes_per_variant).total_bytes);
}

std::size_t usable_bytes(std::size_t free_bytes, const MemoryReserve& reserve) {
    if (!(reserve.fraction >= 0.0 && reserve.fraction <= 1.0)) {
        throw std::invalid_argument("MemoryReserve: fraction muss in [0, 1] liegen");
    }
    const auto proportional =
        static_cast<std::size_t>(static_cast<double>(free_bytes) * reserve.fraction);
    const std::size_t keep = std::max(reserve.min_bytes, proportional);
    return free_bytes > keep ? free_bytes - keep : 0;
}

int max_variants(std::size_t free_bytes, const BatchTopology& t, const OutputRequest& out,
                 const MemoryReserve& reserve, std::size_t extra_bytes_per_variant) {
    validate_topology(t);
    const std::size_t usable = usable_bytes(free_bytes, reserve);
    const auto fits = [&](int v) {
        return estimate_device_bytes(t, v, out, extra_bytes_per_variant) <= usable;
    };
    const std::int64_t blocks = static_cast<std::int64_t>(t.fields) * t.wavelengths;
    const std::int64_t grid_limit = blocks > 0 ? kMaxGridBlocks / blocks : 0;
    int hi = static_cast<int>(std::min<std::int64_t>(INT_MAX, grid_limit));
    if (hi < 1 || !fits(1)) {
        return 0;
    }
    if (fits(hi)) {
        return hi;
    }
    // Invariante: fits(lo), !fits(hi); estimate_device_bytes ist monoton in V.
    int lo = 1;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (fits(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

} // namespace mko
