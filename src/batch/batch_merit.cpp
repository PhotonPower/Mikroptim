// Merit pro Variante: CPU-Referenz des Kernels MERIT.

#include "mikroptim/batch/batch_merit.hpp"

namespace mko {

namespace {

// Merit und versagte Strahlen der Variante v aus den CPU-Residuen (Reihenfolge siehe Header).
void variant_merit(const SystemBatchHost& batch, int v, const ResidualLayout& layout, double& merit,
                   std::int64_t& failed, std::int64_t& hard_failed) {
    const ResidualResult r = residuals_cpu(batch, v, layout); // prüft Variante und Layout
    const bool soft_mode = layout.soft_failure();
    double sum = 0.0;
    for (int f = 0; f < layout.fields(); ++f) {
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const std::int64_t o = layout.offset(f, l);
            double tuple = 0.0;
            for (int k = 0; k < layout.samples(f); ++k) {
                const double x = r.residual(0, o + k);
                tuple += x * x;
            }
            if (soft_mode) {
                double pen = 0.0;
                for (int k = 0; k < layout.samples(f); ++k) {
                    const double p = r.penalty(0, o + k);
                    pen += p * p;
                }
                tuple += pen;
            }
            sum += tuple;
        }
    }
    merit = sum;
    failed = static_cast<std::int64_t>(r.failed_count());
    hard_failed = static_cast<std::int64_t>(r.hard_failed_count());
}

MeritResult make_result(int variants, const ResidualLayout& layout) {
    MeritResult r;
    r.variants = variants;
    r.rays_per_variant = layout.rays_per_variant();
    r.merit.assign(static_cast<std::size_t>(variants), 0.0);
    r.failed.assign(static_cast<std::size_t>(variants), 0);
    r.hard_failed.assign(static_cast<std::size_t>(variants), 0);
    return r;
}

} // namespace

MeritResult merit_cpu(const SystemBatchHost& batch, int v, const ResidualLayout& layout,
                      bool extrema) {
    MeritResult r = make_result(1, layout);
    variant_merit(batch, v, layout, r.merit[0], r.failed[0], r.hard_failed[0]);
    if (extrema) {
        r.extrema = surface_extrema_cpu(batch, v, layout);
    }
    return r;
}

MeritResult merit_cpu(const SystemBatchHost& batch, const ResidualLayout& layout, bool extrema) {
    MeritResult r = make_result(batch.variants(), layout);
    for (int v = 0; v < batch.variants(); ++v) {
        const auto i = static_cast<std::size_t>(v);
        variant_merit(batch, v, layout, r.merit[i], r.failed[i], r.hard_failed[i]);
    }
    if (extrema) {
        r.extrema = surface_extrema_cpu(batch, layout);
    }
    return r;
}

} // namespace mko
