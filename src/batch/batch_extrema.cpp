// Max-Reduktionen pro Variante und Fläche: CPU-Referenz der Kernel MERIT und RESIDUALS.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/trace/surface_extrema.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace mko {

double SurfaceExtremaResult::max_incidence_angle(int v, int s) const {
    return std::acos(std::clamp(min_cos_incidence(v, s), -1.0, 1.0));
}

namespace {

// Max-Reduktionen der Variante v über alle gültigen Strahlen; schreibt 2 · S Werte ab `out`.
void variant_extrema(const SystemBatchHost& batch, int v, const ResidualLayout& layout,
                     double* out) {
    const SystemBatchView<double> view = batch.view();
    const SurfaceTableView<double> table = view.surface_table(v);
    const int S = table.surfaces;
    // Je 1 KB, daher auf dem Heap
    const auto acc = std::make_unique<SurfaceExtrema<double>>();
    const auto ray = std::make_unique<RaySurfaceRecord<double>>();
    acc->reset(S);
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const Vec3<double> center = batch_image_point(view, table, v, f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            for (int k = 0; k < sample_count(q); ++k) {
                double opl = 0.0;
                const TraceStatus st =
                    batch_ray_opl(view, table, l, v, f, l, sample(q, k), center, &opl, ray.get());
                if (st == TraceStatus::Ok) {
                    acc->add_ray(*ray, S);
                }
            }
        }
    }
    for (int s = 0; s < S; ++s) {
        out[2 * s] = std::sqrt(acc->max_height_sq[s]);
        out[2 * s + 1] = acc->min_cos_incidence[s];
    }
}

SurfaceExtremaResult make_result(const SystemBatchHost& batch, int variants) {
    SurfaceExtremaResult r;
    r.variants = variants;
    r.surfaces = batch.surfaces();
    r.values.assign(static_cast<std::size_t>(variants) * static_cast<std::size_t>(r.surfaces) * 2,
                    0.0);
    return r;
}

} // namespace

SurfaceExtremaResult surface_extrema_cpu(const SystemBatchHost& batch, int v,
                                         const ResidualLayout& layout) {
    check_residual_variant(batch, v, layout);
    SurfaceExtremaResult r = make_result(batch, 1);
    variant_extrema(batch, v, layout, r.values.data());
    return r;
}

SurfaceExtremaResult surface_extrema_cpu(const SystemBatchHost& batch,
                                         const ResidualLayout& layout) {
    for (int v = 0; v < batch.variants(); ++v) {
        check_residual_variant(batch, v, layout);
    }
    SurfaceExtremaResult r = make_result(batch, batch.variants());
    for (int v = 0; v < batch.variants(); ++v) {
        variant_extrema(batch, v, layout, r.values.data() + r.index(v, 0));
    }
    return r;
}

} // namespace mko
