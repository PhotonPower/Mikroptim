// CPU-Referenztrace einer Variante eines SystemBatch.

#include "mikroptim/batch/batch_trace_cpu.hpp"
#include "mikroptim/trace/trace.hpp"

#include <stdexcept>
#include <string>

namespace mko {

CpuTraceResult trace_variant_cpu(const SystemBatchHost& batch, int v,
                                 const PupilQuadrature<double>& pupil) {
    if (v < 0 || v >= batch.variants()) {
        throw std::out_of_range("trace_variant_cpu: Variante " + std::to_string(v) +
                                " existiert nicht");
    }
    const SystemBatchView<double> view = batch.view();
    if (batch.source() == SourceKind::Point) {
        // Gleiche Prüfung wie source_ray (cpu_trace.cpp): sin_max = NA / n in (0, 1)
        for (int l = 0; l < batch.wavelengths(); ++l) {
            const double sin_max = batch.source_na / batch.index(v, 0, l);
            if (!(sin_max > 0.0 && sin_max < 1.0)) {
                throw std::invalid_argument(
                    "trace_variant_cpu: NA der Punktquelle muss in (0, n) liegen");
            }
        }
    }
    CpuTraceResult r;
    r.fields = batch.fields();
    r.wavelengths = batch.wavelengths();
    r.samples = sample_count(pupil);
    r.pupil.reserve(static_cast<std::size_t>(r.samples));
    for (int k = 0; k < r.samples; ++k) {
        r.pupil.push_back(sample(pupil, k));
    }
    const std::size_t total = static_cast<std::size_t>(r.fields) *
                              static_cast<std::size_t>(r.wavelengths) *
                              static_cast<std::size_t>(r.samples);
    r.rays.resize(total);
    r.status.resize(total, TraceStatus::Ok);
    r.failed_surface.resize(total, -1);

    const SurfaceTableView<double> table = view.surface_table(v);
    for (int f = 0; f < r.fields; ++f) {
        for (int l = 0; l < r.wavelengths; ++l) {
            for (int k = 0; k < r.samples; ++k) {
                const std::size_t i = r.at(f, l, k);
                Ray<double> ray =
                    batch_source_ray(view, v, f, l, r.pupil[static_cast<std::size_t>(k)]);
                int failed = -1;
                r.status[i] = trace_ray(table, l, ray, nullptr, &failed);
                r.failed_surface[i] = failed;
                r.rays[i] = ray;
            }
        }
    }
    return r;
}

} // namespace mko
