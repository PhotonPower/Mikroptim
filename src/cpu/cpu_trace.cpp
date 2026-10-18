// CPU-Referenztracer: Schleife über Felder × Wellenlängen × Pupillenstützstellen.

#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/trace/trace.hpp"

#include <numbers>
#include <stdexcept>
#include <string>

namespace mko {

std::size_t CpuTraceResult::failed_count() const {
    std::size_t n = 0;
    for (const TraceStatus s : status) {
        if (s != TraceStatus::Ok) {
            ++n;
        }
    }
    return n;
}

int source_field_count(const SystemDescription& system) {
    return system.source.type == SourceType::Collimated
               ? static_cast<int>(system.source.field_angles_deg.size())
               : 1;
}

Ray<double> source_ray(const SystemDescription& system, const HostSurfaceTable& table, int field,
                       int wavelength, const PupilSample<double>& sample) {
    if (field < 0 || field >= source_field_count(system)) {
        throw std::out_of_range("Startstrahl: Feld " + std::to_string(field) + " existiert nicht");
    }
    if (wavelength < 0 || wavelength >= table.wavelengths) {
        throw std::out_of_range("Startstrahl: Wellenlänge " + std::to_string(wavelength) +
                                " existiert nicht");
    }
    if (system.source.type == SourceType::Collimated) {
        const double theta = system.source.field_angles_deg[static_cast<std::size_t>(field)] *
                             (std::numbers::pi / 180.0);
        return collimated_ray(table.semi_diameter.at(0), theta, sample.rho, sample.phi);
    }
    // Medium nach Fläche 0, Zeile 0 der Brechzahltabelle
    const double n_source = table.index.at(static_cast<std::size_t>(wavelength));
    const double sin_max = system.source.numerical_aperture / n_source;
    if (!(sin_max > 0.0 && sin_max < 1.0)) {
        throw std::invalid_argument("Startstrahl: NA der Punktquelle muss in (0, n) liegen");
    }
    const auto& q = system.source.point;
    return point_source_ray(Vec3<double>{q[0], q[1], q[2]}, sin_max, sample.rho, sample.phi);
}

CpuTraceResult trace_system_cpu(const SystemDescription& system, const HostSurfaceTable& table,
                                const PupilQuadrature<double>& pupil) {
    if (table.surfaces != static_cast<int>(system.surfaces.size()) ||
        table.wavelengths != static_cast<int>(system.spectrum.wavelengths_nm.size())) {
        throw std::invalid_argument(
            "CPU-Tracer: Flächentabelle passt nicht zur Systembeschreibung");
    }
    CpuTraceResult r;
    r.fields = source_field_count(system);
    r.wavelengths = table.wavelengths;
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

    const SurfaceTableView<double> view = table.view();
    for (int f = 0; f < r.fields; ++f) {
        for (int l = 0; l < r.wavelengths; ++l) {
            for (int k = 0; k < r.samples; ++k) {
                const std::size_t i = r.at(f, l, k);
                Ray<double> ray =
                    source_ray(system, table, f, l, r.pupil[static_cast<std::size_t>(k)]);
                int failed = -1;
                r.status[i] = trace_ray(view, l, ray, nullptr, &failed);
                r.failed_surface[i] = failed;
                r.rays[i] = ray;
            }
        }
    }
    return r;
}

} // namespace mko
