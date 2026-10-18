// Wellenfrontauswertung auf der CPU: Referenzsphäre, Kolbenabzug, RMS pro Feld und Wellenlänge.

#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/cpu/cpu_trace.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/trace/reference_sphere.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

void check_table(const SystemDescription& system, const HostSurfaceTable& table) {
    if (table.surfaces != static_cast<int>(system.surfaces.size()) ||
        table.wavelengths != static_cast<int>(system.spectrum.wavelengths_nm.size()) ||
        table.thickness.size() != static_cast<std::size_t>(table.surfaces) || table.surfaces < 2) {
        throw std::invalid_argument(
            "Wellenfront: Flächentabelle passt nicht zur Systembeschreibung");
    }
}

} // namespace

std::size_t WavefrontResult::failed_count() const {
    std::size_t n = 0;
    for (const WavefrontMap& m : maps) {
        n += static_cast<std::size_t>(m.failed);
    }
    return n;
}

double target_focal_length(const SystemDescription& system, const GlassCatalog& catalog) {
    double f = 0.0;
    if (system.target && system.target->focal_length) {
        f = *system.target->focal_length;
    } else {
        f = compute_paraxial(system, catalog).focal_length;
    }
    if (!std::isfinite(f) || f == 0.0) {
        throw std::invalid_argument("Wellenfront: Sollbrennweite ist nicht endlich (afokales "
                                    "System?); target.focal_length setzen");
    }
    return f;
}

std::vector<Vec3<double>> ideal_image_points(const SystemDescription& system,
                                             const HostSurfaceTable& table, double focal_length) {
    check_table(system, table);
    const double w = table.thickness[static_cast<std::size_t>(table.surfaces - 1)];
    std::vector<Vec3<double>> points;
    if (system.source.type != SourceType::Collimated) {
        points.push_back({0.0, 0.0, w});
        return points;
    }
    if (!std::isfinite(focal_length)) {
        throw std::invalid_argument("Wellenfront: Brennweite für die Bildhöhen nicht endlich");
    }
    points.reserve(system.source.field_angles_deg.size());
    for (const double deg : system.source.field_angles_deg) {
        const double theta = deg * (std::numbers::pi / 180.0);
        points.push_back({0.0, focal_length * std::tan(theta), w});
    }
    return points;
}

PupilMode wavefront_pupil_mode(const SystemDescription& system, int field,
                               const WavefrontOptions& options) {
    if (field < 0 || field >= source_field_count(system)) {
        throw std::out_of_range("Wellenfront: Feld " + std::to_string(field) + " existiert nicht");
    }
    if (options.pupil_mode) {
        return *options.pupil_mode;
    }
    if (system.source.type == SourceType::Collimated) {
        return system.source.field_angles_deg[static_cast<std::size_t>(field)] == 0.0
                   ? PupilMode::Axis
                   : PupilMode::Half;
    }
    const auto& q = system.source.point;
    if (q[0] == 0.0) {
        return q[1] == 0.0 ? PupilMode::Axis : PupilMode::Half;
    }
    return PupilMode::Full;
}

WavefrontResult evaluate_wavefront(const SystemDescription& system, const HostSurfaceTable& table,
                                   std::span<const Vec3<double>> image_points,
                                   const WavefrontOptions& options) {
    check_table(system, table);
    const int fields = source_field_count(system);
    if (image_points.size() != static_cast<std::size_t>(fields)) {
        throw std::invalid_argument("Wellenfront: " + std::to_string(image_points.size()) +
                                    " Bildpunkte für " + std::to_string(fields) + " Felder");
    }
    const double lambda_ref_mm = system.spectrum.reference_nm * 1e-6;
    if (!(lambda_ref_mm > 0.0)) {
        throw std::invalid_argument("Wellenfront: Referenzwellenlänge muss > 0 sein");
    }

    WavefrontResult res;
    res.fields = fields;
    res.wavelengths = table.wavelengths;
    res.reference_nm = system.spectrum.reference_nm;
    res.maps.reserve(static_cast<std::size_t>(fields) *
                     static_cast<std::size_t>(table.wavelengths));

    const SurfaceTableView<double> view = table.view();
    std::vector<double> weight;
    for (int f = 0; f < fields; ++f) {
        const auto quad = make_pupil_quadrature(options.rings, options.spokes,
                                                wavefront_pupil_mode(system, f, options));
        const int n = sample_count(quad);
        const Vec3<double> center = image_points[static_cast<std::size_t>(f)];
        for (int l = 0; l < table.wavelengths; ++l) {
            WavefrontMap m;
            m.field = f;
            m.wavelength = l;
            m.wavelength_nm = system.spectrum.wavelengths_nm[static_cast<std::size_t>(l)];
            m.image_point = center;
            m.quadrature = quad;
            const auto nu = static_cast<std::size_t>(n);
            m.pupil.resize(nu);
            m.opl.resize(nu);
            m.opd.resize(nu);
            m.opd_waves.resize(nu);
            m.status.resize(nu);
            m.failed_surface.resize(nu);
            weight.resize(nu);
            for (std::size_t k = 0; k < nu; ++k) {
                m.pupil[k] = sample(quad, static_cast<int>(k));
                weight[k] = m.pupil[k].weight;
                Ray<double> ray = source_ray(system, table, f, l, m.pupil[k]);
                int failed = -1;
                m.status[k] = trace_to_reference_sphere(view, l, ray, center, &failed);
                m.failed_surface[k] = failed;
                m.opl[k] = ray.opl;
            }
            const OpdStatistics<double> st =
                opd_statistics(m.opl.data(), weight.data(), m.status.data(), n, m.opd.data());
            m.failed = st.failed;
            m.valid_weight = st.valid_weight;
            if (m.valid()) {
                m.piston = st.piston;
                m.rms = st.rms;
                m.rms_waves = st.rms / lambda_ref_mm;
            }
            for (std::size_t k = 0; k < nu; ++k) {
                if (m.status[k] == TraceStatus::Ok) {
                    m.opd_waves[k] = m.opd[k] / lambda_ref_mm;
                } else {
                    m.opl[k] = WavefrontMap::kNaN;
                    m.opd[k] = WavefrontMap::kNaN;
                    m.opd_waves[k] = WavefrontMap::kNaN;
                }
            }
            res.maps.push_back(std::move(m));
        }
    }
    return res;
}

WavefrontResult evaluate_wavefront(const LoadedSystem& system, const WavefrontOptions& options) {
    double f = WavefrontMap::kNaN;
    if (system.system.source.type == SourceType::Collimated) {
        f = target_focal_length(system.system, system.catalog);
    }
    const std::vector<Vec3<double>> points = ideal_image_points(system.system, system.table, f);
    WavefrontResult res = evaluate_wavefront(system.system, system.table, points, options);
    res.focal_length = f;
    return res;
}

} // namespace mko
