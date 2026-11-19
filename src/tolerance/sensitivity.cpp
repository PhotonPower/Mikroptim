// Empfindlichkeiten (M10-T3): Störungen ±δ und ±h·δ jeder Zufallsgröße eines Toleranzsatzes in
// einem Batch, Bewertung wie in der Tolerierung, Jacobi-Spalten der Wellenfrontreste
// (docs/optimierung.md, Abschnitt „Empfindlichkeiten“).

#include "mikroptim/tolerance/sensitivity.hpp"

#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/tolerance/tolerance_model.hpp"
#include "mikroptim/trace/ray.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(std::int64_t i) {
    return static_cast<std::size_t>(i);
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

void check_default(double v, const char* name) {
    if (!std::isfinite(v) || v < 0.0) {
        throw std::invalid_argument(std::string("Empfindlichkeit: Standardgrenze ") + name +
                                    " muss endlich und >= 0 sein");
    }
}

ToleranceSpec surface_item(ToleranceType type, std::vector<int> surfaces) {
    ToleranceSpec spec;
    spec.type = type;
    spec.selection = SurfaceSelection::List;
    spec.surfaces = std::move(surfaces);
    return spec;
}

// Bewertung einer Variante: Q und Wellenfrontreste ẽ (Länge residuals).
struct VariantValue {
    double q = 0.0;
    std::int64_t failed = 0;
};

VariantValue evaluate(const ResidualLayout& layout, const SystemBatchHost& batch,
                      const ResidualResult& rr, int v, const std::vector<double>& weights,
                      double lambda_ref_mm, std::span<double> e) {
    VariantValue out;
    const int fields = layout.fields();
    const int wavelengths = layout.wavelengths();
    std::size_t at_e = 0;
    std::vector<double> rest;
    double sum = 0.0;
    for (int f = 0; f < fields; ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const auto n = static_cast<std::size_t>(layout.samples(f));
        rest.assign(n, 0.0);
        for (int l = 0; l < wavelengths; ++l) {
            const std::size_t at = rr.ray_index(v, layout.offset(f, l));
            const std::span<const std::uint8_t> status =
                std::span<const std::uint8_t>(rr.status).subspan(at, n);
            const TupleQuality t = tuple_quality(q, std::span<const double>(rr.opd).subspan(at, n),
                                                 status, batch.wavelength_nm(l) * 1e-6, rest);
            const double w = weights[static_cast<std::size_t>(f * wavelengths + l)];
            const double rms = t.rms / lambda_ref_mm;
            sum += w * (rms * rms);
            out.failed += t.failed;
            // W_ok wie in tuple_quality (aufsteigend über k)
            double w_ok = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                if (static_cast<TraceStatus>(status[k]) == TraceStatus::Ok) {
                    w_ok += sample(q, static_cast<int>(k)).weight;
                }
            }
            for (std::size_t k = 0; k < n; ++k) {
                double r = 0.0;
                if (w_ok > 0.0 && static_cast<TraceStatus>(status[k]) == TraceStatus::Ok) {
                    const double wk = sample(q, static_cast<int>(k)).weight;
                    r = std::sqrt(w * wk / w_ok) * (rest[k] / lambda_ref_mm);
                }
                e[at_e + k] = r;
            }
            at_e += n;
        }
    }
    out.q = sum;
    return out;
}

} // namespace

Tolerances default_sensitivity_tolerances(const SystemDescription& system,
                                          const SensitivityDefaults& d) {
    check_default(d.element_decenter_mm, "element_decenter_mm");
    check_default(d.element_tilt_arcmin, "element_tilt_arcmin");
    check_default(d.radius_relative, "radius_relative");
    check_default(d.thickness_mm, "thickness_mm");
    check_default(d.index, "index");
    if (!(d.radius_relative < 1.0)) {
        throw std::invalid_argument("Empfindlichkeit: radius_relative muss < 1 sein");
    }
    if (!(d.element_tilt_arcmin < 90.0 * 60.0)) {
        throw std::invalid_argument("Empfindlichkeit: Kippwinkel muss < 90° sein");
    }
    if (d.element_pivot == ElementPivot::Offset) {
        throw std::invalid_argument("Empfindlichkeit: Drehpunkt first, center oder last");
    }
    Tolerances t;
    t.distribution = ToleranceDistribution::Uniform;
    t.compensator = CompensatorType::None;
    const int count = static_cast<int>(system.surfaces.size());
    const std::vector<std::array<int, 2>> elements = all_elements(system);
    if (!elements.empty() && d.element_decenter_mm > 0.0) {
        ToleranceSpec spec;
        spec.type = ToleranceType::ElementDecenter;
        spec.element_selection = SurfaceSelection::List;
        spec.elements = elements;
        spec.delta = d.element_decenter_mm;
        t.items.push_back(spec);
    }
    if (!elements.empty() && d.element_tilt_arcmin > 0.0) {
        ToleranceSpec spec;
        spec.type = ToleranceType::ElementTilt;
        spec.element_selection = SurfaceSelection::List;
        spec.elements = elements;
        spec.arcmin = d.element_tilt_arcmin;
        spec.pivot = d.element_pivot;
        t.items.push_back(spec);
    }
    std::vector<int> spheres;
    std::vector<int> gaps;
    std::vector<int> glasses;
    for (int s = 0; s < count; ++s) {
        const Surface& sf = system.surfaces[static_cast<std::size_t>(s)];
        if (sf.type == SurfaceType::Sphere && sf.curvature != 0.0) {
            spheres.push_back(s);
        }
        if (s < count - 1) {
            gaps.push_back(s);
        }
        if (sf.material.kind != MaterialKind::Air) {
            glasses.push_back(s);
        }
    }
    if (!spheres.empty() && d.radius_relative > 0.0) {
        ToleranceSpec spec = surface_item(ToleranceType::Radius, spheres);
        spec.relative = d.radius_relative;
        t.items.push_back(spec);
    }
    if (!gaps.empty() && d.thickness_mm > 0.0) {
        ToleranceSpec spec = surface_item(ToleranceType::Thickness, gaps);
        spec.delta = d.thickness_mm;
        t.items.push_back(spec);
    }
    if (!glasses.empty() && d.index > 0.0) {
        ToleranceSpec spec = surface_item(ToleranceType::Index, glasses);
        spec.delta = d.index;
        t.items.push_back(spec);
    }
    return t;
}

SensitivityResult run_sensitivity_analysis(const SystemDescription& system,
                                           const SystemBatchHost& nominal,
                                           const SensitivitySettings& settings) {
    if (!(settings.step > 0.0 && settings.step <= 1.0)) {
        throw std::invalid_argument("Empfindlichkeit: Schrittweite h muss in (0, 1] liegen");
    }
    if (nominal.variants() != 1) {
        throw std::invalid_argument("Empfindlichkeit: Ausgangsbatch muss genau eine Variante "
                                    "haben");
    }
    const SystemBatchHost base = make_non_coaxial(nominal);
    const ToleranceModel model(system, base.topology());
    ResidualSettings eval = settings.quadrature;
    eval.failure = FailureMode::Hard;
    const ResidualLayout layout(base.topology(), eval);

    SensitivityResult r;
    r.tolerances.assign(model.tolerances().begin(), model.tolerances().end());
    r.fields = base.fields();
    r.wavelengths = base.wavelengths();
    r.reference_nm = base.reference_wavelength_nm;
    r.step = settings.step;
    const int draws = model.draws();
    r.variants = 1 + 4 * draws;
    int residuals = 0;
    for (int f = 0; f < layout.fields(); ++f) {
        residuals += layout.samples(f) * layout.wavelengths();
    }
    r.residuals = residuals;

    // Varianten: 0 nominal, 1 + 4d + {0, 1, 2, 3} für ξ_d = +1, −1, +h, −h
    SystemBatchHost batch;
    batch.assign_variants(base, 0, 1);
    batch.resize(r.variants);
    std::vector<double> xi(static_cast<std::size_t>(draws), 0.0);
    batch.copy_variant_from(base, 0, 0);
    model.apply(batch, 0, xi);
    const double values[4] = {1.0, -1.0, settings.step, -settings.step};
    for (int d = 0; d < draws; ++d) {
        for (int j = 0; j < 4; ++j) {
            const int v = 1 + 4 * d + j;
            xi[static_cast<std::size_t>(d)] = values[j];
            batch.copy_variant_from(base, 0, v);
            model.apply(batch, v, xi);
        }
        xi[static_cast<std::size_t>(d)] = 0.0;
    }

    ResidualResult rr;
    if (settings.trace) {
        settings.trace(batch, layout, rr);
    } else {
        tolerance_trace_cpu(batch, layout, rr, settings.threads);
    }
    const std::size_t need = to_size(layout.rays_per_variant()) * to_size(r.variants);
    if (rr.opd.size() != need || rr.status.size() != need ||
        rr.rays_per_variant != layout.rays_per_variant()) {
        throw std::runtime_error("Empfindlichkeit: Trace lieferte OPD und Status nicht in der "
                                 "erwarteten Größe");
    }

    const std::vector<double> weights = tolerance_tuple_weights(base);
    const double lambda_ref_mm = base.reference_wavelength_nm * 1e-6;
    const auto len = static_cast<std::size_t>(residuals);
    std::vector<double> e0(len);
    const VariantValue nominal_value = evaluate(layout, batch, rr, 0, weights, lambda_ref_mm, e0);
    r.q0 = nominal_value.q;
    r.failed_rays = nominal_value.failed;

    std::vector<double> ep(len);
    std::vector<double> em(len);
    std::vector<double> scratch(len);
    int tolerance = 0;
    int component = 0;
    for (int d = 0; d < draws; ++d) {
        while (tolerance + 1 < static_cast<int>(r.tolerances.size()) &&
               d >= model.draw_offset(tolerance + 1)) {
            ++tolerance;
        }
        component = d - model.draw_offset(tolerance);
        SensitivityComponent c;
        c.tolerance = tolerance;
        c.component = component;
        const int v = 1 + 4 * d;
        const VariantValue plus = evaluate(layout, batch, rr, v, weights, lambda_ref_mm, scratch);
        const VariantValue minus =
            evaluate(layout, batch, rr, v + 1, weights, lambda_ref_mm, scratch);
        const VariantValue hp = evaluate(layout, batch, rr, v + 2, weights, lambda_ref_mm, ep);
        const VariantValue hm = evaluate(layout, batch, rr, v + 3, weights, lambda_ref_mm, em);
        c.q_plus = plus.q;
        c.q_minus = minus.q;
        c.linear = 0.5 * (plus.q - minus.q);
        c.quadratic = 0.5 * ((plus.q + minus.q) - 2.0 * r.q0);
        c.failed_rays = plus.failed + minus.failed + hp.failed + hm.failed;
        // j = (ẽ(+h) − ẽ(−h)) / (2h); g = 2 ẽ₀ᵀ j, ‖j‖² (Summen aufsteigend über k)
        const double inv = 1.0 / (2.0 * settings.step);
        double g = 0.0;
        double jj = 0.0;
        for (std::size_t k = 0; k < len; ++k) {
            const double j = (ep[k] - em[k]) * inv;
            g += e0[k] * j;
            jj += j * j;
        }
        c.jacobian_linear = 2.0 * g;
        c.jacobian_quadratic = jj;
        if (!std::isfinite(hp.q) || !std::isfinite(hm.q) || !std::isfinite(r.q0)) {
            c.jacobian_linear = kNaN;
            c.jacobian_quadratic = kNaN;
        }
        r.components.push_back(c);
    }
    return r;
}

std::vector<SensitivitySummary> summarize_sensitivity(const SensitivityResult& result) {
    std::vector<SensitivitySummary> out(result.tolerances.size());
    std::vector<int> count(result.tolerances.size(), 0);
    for (std::size_t k = 0; k < out.size(); ++k) {
        out[k].tolerance = static_cast<int>(k);
        out[k].worst_delta_q = -std::numeric_limits<double>::infinity();
    }
    for (const SensitivityComponent& c : result.components) {
        SensitivitySummary& s = out[static_cast<std::size_t>(c.tolerance)];
        const double dp = c.q_plus - result.q0;
        const double dm = c.q_minus - result.q0;
        if (std::isnan(dp) || std::isnan(dm) || std::isnan(s.worst_delta_q)) {
            s.worst_delta_q = kNaN;
        } else {
            s.worst_delta_q = std::max({s.worst_delta_q, dp, dm});
        }
        s.mean_quadratic += c.quadratic;
        s.mean_jacobian_quadratic += c.jacobian_quadratic;
        s.failed_rays += c.failed_rays;
        ++count[static_cast<std::size_t>(c.tolerance)];
    }
    for (std::size_t k = 0; k < out.size(); ++k) {
        SensitivitySummary& s = out[k];
        if (count[k] > 0) {
            s.mean_quadratic /= count[k];
            s.mean_jacobian_quadratic /= count[k];
        }
        s.worst_rms = std::sqrt(result.q0 + s.worst_delta_q);
    }
    return out;
}

} // namespace mko
